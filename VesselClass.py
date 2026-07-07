import os
import json
import numpy as np
import pandas as pd
import copy
from utils.vessel_io import N_BAY, STSE_BAY_PAIRS, BAYPLAN_COLUMNS
from utils.vessel_io import _BIG_BAY_OF_B0


class Vessel:
    """
    CSP配载规划的数据层。
    - 静态几何层：从full_slot_table派生一次，之后只读
    - 动态状态层：self.cell，每个(bay,lr,hd)存一条记录{"POD":.., "POL":.., "GP_count":.., "RF_count":..}，
      是cell颗粒度的"配载单"。一个cell只对应一个POD，但内部可以同时装这个POD的
      GP和RF两部分箱量（RF额度优先用满这个cell真实的reefer插座数capacity_rf，
      剩余容量再装同一个POD的GP箱），不再是"整个cell要么GP要么RF"的互斥选择。
      搜索过程中读写，支持回溯。

    坐标系：
        bay  : 0-base，只含有效大箱bay（偶数idx）
        lr   : 0=left(row 0-4), 1=right(row 5-10)
        hd   : 0=hold(tier 0-3), 1=deck(tier 4-9)
    """

    _EMPTY_RECORD = {"POD": -1, "POL": -1, "GP_count": 0, "RF_count": 0}
    # 未赋值cell的记录模板。GP_count/RF_count=实际装的箱量（<=各自capacity，never超装）。
    # 注意：每个cell必须持有独立的dict实例，
    # 不能用np.full(shape, {...})批量填充——那样所有cell会共享同一个dict对象，
    # 改一个牵动全部。__init__里逐个构造。

    def __init__(self, full_slot_table: pd.DataFrame, cbf: dict, current_pol: int = 0, tail_threshold: int = 5):
        if full_slot_table is None:
            raise ValueError("full_slot_table 不能为空")
        if cbf is None:
            raise ValueError("cbf 不能为空")

        # ── 静态几何层（只读，从full_slot_table派生一次） ──────────────────

        self.full_slot_table = full_slot_table
        # 供proj_cell_to_vessel展开回slot级坐标用

        is_valid, capacity_total, capacity_rf = self.proj_vessel_to_cell(full_slot_table)

        self.is_valid = is_valid.astype(bool)
        # is_valid[bay][lr][hd]: 该cell是否物理可用（capacity_total > 0）

        self.capacity_total = capacity_total.astype(int)
        # capacity_total[bay][lr][hd]: 该cell内有效40ft槽位总数
        # 用于GP类型赋值时的cbf扣减量

        self.capacity_rf = capacity_rf.astype(int)
        # capacity_rf[bay][lr][hd]: 该cell内有RF插座的槽位数
        # 用于RF类型赋值时的cbf扣减量

        self.has_reefer = self.capacity_rf > 0
        # has_reefer[bay][lr][hd]: bool，由capacity_rf推导，方便候选集过滤
        # has_reefer=True的cell候选集包含(POD, "RF")对, =False的cell只允许(POD, "GP")对

        self.n_bay = self.is_valid.shape[0]
        # 搜索空间的bay数量，测试时按传入的full_slot_table定，STSE时7

        # ── 动态状态层（搜索变量） ─────────────────────────────────────

        self.cell = np.empty((self.n_bay, 2, 2), dtype=object)
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    self.cell[bay, lr, hd] = dict(self._EMPTY_RECORD)
        # cell[bay][lr][hd] = {"POD": pod或-1, "type": "GP"/"RF"/None, "POL": 装货港或-1}
        # POD=-1表示有效但未赋值；POL是assign()时记录的真实current_pol，
        # 不等同于所属快照的港口——同一份snapshot里可能混有更早港口装、还未卸的货

        self.cbf = cbf
        # 全航次cbf，格式：{POL: {POD: {"GP": count, "RF": count}}}
        # 内部通过current_pol指针取当前港口的切片，不在换港时替换整个dict
        
        all_ports = set(self.cbf.keys())
        for pod_counts in self.cbf.values():
            all_ports.update(pod_counts.keys())
        self.port_min = min(all_ports)
        self.port_max = max(all_ports)
        self.n_ports = self.port_max - self.port_min + 1

        self.current_pol = current_pol
        # 当前装载港口编号，指向cbf的第一层key
        # 换港时只更新这个指针，cbf本身不变
        
        self.tail_threshold = tail_threshold
        # 单个POD剩余需求 <= 此阈值时，视为"尾货"，不再主动占用cell，
        # 留到求解结束后统一打印，交给后续人工/专门策略处理

    # ── 构造 ───────────────────────────────────────────────────────────

    @classmethod
    def load_vessel(cls, geometry_dir: str, cbf_json_path: str, current_pol: int = None) -> "Vessel":
        """
        从真实船型数据构造Vessel。
        geometry_dir: 含 full_slot_table.csv 的目录
                      （vessel_io.build_vessel_geometry + find_can_40ft/20ft/reefer 之后落盘的产物，
                       已含 bay_idx/row_idx/tier_idx/lr/hd/can_40ft/can_20ft/can_reefer 列）
        cbf_json_path: cbf.json路径（vessel_io.batch_parse_cbf产出，json的key是字符串，这里转回int）
        current_pol: 不传则用cbf里最小的POL作为起始港口（cbf本身决定航次从哪个港开始，
                     不应该由调用方硬编码猜测，避免像POL从1而非0开始时KeyError）
        """
        slots = pd.read_csv(os.path.join(geometry_dir, "full_slot_table.csv"))

        with open(cbf_json_path) as f:
            raw_cbf = json.load(f)
        cbf = {
            int(pol): {int(pod): counts for pod, counts in pod_counts.items()}
            for pol, pod_counts in raw_cbf.items()
        }

        if current_pol is None:
            current_pol = min(cbf.keys())

        return cls(full_slot_table=slots, cbf=cbf, current_pol=current_pol)

    # ── 查询方法 ───────────────────────────────────────────────────────

    def get_candidates(self, bay, lr, hd) -> set:
        """
        返回候选POD集合（不再区分type——一个POD候选意味着这个cell可以同时
        用它的GP和/或RF箱量，具体内部怎么拆在assign()里决定）。
        四层过滤：舱盖物理限制（hold在deck下方，deck一旦有货hold就无法再装，
        deck方向没有对称限制）、不能翻箱（按相对current_pol的挂靠距离比较，
        允许环线绕圈）、这个POD在这里至少有一种箱量能放（GP需求超过尾货阈值，
        或有reefer能力的RF）、有cbf余量。
        """
        if not self.is_valid[bay, lr, hd]:
            return set()

        current_cbf = self.cbf[self.current_pol]  # {POD: {"GP": n, "RF": n}}

        other_hd = 1 - hd
        other_pod = self.cell[bay, lr, other_hd]["POD"]

        # 舱盖硬约束：hold(hd=0)物理上被deck(hd=1)的舱盖盖住，deck一旦有货
        # （不管是本港刚装的还是更早港口还没卸的），hold就无法再装任何新货，
        # 除非deck先清空——这不是no-overstow的排序问题，是舱盖结构本身的
        # 物理限制，deck方向没有对称限制（hold空或不空，都不妨碍往deck装货）。
        if hd == 0 and other_pod != -1:
            return set()

        def rel_rank(pod):
            """相对current_pol的挂靠距离，允许绕圈（用port_min把港口范围平移到0起点）。"""
            c = (self.current_pol - self.port_min) % self.n_ports
            p = (pod - self.port_min) % self.n_ports
            return (p - c) if p >= c else (p - c + self.n_ports)

        # 走到这里，要么other_pod==-1，要么hd==1且hold已有货——
        # 此时deck候选必须比hold的货早卸（距离更小）
        other_rank = rel_rank(other_pod) if other_pod != -1 else None

        candidates = set()
        for pod, counts in current_cbf.items():
            if other_rank is not None:
                new_rank = rel_rank(pod)
                if new_rank > other_rank:
                    continue

            has_gp_demand = counts.get("GP", 0) > self.tail_threshold
            has_rf_demand = self.has_reefer[bay, lr, hd] and counts.get("RF", 0) > 0
            if has_gp_demand or has_rf_demand:
                candidates.add(pod)

        return candidates

    def remaining_pods(self) -> set:
        """当前POL中cbf总量>尾货阈值的POD集合（阈值以下的尾货不阻塞港口收尾）。"""
        return {
            pod for pod, counts in self.cbf[self.current_pol].items()
            if counts.get("GP", 0) + counts.get("RF", 0) > self.tail_threshold
        }

    def port_complete(self) -> bool:
        """当前POL的cbf是否全部分配完毕。"""
        return len(self.remaining_pods()) == 0

    def total_remaining(self) -> int:
        """当前POL的cbf剩余总箱量。"""
        return sum(
            c.get("GP", 0) + c.get("RF", 0)
            for c in self.cbf[self.current_pol].values()
        )

    # ── 赋值与撤销 ─────────────────────────────────────────────────────

    def assign(self, bay, lr, hd, pod):
        """
        赋值cell给pod，记录装货港current_pol。
        内部先用这个cell真实的reefer插座数(capacity_rf)优先满足pod的RF需求
        （取min(capacity_rf, 剩余RF需求)，不超装、不多分），
        剩余容量(capacity_total - 实际用掉的RF)接着满足同一个pod的GP需求。
        一个cell只对应一个pod，但GP/RF两部分都可能同时非零。
        实际用掉的GP_count/RF_count都记在cell记录里，unassign时精确按这两个数值加回。
        """
        cap_total = self.capacity_total[bay, lr, hd]
        cap_rf = self.capacity_rf[bay, lr, hd]

        rf_remaining = self.cbf[self.current_pol][pod].get("RF", 0)
        rf_used = min(cap_rf, rf_remaining)

        gp_remaining = self.cbf[self.current_pol][pod].get("GP", 0)
        gp_capacity = cap_total - rf_used
        gp_used = min(gp_capacity, gp_remaining)

        self.cell[bay, lr, hd] = {
            "POD": pod, "POL": self.current_pol,
            "GP_count": gp_used, "RF_count": rf_used,
        }
        self.cbf[self.current_pol][pod]["RF"] = rf_remaining - rf_used
        self.cbf[self.current_pol][pod]["GP"] = gp_remaining - gp_used

        # print(f"[assign] POL={self.current_pol} POD={pod} cell=({bay},{lr},{hd}) "
        #       f"装GP={gp_used} 装RF={rf_used}  →  剩余cbf[POD={pod}]="
        #       f"{self.cbf[self.current_pol][pod]}")
        
    def unassign(self, bay, lr, hd, pod):
        """撤销赋值，把cell记录里实际用掉的GP_count/RF_count分别加回cbf，精确恢复。"""
        record = self.cell[bay, lr, hd]
        self.cbf[self.current_pol][pod]["RF"] = self.cbf[self.current_pol][pod].get("RF", 0) + record["RF_count"]
        self.cbf[self.current_pol][pod]["GP"] = self.cbf[self.current_pol][pod].get("GP", 0) + record["GP_count"]
        self.cell[bay, lr, hd] = dict(self._EMPTY_RECORD)

    # ── 多港口 ─────────────────────────────────────────────────────────

    def discharge(self, arriving_pod) -> list:
        """卸载arriving_pod的所有cell，返回记录供undischarge回溯。"""
        discharged = []
        total_gp, total_rf = 0, 0
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    if self.cell[bay, lr, hd]["POD"] == arriving_pod:
                        record = self.cell[bay, lr, hd]
                        total_gp += record["GP_count"]
                        total_rf += record["RF_count"]
                        discharged.append((bay, lr, hd, dict(record)))
                        self.cell[bay, lr, hd] = dict(self._EMPTY_RECORD)
        # print(f"[discharge] POD={arriving_pod} 到港：卸了{len(discharged)}个cell，"
        #       f"共GP={total_gp} RF={total_rf}")
        return discharged

    def undischarge(self, discharged: list):
        """精确恢复discharge的cell，不动cbf。"""
        for bay, lr, hd, record in discharged:
            self.cell[bay, lr, hd] = dict(record)

    def advance_pol(self):
        """换港：current_pol指针+1。"""
        self.current_pol += 1

    # ── 快照 ───────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """返回动态状态的深拷贝，用于跨港回溯或记录departure状态。"""
        return {
            "cell":        copy.deepcopy(self.cell),
            "cbf":         copy.deepcopy(self.cbf),
            "current_pol": self.current_pol,
        }

    def restore(self, snap: dict):
        """从snapshot恢复动态状态。"""
        self.cell        = copy.deepcopy(snap["cell"])
        self.cbf         = snap["cbf"]
        self.current_pol = snap["current_pol"]

    # ── 导出 ───────────────────────────────────────────────────────────

    def export_cell_state(self) -> dict:
        """导出已赋值cell的状态，供viz展开到slot层面。
        返回 {(bay, lr, hd): {"POD":.., "POL":.., "GP_count":.., "RF_count":..}}，只含POD != -1的cell。"""
        result = {}
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    record = self.cell[bay, lr, hd]
                    if record["POD"] != -1:
                        result[(bay, lr, hd)] = dict(record)
        return result

    @staticmethod
    def build_vessel_cell(slots: pd.DataFrame, flag_col: str) -> np.ndarray:
        """把slots按flag_col筛选后，聚合成(N_BAY,2,2)的cell田字格。
        """
        cell = np.zeros((N_BAY, 2, 2), dtype=int)
        hits = slots[slots[flag_col]]
        for bay_idx, lr, hd in zip(hits.bay_idx, hits.lr, hits.hd):
            big_bay = _BIG_BAY_OF_B0.get(bay_idx)
            if big_bay is None:
                continue
            cell[big_bay, lr, hd] += 1
        return cell

    @staticmethod
    def build_init_state(slots: pd.DataFrame) -> pd.DataFrame:
        """空船初始状态：船上还没有任何箱子"""
        return pd.DataFrame(columns=BAYPLAN_COLUMNS)

    @staticmethod
    def proj_vessel_to_cell(slots: pd.DataFrame):
        """
        slot级 -> cell级(N_BAY,2,2)投影。
        要求slots已含can_40ft/can_reefer列（build_vessel_geometry + find_can_40ft +
        find_can_reefer之后的产物），聚合出Vessel构造需要的三个静态几何数组。
        返回 (is_valid, capacity_total, capacity_rf)。
        """
        capacity_total = Vessel.build_vessel_cell(slots, "can_40ft")
        is_valid = capacity_total > 0
        slots = slots.copy()
        slots["can_reefer_40ft"] = slots["can_40ft"] & slots["can_reefer"]
        capacity_rf = Vessel.build_vessel_cell(slots, "can_reefer_40ft")
        return is_valid, capacity_total, capacity_rf

    def proj_cell_to_vessel(self, cell_state=None) -> pd.DataFrame:
        """
        cell级解 -> slot级DataFrame投影。
        cell_state: 不传则用当前self.cell；传则接受snapshot()格式的dict（取其"cell"）。

        返回列：bay_idx, row_idx, tier_idx, lr, hd, can_40ft, can_20ft, can_reefer,
               POL, POD, GP_count, RF_count

        分配到具体物理槽位的规则（cell级解 -> slot级的合理近似复原，
        不是真实精确到箱位的CSP解，只是按装载习惯把cell总量摊到槽位上，
        摊不满的槽位保持真正的空(POD=-1)，不再用广播制造"整格同色但没装满"的误导）：
            1. 先摊RF需求：只在can_reefer=True的槽位里，按"从下往上、从中间到两边"
               的顺序，摊满RF_count个槽位（每个槽位RF_count只会是0或1）。
            2. 再摊GP需求：在这个cell剩下的所有槽位里（含没被RF用到的reefer槽位），
               按同样顺序摊满GP_count个槽位。
            3. 摊不满的槽位，POD保持-1。
        """
        if self.full_slot_table is None:
            raise ValueError("此Vessel无full_slot_table，无法投影，需通过Vessel.load_vessel()构造")

        cell = self.cell if cell_state is None else cell_state["cell"]

        slots = self.full_slot_table.copy()
        slots["POL"] = -1
        slots["POD"] = -1
        slots["GP_count"] = 0
        slots["RF_count"] = 0

        for big_bay, (b0, b1) in enumerate(STSE_BAY_PAIRS):
            b0_can40 = (slots.bay_idx == b0) & slots.can_40ft
            for lr in (0, 1):
                for hd in (0, 1):
                    record = cell[big_bay, lr, hd]
                    pod = record["POD"]
                    if pod == -1:
                        continue
                    pol = record["POL"]
                    gp_count = record["GP_count"]
                    rf_count = record["RF_count"]

                    b0_mask = b0_can40 & (slots.lr == lr) & (slots.hd == hd)
                    cell_idx = list(slots.index[b0_mask])
                    if not cell_idx:
                        continue

                    # 摆放顺序：从下往上(tier_idx升序)、从中间到两边。
                    # lr=0这一侧假设row_idx越大越靠中线(降序=从中间到两边)，
                    # lr=1这一侧假设row_idx越小越靠中线(升序=从中间到两边)——
                    # 如果实际方向反了，把下面row_reverse的条件互换即可。
                    row_reverse = (lr == 0)
                    cell_idx.sort(key=lambda idx: (
                        slots.at[idx, "tier_idx"],
                        -slots.at[idx, "row_idx"] if row_reverse else slots.at[idx, "row_idx"],
                    ))

                    # 第一步：只在can_reefer=True的槽位里摊RF需求
                    reefer_idx = [idx for idx in cell_idx if slots.at[idx, "can_reefer"]]
                    used_rf_idx = reefer_idx[:rf_count]

                    # 第二步：剩下所有槽位（含没被RF用到的reefer槽位）摊GP需求
                    remaining_idx = [idx for idx in cell_idx if idx not in used_rf_idx]
                    used_gp_idx = remaining_idx[:gp_count]

                    for idx in used_rf_idx:
                        slots.at[idx, "POL"] = pol
                        slots.at[idx, "POD"] = pod
                        slots.at[idx, "RF_count"] = 1
                    for idx in used_gp_idx:
                        slots.at[idx, "POL"] = pol
                        slots.at[idx, "POD"] = pod
                        slots.at[idx, "GP_count"] = 1

                    # b1侧：镜像写回，(row_idx,tier_idx)与b0侧完全一致
                    used_idx = used_rf_idx + used_gp_idx
                    if not used_idx:
                        continue
                    rt_to_type = {
                        (slots.at[idx, "row_idx"], slots.at[idx, "tier_idx"]):
                            ("RF" if idx in used_rf_idx else "GP")
                        for idx in used_idx
                    }
                    for idx in slots.index[slots.bay_idx == b1]:
                        key = (slots.at[idx, "row_idx"], slots.at[idx, "tier_idx"])
                        if key in rt_to_type:
                            slots.at[idx, "POL"] = pol
                            slots.at[idx, "POD"] = pod
                            if rt_to_type[key] == "RF":
                                slots.at[idx, "RF_count"] = 1
                            else:
                                slots.at[idx, "GP_count"] = 1

        return slots[["bay_idx", "row_idx", "tier_idx", "lr", "hd",
                      "can_40ft", "can_20ft", "can_reefer", "POL", "POD", "GP_count", "RF_count"]]

    def export_bayplan(self, snapshots: dict, out_dir: str, port_names: dict = None, if_plot_phy: bool = False) -> list:
        """
        遍历snapshots（solve()产出的{POL: snapshot_dict}），对每个POL调用proj_cell_to_vessel，
        存成{POL}_{港口码}_DEP_bayplan.csv，同时调用utils.viz.plot_bayplan画一张png，
        都落盘到out_dir，返回写出的文件路径列表（csv和png交替）。
        port_names: 可选{POL: 三字码}，不传则用POL数字编号命名。

        导出前先打印各POL剩余的cbf（GP/RF计数非0的部分），纯诊断信息：
        可能是capacity取整产生的余量（正数=没放完，负数=capacity超出实际需求的超额扣减），
        不影响已完成的求解结果，也不在这里做任何修正。
        """
        print("[export_bayplan] 各POL剩余cbf（0表示刚好分配完）：")
        for pol, pod_counts in sorted(self.cbf.items()):
            leftover = {
                pod: counts for pod, counts in pod_counts.items()
                if counts.get("GP", 0) != 0 or counts.get("RF", 0) != 0
            }
            if leftover:
                print(f"  POL={pol}: {leftover}")

        from utils.viz import plot_bayplan, _default_port_colors

        os.makedirs(out_dir, exist_ok=True)
        paths = []

        from utils.vessel_io import STSE_PORT_COLORS

        # 所有港口共用一套POD颜色映射，方便跨港口对比同一POD在不同图里颜色一致。
        # 优先用手动指定的STSE_PORT_COLORS(按港口三字码查)，查不到的POD用自动色板。
        all_pods = set()
        for snap in snapshots.values():
            for record in snap["cell"].flatten():
                if record["POD"] != -1:
                    all_pods.add(record["POD"])

        fallback_colors = _default_port_colors(all_pods)
        port_colors = {}
        for pod in all_pods:
            code = port_names.get(pod) if port_names else None
            port_colors[pod] = STSE_PORT_COLORS.get(code, fallback_colors[pod])

        for pol in sorted(snapshots.keys()):
            code = port_names.get(pol, str(pol)) if port_names else str(pol)
            df = self.proj_cell_to_vessel(cell_state=snapshots[pol])

            csv_path = os.path.join(out_dir, f"{pol}_{code}_DEP_bayplan.csv")
            df.to_csv(csv_path, index=False)
            paths.append(csv_path)

            png_paths = plot_bayplan(
                df, title=f"POL={pol} ({code}) departure",
                filename=f"{pol}_{code}_DEP_bayplan.png",
                save_dir=out_dir, port_colors=port_colors, port_names=port_names,
                if_plot_phy=if_plot_phy,
            )
            paths.extend(png_paths)

        return paths
    
    def verify_reefer_allocation(vessel, snapshots):
        for pol, snap in sorted(snapshots.items()):
            cell = snap["cell"]
            total_rf_used = sum(
                cell[b, l, h]["RF_count"]
                for b in range(vessel.n_bay) for l in range(2) for h in range(2)
            )
            total_rf_capacity = sum(
                vessel.capacity_rf[b, l, h]
                for b in range(vessel.n_bay) for l in range(2) for h in range(2)
                if cell[b, l, h]["POD"] != -1  # 只统计已赋值的cell
            )
            print(f"POL={pol}: RF实际用量={total_rf_used}, "
                f"已赋值cell的RF槽位上限={total_rf_capacity}, "
                f"差值(槽位空闲)={total_rf_capacity - total_rf_used}")