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
    - 动态状态层：self.cell，每个(bay,lr,hd)存一条记录{"POD":.., "type":.., "POL":..}，
      是cell颗粒度的"配载单"，字段和proj_cell_to_vessel/export_bayplan导出的列名对齐。
      搜索过程中读写，支持回溯。

    坐标系：
        bay  : 0-base，只含有效大箱bay（偶数idx）
        lr   : 0=left(row 0-4), 1=right(row 5-10)
        hd   : 0=hold(tier 0-3), 1=deck(tier 4-9)
    """

    _EMPTY_RECORD = {"POD": -1, "type": None, "POL": -1, "count": 0}
    # 未赋值cell的记录模板。count=实际装的箱量（<=capacity，never超装）。
    # 注意：每个cell必须持有独立的dict实例，
    # 不能用np.full(shape, {...})批量填充——那样所有cell会共享同一个dict对象，
    # 改一个牵动全部。__init__里逐个构造。

    def __init__(self, full_slot_table: pd.DataFrame, cbf: dict, current_pol: int = 0):
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

        self.current_pol = current_pol
        # 当前装载港口编号，指向cbf的第一层key
        # 换港时只更新这个指针，cbf本身不变

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
        """返回(POD, type)候选对集合，三层过滤：不能翻箱、可装特殊箱（reefer）、有cbf余量。"""
        if not self.is_valid[bay, lr, hd]:
            return set()

        current_cbf = self.cbf[self.current_pol]  # {POD: {"GP": n, "RF": n}}

        # 第一层：no-overstow，同一(bay, lr)的hold/deck POD约束
        other_hd = 1 - hd
        other_pod = self.cell[bay, lr, other_hd]["POD"]

        pod_lo, pod_hi = 0, max(current_cbf.keys()) if current_cbf else 0
        if other_pod != -1:
            if hd == 0:   # 当前是hold，必须 >= deck的POD
                pod_lo = other_pod
            else:          # 当前是deck，必须 <= hold的POD
                pod_hi = other_pod

        # 第二层：capabilities + 第三层：cbf余量，同时过滤
        candidates = set()
        for pod, counts in current_cbf.items():
            if not (pod_lo <= pod <= pod_hi):
                continue
            if counts.get("GP", 0) > 0:
                candidates.add((pod, "GP"))
            if self.has_reefer[bay, lr, hd] and counts.get("RF", 0) > 0:
                candidates.add((pod, "RF"))

        return candidates

    def remaining_pods(self) -> set:
        """当前POL中cbf总量>0的POD集合。"""
        return {
            pod for pod, counts in self.cbf[self.current_pol].items()
            if counts.get("GP", 0) + counts.get("RF", 0) > 0
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

    def assign(self, bay, lr, hd, pod, ctype):
        """
        赋值cell，记录装货港current_pol。
        扣减cbf时取min(capacity, 当前剩余量)——容量再大也不能超装，
        剩余量不够填满整个cell时就按实际剩余量装，cbf扣到0为止，不会变负数。
        实际装的箱量记在cell的"count"字段里，unassign时精确按这个数值加回，
        不能重新用capacity推算（因为可能小于capacity）。
        """
        cap = self.capacity_rf[bay, lr, hd] if ctype == "RF" else self.capacity_total[bay, lr, hd]
        remaining = self.cbf[self.current_pol][pod][ctype]
        used = min(cap, remaining)
        self.cell[bay, lr, hd] = {"POD": pod, "type": ctype, "POL": self.current_pol, "count": used}
        self.cbf[self.current_pol][pod][ctype] = remaining - used

    def unassign(self, bay, lr, hd, pod, ctype):
        """撤销赋值，把cell记录里实际装的count加回cbf（不是capacity），精确恢复。"""
        used = self.cell[bay, lr, hd]["count"]
        self.cbf[self.current_pol][pod][ctype] += used
        self.cell[bay, lr, hd] = dict(self._EMPTY_RECORD)

    # ── 多港口 ─────────────────────────────────────────────────────────

    def discharge(self, arriving_pod) -> list:
        """卸载arriving_pod的所有cell，返回记录供undischarge回溯。"""
        discharged = []
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    if self.cell[bay, lr, hd]["POD"] == arriving_pod:
                        discharged.append((bay, lr, hd, dict(self.cell[bay, lr, hd])))
                        self.cell[bay, lr, hd] = dict(self._EMPTY_RECORD)
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
        返回 {(bay, lr, hd): {"POD":.., "type":.., "POL":.., "count":..}}，只含POD != -1的cell。
        count是实际装的箱量（<=capacity，不一定填满整个cell）。"""
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
        cell级解 -> slot级DataFrame投影（proj_vessel_to_cell的逆方向）。
        cell_state: 不传则用当前self.cell；传则接受snapshot()格式的dict（取其"cell"）。

        返回列：bay_idx, row_idx, tier_idx, lr, hd, can_40ft, can_20ft, POL, POD, type
        POL是该cell实际装货时的current_pol（assign()时精确记录），不是快照所属港口，
        两者在同一份snapshot里可能不同（更早港口装、还未卸的货会保留原始POL）。

        can_40ft只标在每对STSE_BAY_PAIRS的b0一侧（见vessel_io.find_can_40ft），
        一个40ft箱物理上同时占用b0和b1两侧的槽位，所以每个cell的解要同时写回
        b0侧can_40ft=True的行，以及b1侧(row_idx,tier_idx)相同的对应行。
        """
        if self.full_slot_table is None:
            raise ValueError("此Vessel无full_slot_table，无法投影，需通过Vessel.load_vessel()构造")

        cell = self.cell if cell_state is None else cell_state["cell"]

        slots = self.full_slot_table.copy()
        slots["POL"] = -1
        slots["POD"] = -1
        slots["type"] = None

        for big_bay, (b0, b1) in enumerate(STSE_BAY_PAIRS):
            b0_can40 = (slots.bay_idx == b0) & slots.can_40ft
            for lr in (0, 1):
                for hd in (0, 1):
                    record = cell[big_bay, lr, hd]
                    pod = record["POD"]
                    if pod == -1:
                        continue
                    ctype = record["type"]
                    pol = record["POL"]

                    # b0侧：can_40ft标记的行，直接按(lr,hd)取这个cell对应的所有物理槽位
                    b0_mask = b0_can40 & (slots.lr == lr) & (slots.hd == hd)
                    slots.loc[b0_mask, "POL"] = pol
                    slots.loc[b0_mask, "POD"] = pod
                    slots.loc[b0_mask, "type"] = ctype

                    # b1侧：同一批40ft箱占用的另一半，(row_idx,tier_idx)与b0侧完全一致
                    rt_pairs = set(
                        zip(slots.loc[b0_mask, "row_idx"], slots.loc[b0_mask, "tier_idx"])
                    )
                    if not rt_pairs:
                        continue
                    b1_mask = (slots.bay_idx == b1) & slots.apply(
                        lambda r: (r.row_idx, r.tier_idx) in rt_pairs, axis=1
                    )
                    slots.loc[b1_mask, "POL"] = pol
                    slots.loc[b1_mask, "POD"] = pod
                    slots.loc[b1_mask, "type"] = ctype

        return slots[["bay_idx", "row_idx", "tier_idx", "lr", "hd",
                      "can_40ft", "can_20ft", "POL", "POD", "type"]]

    def export_bayplan(self, snapshots: dict, out_dir: str, port_names: dict = None) -> list:
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

        # 所有港口共用一套POD颜色映射，方便跨港口对比同一POD在不同图里颜色一致
        all_pods = set()
        for snap in snapshots.values():
            for record in snap["cell"].flatten():
                if record["POD"] != -1:
                    all_pods.add(record["POD"])
        port_colors = _default_port_colors(all_pods)

        for pol in sorted(snapshots.keys()):
            code = port_names.get(pol, str(pol)) if port_names else str(pol)
            df = self.proj_cell_to_vessel(cell_state=snapshots[pol])

            csv_path = os.path.join(out_dir, f"{pol}_{code}_DEP_bayplan.csv")
            df.to_csv(csv_path, index=False)
            paths.append(csv_path)

            png_path = plot_bayplan(
                df, title=f"POL={pol} ({code}) departure",
                filename=f"{pol}_{code}_DEP_bayplan.png",
                save_dir=out_dir, port_colors=port_colors,
            )
            paths.append(png_path)

        return paths