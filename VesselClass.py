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

    def __init__(self, full_slot_table: pd.DataFrame, cbf: dict, current_pol: int = 0, tail_threshold: int = 4):
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
        
        self.capacity_hc = self._derive_capacity_hc(full_slot_table)
        # capacity_hc[bay][lr][hd]: 该cell内高箱(HC/HR)槽位上限。
        # hold每row: min(n,2)；deck每row: n-1（n=这一摞的can_40ft槽位数）

        self.has_reefer = self.capacity_rf > 0
        # has_reefer[bay][lr][hd]: bool，由capacity_rf推导，方便候选集过滤
        # has_reefer=True的cell候选集包含(POD, "RF")对, =False的cell只允许(POD, "GP")对

        self.n_bay = self.is_valid.shape[0]
        # 搜索空间的bay数量，测试时按传入的full_slot_table定，STSE时7

        self.bay_capacity_share = self.capacity_total.sum(axis=(1, 2)) / self.capacity_total.sum()
        # bay_capacity_share[bay]: 该bay物理容量(capacity_total)占全船总容量的比例，
        # 静态、只算一次，供CSP_solver的CI cell层评分把"预算"按容量比例分摊到各bay用

        self.current_port_bay_load = np.zeros(self.n_bay, dtype=int)
        # current_port_bay_load[bay]: 当前港口这个bay累计的吊车作业量
        # （卸箱+本港装箱），供CSP_solver的CI打分用。换港时通过reset_port_bay_load
        # 重置为新港口的卸箱量，之后assign/unassign原地增减，不重新扫全船。

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

        self.cbf_original = copy.deepcopy(cbf)
        # 航次开始前的原始需求快照，只读；assign/unassign只会修改self.cbf，
        # 不会碰这份拷贝，专供后续箱子层CI打分计算各POD总需求用

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

        self.port_budget = self.total_remaining()
        # 这一港开始装货前的剩余总量快照，港内固定不变，供CI cell层评分
        # 按bay_capacity_share分摊"应得预算"用。换港时在reset_port_bay_load里
        # 用同样的方式重新赋值。

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

    @staticmethod
    def _stack_hc_cap(n: int, hd: int = None) -> int:
        """
        一摞(同一row_idx方向叠放的can_40ft槽位集合)的HC配额quota(n)，
        _derive_capacity_hc(静态capacity_hc)和proj_cell_to_vessel(实际贴标签时
        按摞贪心分配)都要用同一条公式，抽在这里只维护一份：
            quota(n) = n - 1 （n >= 2）
            quota(n) = 1     （n == 1）
        hold和deck共用同一公式，不再区分两级配额；hd参数只为兼容旧调用签名保留，
        不参与计算。
        """
        if n <= 0:
            return 0
        return n - 1 if n >= 2 else 1

    @staticmethod
    def _derive_capacity_hc(slots: pd.DataFrame) -> np.ndarray:
        """
        按(bay_idx, row_idx, lr, hd)分组算出每一row的can_40ft槽位数n，
        再用_stack_hc_cap(n, hd)算这一摞的HC容量上限，按big_bay累加。
        """
        capacity_hc = np.zeros((N_BAY, 2, 2), dtype=int)
        can40 = slots[slots["can_40ft"]]

        stack_counts = can40.groupby(["bay_idx", "row_idx", "lr", "hd"]).size()
        for (bay_idx, row_idx, lr, hd), n in stack_counts.items():
            big_bay = _BIG_BAY_OF_B0.get(bay_idx)
            if big_bay is None:
                continue
            capacity_hc[big_bay, lr, hd] += Vessel._stack_hc_cap(n, hd)

        return capacity_hc.astype(int)
        
    # ── 查询方法 ───────────────────────────────────────────────────────

    def rel_rank(self, pod):
        """相对current_pol的挂靠距离，允许绕圈（用port_min把港口范围平移到0起点）。"""
        c = (self.current_pol - self.port_min) % self.n_ports
        p = (pod - self.port_min) % self.n_ports
        return (p - c) if p >= c else (p - c + self.n_ports)

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

        # 走到这里，要么other_pod==-1，要么hd==1且hold已有货——
        # 此时deck候选必须比hold的货早卸（距离更小）
        other_rank = self.rel_rank(other_pod) if other_pod != -1 else None

        candidates = set()
        for pod, counts in current_cbf.items():
            if other_rank is not None:
                new_rank = self.rel_rank(pod)
                if new_rank > other_rank:
                    continue
            has_gp_demand = (counts.get("GP", 0) + counts.get("HC", 0)) > self.tail_threshold
            has_rf_demand = self.has_reefer[bay, lr, hd] and (counts.get("RF", 0) + counts.get("HR", 0)) > 0
            if has_gp_demand or has_rf_demand:
                candidates.add(pod)

        return candidates
    
    def remaining_pods(self) -> set:
        return {
            pod for pod, counts in self.cbf[self.current_pol].items()
            if (counts.get("GP", 0) + counts.get("HC", 0)) > self.tail_threshold
            or (counts.get("RF", 0) + counts.get("HR", 0)) > 0
        }

    def port_complete(self) -> bool:
        """当前POL的cbf是否全部分配完毕。"""
        return len(self.remaining_pods()) == 0

    def total_remaining(self) -> int:
        """当前POL的cbf剩余总箱量。"""
        return sum(
            sum(c.get(k, 0) for k in ("GP", "HC", "RF", "HR"))
            for c in self.cbf[self.current_pol].values()
        )


    # ── 赋值与撤销 ─────────────────────────────────────────────────────
    def assign(self, bay, lr, hd, pod):
        cap_total = int(self.capacity_total[bay, lr, hd])
        cap_rf = int(self.capacity_rf[bay, lr, hd])

        demand = self.cbf[self.current_pol][pod]
        rf_demand, hr_demand = demand.get("RF", 0), demand.get("HR", 0)
        gp_demand, hc_demand = demand.get("GP", 0), demand.get("HC", 0)

        rf_total_remaining = rf_demand + hr_demand
        rf_used = min(cap_rf, rf_total_remaining)
        rf_deduct_rf = min(rf_demand, rf_used)
        rf_deduct_hr = rf_used - rf_deduct_rf

        gp_total_remaining = gp_demand + hc_demand
        gp_capacity = cap_total - rf_used
        gp_used = min(gp_capacity, gp_total_remaining)
        gp_deduct_gp = min(gp_demand, gp_used)
        gp_deduct_hc = gp_used - gp_deduct_gp

        self.cell[bay, lr, hd] = {
            "POD": pod, "POL": self.current_pol,
            "GP_count": gp_used, "RF_count": rf_used,
            "_gp_from_gp": gp_deduct_gp, "_gp_from_hc": gp_deduct_hc,
            "_rf_from_rf": rf_deduct_rf, "_rf_from_hr": rf_deduct_hr,
        }
        demand["GP"] = gp_demand - gp_deduct_gp
        demand["HC"] = hc_demand - gp_deduct_hc
        demand["RF"] = rf_demand - rf_deduct_rf
        demand["HR"] = hr_demand - rf_deduct_hr

        self.current_port_bay_load[bay] += gp_used + rf_used

    def unassign(self, bay, lr, hd, pod):
        record = self.cell[bay, lr, hd]
        demand = self.cbf[self.current_pol][pod]
        demand["GP"] = demand.get("GP", 0) + record["_gp_from_gp"]
        demand["HC"] = demand.get("HC", 0) + record["_gp_from_hc"]
        demand["RF"] = demand.get("RF", 0) + record["_rf_from_rf"]
        demand["HR"] = demand.get("HR", 0) + record["_rf_from_hr"]

        self.current_port_bay_load[bay] -= record["GP_count"] + record["RF_count"]

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

    def reset_port_bay_load(self, discharged: list):
        """换港后调用：把current_port_bay_load重置为这一港的初始卸箱量
        （按bay汇总discharge()返回的记录），后续assign/unassign在此基础上
        累加/累减本港新装的部分。回溯时如果这一港整体失败，调用方需要自己
        把current_port_bay_load恢复成换港前的备份，这个方法不负责回溯。
        """
        self.current_port_bay_load = np.zeros(self.n_bay, dtype=int)
        for bay, lr, hd, record in discharged:
            self.current_port_bay_load[bay] += record["GP_count"] + record["RF_count"]

        # current_pol可能已经越过最后一个真实POL(比如末港discharge后的哨兵态)，
        # 此时self.cbf里没有这个key，total_remaining()会KeyError——但solve()
        # 马上就会在下一次递归开头判定current_pol > max(cbf.keys())并返回，
        # port_budget在那之前不会被读取，这里安全地留空即可。
        self.port_budget = self.total_remaining() if self.current_pol in self.cbf else 0

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

    def proj_cell_to_vessel(self, cell_state=None, original_cbf=None, cbf_with_20=None) -> pd.DataFrame:
        """
        cell级解 -> slot级DataFrame投影。
        cell_state: 不传则用当前self.cell；传则接受snapshot()格式的dict（取其"cell"）。
        original_cbf: 航次开始前（solve()扣减之前）的原始cbf，用于按(POL,POD)读取
                      HC/HR原始总demand，作为贴HC标签的预算池来源（见下）。
        cbf_with_20: 可选，vessel_io.batch_parse_cbf_with_20的输出格式（int-keyed
                     {POL:{POD:{"20GP":n,"20HC":n,...}}}，跟original_cbf一样由调用方
                     从json加载后把key转成int）。不传则跳过第三步20ft relabel，
                     is_20ft保持全False，不影响任何现有调用方。

        返回列：bay_idx, row_idx, tier_idx, lr, hd, can_40ft, can_20ft, can_reefer,
               POL, POD, GP_count, RF_count, is_hc, is_20ft（默认恒False，见第三步）

        分配到具体物理槽位的规则（cell级解 -> slot级的合理近似复原，
        不是真实精确到箱位的CSP解，只是按装载习惯把cell总量摊到槽位上，
        摊不满的槽位保持真正的空(POD=-1)，不再用广播制造"整格同色但没装满"的误导）：
            1. 先摊RF需求：只在can_reefer=True的槽位里，按"从下往上、从中间到两边"
               的顺序，摊满RF_count个槽位（每个槽位RF_count只会是0或1）。
            2. GP/HC由下面的第二步（两遍法）统一决定，不在这里摊。

        第二步（两遍法：Pass A分HC，Pass B补GP）：按(POL,POD)分组，把该组占用的
        所有摞（row，跨host cell合并）按hd拆成hold摞/deck摞两批。每摞的
        n_eff = 这摞物理40ft槽位数 - 这摞里第一步已摊上RF的槽位数，只跟reefer
        占用有关，不涉及HC。

        排序规则（Pass A/B共用）：每组（hold/deck各自）内部，第一步已分配过
        RF的摞排最前面，其余摞按(bay_idx, row_idx)排序后套一个从两端向中间
        交替的兜底顺序（如[0,7,1,6,2,5,3,4]，见_zigzag_order，占位实现）。
        遍历顺序总是先所有hold摞（按上述排序），再所有deck摞（按上述排序）。

        Pass A（分HC，纯dry，不区分RF/HR）：hc_budget取original_cbf该(POL,POD)
        的HC demand。quota(hd, n_eff)：hold查表{0:0,1:1,2:2,3:2,4:2}（n_eff>4
        时quota先按2处理，TODO）；deck为n_eff-1（n_eff<=0则0）。按排序遍历，
        每摞place=min(quota, hc_budget)个（从这摞摊完RF后剩下的槽位里按
        tier_idx升序取），标is_hc=True，hc_budget -= place；hc_budget耗尽
        立即整体停止，后面没轮到的摞hc_used视为0。

        Pass B（补GP）：gp_budget取original_cbf该(POL,POD)的GP demand。按
        同样排序遍历所有摞：先把这摞Pass A贴过is_hc的槽位落定GP_count=1
        （dry高箱，不占用gp_budget）；再算这摞avail——hold摞avail=n_eff-hc_used
        （无结构性封顶）；deck摞若hc_used>0则avail=(n_eff-1)-hc_used（沾过HC
        的摞总占用封顶在n_eff-1），否则avail=n_eff（完全没分到HC则不封顶）。
        place=min(avail, gp_budget)个（从这摞剩下的槽位里按tier_idx升序取），
        标为GP（POD/POL写上，GP_count=1，RF_count=0），gp_budget -= place。

        Pass A/B结束后不做任何回退/腾空/记账：多出的hc_budget/gp_budget和
        没填满的摞容量都不写回self.cbf，尾箱统计交给后续单独的逻辑处理。
        Pass A/B结束后把is_hc和GP_count/RF_count按(row_idx,tier_idx)镜像
        写到b1侧对应槽位。

        第三步（post-solve 20ft relabel，纯打标签，不改GP_count/RF_count/POD/POL/is_hc）：
        cbf_with_20非None时才跑。按record["POL"]（不是本函数外层调用方遍历到的导出快照
        POL——同一批未卸货物会原样出现在装船后所有后续POL快照里，只有record["POL"]即这批
        货实际装船港才能对上cbf_with_20的需求量，跟第二步HC贴标签用info["pol"]=record["POL"]
        是同一原则）分组，每组floor((20GP+20HC)/2)（HC不单独区分，一视同仁并入GP）= 需要
        relabel成"20ft对"的cell数量，只在b0侧（bay_idx==b0）、POL/POD匹配、GP_count==1
        （is_hc不管）的slot里选，按(big_bay,lr,hd)自然顺序分组、组内按(tier_idx,row_idx)
        升序挑选，选满一组的quota用完为止；选中的b0侧slot镜像同步到b1侧同(row,tier)位置。
        候选池不够用（正常不应发生，40ft GP_count本身就是20ft折算来的）时不报错，直接
        跳过剩余数量，留给调用方/校验脚本核对。
        """
        if self.full_slot_table is None:
            raise ValueError("此Vessel无full_slot_table，无法投影，需通过Vessel.load_vessel()构造")
        if original_cbf is None:
            raise ValueError("proj_cell_to_vessel需要original_cbf来确定HC/GP两遍法预算池")

        cell = self.cell if cell_state is None else cell_state["cell"]

        slots = self.full_slot_table.copy()
        slots["POL"] = -1
        slots["POD"] = -1
        slots["GP_count"] = 0
        slots["RF_count"] = 0
        slots["is_hc"] = False
        # 预留给post-solve补丁模块：把某个已分配的40ft cell relabel成2个独立20ft箱时，
        # 用这一列标记哪些槽位属于被拆分出的20ft箱。当前proj_cell_to_vessel完全没有
        # 拆分逻辑，这一列恒为False，纯占位。
        slots["is_20ft"] = False

        pod_rows = []  # 供第二步Pass A/B使用，按(POL,POD)分组，每个元素是一个摞(row)

        for big_bay, (b0, b1) in enumerate(STSE_BAY_PAIRS):
            b0_can40 = (slots.bay_idx == b0) & slots.can_40ft
            for lr in (0, 1):
                for hd in (0, 1):
                    record = cell[big_bay, lr, hd]
                    pod = record["POD"]
                    if pod == -1:
                        continue
                    pol = record["POL"]
                    rf_count = record["RF_count"]

                    b0_mask = b0_can40 & (slots.lr == lr) & (slots.hd == hd)
                    cell_idx = list(slots.index[b0_mask])
                    if not cell_idx:
                        continue

                    # 摆放顺序：从下往上(tier_idx升序)、从中间到两边。
                    # lr=0这一侧假设row_idx越大越靠中线(降序=从中间到两边)，
                    # lr=1这一侧假设row_idx越小越靠中线(升序=从中间到两边)——
                    # 如果实际方向反了，把下面row_reverse的条件互换即可。
                    #
                    # 按row(摞)为单位顺序分摊：先分row、按row_reverse定的
                    # "从中间到两侧"顺序把row排好处理序列；rf_count是这个cell
                    # 所有row共享的预算，严格按row顺序逐个处理——每个row从
                    # can_reefer槽位里(row内tier_idx升序)吃掉min(该row reefer
                    # 槽位数,剩余rf_count)。GP/HC不在这里摊，交给下面第二步
                    # Pass A/B按(POL,POD)分组统一决定。
                    row_reverse = (lr == 0)
                    dist_row_groups = {}
                    for idx in cell_idx:
                        dist_row_groups.setdefault(slots.at[idx, "row_idx"], []).append(idx)
                    for row_idx, idx_list in dist_row_groups.items():
                        idx_list.sort(key=lambda idx: slots.at[idx, "tier_idx"])
                    dist_row_order = sorted(
                        dist_row_groups.keys(),
                        key=lambda r: (-r if row_reverse else r),
                    )

                    used_rf_idx = []
                    rf_remaining = rf_count
                    row_rf_map = {}
                    for row_idx in dist_row_order:
                        row_idx_list = dist_row_groups[row_idx]
                        if rf_remaining <= 0:
                            row_rf_map[row_idx] = []
                            continue
                        row_reefer_idx = [idx for idx in row_idx_list if slots.at[idx, "can_reefer"]]
                        take_rf = min(len(row_reefer_idx), rf_remaining)
                        row_used_rf = row_reefer_idx[:take_rf]
                        row_rf_map[row_idx] = row_used_rf
                        used_rf_idx.extend(row_used_rf)
                        rf_remaining -= take_rf

                    for idx in used_rf_idx:
                        slots.at[idx, "POL"] = pol
                        slots.at[idx, "POD"] = pod
                        slots.at[idx, "RF_count"] = 1

                    # 摞 = 同一row_idx方向叠放的can_40ft槽位集合，收集给第二步
                    # Pass A/B按(POL,POD)跨host cell合并使用。
                    for row_idx, idx_list in dist_row_groups.items():
                        pod_rows.append({
                            "pol": pol, "pod": pod, "hd": hd, "b1": b1,
                            "bay_idx": big_bay, "row_idx": row_idx,
                            "idx_list": idx_list,
                            "rf_idx": row_rf_map.get(row_idx, []),
                        })

                    # b1侧：镜像写回RF计数，(row_idx,tier_idx)与b0侧完全一致。
                    # b0/b1写入相同的POD是当前实现的选择（因为一个40ft箱天然占用
                    # 镜像的两个20ft物理位置），不是slot级表结构的硬约束——这张表
                    # 本身是逐20ft-slot记录的，将来如果要把某个40ft cell拆成两个
                    # 独立20ft箱（可能分给不同POD），完全可以把这里改成b0/b1各自
                    # 独立写入，不需要改表结构，只需要改这段赋值逻辑本身。
                    if not used_rf_idx:
                        continue
                    rt_to_rf = {
                        (slots.at[idx, "row_idx"], slots.at[idx, "tier_idx"])
                        for idx in used_rf_idx
                    }
                    for idx in slots.index[slots.bay_idx == b1]:
                        key = (slots.at[idx, "row_idx"], slots.at[idx, "tier_idx"])
                        if key in rt_to_rf:
                            slots.at[idx, "POL"] = pol
                            slots.at[idx, "POD"] = pod
                            slots.at[idx, "RF_count"] = 1

        # 第二步：两遍法，按(POL,POD)分组，摞(row)级分配（见上方docstring）。
        pod_groups = {}
        for row in pod_rows:
            pod_groups.setdefault((row["pol"], row["pod"]), []).append(row)

        HOLD_HC_QUOTA_TABLE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}

        def _hc_quota(hd, n_eff):
            if n_eff <= 0:
                return 0
            if hd == 0:
                if n_eff in HOLD_HC_QUOTA_TABLE:
                    return HOLD_HC_QUOTA_TABLE[n_eff]
                # TODO: n_eff超过表范围(>4)时quota先按2处理，后续可能需要重新定义
                return 2
            return n_eff - 1

        def _zigzag_order(items):
            # TODO: 这个兜底顺序目前是占位实现，后续可能优化
            ordered = []
            lo, hi = 0, len(items) - 1
            take_lo = True
            while lo <= hi:
                if take_lo:
                    ordered.append(items[lo])
                    lo += 1
                else:
                    ordered.append(items[hi])
                    hi -= 1
                take_lo = not take_lo
            return ordered

        def _order_rows(rows):
            rf_rows = sorted(
                (r for r in rows if r["rf_idx"]),
                key=lambda r: (r["bay_idx"], r["row_idx"]),
            )
            other_rows = sorted(
                (r for r in rows if not r["rf_idx"]),
                key=lambda r: (r["bay_idx"], r["row_idx"]),
            )
            return rf_rows + _zigzag_order(other_rows)

        for (pol, pod), rows in pod_groups.items():
            for row in rows:
                row["n_eff"] = len(row["idx_list"]) - len(row["rf_idx"])
                rf_set = set(row["rf_idx"])
                row["remaining_idx"] = [idx for idx in row["idx_list"] if idx not in rf_set]
                row["hc_idx"] = []
                row["gp_idx"] = []

            hold_rows = [r for r in rows if r["hd"] == 0]
            deck_rows = [r for r in rows if r["hd"] == 1]
            ordered_rows = _order_rows(hold_rows) + _order_rows(deck_rows)

            # Pass A：分配HC（纯dry，不区分RF/HR）
            hc_budget = original_cbf.get(pol, {}).get(pod, {}).get("HC", 0)
            for row in ordered_rows:
                if hc_budget <= 0:
                    break
                quota = _hc_quota(row["hd"], row["n_eff"])
                place = min(quota, hc_budget)
                take_idx = row["remaining_idx"][:place]
                for idx in take_idx:
                    slots.at[idx, "POL"] = pol
                    slots.at[idx, "POD"] = pod
                    slots.at[idx, "is_hc"] = True
                row["hc_idx"] = take_idx
                row["remaining_idx"] = row["remaining_idx"][place:]
                hc_budget -= place

            # Pass B：补GP，同时把Pass A贴过is_hc的槽位落定GP_count=1（dry高箱）
            gp_budget = original_cbf.get(pol, {}).get(pod, {}).get("GP", 0)
            for row in ordered_rows:
                for idx in row["hc_idx"]:
                    slots.at[idx, "GP_count"] = 1
                    slots.at[idx, "RF_count"] = 0

                hc_used = len(row["hc_idx"])
                if row["hd"] == 0:
                    avail = row["n_eff"] - hc_used
                else:
                    avail = (row["n_eff"] - 1 - hc_used) if hc_used > 0 else row["n_eff"]
                avail = max(0, avail)
                place = min(avail, gp_budget)
                take_idx = row["remaining_idx"][:place]
                for idx in take_idx:
                    slots.at[idx, "POL"] = pol
                    slots.at[idx, "POD"] = pod
                    slots.at[idx, "GP_count"] = 1
                    slots.at[idx, "RF_count"] = 0
                row["gp_idx"] = take_idx
                gp_budget -= place

            # b1侧：镜像is_hc和GP_count/RF_count，(row_idx,tier_idx)与b0侧一致。
            for row in rows:
                touched = row["hc_idx"] + row["gp_idx"]
                if not touched:
                    continue
                b1 = row["b1"]
                for idx in touched:
                    row_i = slots.at[idx, "row_idx"]
                    tier_i = slots.at[idx, "tier_idx"]
                    gp_val = int(slots.at[idx, "GP_count"])
                    rf_val = int(slots.at[idx, "RF_count"])
                    is_hc_val = bool(slots.at[idx, "is_hc"])
                    for mirror_idx in slots.index[
                        (slots.bay_idx == b1) & (slots.row_idx == row_i) & (slots.tier_idx == tier_i)
                    ]:
                        slots.at[mirror_idx, "POL"] = pol
                        slots.at[mirror_idx, "POD"] = pod
                        slots.at[mirror_idx, "GP_count"] = gp_val
                        slots.at[mirror_idx, "RF_count"] = rf_val
                        slots.at[mirror_idx, "is_hc"] = is_hc_val

        # 第三步：post-solve 20ft relabel（见上方docstring）。不改GP_count/RF_count/
        # POD/POL/is_hc，只在最终已定型的slots状态上打is_20ft标签。
        if cbf_with_20 is not None:
            pol_pod_pairs = set(
                zip(slots.loc[slots.POD != -1, "POL"], slots.loc[slots.POD != -1, "POD"])
            )
            for pol, pod in sorted(pol_pod_pairs):
                demand20 = cbf_with_20.get(pol, {}).get(pod, {})
                pairs_needed = (demand20.get("20GP", 0) + demand20.get("20HC", 0)) // 2
                if pairs_needed <= 0:
                    continue

                for big_bay, (b0, b1) in enumerate(STSE_BAY_PAIRS):
                    if pairs_needed <= 0:
                        break
                    for lr in (0, 1):
                        if pairs_needed <= 0:
                            break
                        for hd in (0, 1):
                            if pairs_needed <= 0:
                                break
                            group_mask = (
                                (slots.bay_idx == b0) & (slots.lr == lr) & (slots.hd == hd)
                                & (slots.POL == pol) & (slots.POD == pod) & (slots.GP_count == 1)
                                & (~slots.is_20ft)
                            )
                            group_idx = list(slots.index[group_mask])
                            group_idx.sort(key=lambda idx: (
                                slots.at[idx, "tier_idx"], slots.at[idx, "row_idx"],
                            ))
                            take_idx = group_idx[:pairs_needed]
                            for idx in take_idx:
                                slots.at[idx, "is_20ft"] = True
                                release_row = slots.at[idx, "row_idx"]
                                release_tier = slots.at[idx, "tier_idx"]
                                for mirror_idx in slots.index[
                                    (slots.bay_idx == b1)
                                    & (slots.row_idx == release_row)
                                    & (slots.tier_idx == release_tier)
                                ]:
                                    slots.at[mirror_idx, "is_20ft"] = True
                            pairs_needed -= len(take_idx)

                if pairs_needed > 0:
                    print(f"[proj_cell_to_vessel][20ft relabel] POL={pol} POD={pod} "
                          f"候选GP slot不够，缺 {pairs_needed} 对(={pairs_needed * 2}个20ft箱)未能relabel")

        return slots[["bay_idx", "row_idx", "tier_idx", "lr", "hd",
                      "can_40ft", "can_20ft", "can_reefer", "POL", "POD", "GP_count", "RF_count", "is_hc",
                      "is_20ft"]]

    def export_bayplan(self, snapshots: dict, out_dir: str, original_cbf: dict, port_names: dict = None, if_csv: bool = False, if_plot_phy: bool = False, cbf_with_20: dict = None) -> list:
        """
        遍历snapshots（solve()产出的{POL: snapshot_dict}），对每个POL调用proj_cell_to_vessel，
        存成{POL}_{港口码}_DEP_bayplan.csv，同时调用utils.viz.plot_bayplan画一张png，
        都落盘到out_dir，返回写出的文件路径列表（csv和png交替）。
        port_names: 可选{POL: 三字码}，不传则用POL数字编号命名。
        cbf_with_20: 可选，原样透传给proj_cell_to_vessel做第三步20ft relabel
                     （见proj_cell_to_vessel docstring）；不传则不relabel，行为不变。

        导出前先打印各POL剩余的cbf（GP/RF计数非0的部分），纯诊断信息：
        可能是capacity取整产生的余量（正数=没放完，负数=capacity超出实际需求的超额扣减），
        不影响已完成的求解结果，也不在这里做任何修正。
        """
        print("[export_bayplan]")

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
            df = self.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf, cbf_with_20=cbf_with_20)
            if if_csv:
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