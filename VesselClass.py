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

        self._hc_cbf_writeback_seen = set()
        # proj_cell_to_vessel记账去重用：{(POL, POD), ...}。同一批货物在被discharge
        # 之前会原样出现在它装船后所有后续POL的snapshot里，deck降级回退/预算分不完
        # 回退这两处写self.cbf的副作用必须对同一个(POL,POD)分组只生效一次，
        # 否则export_bayplan对每个POL都调用一次proj_cell_to_vessel会导致重复计入。

        self._tail_source2_log = []
        # 尾箱来源2（HC降级挤出腾空）诊断日志：每次proj_cell_to_vessel触发deck摞
        # 腾空就追加一条(POL, POD)记录，不去重（同一(POL,POD)在多个POL快照里
        # 重算是幂等的物理动作，每次真实发生都要记一笔，跟_hc_cbf_writeback_seen
        # 控制的cbf写回去重是两回事）。供utils/tail.py统计用，只读不影响求解逻辑。

        self._tail_source3_log = []
        # 尾箱来源3（HC/RF贴标签预算池分不完回退）诊断日志：proj_cell_to_vessel
        # 里"预算池分不完，把这部分HC/HR demand回退进cbf余量"那一段触发时追加
        # 一条(POL, POD, gp_hc_budget_leftover, rf_hc_budget_leftover)记录。
        # 跟来源2(deck-squeeze)是两条独立的写回路径，故意不合并成一个log：
        # 来源2每次触发固定回退1个GP名额(数量恒为1)，来源3每个(POL,POD)分组
        # 只触发一次但回退量可以是任意正整数(gp_hc_budget/rf_hc_budget剩多少
        # 就回退多少)，两者的记录粒度和字段形状本来就不一样，分开存更直接，
        # 不用靠一个kind字段再反向拆分成两种不同shape的tuple。

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
            2. 再摊GP需求：在这个cell剩下的所有槽位里（含没被RF用到的reefer槽位），
               按同样顺序摊满GP_count个槽位。
            3. 摊不满的槽位，POD保持-1。

        第二步（贴HC标签）：预算池按(POL,POD)粒度，取original_cbf里这个POL/POD
        的原始HC/HR总demand
        RF预算(rf_hc_budget)和GP预算(gp_hc_budget)分开走，
        但共享同一套摞级配额quota(n)（_stack_hc_cap，hold/deck统一公式）和同一个
        occupied集合判断逻辑，贴标顺序沿用先RF后GP。按(POL,POD)分组后，把该组
        占用的所有摞（跨host cell合并）按hd拆成hold摞/deck摞两批，分四步分配：
            Step1（hold）：所有hold摞按quota(n)降序贪心贴标，每摞最多贴quota(n)个，
                不要求整摞同质（quota个高箱+其余普箱共存）。
            Step2（deck）：所有deck摞按quota(n)降序贪心处理。若当前预算能覆盖
                这一摞的整个quota(n)，则整摞转HC：贴满quota(n)个高箱；若这一摞
                贴标前是摆满的(occupied==n)，把多出的1个slot(tier_idx最大)腾空，
                对应demand按GP回退回cbf余量。若预算不足以覆盖整摞quota(n)但仍>0，
                视为"收尾摞"：贴min(预算,quota(n))个高箱，其余slot维持原样的
                GP/RF（不腾空、不退回），预算清零后立即停止处理后续deck摞——
                这种收尾摞混装每个(POL,POD)分组最多发生一次。
            Step3（hold二次扫描）：Step1+Step2跑完后预算仍>0，回到hold摞（跳过
                Step1已贴满quota上限的摞），套用跟Step2收尾摞相同的规则继续贴，
                直到预算耗尽或hold摞用尽（不限次数，不要求GP需求为0才触发）。
            Step4：分不完的gp_hc_budget/rf_hc_budget回退进cbf余量。
        排序用贪心（quota(n)降序），不做背包最优匹配。

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
            raise ValueError("proj_cell_to_vessel需要original_cbf来确定HC/HR贴标预算池")

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

        cell_infos = []  # 供第二步HC贴标使用，按(POL,POD)分组

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
                    #
                    # 按row(摞)为单位顺序分摊，不能用一个跨row的全局tier排序
                    # 统一切片：不同row的tier_idx取值范围可能完全不重叠(例如
                    # 一个row是4-7、另一个row是0-3)，如果对整个cell的槽位按
                    # 绝对tier_idx全局排序后统一做reefer_idx[:rf_count]/
                    # remaining_idx[:gp_count]前缀切片，会出现某个row自己的
                    # 低tier在全局排序里排到后面、被别的row先抢走预算，而这个
                    # row自己更高的tier在全局序列里反而排到前面被填上——产生
                    # 摞内tier不连续的物理不可能状态(低tier空、高tier占用)。
                    # 于是改成：先分row、按row_reverse定的"从中间到两侧"顺序把
                    # row排好处理序列；rf_count/gp_count是这个cell所有row共享
                    # 的预算，严格按row顺序逐个处理——每个row先从can_reefer槽位
                    # 里(row内tier_idx升序)吃掉min(该row reefer槽位数,剩余
                    # rf_count)，再从row剩下的槽位(同样tier_idx升序)吃掉
                    # min(该row剩余槽位数,剩余gp_count)；哪个row没被这两步
                    # 填满(说明预算已在这个row耗尽)，从这个row往后的所有row
                    # 都不再分配——不允许后面的row插队使用本该属于前面某个
                    # row的预算，从而保证每个row内部占用一定是从最低tier开始
                    # 连续的一段。
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

                    used_rf_idx, used_gp_idx = [], []
                    rf_remaining, gp_remaining = rf_count, gp_count
                    for row_idx in dist_row_order:
                        if rf_remaining <= 0 and gp_remaining <= 0:
                            break
                        row_idx_list = dist_row_groups[row_idx]

                        row_reefer_idx = [idx for idx in row_idx_list if slots.at[idx, "can_reefer"]]
                        take_rf = min(len(row_reefer_idx), rf_remaining)
                        row_used_rf = row_reefer_idx[:take_rf]

                        row_remaining_idx = [idx for idx in row_idx_list if idx not in row_used_rf]
                        take_gp = min(len(row_remaining_idx), gp_remaining)
                        row_used_gp = row_remaining_idx[:take_gp]

                        used_rf_idx.extend(row_used_rf)
                        used_gp_idx.extend(row_used_gp)
                        rf_remaining -= take_rf
                        gp_remaining -= take_gp

                        if take_rf + take_gp < len(row_idx_list):
                            # 这个row没填满，预算已耗尽——后续row不再分配
                            break

                    for idx in used_rf_idx:
                        slots.at[idx, "POL"] = pol
                        slots.at[idx, "POD"] = pod
                        slots.at[idx, "RF_count"] = 1
                    for idx in used_gp_idx:
                        slots.at[idx, "POL"] = pol
                        slots.at[idx, "POD"] = pod
                        slots.at[idx, "GP_count"] = 1

                    # 摞 = 同一row_idx方向叠放的can_40ft槽位集合（与_derive_capacity_hc
                    # 的分组口径一致），stack_hc = hold摞min(n,2) / deck摞n-1。
                    row_groups = {}
                    for idx in cell_idx:
                        row_groups.setdefault(slots.at[idx, "row_idx"], []).append(idx)
                    for row_idx, idx_list in row_groups.items():
                        idx_list.sort(key=lambda idx: slots.at[idx, "tier_idx"])

                    stack_hc_remaining = {
                        row_idx: self._stack_hc_cap(len(idx_list), hd)
                        for row_idx, idx_list in row_groups.items()
                    }
                    row_order = sorted(row_groups.keys(), key=lambda r: stack_hc_remaining[r], reverse=True)

                    cell_infos.append({
                        "pol": pol, "pod": pod, "b1": b1, "hd": hd,
                        "cap_hc": int(self.capacity_hc[big_bay, lr, hd]),
                        "row_groups": row_groups,
                        "row_order": row_order,
                        "stack_hc_remaining": stack_hc_remaining,
                        "used_rf_set": set(used_rf_idx),
                        "used_gp_set": set(used_gp_idx),
                    })

                    # b1侧：镜像写回GP/RF计数，(row_idx,tier_idx)与b0侧完全一致。
                    # b0/b1写入相同的POD是当前实现的选择（因为一个40ft箱天然占用
                    # 镜像的两个20ft物理位置），不是slot级表结构的硬约束——这张表
                    # 本身是逐20ft-slot记录的，将来如果要把某个40ft cell拆成两个
                    # 独立20ft箱（可能分给不同POD），完全可以把这里改成b0/b1各自
                    # 独立写入，不需要改表结构，只需要改这段赋值逻辑本身。
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

        # 第二步：贴HC标签 + 降级回退，按(POL,POD)分组，摞(row)级四步分配（见上方docstring）。
        pod_groups = {}
        for info in cell_infos:
            pod_groups.setdefault((info["pol"], info["pod"]), []).append(info)

        for (pol, pod), infos in pod_groups.items():
            # 同一个(POL,POD)分组在被discharge之前会原样出现在后续每个POL的
            # snapshot里，is_hc贴标签本身可以每次都重算(幂等)，但写回self.cbf
            # 这个副作用只能对同一分组生效一次，否则会被重复计入——写回方式
            # 已经从"多次+="改成"一次性总账赋值"，重复生效的风险也从"多算几次
            # +=1"变成"拿本次调用已经写过的最终值当baseline、在上面再叠加一次
            # gp_released_count/gp_hc_budget"，同样会出错，所以这层去重依然
            # 必须保留，只是保护对象变了（见下面baseline_hc/baseline_hr的注释）。
            already_written = (pol, pod) in self._hc_cbf_writeback_seen

            # 总账公式的起点：这个分组开始处理前，self.cbf里已有的残量("来源1"
            # tail_threshold尾量，solve()结束、任何proj_cell_to_vessel调用之前
            # 就存在)。必须在本次调用对这个分组做任何写回之前读取——同一分组
            # 只会真正写回一次(already_written保护)，所以只有第一次调用时这里
            # 读到的才是真正的"处理前"基线；重复调用时这个值已经是上一次算出的
            # 最终结果，但反正下面的写回也会被already_written跳过，不会用错。
            cbf_demand_before = self.cbf[pol][pod]
            baseline_hc = cbf_demand_before.get("HC", 0)
            baseline_hr = cbf_demand_before.get("HR", 0)
            # 总账公式的另外两项：这个分组释放的leftover slot数，按释放前是GP
            # 来源还是RF来源分开累计，取代原来_settle_row里逐次的self.cbf+=1。
            gp_released_count = 0
            rf_released_count = 0

            demand = original_cbf.get(pol, {}).get(pod, {})
            gp_hc_budget = demand.get("HC", 0)
            rf_hc_budget = demand.get("HR", 0)
            # 真GP预算池：取original_cbf的原始"GP"字段本身(不是GP+HC合并值)，
            # 跟gp_hc_budget/rf_hc_budget一样是这个(POL,POD)分组共享、贯穿
            # 所有hold/deck摞的池子，专门核对"贴完HC quota之后剩下的slot"。
            gp_true_budget = demand.get("GP", 0)
            # 真RF预算池：跟gp_true_budget完全对称，取original_cbf的原始
            # "RF"字段本身，核对reefer摞里"贴完HC quota之后剩下的slot"该留
            # 真RF还是该清空退回HR借位。
            rf_true_budget = demand.get("RF", 0)

            # 把该分组占用的所有摞（跨host cell合并）按hd拆成hold摞/deck摞两批。
            hold_stacks, deck_stacks = [], []
            for info in infos:
                for row_idx, idx_list in info["row_groups"].items():
                    n = len(idx_list)
                    stack = {
                        "info": info, "idx_list": idx_list,
                        "quota": self._stack_hc_cap(n),
                        "used_rf": [idx for idx in idx_list if idx in info["used_rf_set"]],
                        "used_gp": [idx for idx in idx_list if idx in info["used_gp_set"]],
                        "hc_tagged": set(),
                    }
                    (hold_stacks if info["hd"] == 0 else deck_stacks).append(stack)

            hold_stacks.sort(key=lambda s: s["quota"], reverse=True)
            deck_stacks.sort(key=lambda s: s["quota"], reverse=True)

            def _tag_stack(stack, cap):
                """在这一摞里最多贴cap个高箱标签（先RF后GP，各自受各自预算池和
                自身occupied集合限制），就地更新is_hc/stack["hc_tagged"]和
                外层rf_hc_budget/gp_hc_budget，返回本次实际贴的数量。"""
                nonlocal rf_hc_budget, gp_hc_budget
                remaining_cap = cap

                rf_untagged = [idx for idx in stack["used_rf"] if idx not in stack["hc_tagged"]]
                take_rf = min(remaining_cap, len(rf_untagged), rf_hc_budget)
                for idx in rf_untagged[:take_rf]:
                    slots.at[idx, "is_hc"] = True
                    stack["hc_tagged"].add(idx)
                rf_hc_budget -= take_rf
                remaining_cap -= take_rf

                gp_untagged = [idx for idx in stack["used_gp"] if idx not in stack["hc_tagged"]]
                take_gp = min(remaining_cap, len(gp_untagged), gp_hc_budget)
                for idx in gp_untagged[:take_gp]:
                    slots.at[idx, "is_hc"] = True
                    stack["hc_tagged"].add(idx)
                gp_hc_budget -= take_gp

                return take_rf + take_gp

            def _settle_row(stack):
                """联合结算这一摞(row)里没被贴上HC标签的GP+RF槽位——GP/RF
                合并处理，不再各自独立决定去留，避免同一row内两者各自结算
                互不知情、导致低tier(比如恰好被RF占用)被释放而高tier(被GP
                占用)却保留下来的物理不可能悬空态(per-row分摊已经保证RF/
                GP不会跨row交错，所以这里的协调只需要限定在单个row内部)。

                kept_rf/kept_gp的计算方式跟原来独立版完全一致：各自把
                leftover(未被贴HC标签的occupied slot)跟rf_true_budget/
                gp_true_budget比较，min(leftover数, 预算)个"合法保留为真
                GP/RF"，预算不受"两个HC/RF预算池是否已耗尽"提前退出条件
                限制，budget只降不升，一旦被这个(POL,POD)分组处理过就是
                最终结果，可以立即结算。

                跟独立版的区别在于物理slot的保留/释放决定：把这一摞leftover
                的GP+RF物理slot合并、按tier_idx从低到高排序，只保留最底部
                keep_count=kept_rf+kept_gp个，其余(更高tier的)全部释放
                (POD/POL/GP_count/RF_count/is_hc清空)，按slot原本的类型
                (释放前是RF还是GP)累加进外层gp_released_count/rf_released_count
                （不再直接写self.cbf——分组循环结束后用baseline_hc/baseline_hr+
                released_count+leftover budget的总账公式一次性赋值，见调用方）。
                保留区间内
                同样从最底部开始，优先把can_reefer=True的slot贴上RF(最多
                kept_rf个)，其余保留slot贴GP——同一物理slot的类型可能因此
                从GP变成RF或反过来，GP_count/RF_count要跟着同步更新，避免
                字段和实际类型对不上。结算完把stack["used_rf"]/["used_gp"]
                更新为保留区间内实际最终的类型划分，供Step3二次扫描/HC
                镜像使用。"""
                nonlocal gp_true_budget, rf_true_budget, gp_released_count, rf_released_count
                leftover_rf = [idx for idx in stack["used_rf"] if idx not in stack["hc_tagged"]]
                leftover_gp = [idx for idx in stack["used_gp"] if idx not in stack["hc_tagged"]]
                leftover_rf_set = set(leftover_rf)

                kept_rf = min(len(leftover_rf), rf_true_budget)
                kept_gp = min(len(leftover_gp), gp_true_budget)
                rf_true_budget -= kept_rf
                gp_true_budget -= kept_gp
                keep_count = kept_rf + kept_gp

                pool = leftover_rf + leftover_gp
                pool.sort(key=lambda idx: slots.at[idx, "tier_idx"])
                keep_slots = pool[:keep_count]
                release_slots = pool[keep_count:]

                def _mirror_idxs(idx):
                    b1 = stack["info"]["b1"]
                    row_i = slots.at[idx, "row_idx"]
                    tier_i = slots.at[idx, "tier_idx"]
                    return [idx] + list(slots.index[
                        (slots.bay_idx == b1)
                        & (slots.row_idx == row_i)
                        & (slots.tier_idx == tier_i)
                    ])

                rf_assigned = 0
                for idx in keep_slots:
                    if rf_assigned < kept_rf and bool(slots.at[idx, "can_reefer"]):
                        for target_idx in _mirror_idxs(idx):
                            slots.at[target_idx, "RF_count"] = 1
                            slots.at[target_idx, "GP_count"] = 0
                        rf_assigned += 1
                    else:
                        for target_idx in _mirror_idxs(idx):
                            slots.at[target_idx, "GP_count"] = 1
                            slots.at[target_idx, "RF_count"] = 0

                for idx in release_slots:
                    was_rf = idx in leftover_rf_set
                    for target_idx in _mirror_idxs(idx):
                        slots.at[target_idx, "POL"] = -1
                        slots.at[target_idx, "POD"] = -1
                        slots.at[target_idx, "GP_count"] = 0
                        slots.at[target_idx, "RF_count"] = 0
                        slots.at[target_idx, "is_hc"] = False
                    # 不再直接写self.cbf——只在内存里累计这个分组释放了多少个
                    # GP来源/RF来源的slot，真正的self.cbf写回挪到分组循环结束后
                    # 的总账公式一次性完成（见下方baseline_hc/baseline_hr那段）。
                    # 这里不需要already_written保护：计数本身跟is_hc贴标签一样
                    # 是幂等的纯计算，重复调用也只是重算出同样的数字，真正的
                    # 副作用(写self.cbf)由末尾唯一一处already_written保护。
                    if was_rf:
                        rf_released_count += 1
                    else:
                        gp_released_count += 1

                stack["used_rf"] = [idx for idx in keep_slots if slots.at[idx, "RF_count"] == 1]
                stack["used_gp"] = [idx for idx in keep_slots if slots.at[idx, "GP_count"] == 1]

            # ── Step1：hold摞，按quota(n)降序贪心，每摞最多贴quota(n)个。
            # 结算gp_true_budget不受"两个HC/RF预算池是否已耗尽"这个提前退出
            # 条件限制——budget只降不升，一旦耗尽，后面没被_tag_stack碰到的
            # hold摞里的GP也永远不会再有机会被贴成HC，同样需要立即结算，
            # 否则会在耗尽点之后残留大量"名义是GP、实际没被gp_true_budget
            # 核销"的phantom GP。──
            for stack in hold_stacks:
                if not (rf_hc_budget <= 0 and gp_hc_budget <= 0):
                    _tag_stack(stack, stack["quota"])
                _settle_row(stack)

            # ── Step2：deck摞，按quota(n)降序，整摞转HC(预算够)或收尾摞混装(不够，
            # 只允许触发一次)。"只允许触发一次"只约束贴HC标签这个动作——
            # deck_tail_used置位之后不再尝试贴标签，但每一摞(包括被跳过贴标签
            # 的)仍然要走gp_true_budget结算，否则quota之外/被跳过的deck摞会
            # 残留大量没被gp_true_budget核销的phantom GP ──
            deck_tail_used = False
            for stack in deck_stacks:
                quota = stack["quota"]
                if not deck_tail_used and not (rf_hc_budget <= 0 and gp_hc_budget <= 0):
                    # 前置判断：不实际贴标，先算这一摞当前的occupied+预算最多能贴满
                    # 多少个，判断是否够覆盖整摞quota(n)。
                    rf_avail = len([idx for idx in stack["used_rf"] if idx not in stack["hc_tagged"]])
                    gp_avail = len([idx for idx in stack["used_gp"] if idx not in stack["hc_tagged"]])
                    take_rf_sim = min(quota, rf_avail, rf_hc_budget)
                    take_gp_sim = min(quota - take_rf_sim, gp_avail, gp_hc_budget)
                    achievable = take_rf_sim + take_gp_sim

                    if achievable >= quota:
                        # 整摞转HC：贴满quota(n)个高箱。deck摞的HC是阶跃式扣减：
                        # 这一摞贴标签前如果是摆满的(occupied==n)，就要把这一摞里
                        # tier_idx最大(最后摆放)的那个占用槽位腾空，对应demand
                        # 按GP回退回cbf余量；如果贴标签前本就不满(occupied==quota)，
                        # 则不需要腾空。
                        _tag_stack(stack, quota)

                        n = len(stack["idx_list"])
                        occupied = len(stack["used_rf"]) + len(stack["used_gp"])
                        if occupied == n:
                            idx_release = stack["idx_list"][-1]
                            release_row = slots.at[idx_release, "row_idx"]
                            release_tier = slots.at[idx_release, "tier_idx"]
                            b1 = stack["info"]["b1"]

                            # print(f"[尾箱来源2] deck摞腾空回退触发: POL={pol}, POD={pod}, 数量=1")
                            self._tail_source2_log.append((pol, pod))

                            for target_idx in [idx_release] + list(slots.index[
                                (slots.bay_idx == b1)
                                & (slots.row_idx == release_row)
                                & (slots.tier_idx == release_tier)
                            ]):
                                slots.at[target_idx, "POL"] = -1
                                slots.at[target_idx, "POD"] = -1
                                slots.at[target_idx, "GP_count"] = 0
                                slots.at[target_idx, "RF_count"] = 0
                                slots.at[target_idx, "is_hc"] = False
                            # idx_release已经被这里的腾空+GP回退处理过，若它属于
                            # used_gp/used_rf，必须从列表里摘除，避免下面
                            # _settle_leftover_gp/_settle_leftover_rf把同一个
                            # 已清空的slot再核销一次(GP/RF和HC/HR重复回退)。
                            if idx_release in stack["used_gp"]:
                                stack["used_gp"].remove(idx_release)
                            if idx_release in stack["used_rf"]:
                                stack["used_rf"].remove(idx_release)

                            if not already_written:
                                cbf_demand = self.cbf[pol][pod]
                                cbf_demand["GP"] = cbf_demand.get("GP", 0) + 1
                    elif achievable > 0:
                        # 收尾摞混装：贴min(预算,quota(n))个高箱，其余occupied slot
                        # 交给下面的gp_true_budget结算决定去留。预算清零后不再对
                        # 后续deck摞尝试贴标签——每个(POL,POD)分组最多发生一次。
                        _tag_stack(stack, quota)
                        deck_tail_used = True

                _settle_row(stack)

            # ── Step3：预算仍有剩余，回到hold摞二次扫描（跳过Step1已贴满quota
            # 上限的摞），套用跟Step2收尾摞相同的规则继续贴，不限次数，直到
            # 预算耗尽或hold摞用尽 ──
            for stack in hold_stacks:
                if rf_hc_budget <= 0 and gp_hc_budget <= 0:
                    break
                remaining_quota = stack["quota"] - len(stack["hc_tagged"])
                if remaining_quota <= 0:
                    continue
                _tag_stack(stack, remaining_quota)
                _settle_row(stack)

            # b1侧：镜像is_hc标签，(row_idx,tier_idx)与b0侧完全一致。同上——
            # 这里镜像的前提是b0/b1两侧共享同一个POD（一个40ft箱占用镜像的两个
            # 20ft物理位置），不是表结构强制的；这张表逐20ft-slot记录，未来拆分
            # 40ft cell成两个独立20ft箱时，is_hc同样可以按各自的箱身独立贴标。
            for stack in hold_stacks + deck_stacks:
                if not stack["hc_tagged"]:
                    continue
                b1 = stack["info"]["b1"]
                rt_hc = {
                    (slots.at[idx, "row_idx"], slots.at[idx, "tier_idx"])
                    for idx in stack["hc_tagged"]
                }
                for idx in slots.index[slots.bay_idx == b1]:
                    key = (slots.at[idx, "row_idx"], slots.at[idx, "tier_idx"])
                    if key in rt_hc:
                        slots.at[idx, "is_hc"] = True

            if gp_hc_budget > 0 or rf_hc_budget > 0:
                # 预算池分不完——跟deck-squeeze一样是幂等的计算结果，每次
                # proj_cell_to_vessel重算这个(POL,POD)分组都会得到同样的
                # gp_hc_budget/rf_hc_budget，所以每次触发都记一笔（不受
                # already_written限制），真正写回self.cbf才需要去重一次。这段
                # 记录时机不受下面写回方式重构的影响：gp_hc_budget/rf_hc_budget
                # 在Step1/2/3结束后就是最终的leftover值，跟总账公式用的是
                # 同一份数字，只是总账公式把它跟baseline/released_count合并
                # 一起赋值，不再单独写一次self.cbf，日志本身照常按老条件触发。
                self._tail_source3_log.append((pol, pod, gp_hc_budget, rf_hc_budget))

            if not already_written:
                # 总账公式：一次性把这个(POL,POD)分组最终的HC/HR余量算出来，
                # 取代原来_settle_row逐次+=和这里leftover单独+=两条独立写回
                # 路径——三项加总正好是完整的账：baseline(分组处理前self.cbf
                # 里的残量) + 本组settle_row释放的leftover slot数(按GP/RF来源
                # 分开) + budget池分不完的leftover。赋值本身天然幂等，
                # already_written只是防止拿本次调用已经写过的最终值当baseline
                # 重复叠加，不是防止"重复+="（已经没有+=了）。
                cbf_demand = self.cbf[pol][pod]
                cbf_demand["HC"] = baseline_hc + gp_released_count + gp_hc_budget
                cbf_demand["HR"] = baseline_hr + rf_released_count + rf_hc_budget

            self._hc_cbf_writeback_seen.add((pol, pod))

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