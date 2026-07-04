import os
import json
import numpy as np
import pandas as pd
import copy
from utils.vessel_io import proj_vessel_to_cell


class Vessel:
    """
    CSP配载规划的数据层。
    - 静态几何层：从船型文件初始化一次，之后只读
    - 动态状态层：搜索过程中读写，支持回溯

    坐标系：
        bay  : 0-base，只含有效大箱bay（偶数idx）
        lr   : 0=left(row 0-4), 1=right(row 5-10)
        hd   : 0=hold(tier 0-3), 1=deck(tier 4-9)
    """

    def __init__(
        self,
        is_valid: np.ndarray = None,        # shape (n_bay, 2, 2), bool
        capacity_total: np.ndarray = None,  # shape (n_bay, 2, 2), int
        capacity_rf: np.ndarray = None,      # shape (n_bay, 2, 2), int
        cbf: dict = None,                    # {POL: {POD: {"GP": count, "RF": count}}}
        current_pol: int = 0,
        full_slot_table: pd.DataFrame = None,  # 真实数据来源，只读，供proj_cell_to_vessel使用
    ):
        # ── 静态几何层（只读） ──────────────────────────────────────────
        # 两条构造路径：
        #   1) 直接传is_valid/capacity_total/capacity_rf三个数组（合成测试数据用）
        #   2) 只传full_slot_table，内部调用proj_vessel_to_cell派生一次（真实数据用）
        # 派生只在构造时发生一次，之后是普通numpy数组，不影响solve()内层循环的访问速度。

        if full_slot_table is not None:
            is_valid, capacity_total, capacity_rf = proj_vessel_to_cell(full_slot_table)

        if is_valid is None or capacity_total is None or capacity_rf is None:
            raise ValueError("必须提供 is_valid/capacity_total/capacity_rf 三个数组，或提供 full_slot_table 由其派生")
        if cbf is None:
            raise ValueError("cbf 不能为空")

        self.full_slot_table = full_slot_table
        # 只有load_vessel构造的Vessel才有此表，供proj_cell_to_vessel展开回slot级坐标
        # 合成测试数据构造的Vessel此属性为None

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
        # 搜索空间的bay数量，测试时4，STSE时7

        # ── 动态状态层（搜索变量） ─────────────────────────────────────

        self.vessel_pod = np.full((self.n_bay, 2, 2), -1, dtype=int)
        # vessel_pod[bay][lr][hd]: 已赋值的POD编号
        # -1 = 有效但未赋值（is_valid=True且尚未分配）

        self.vessel_type = np.full((self.n_bay, 2, 2), None, dtype=object)
        # vessel_type[bay][lr][hd]: 已赋值的货物类型，"GP" / "RF" / None
        # None表示未赋值，和vessel_pod=-1同步

        self.cbf = cbf
        # 全航次cbf，格式：{POL: {POD: {"GP": count, "RF": count}}}
        # 内部通过current_pol指针取当前港口的切片，不在换港时替换整个dict

        self.current_pol = current_pol
        # 当前装载港口编号，指向cbf的第一层key
        # 换港时只更新这个指针，cbf本身不变

    # ── 构造 ───────────────────────────────────────────────────────────

    @classmethod
    def load_vessel(cls, geometry_dir: str, cbf_json_path: str, current_pol: int = 0) -> "Vessel":
        """
        从真实船型数据构造Vessel。
        geometry_dir: 含 full_slot_table.csv 的目录
                      （vessel_io.build_vessel_geometry + find_can_40ft/20ft/reefer 之后落盘的产物，
                       已含 bay_idx/row_idx/tier_idx/lr/hd/can_40ft/can_20ft/can_reefer 列）
        cbf_json_path: cbf.json路径（vessel_io.batch_parse_cbf产出，json的key是字符串，这里转回int）
        """
        slots = pd.read_csv(os.path.join(geometry_dir, "full_slot_table.csv"))

        with open(cbf_json_path) as f:
            raw_cbf = json.load(f)
        cbf = {
            int(pol): {int(pod): counts for pod, counts in pod_counts.items()}
            for pol, pod_counts in raw_cbf.items()
        }

        return cls(cbf=cbf, current_pol=current_pol, full_slot_table=slots)

    # ── 查询方法 ───────────────────────────────────────────────────────
 
    def get_candidates(self, bay, lr, hd) -> set:
        """返回(POD, type)候选对集合，三层过滤：不能翻箱、可装特殊箱（reefer）、有cbf余量。"""
        if not self.is_valid[bay, lr, hd]:
            return set()
 
        current_cbf = self.cbf[self.current_pol]  # {POD: {"GP": n, "RF": n}}
 
        # 第一层：no-overstow，同一(bay, lr)的hold/deck POD约束
        other_hd = 1 - hd
        other_pod = self.vessel_pod[bay, lr, other_hd]
 
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
        """赋值cell，同步扣减cbf。ctype='GP'扣capacity_total，'RF'扣capacity_rf。"""
        self.vessel_pod[bay, lr, hd] = pod
        self.vessel_type[bay, lr, hd] = ctype
        cap = self.capacity_rf[bay, lr, hd] if ctype == "RF" else self.capacity_total[bay, lr, hd]
        self.cbf[self.current_pol][pod][ctype] -= cap
 
    def unassign(self, bay, lr, hd, pod, ctype):
        """撤销赋值，精确恢复cbf。"""
        self.vessel_pod[bay, lr, hd] = -1
        self.vessel_type[bay, lr, hd] = None
        cap = self.capacity_rf[bay, lr, hd] if ctype == "RF" else self.capacity_total[bay, lr, hd]
        self.cbf[self.current_pol][pod][ctype] += cap

    # ── 多港口 ─────────────────────────────────────────────────────────
 
    def discharge(self, arriving_pod) -> list:
        """卸载arriving_pod的所有cell，返回记录供undischarge回溯。"""
        discharged = []
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    if self.vessel_pod[bay, lr, hd] == arriving_pod:
                        discharged.append((bay, lr, hd, arriving_pod, self.vessel_type[bay, lr, hd]))
                        self.vessel_pod[bay, lr, hd] = -1
                        self.vessel_type[bay, lr, hd] = None
        return discharged
 
    def undischarge(self, discharged: list):
        """精确恢复discharge的cell，不动cbf。"""
        for bay, lr, hd, pod, ctype in discharged:
            self.vessel_pod[bay, lr, hd] = pod
            self.vessel_type[bay, lr, hd] = ctype
 
    def advance_pol(self):
        """换港：current_pol指针+1。"""
        self.current_pol += 1
 
    # ── 快照 ───────────────────────────────────────────────────────────
 
    def snapshot(self) -> dict:
        """返回动态状态的深拷贝，用于跨港回溯或记录departure状态。"""
        return {
            "vessel_pod":  self.vessel_pod.copy(),
            "vessel_type": self.vessel_type.copy(),
            "cbf":         copy.deepcopy(self.cbf),
            "current_pol": self.current_pol,
        }
 
    def restore(self, snap: dict):
        """从snapshot恢复动态状态。"""
        self.vessel_pod  = snap["vessel_pod"].copy()
        self.vessel_type = snap["vessel_type"].copy()
        self.cbf         = snap["cbf"]
        self.current_pol = snap["current_pol"]
 
    # ── 导出 ───────────────────────────────────────────────────────────
 
    def export_cell_state(self) -> dict:
        """导出已赋值cell的状态，供viz展开到slot层面。
        返回 {(bay, lr, hd): (pod, ctype)}，只含vessel_pod != -1的cell。"""
        result = {}
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    pod = self.vessel_pod[bay, lr, hd]
                    if pod != -1:
                        result[(bay, lr, hd)] = (pod, self.vessel_type[bay, lr, hd])
        return result

    def build_vessel_cell(slots: pd.DataFrame, flag_col: str) -> np.ndarray:
        """把slots按flag_col筛选后，聚合成(7,2,2)的cell田字格。
        """
        cell = np.zeros((N_BAY, 2, 2), dtype=int)
        hits = slots[slots[flag_col]]
        for bay_idx, lr, hd in zip(hits.bay_idx, hits.lr, hits.hd):
            big_bay = _BIG_BAY_OF_B0.get(bay_idx)
            if big_bay is None:
                continue
            cell[big_bay, lr, hd] += 1
        return cell

    def build_init_state(slots: pd.DataFrame) -> pd.DataFrame:
        """空船初始状态：船上还没有任何箱子"""
        return pd.DataFrame(columns=BAYPLAN_COLUMNS)

    def proj_vessel_to_cell(slots: pd.DataFrame):
        """
        slot级 -> cell级(N_BAY,2,2)投影。
        要求slots已含can_40ft/can_reefer列（build_vessel_geometry + find_can_40ft +
        find_can_reefer之后的产物），聚合出Vessel构造需要的三个静态几何数组。
        返回 (is_valid, capacity_total, capacity_rf)。
        """
        capacity_total = build_vessel_cell(slots, "can_40ft")
        is_valid = capacity_total > 0
        slots = slots.copy()
        slots["can_reefer_40ft"] = slots["can_40ft"] & slots["can_reefer"]
        capacity_rf = build_vessel_cell(slots, "can_reefer_40ft")
        return is_valid, capacity_total, capacity_rf
    
    def proj_cell_to_vessel(self, cell_state: dict = None) -> pd.DataFrame:
        """
        cell级解 -> slot级DataFrame投影（proj_vessel_to_cell的逆方向）。
        cell_state: 不传则用当前self.vessel_pod/self.vessel_type；
                    传则接受snapshot()格式的dict（取其"vessel_pod"/"vessel_type"）。

        TODO 待实现：
          1. 若self.full_slot_table is None，报错："此Vessel无full_slot_table，无法投影"
          2. can_40ft=True的行只标在STSE_BAY_PAIRS每对的b0一侧（见vessel_io.find_can_40ft），
             需要用vessel_io.STSE_BAY_PAIRS把同一cell的解同时写回b0和b1两侧对应的
             (row_idx, tier_idx)行，否则展开出来的bayplan会缺一半物理槽位
          3. 返回列结构对齐parse_asc_file的输出（bay_idx,row_idx,tier_idx,POD,type等）
        """
        raise NotImplementedError

    def export_bayplan(self, snapshots: dict, out_dir: str, port_names: dict = None) -> list:
        """
        遍历snapshots（solve()产出的{POL: snapshot_dict}），对每个POL调用proj_cell_to_vessel，
        存成{POL}_{港口码}_DEP_bayplan.csv，落盘到out_dir，返回写出的文件路径列表。
        port_names: 可选{POL: 三字码}，不传则用POL数字编号命名。

        TODO 待proj_cell_to_vessel实现后再填充
        """
        raise NotImplementedError