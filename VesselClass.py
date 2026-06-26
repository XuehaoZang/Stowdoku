import numpy as np
import copy


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
        is_valid: np.ndarray,        # shape (n_bay, 2, 2), bool
        capacity_total: np.ndarray,  # shape (n_bay, 2, 2), int
        capacity_rf: np.ndarray,     # shape (n_bay, 2, 2), int
        cbf: dict,                   # {POL: {POD: {"GP": count, "RF": count}}}
        current_pol: int = 0,
    ):
        # ── 静态几何层（只读） ──────────────────────────────────────────
        
        self.is_valid = is_valid.astype(bool)
        # is_valid[bay][lr][hd]: 该cell是否物理可用（capacity_total > 0）
        # 来源：几何文件STSE_slots_idx.csv，由外部pipeline提取后传入

        self.capacity_total = capacity_total.astype(int)
        # capacity_total[bay][lr][hd]: 该cell内有效40ft槽位总数
        # 用于GP类型赋值时的cbf扣减量

        self.capacity_rf = capacity_rf.astype(int)
        # capacity_rf[bay][lr][hd]: 该cell内有RF插座的槽位数
        # 用于RF类型赋值时的cbf扣减量

        self.has_reefer = capacity_rf > 0
        # has_reefer[bay][lr][hd]: bool，由capacity_rf推导，方便候选集过滤
        # has_reefer=True的cell候选集包含(POD, "RF")对, =False的cell只允许(POD, "GP")对

        self.n_bay = is_valid.shape[0]
        # 搜索空间的bay数量，测试时4，STSE时8

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