import numpy as np
import copy
import random
from VesselClass import Vessel
from utils.viz import print_vessel
# import sys
# sys.setrecursionlimit(10000)

def cal_candidates(vessel: Vessel) -> dict:
    """
    计算所有valid且未赋值的cell的候选集。
    某个具体cell候选为空（比如被舱盖约束锁死、或只是这个cell放不下剩余需求）
    只代表这个cell这次不用、留空，不代表整体失败。
    只有扫完所有cell后，choices整体为空、且仍有超过尾货阈值的需求没处理完，
    才是真正的死路（没有任何地方能再装下去了）。
    """
    choices = {}
    for bay in range(vessel.n_bay):
        for lr in range(2):
            for hd in range(2):
                if not vessel.is_valid[bay, lr, hd]:
                    continue
                if vessel.cell[bay, lr, hd]["POD"] != -1:
                    continue
                cands = vessel.get_candidates(bay, lr, hd)
                if cands:
                    choices[(bay, lr, hd)] = cands

    if not choices and len(vessel.remaining_pods()) > 0:
        return None  # 真正的dead cell：没有任何cell能接下剩余需求
    return choices


def _ci_marginal_cost(vessel: Vessel, bay: int, lr: int, hd: int) -> int:
    """
    TODO 需要继续思考有没有更好的算法来建模CI问题。
    假设往这个cell装满(用capacity_total估算，不用实际会装多少)，
    相邻bay对里最挤的那一对会变成多大——用来在mrv_select里挑"往哪个bay放
    对本港CI伤害最小"的格子。

    刻意不用assign()真实会用掉的量(gp_used+rf_used)去估算：如果用真实量，
    这个打分会被_pod_try_order选哪个POD牵着走——选一个剩余需求很少的POD，
    这一步实际装得少，打分会显得"贡献小、很均衡"，但那只是这个cell被浪费了
    大半容量，不是真的均衡。用capacity_total不依赖POD是谁，天然避开这个博弈。
    """
    hypothetical = vessel.current_port_bay_load.copy()
    hypothetical[bay] += vessel.capacity_total[bay, lr, hd]
    if len(hypothetical) < 2:
        return int(hypothetical.sum())
    return int(max(hypothetical[i] + hypothetical[i + 1] for i in range(len(hypothetical) - 1)))


def mrv_select(choices: dict, vessel: Vessel):
    """
    MRV选择：优先has_reefer且候选中有POD真的还需要RF的cell（保证冰箱能放），
    其次优先hold(hd=0)而非deck(hd=1)——hold一旦被deck盖住就永久锁死，
    先填hold能避免不必要地触发舱盖约束、减少回溯，
    再按CI边际代价升序（往这个bay放，本港最挤的相邻bay对会变多大，越小越好），
    组内最后按候选集大小升序，全部打平时用随机数打散（避免choices字典按bay
    从小到大插入、min()对打平情况总是确定性选第一个这个副作用，之前验证过
    这个副作用确实存在——加了这个随机tie-break之后CI有明显变化）。
    返回 (bay, lr, hd)
    """
    def priority(item):
        (bay, lr, hd), cands = item
        current_cbf = vessel.cbf[vessel.current_pol]
        has_rf_need = vessel.has_reefer[bay, lr, hd] and any(
            current_cbf[pod].get("RF", 0) > 0 for pod in cands
        )
        ci_cost = _ci_marginal_cost(vessel, bay, lr, hd)
        return (0 if has_rf_need else 1, 0 if hd == 0 else 1, ci_cost, len(cands), random.random())

    return min(choices.items(), key=priority)[0]


def _pod_try_order(cands, vessel, bay, lr, hd):
    """
    候选POD的尝试顺序：如果这个cell有reefer能力，优先尝试还有RF需求的POD
    （避免reefer cell被先分给一个只有GP需求的POD，白白浪费这个cell的reefer额度），
    其余按POD数值升序。
    """
    current_cbf = vessel.cbf[vessel.current_pol]
    has_reefer_here = vessel.has_reefer[bay, lr, hd]

    def key(pod):
        rf_need = has_reefer_here and current_cbf[pod].get("RF", 0) > 0
        return (0 if rf_need else 1, pod)

    return sorted(cands, key=key)

def _total_assigned(vessel: Vessel) -> int:
    """统计当前vessel.cell里已经装了多少箱(GP+RF)，作为"解的完整度"指标。"""
    total = 0
    for bay in range(vessel.n_bay):
        for lr in range(2):
            for hd in range(2):
                rec = vessel.cell[bay, lr, hd]
                if rec["POD"] != -1:
                    total += rec["GP_count"] + rec["RF_count"]
    return total

_solve_call_count = [0]

def solve(vessel: Vessel, is_debug=False, snapshots=None, best=None) -> bool:
    """
    统一大递归：装载 + 换港，discharge作为递归中的特殊节点。
    vessel内部维护current_pol和cbf状态。
    best: dict容器 {"assigned": int, "vessel": Vessel或None}，
          记录搜索过程中见过的、已装箱数最多的状态快照，用于失败时输出最优近似解。
    """
    _solve_call_count[0] += 1
    # if _solve_call_count[0] % 500 == 0:
    #     print(f"[depth debug] 已调用{_solve_call_count[0]}次, current_pol={vessel.current_pol}, "
    #           f"total_remaining={vessel.total_remaining()}")
    
    if snapshots is None:
        snapshots = {}
    if best is None:
        best = {"assigned": -1, "vessel": None}

    # 终止条件：已超过最后一个港口
    if vessel.current_pol > max(vessel.cbf.keys()):
        return True

    # 当前港装完 → 换港
    if vessel.port_complete():
        snapshots[vessel.current_pol] = vessel.snapshot()
        prev_port_bay_load = vessel.current_port_bay_load.copy()
        # 备份换港前的负载表，这一港如果整体失败要精确恢复，不能留着新港口的脏状态

        vessel.advance_pol()
        discharged = vessel.discharge(vessel.current_pol)
        vessel.reset_port_bay_load(discharged)

        if solve(vessel, is_debug, snapshots, best):
            return True

        # 下一港失败，回溯
        vessel.undischarge(discharged)
        vessel.current_pol -= 1
        vessel.current_port_bay_load = prev_port_bay_load
        del snapshots[vessel.current_pol]

        if is_debug:
            print(f"[Backtrack] 回溯到POL={vessel.current_pol}")
        return False

    # 计算候选集
    choices = cal_candidates(vessel)
    if choices is None:
        return False  # dead cell，cbf有余量但某cell放不下

    if not choices:
        # 所有空cell的cbf候选都已耗尽，等价于port_complete
        # 下一次递归开头的port_complete()会处理换港
        return solve(vessel, is_debug, snapshots, best)

    # MRV选位置
    pos = mrv_select(choices, vessel)
    bay, lr, hd = pos

    for pod in _pod_try_order(choices[pos], vessel, bay, lr, hd):
        vessel.assign(bay, lr, hd, pod)

        current_total = _total_assigned(vessel)
        if current_total > best["assigned"]:
            best["assigned"] = current_total
            best["vessel"] = copy.deepcopy(vessel)

        if solve(vessel, is_debug, snapshots, best):
            return True

        vessel.unassign(bay, lr, hd, pod)

    return False


if __name__ == "__main__":
    import pandas as pd

    def _make_pair_rows(b0, b1, cells):
        """cells: {(lr,hd): {"capacity": 0或1, "reefer": bool}}。
        capacity=1时给这对bay各生成一行占位slot，capacity=0则不生成（对应is_valid=False）。"""
        rows = []
        for (lr, hd), spec in cells.items():
            if spec["capacity"] == 0:
                continue
            row_idx = 0 if lr == 0 else 5
            tier_idx = 0 if hd == 0 else 4
            for bay_idx in (b0, b1):
                rows.append({
                    "bay_idx": bay_idx, "row_idx": row_idx, "tier_idx": tier_idx,
                    "lr": lr, "hd": hd,
                    "can_40ft": True, "can_20ft": False, "can_reefer": spec["reefer"],
                })
        return rows

    # 复现原先4-bay测试场景的is_valid/capacity/reefer分布，
    # 分别对应STSE_BAY_PAIRS的前4对(2,3)/(4,5)/(6,7)/(8,9)（big_bay 0-3）
    # 后3对(big_bay 4-6)不给数据，capacity_total=0，永远is_valid=False，不参与搜索
    rows = []
    rows += _make_pair_rows(2, 3, {
        (0, 0): {"capacity": 0, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 1, "reefer": False},
        (1, 1): {"capacity": 1, "reefer": True},
    })
    rows += _make_pair_rows(4, 5, {
        (0, 0): {"capacity": 1, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 1, "reefer": False},
        (1, 1): {"capacity": 1, "reefer": False},
    })
    rows += _make_pair_rows(6, 7, {
        (0, 0): {"capacity": 1, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 1, "reefer": True},
        (1, 1): {"capacity": 1, "reefer": False},
    })
    rows += _make_pair_rows(8, 9, {
        (0, 0): {"capacity": 1, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 0, "reefer": False},
        (1, 1): {"capacity": 1, "reefer": False},
    })
    full_slot_table = pd.DataFrame(rows)

    cbf = {
        0: {
            1: {"GP": 7, "RF": 1},
            2: {"GP": 5, "RF": 1},
        },
        1: {
            3: {"GP": 7, "RF": 1},
        },
    }

    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    vessel_init = copy.deepcopy(vessel)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)

    if success:
        print("\n──── Solution Found ────")
        result_vessel = vessel
    else:
        print("\n──── No Full Solution — 输出搜索过程中最优的近似解 ────")
        result_vessel = best["vessel"]

    if result_vessel is not None:
        print(f"共装箱数: {_total_assigned(result_vessel)}")
        print("剩余cbf（未能装上的部分）：")
        for pol, pod_dict in sorted(result_vessel.cbf.items()):
            for pod, counts in sorted(pod_dict.items()):
                if counts.get("GP", 0) > 0 or counts.get("RF", 0) > 0:
                    print(f"    POL={pol} POD={pod}: {counts}")
        print("[final state]")
        print_vessel(result_vessel)
    else:
        print("连一个箱子都没能装上")