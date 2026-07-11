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


def _ci_cell_score(vessel: Vessel, bay: int, lr: int, hd: int) -> float:
    """
    cell层CI评分：只用cell固有属性(capacity_total)和聚合状态(current_port_bay_load)，
    不看实际分配的POD/demand数量。
    假设往这个cell装满(用capacity_total估算)，跟bay直接相邻的1~2个pair
    (bay-1,bay)和(bay,bay+1)里，"pair实际负荷 vs pair按容量比例应得预算"
    的超出量，取两侧较大的一个；用max(0, ...)封底，未超预算记0。
    船头/船尾只有一侧邻居时，退化成单bay超出量。
    """
    load = vessel.current_port_bay_load
    budget = vessel.port_budget

    def target(b):
        return budget * vessel.bay_capacity_share[b]

    hypothetical_self = load[bay] + vessel.capacity_total[bay, lr, hd]

    pair_costs = []
    if bay - 1 >= 0:
        pair_costs.append(max(0.0,
            (load[bay - 1] + hypothetical_self) - (target(bay - 1) + target(bay))))
    if bay + 1 < vessel.n_bay:
        pair_costs.append(max(0.0,
            (hypothetical_self + load[bay + 1]) - (target(bay) + target(bay + 1))))
    if not pair_costs:
        pair_costs.append(max(0.0, hypothetical_self - target(bay)))

    return max(pair_costs)


def mrv_select(choices: dict, vessel: Vessel, ci_enabled=True):
    """
    原始数独的方式是根据现在已知方格的信息确定其余方格的约束信息，从候选集最少的方格开始尝试，这里主要考虑在多种约束情况下设计剪枝规则
    选格子阶段:
    1. 特殊箱判断       -->  优先看has_reefer的（当仍有Reefer需求时）  --> 剪枝：放完GP但是RF放不了
    2. 封舱判断         -->  优先看hold 或 已占用hold上deck           --> 剪枝：直接装完deck导致封舱
    （待定3. 高箱判断         -->  优先看has_hicube的（当仍有HC需求时）      --> 剪枝：分配的位置放不进这么多高箱(暂时不加高箱规则)）
    4. CI评分           -->  优先看cell层CI评分较低的（不启用时恒为0）    --> 剪枝：避免把负荷堆到已经紧张的相邻bay对
    5. 候选集排序       -->  优先看候选可能最少的                      --> 剪枝：加快搜索
    6. 随机数打散
    ci_enabled=False时用于消融实验，ci_score恒为0，排序退化成不含CI项的版本。
    返回 (bay, lr, hd)
    """
    def priority(item):
        (bay, lr, hd), cands = item
        current_cbf = vessel.cbf[vessel.current_pol]
        has_rf_need = vessel.has_reefer[bay, lr, hd] and any(
            current_cbf[pod].get("RF", 0) + current_cbf[pod].get("HR", 0) > 0 for pod in cands
        )
        is_dead_slot = hd == 1 and vessel.cell[bay, lr, 0]["POD"] == -1
        ci_score = _ci_cell_score(vessel, bay, lr, hd) if ci_enabled else 0
        return (0 if has_rf_need else 1, 0 if not is_dead_slot else 1, ci_score, len(cands), random.random())

    return min(choices.items(), key=priority)[0]

def _pod_try_order(cands, vessel, bay, lr, hd):
    """
    选箱子来填格子阶段：_pod_try_order
    1. 特殊箱匹配：哪个港口有reefer箱子，根据格子的冰箱容量进行匹配
    2. CI打分（往这个bay放POD=?的箱子可以改善整体CI？）（TODO CI评估函数部分需要继续推敲）
    3. 箱重匹配（旨在让空箱上浮（甲板上堆高）重箱下沉（舱底））（TODO 未来实现）
    4. 重量平衡（往这个bay放POD=?的箱子可以改善重量平衡？）（TODO 未来实现）
    5. 按照POD rel_rank降序（先装目的地远的箱子 TODO 先远后近是好的策略吗）
    """
    current_cbf = vessel.cbf[vessel.current_pol]
    has_reefer_here = vessel.has_reefer[bay, lr, hd]

    def key(pod):
        rf_need = has_reefer_here and current_cbf[pod].get("RF", 0) + current_cbf[pod].get("HR", 0) > 0
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

def solve(vessel: Vessel, is_debug=False, snapshots=None, best=None, ci_enabled=True) -> bool:
    """
    统一大递归：装载 + 换港，discharge作为递归中的特殊节点。
    vessel内部维护current_pol和cbf状态。
    best: dict容器 {"assigned": int, "vessel": Vessel或None}，
          记录搜索过程中见过的、已装箱数最多的状态快照，用于失败时输出最优近似解。
    ci_enabled: 传给mrv_select，控制是否启用CI cell层评分，供消融实验用。
    CI的事后评估交给utils.evaluate.evaluate_crane_intensity（基于solve()跑完后的
    snapshots算，跨港口口径统一），不在这里自己攒诊断数据。
    """
    _solve_call_count[0] += 1
    if _solve_call_count[0] % 100000 == 0:
        print(f"[depth debug] 已调用{_solve_call_count[0]}次, current_pol={vessel.current_pol}, "
              f"total_remaining={vessel.total_remaining()}")

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

        vessel.advance_pol()
        discharged = vessel.discharge(vessel.current_pol)
        vessel.reset_port_bay_load(discharged)

        if solve(vessel, is_debug, snapshots, best, ci_enabled):
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
        return solve(vessel, is_debug, snapshots, best, ci_enabled)

    # MRV选位置
    pos = mrv_select(choices, vessel, ci_enabled)
    bay, lr, hd = pos

    for pod in _pod_try_order(choices[pos], vessel, bay, lr, hd):
        vessel.assign(bay, lr, hd, pod)

        current_total = _total_assigned(vessel)
        if current_total > best["assigned"]:
            best["assigned"] = current_total
            best["vessel"] = copy.deepcopy(vessel)

        if solve(vessel, is_debug, snapshots, best, ci_enabled):
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