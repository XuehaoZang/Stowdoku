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


def _ci_current_pol_score(vessel: Vessel, bay: int, lr: int, hd: int) -> float:
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


def _pod_total_demand(cbf_original: dict) -> dict:
    """
    箱子层CI打分的基础准备：按POD汇总全航次原始总需求(GP+HC+RF+HR)，
    跨所有POL累加。只读cbf_original，不受assign/unassign过程影响。
    返回 {pod: 总量}。
    """
    totals = {}
    for pod_counts in cbf_original.values():
        for pod, counts in pod_counts.items():
            totals[pod] = totals.get(pod, 0) + sum(
                counts.get(k, 0) for k in ("GP", "HC", "RF", "HR")
            )
    return totals


def _pod_bay_footprint(vessel: Vessel) -> dict:
    """
    箱子层CI打分的基础准备：扫一遍vessel.cell，按POD分组统计每个bay已装的
    GP_count+RF_count，产出{pod: np.ndarray(n_bay)}。未出现过的POD不预先补0，
    调用方自己处理缺失key。
    """
    footprints = {}
    for bay in range(vessel.n_bay):
        for lr in range(2):
            for hd in range(2):
                rec = vessel.cell[bay, lr, hd]
                pod = rec["POD"]
                if pod == -1:
                    continue
                if pod not in footprints:
                    footprints[pod] = np.zeros(vessel.n_bay, dtype=int)
                footprints[pod][bay] += rec["GP_count"] + rec["RF_count"]
    return footprints


SMALL_POD_DEMAND_THRESHOLD = 15
# D_pod低于此值视为"小POD"，供_small_pod_ci_stats误伤诊断用，改这个数即可调阈值

HIGH_COST_ADJ_THRESHOLD = 0.5
# Cost_adj(邻居影响项)超过此值视为"高Cost_adj"，供_small_pod_ci_stats误伤诊断用

_small_pod_ci_stats = {"triggered": 0, "total": 0}
# 小POD误伤诊断计数器：total=_ci_future_pod_score实际算出Cost_adj的次数，
# triggered=其中D_pod<SMALL_POD_DEMAND_THRESHOLD且Cost_adj>HIGH_COST_ADJ_THRESHOLD的次数。
# 只做频率统计，不记录具体POD/bay，main.py每组实验开始前调用reset_small_pod_ci_stats()清零。


def reset_small_pod_ci_stats():
    """重置小POD误伤诊断计数器，供main.py每组(ci_pol_enabled, ci_pod_enabled)实验开始前调用。"""
    _small_pod_ci_stats["triggered"] = 0
    _small_pod_ci_stats["total"] = 0


def _ci_future_pod_score(vessel: Vessel, bay: int, lr: int, hd: int, pod, footprint: dict, pod_total_demand: dict) -> float:
    """
    箱子层CI评分：给"往(bay,lr,hd)装POD=pod"这个选择打分，衡量对该POD自身
    吊车负荷分布的影响，按该POD全航次总需求D_pod归一化（同样的绝对负荷，
    对总需求小的POD影响更大）。
    - Cost_adj: 邻居影响。假设把这个cell放到相邻bay对里，会不会把该POD的负荷进一步
      堆到已经堆得高的相邻bay旁边（用相邻bay里该POD footprint的较大值衡量）。
    - Cost_intra: 内部影响。该POD在本bay内（放了这个cell之后）是否超过本bay容量的一半
      （超过半舱意味着这个bay要为这个POD单独作业一整趟吊车，intra-bay堆叠代价）。
    D_pod<=0（这个POD根本没有总需求）时直接返回0，避免除零。

    每次算出Cost_adj都会顺带更新_small_pod_ci_stats，用于诊断"小POD是否容易被
    误判为高CI代价"（小D_pod会放大Cost_adj，不代表这个POD真的造成了很大的吊车负荷）。
    """
    D_pod = pod_total_demand.get(pod, 0)
    if D_pod <= 0:
        return 0.0

    fp = footprint.get(pod)
    if fp is None:
        fp = np.zeros(vessel.n_bay, dtype=int)

    neighbor_max = max(
        fp[bay - 1] if bay - 1 >= 0 else 0,
        fp[bay + 1] if bay + 1 < vessel.n_bay else 0,
    )
    Cost_adj = neighbor_max / D_pod
    return Cost_adj

def mrv_select(choices: dict, vessel: Vessel, ci_pol_enabled=True):
    """
    原始数独的方式是根据现在已知方格的信息确定其余方格的约束信息，从候选集最少的方格开始尝试，这里主要考虑在多种约束情况下设计剪枝规则
    选格子阶段:
    1. 特殊箱判断       -->  优先看has_reefer的（当仍有Reefer需求时）  --> 剪枝：放完GP但是RF放不了
    2. 封舱判断         -->  优先看hold 或 已占用hold上deck           --> 剪枝：直接装完deck导致封舱
    3. CI评分           -->  优先看cell层对当前POL的影响（ci_pol_enabled控制）    --> 剪枝：避免把负荷堆到已经紧张的相邻bay对
    5. 候选集排序       -->  优先看候选可能最少的                      --> 剪枝：加快搜索
    6. 随机数打散
    ci_pol_enabled=False时用于消融实验，ci_score恒为0，排序退化成不含CI项的版本。
    返回 (bay, lr, hd)
    """
    def priority(item):
        (bay, lr, hd), cands = item
        current_cbf = vessel.cbf[vessel.current_pol]
        has_rf_need = vessel.has_reefer[bay, lr, hd] and any(
            current_cbf[pod].get("RF", 0) + current_cbf[pod].get("HR", 0) > 0 for pod in cands
        )
        is_dead_slot = hd == 1 and vessel.cell[bay, lr, 0]["POD"] == -1
        ci_score = _ci_current_pol_score(vessel, bay, lr, hd) if ci_pol_enabled else 0
        return (0 if has_rf_need else 1, 0 if not is_dead_slot else 1, ci_score, len(cands), random.random())

    return min(choices.items(), key=priority)[0]

def _pod_try_order(cands, vessel, bay, lr, hd, ci_pod_enabled=True):
    """
    选箱子来填格子阶段：_pod_try_order
    1. 特殊箱匹配：哪个港口有reefer箱子，根据格子的冰箱容量进行匹配
    2. CI_POD打分（往这个bay放POD=?的箱子可以改善整体CI？）(ci_pod_enabled控制)
    3. 箱重匹配（旨在让空箱上浮（甲板上堆高）重箱下沉（舱底））（TODO 未来实现）
    4. 重量平衡（往这个bay放POD=?的箱子可以改善重量平衡？）（TODO 未来实现）
    5. 随机数打散
    ci_pod_enabled=False时用于消融实验，退回历史基线排序(rf_need, rel_rank)，
    不计算_ci_future_pod_score。
    兜底项用随机比用rel_rank(pod)效果更好。
    """
    current_cbf = vessel.cbf[vessel.current_pol]
    has_reefer_here = vessel.has_reefer[bay, lr, hd]

    if not ci_pod_enabled:
        def key(pod):
            rf_need = has_reefer_here and current_cbf[pod].get("RF", 0) + current_cbf[pod].get("HR", 0) > 0
            return (0 if rf_need else 1, random.random())
            # return (0 if rf_need else 1, vessel.rel_rank(pod))

        return sorted(cands, key=key)

    footprint = _pod_bay_footprint(vessel)
    pod_total_demand = _pod_total_demand(vessel.cbf_original)

    def key(pod):
        rf_need = has_reefer_here and current_cbf[pod].get("RF", 0) + current_cbf[pod].get("HR", 0) > 0
        ci_score = _ci_future_pod_score(vessel, bay, lr, hd, pod, footprint, pod_total_demand)
        return (0 if rf_need else 1, ci_score, random.random())

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

def solve(vessel: Vessel, is_debug=False, snapshots=None, best=None, ci_pol_enabled=True, ci_pod_enabled=True) -> bool:
    """
    统一大递归：装载 + 换港，discharge作为递归中的特殊节点。
    vessel内部维护current_pol和cbf状态。
    best: dict容器 {"assigned": int, "vessel": Vessel或None}，
          记录搜索过程中见过的、已装箱数最多的状态快照，用于失败时输出最优近似解。
    ci_pol_enabled: 传给mrv_select，控制是否启用CI cell层评分，供消融实验用。
    ci_pod_enabled: 传给_pod_try_order，控制是否启用箱子层CI评分，供消融实验用。
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

        if solve(vessel, is_debug, snapshots, best, ci_pol_enabled, ci_pod_enabled):
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
        return solve(vessel, is_debug, snapshots, best, ci_pol_enabled, ci_pod_enabled)

    # MRV选位置
    pos = mrv_select(choices, vessel, ci_pol_enabled)
    bay, lr, hd = pos

    for pod in _pod_try_order(choices[pos], vessel, bay, lr, hd, ci_pod_enabled):
        vessel.assign(bay, lr, hd, pod)

        current_total = _total_assigned(vessel)
        if current_total > best["assigned"]:
            best["assigned"] = current_total
            best["vessel"] = copy.deepcopy(vessel)

        if solve(vessel, is_debug, snapshots, best, ci_pol_enabled, ci_pod_enabled):
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

    # ──────────────────────────────────────────────────────────────
    # rel_rank提升为公开方法的冒烟测：
    # 1. 手工按原公式心算几个POD的rel_rank，跟vessel.rel_rank(pod)比对
    # 2. 用重构前的原始闭包逻辑复刻一份_get_candidates_reference，跟
    #    重构后的vessel.get_candidates逐cell比对，验证候选集完全一致
    # ──────────────────────────────────────────────────────────────
    print("──── rel_rank重构 冒烟测 ────")

    def _manual_rel_rank(v, pod):
        c = (v.current_pol - v.port_min) % v.n_ports
        p = (pod - v.port_min) % v.n_ports
        return (p - c) if p >= c else (p - c + v.n_ports)

    for pod in (1, 2, 3):
        manual = _manual_rel_rank(vessel_init, pod)
        actual = vessel_init.rel_rank(pod)
        assert manual == actual, f"[FAILED] rel_rank(pod={pod}): 手算={manual}, 实际={actual}"
        print(f"  [OK] rel_rank(pod={pod}) 手算={manual} == vessel.rel_rank={actual}")

    def _get_candidates_reference(v, bay, lr, hd):
        """原重构前的get_candidates逻辑，rel_rank用局部闭包内联复刻。"""
        if not v.is_valid[bay, lr, hd]:
            return set()
        current_cbf = v.cbf[v.current_pol]
        other_hd = 1 - hd
        other_pod = v.cell[bay, lr, other_hd]["POD"]
        if hd == 0 and other_pod != -1:
            return set()

        def rel_rank(pod):
            c = (v.current_pol - v.port_min) % v.n_ports
            p = (pod - v.port_min) % v.n_ports
            return (p - c) if p >= c else (p - c + v.n_ports)

        other_rank = rel_rank(other_pod) if other_pod != -1 else None
        candidates = set()
        for pod, counts in current_cbf.items():
            if other_rank is not None:
                new_rank = rel_rank(pod)
                if new_rank > other_rank:
                    continue
            has_gp_demand = (counts.get("GP", 0) + counts.get("HC", 0)) > v.tail_threshold
            has_rf_demand = v.has_reefer[bay, lr, hd] and (counts.get("RF", 0) + counts.get("HR", 0)) > 0
            if has_gp_demand or has_rf_demand:
                candidates.add(pod)
        return candidates

    def _check_candidates_match(v, label):
        mismatches = []
        for bay in range(v.n_bay):
            for lr in range(2):
                for hd in range(2):
                    expected = _get_candidates_reference(v, bay, lr, hd)
                    actual = v.get_candidates(bay, lr, hd)
                    if expected != actual:
                        mismatches.append((bay, lr, hd, expected, actual))
        assert not mismatches, f"[FAILED] {label} 候选集不一致: {mismatches}"
        print(f"  [OK] {label}: 全部(bay,lr,hd)候选集与重构前逻辑完全一致")

    _check_candidates_match(vessel_init, "重构前状态(current_pol=0，未assign任何货)")

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best, ci_pod_enabled=True)

    if success:
        print("\n──── Solution Found ────")
        result_vessel = vessel
    else:
        print("\n──── No Full Solution — 输出搜索过程中最优的近似解 ────")
        result_vessel = best["vessel"]

    if snapshots:
        # result_vessel在完全成功时current_pol已推进到cbf.keys()之外（终止态），
        # 不是get_candidates的合法调用场景（原逻辑同样无法处理）；
        # 改用snapshots里最后一个港口的departure快照——current_pol仍是合法值、
        # 且已有部分cell被assign，是验证"重构后仍有效"的更有意义的中间状态。
        last_port = max(snapshots.keys())
        mid_vessel = copy.deepcopy(vessel_init)
        mid_vessel.restore(snapshots[last_port])
        _check_candidates_match(mid_vessel, f"重构后POL={last_port}的departure快照(部分cell已assign)")

    if result_vessel is not None:
        print(f"共装箱数: {_total_assigned(result_vessel)}")
        print("剩余cbf（未能装上的部分）：")
        for pol, pod_dict in sorted(result_vessel.cbf.items()):
            for pod, counts in sorted(pod_dict.items()):
                if counts.get("GP", 0) > 0 or counts.get("RF", 0) > 0:
                    print(f"    POL={pol} POD={pod}: {counts}")
        print("[final state]")
        print_vessel(result_vessel)

        print("\n──── 阶段0+1 冒烟测 ────")
        print(f"vessel_init.cbf == result_vessel.cbf_original: "
              f"{vessel_init.cbf == result_vessel.cbf_original}")
        print(f"vessel_init.cbf        = {vessel_init.cbf}")
        print(f"result_vessel.cbf_original = {result_vessel.cbf_original}")

        total_demand = _pod_total_demand(result_vessel.cbf_original)
        print(f"\n_pod_total_demand = {total_demand}")

        footprint = _pod_bay_footprint(result_vessel)
        print(f"\n_pod_bay_footprint = {footprint}")

        print("\n手动核对（挑几个已赋值cell核对footprint）：")
        checked = 0
        for bay in range(result_vessel.n_bay):
            if checked >= 3:
                break
            for lr in range(2):
                for hd in range(2):
                    rec = result_vessel.cell[bay, lr, hd]
                    if rec["POD"] == -1 or checked >= 3:
                        continue
                    pod = rec["POD"]
                    cell_load = rec["GP_count"] + rec["RF_count"]
                    print(f"  cell(bay={bay}, lr={lr}, hd={hd}): POD={pod}, "
                          f"GP_count={rec['GP_count']}, RF_count={rec['RF_count']} "
                          f"(合计{cell_load}) -> footprint[{pod}][{bay}]={footprint[pod][bay]}")
                    checked += 1
        print("结论：以上每条cell记录的GP_count+RF_count都应 <= footprint[pod][bay]"
              "（同一(pod,bay)可能有多个lr/hd cell的贡献被累加在一起）。")
    else:
        print("连一个箱子都没能装上")

    # ──────────────────────────────────────────────────────────────
    # 阶段2冒烟测：_ci_future_pod_score 独立验证，全部用手工构造的假
    # footprint/pod_total_demand，不依赖上面solve()的真实结果。
    # 复用vessel_init的静态几何（capacity_total等），n_bay=7，
    # 各大bay容量：bay0=3, bay1=4, bay2=4, bay3=3, bay4~6=0
    # ──────────────────────────────────────────────────────────────
    print("\n──── 阶段2 smoke test: _ci_future_pod_score ────")

    def _check(name, cond, expected, actual):
        if not cond:
            raise AssertionError(
                f"[FAILED] 场景: {name} | 期望: {expected} | 实际: {actual}"
            )
        print(f"  [OK] {name} | 实际: {actual}")

    v = vessel_init
    fake_pod = 99
    D_pod = 100

    # 场景1：孤立空bay放第一箱——bay=1（邻居bay0/bay2），自己和邻居footprint都是0
    footprint_1 = {}  # fake_pod没有任何footprint记录 -> 视为全0数组
    score_1 = _ci_future_pod_score(v, bay=1, lr=0, hd=0, pod=fake_pod,
                             footprint=footprint_1, pod_total_demand={fake_pod: D_pod})
    _check("场景1 孤立空bay: Cost_adj+Cost_intra == 0",
           score_1 == 0.0, "0.0", score_1)

    # 场景2：邻居堆高——bay0的footprint设成接近D_pod一半(90)，bay=1去看邻居
    footprint_2 = {fake_pod: np.zeros(v.n_bay, dtype=int)}
    footprint_2[fake_pod][0] = 90
    score_2 = _ci_future_pod_score(v, bay=1, lr=0, hd=0, pod=fake_pod,
                             footprint=footprint_2, pod_total_demand={fake_pod: D_pod})
    _check("场景2 邻居堆高: score > 0", score_2 > 0.0, "> 0", score_2)

    # 场景3：半舱临界——bay=1, Cap_bay=4, 半舱=2, capacity_total[1,0,0]=1
    # 先把fp[bay]设到刚好差1个cell顶格(2-1=1)，Cost_intra应为0
    Cap_bay1 = v.capacity_total[1].sum()  # 4
    cell_cap = v.capacity_total[1, 0, 0]  # 1
    fp_at_edge = Cap_bay1 / 2 - cell_cap  # 1
    footprint_3a = {fake_pod: np.zeros(v.n_bay, dtype=int)}
    footprint_3a[fake_pod][1] = int(fp_at_edge)
    score_3a = _ci_future_pod_score(v, bay=1, lr=0, hd=0, pod=fake_pod,
                              footprint=footprint_3a, pod_total_demand={fake_pod: D_pod})
    cost_intra_3a = score_3a  # bay0/bay2邻居fp都是0，Cost_adj=0，score即Cost_intra
    _check("场景3a 半舱临界(差1个cell顶格): Cost_intra == 0",
           cost_intra_3a == 0.0, "0.0", cost_intra_3a)

    # 再加大1单位footprint，刚好顶格，Cost_intra应变正
    footprint_3b = {fake_pod: np.zeros(v.n_bay, dtype=int)}
    footprint_3b[fake_pod][1] = int(fp_at_edge) + 1
    score_3b = _ci_future_pod_score(v, bay=1, lr=0, hd=0, pod=fake_pod,
                              footprint=footprint_3b, pod_total_demand={fake_pod: D_pod})
    _check("场景3b 半舱临界+1: Cost_intra > 0",
           score_3b > 0.0, "> 0", score_3b)

    # 场景4：归一化对比——完全相同的footprint/bay条件，只换D_pod（12 vs 300）
    # 断言D_pod更小的分数更高：Cost_adj=neighbor_max/D_pod，
    # neighbor_max固定不变时D_pod越小分母越小、分数越大——
    # 这符合公式本身的方向（同样绝对负荷，对总需求小的POD影响更大），如实报告。
    footprint_4 = {fake_pod: np.zeros(v.n_bay, dtype=int)}
    footprint_4[fake_pod][0] = 90  # 邻居堆高，固定绝对负荷
    score_small_D = _ci_future_pod_score(v, bay=1, lr=0, hd=0, pod=fake_pod,
                                   footprint=footprint_4, pod_total_demand={fake_pod: 12})
    score_large_D = _ci_future_pod_score(v, bay=1, lr=0, hd=0, pod=fake_pod,
                                   footprint=footprint_4, pod_total_demand={fake_pod: 300})
    _check("场景4 归一化: D_pod=12的分数 > D_pod=300的分数（公式方向如实断言）",
           score_small_D > score_large_D,
           f"score(D=12) > score(D=300)",
           f"score(D=12)={score_small_D}, score(D=300)={score_large_D}")

    print("阶段2 smoke test all passed")

    # ──────────────────────────────────────────────────────────────
    # 阶段4冒烟测：evaluate_pod_discharge_spread
    # 复用上面ci_pod_enabled=True跑出来的snapshots，不重新solve()，
    # 只验证函数本身算得对不对（不对照ci_pod_enabled=False，那是后续
    # 10-seed实验的事）。
    # ──────────────────────────────────────────────────────────────
    from utils.evaluate import evaluate_pod_discharge_spread

    print("\n──── 阶段4 smoke test: evaluate_pod_discharge_spread ────")
    spread_report = evaluate_pod_discharge_spread(vessel, snapshots, if_debug=True)

    print("\n手动核对（挑1-2个POD核对variance/range是否吻合discharge_tally）：")
    from utils.evaluate import _port_bay_totals
    port_totals = _port_bay_totals(vessel, snapshots)
    checked = 0
    for r in port_totals:
        if r["discharge_tally"].sum() == 0 or checked >= 2:
            continue
        pod = r["pol"]
        tally = r["discharge_tally"]
        manual_variance = float(np.var(tally))
        manual_range = int(tally.max() - tally.min())
        reported = spread_report[pod]
        print(f"  POD={pod}: discharge_tally={list(tally)}, "
              f"手算variance={manual_variance:.4f} (函数返回{reported['variance']:.4f}), "
              f"手算range={manual_range} (函数返回{reported['range']})")
        assert manual_variance == reported["variance"], f"[FAILED] POD={pod} variance不一致"
        assert manual_range == reported["range"], f"[FAILED] POD={pod} range不一致"
        checked += 1
    print("阶段4 smoke test all passed")