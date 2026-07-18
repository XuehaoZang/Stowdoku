"""
debug/verify_cross_pol_is_hc_stability.py - 独立校验脚本，不改动VesselClass.py。

复用main.py的ensure_geometry/ensure_cbf，跑一遍跟其它debug脚本同款的seed=8245
全流程求解，拿到snapshots/original_cbf/result_vessel。

任务4（跨港口一致性）：proj_cell_to_vessel的docstring声称，同一个(POL,POD)
分组在被discharge之前，会原样出现在它存活区间内的每一张POL快照里，且
is_hc贴标签是"幂等"重算——同一批物理slot在不同POL快照的投影里，is_hc
应该完全不变。这个脚本独立验证这条声明：

1. 从原始cbf里枚举所有(POL,POD)demand分组，对每个分组扫描所有快照的cell
   数组，找出它"在船未卸货"(cell记录里POL==此POL且POD==此POD)的快照POL
   连续区间（连续 = 在snapshots.keys()排序后的序列里前后相邻，不是POL数值
   相邻，因为可能有整数港口被跳过/绕圈）。
2. 挑出其中3个存活区间长度>=3的分组(POL,POD)。
3. 对每个挑中的分组，用区间内第一张快照投影出它占用的物理slot坐标集合
   (bay_idx, row_idx, tier_idx)(只取b0侧，b1只是镜像写回，不是独立箱子)。
4. 对区间内的每一张快照分别调用proj_cell_to_vessel，读出这批slot坐标各自
   的is_hc值，断言：同一个slot坐标在区间内所有快照里is_hc值必须完全一致。
   不一致则打印(POD, slot坐标, 各POL对应的is_hc值)。
"""
import copy
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ensure_geometry, ensure_cbf
from VesselClass import Vessel
from CSP_solver import solve
from utils.vessel_io import STSE_BAY_PAIRS


def _group_present(cell_state, pol, pod):
    """在这张快照的cell数组里，是否存在POL==pol且POD==pod的在船记录。"""
    cell = cell_state["cell"]
    for record in cell.flatten():
        if record["POL"] == pol and record["POD"] == pod:
            return True
    return False


def _longest_consecutive_run(sorted_pols, alive_set):
    """在sorted_pols(所有快照POL排序后的序列)里，找alive_set成员组成的最长
    连续子串(按序列里的前后相邻关系，不是POL数值相邻)，返回(start_idx, end_idx)
    闭区间下标，或None(没有任何alive)。"""
    best = None
    cur_start = None
    for i, pol in enumerate(sorted_pols):
        if pol in alive_set:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if best is None or length > (best[1] - best[0] + 1):
                    best = (cur_start, i - 1)
                cur_start = None
    if cur_start is not None:
        length = len(sorted_pols) - cur_start
        if best is None or length > (best[1] - best[0] + 1):
            best = (cur_start, len(sorted_pols) - 1)
    return best


def main():
    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()

    vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)
    original_cbf = copy.deepcopy(vessel.cbf)

    random.seed(8245)
    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    if result_vessel is None:
        print("solve()失败：连一个箱子都没能装上，无法继续校验")
        return

    sorted_pols = sorted(snapshots.keys())
    print(f"solve success={success}, snapshots覆盖POL={sorted_pols}")

    b0_set = {b0 for (b0, b1) in STSE_BAY_PAIRS}

    # ── 1+2：枚举所有(POL,POD)分组，找存活区间长度>=3的候选 ──
    candidates = []  # (run_length, pol_load, pod, run_pols)
    for pol_load, pods in original_cbf.items():
        for pod in pods.keys():
            alive_set = {
                p for p in sorted_pols
                if _group_present(snapshots[p], pol_load, pod)
            }
            if not alive_set:
                continue
            run = _longest_consecutive_run(sorted_pols, alive_set)
            if run is None:
                continue
            start_idx, end_idx = run
            run_pols = sorted_pols[start_idx:end_idx + 1]
            if len(run_pols) >= 3:
                candidates.append((len(run_pols), pol_load, pod, run_pols))

    candidates.sort(key=lambda c: c[0], reverse=True)
    print(f"\n找到{len(candidates)}个存活区间长度>=3的(POL,POD)分组，取前3个:")
    picked = candidates[:3]
    for length, pol_load, pod, run_pols in picked:
        print(f"  (POL={pol_load}, POD={pod}): 存活区间长度={length}, run_pols={run_pols}")

    if len(picked) < 3:
        print(f"\n[WARN] 只找到{len(picked)}个满足条件(>=3)的分组，不足3个，仍继续校验已找到的这些。")

    # ── 3+4：对每个挑中的分组做跨快照is_hc一致性校验 ──
    total_slots_checked = 0
    total_mismatches = 0
    mismatch_msgs = []

    for length, pol_load, pod, run_pols in picked:
        first_pol = run_pols[0]
        df_first = result_vessel.proj_cell_to_vessel(
            cell_state=snapshots[first_pol], original_cbf=original_cbf
        )
        df_first_b0 = df_first[df_first.bay_idx.isin(b0_set)]
        group_rows = df_first_b0[
            (df_first_b0["POL"] == pol_load) & (df_first_b0["POD"] == pod)
        ]
        slot_coords = list(zip(
            group_rows["bay_idx"], group_rows["row_idx"], group_rows["tier_idx"]
        ))
        print(f"\n(POL={pol_load}, POD={pod}) run={run_pols}: 占用物理slot数={len(slot_coords)}")

        # 逐快照收集这批slot坐标的is_hc值
        is_hc_by_pol = {}
        for p in run_pols:
            df = result_vessel.proj_cell_to_vessel(
                cell_state=snapshots[p], original_cbf=original_cbf
            )
            df_b0 = df[df.bay_idx.isin(b0_set)]
            lookup = {
                (row.bay_idx, row.row_idx, row.tier_idx): bool(row.is_hc)
                for row in df_b0[df_b0.bay_idx.isin({c[0] for c in slot_coords})].itertuples()
            }
            is_hc_by_pol[p] = lookup

        for coord in slot_coords:
            total_slots_checked += 1
            values = {}
            for p in run_pols:
                if coord not in is_hc_by_pol[p]:
                    values[p] = None
                else:
                    values[p] = is_hc_by_pol[p][coord]
            distinct = set(values.values())
            if len(distinct) > 1:
                total_mismatches += 1
                mismatch_msgs.append(
                    f"POD={pod}(POL={pol_load}), slot坐标(bay,row,tier)={coord}: "
                    f"各POL的is_hc值={values}"
                )

    print("\n" + "=" * 70)
    print("跨港口一致性 - 不一致明细")
    print("=" * 70)
    if mismatch_msgs:
        for m in mismatch_msgs:
            print("  " + m)
    else:
        print("  (无)")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"校验分组数 = {len(picked)}")
    print(f"校验slot总数(跨分组求和，每个slot按1次计) = {total_slots_checked}")
    print(f"不一致slot数 = {total_mismatches}")
    print(f"\n最终结论: {'[OK] 全部一致' if total_mismatches == 0 else '[FAIL] 存在跨快照is_hc不一致，见上方明细'}")

    assert total_mismatches == 0, f"存在{total_mismatches}处跨快照is_hc不一致，见上方打印"


if __name__ == "__main__":
    main()
