"""
临时验证脚本：核对 proj_cell_to_vessel 新增的第三步20ft relabel。跑完可以删。

检查内容：
    1. 传cbf_with_20 vs 不传，两次投影结果只有is_20ft列不同，其余列完全一致。
    2. 每个(record POL, POD)实际被标记is_20ft=True的slot数，应等于
       cbf_with_20里对应floor((20GP+20HC)/2)*2；对不上的（候选池不够用）单独列出。
    3. 幂等性：同一个未discharge的(record POL, POD)，从两个不同的导出快照分别跑
       一遍relabel，选中的slot集合（(bay_idx,row_idx,tier_idx)）必须完全相同。

用法（从repo根目录运行）：
    python debug/verify_20ft_relabel.py
"""

import os
import sys
import json
import random
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VesselClass import Vessel
from CSP_solver import solve

GEOMETRY_DIR = "data/STSE/geometry"
CBF_JSON = "data/STSE/cbf/cbf.json"
CBF_WITH_20_JSON = "data/STSE/cbf/cbf_with_20.json"


def load_int_keyed(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        int(k1): {int(k2): v2 for k2, v2 in v1.items()}
        for k1, v1 in raw.items()
    }


def main():
    base_vessel = Vessel.load_vessel(GEOMETRY_DIR, CBF_JSON)
    cbf_with_20 = load_int_keyed(CBF_WITH_20_JSON)

    random.seed(8245)
    vessel = copy.deepcopy(base_vessel)
    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best,
                     ci_pol_enabled=True, ci_pod_enabled=True)
    result_vessel = vessel if success else best["vessel"]
    if result_vessel is None or not snapshots:
        print("求解失败，无法验证。")
        return

    original_cbf = copy.deepcopy(base_vessel.cbf)

    # ── 检查1：有/无cbf_with_20，只有is_20ft列不同 ────────────────────
    pol0 = sorted(snapshots.keys())[0]
    df_plain = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol0], original_cbf=original_cbf)
    df_relabel = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol0], original_cbf=original_cbf,
                                                     cbf_with_20=cbf_with_20)

    other_cols = [c for c in df_plain.columns if c != "is_20ft"]
    cols_match = (df_plain[other_cols].reset_index(drop=True) == df_relabel[other_cols].reset_index(drop=True)).all().all()
    plain_all_false = (~df_plain["is_20ft"]).all()
    print(f"[检查1] 其余列完全一致: {cols_match}；不传cbf_with_20时is_20ft恒False: {plain_all_false}；"
          f"传了之后is_20ft=True的行数: {int(df_relabel['is_20ft'].sum())}")

    # ── 检查2：按(record POL, POD)统计relabel数量 vs 期望值 ───────────────
    print("\n[检查2] POL POD 实际relabel数 期望数(floor((20GP+20HC)/2)*2) 差值")
    shortfalls = []
    grouped = df_relabel[df_relabel.POD != -1].groupby(["POL", "POD"])
    for (pol, pod), sub in grouped:
        actual = int(sub["is_20ft"].sum())
        demand20 = cbf_with_20.get(pol, {}).get(pod, {})
        expected = ((demand20.get("20GP", 0) + demand20.get("20HC", 0)) // 2) * 2
        diff = actual - expected
        print(f"  {pol:<5}{pod:<5}{actual:<10}{expected:<10}{diff}")
        if diff != 0:
            shortfalls.append((pol, pod, actual, expected, diff))

    if shortfalls:
        print(f"\n!! 有{len(shortfalls)}组POL/POD relabel数量对不上期望值（候选GP slot不够用，"
              f"和奇数尾箱一起属于已知的边界情况）：")
        for pol, pod, actual, expected, diff in shortfalls:
            print(f"  POL={pol} POD={pod} 实际={actual} 期望={expected} 差={diff}")
    else:
        print("\n全部(POL,POD)组relabel数量与期望值完全吻合。")

    # ── 检查3：幂等性 —— 同一个未discharge的(record POL, POD)在两个不同的
    # 导出快照里跑relabel，选中的slot集合必须完全相同 ────────────────────
    print("\n[检查3] 幂等性检查")
    pols_sorted = sorted(snapshots.keys())
    if len(pols_sorted) < 2:
        print("  只有一个POL快照，跳过幂等性检查。")
    else:
        dfs = {
            pol: result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf,
                                                      cbf_with_20=cbf_with_20)
            for pol in pols_sorted
        }
        # 找出至少在两个快照里都出现过POD!=-1的(record POL, POD)组合
        pairs_per_snapshot = {
            pol: set(zip(df.loc[df.POD != -1, "POL"], df.loc[df.POD != -1, "POD"]))
            for pol, df in dfs.items()
        }
        checked, mismatches = 0, []
        for i in range(len(pols_sorted)):
            for j in range(i + 1, len(pols_sorted)):
                pol_a, pol_b = pols_sorted[i], pols_sorted[j]
                common_pairs = pairs_per_snapshot[pol_a] & pairs_per_snapshot[pol_b]
                for record_pol, pod in common_pairs:
                    df_a, df_b = dfs[pol_a], dfs[pol_b]
                    set_a = set(map(tuple, df_a.loc[
                        (df_a.POL == record_pol) & (df_a.POD == pod) & df_a.is_20ft,
                        ["bay_idx", "row_idx", "tier_idx"]
                    ].values))
                    set_b = set(map(tuple, df_b.loc[
                        (df_b.POL == record_pol) & (df_b.POD == pod) & df_b.is_20ft,
                        ["bay_idx", "row_idx", "tier_idx"]
                    ].values))
                    checked += 1
                    if set_a != set_b:
                        mismatches.append((record_pol, pod, pol_a, pol_b, set_a, set_b))

        print(f"  跨快照对比了{checked}个(record POL, POD)组合。")
        if mismatches:
            print(f"  !! 有{len(mismatches)}处不幂等：")
            for record_pol, pod, pol_a, pol_b, set_a, set_b in mismatches:
                print(f"    record_POL={record_pol} POD={pod}: 快照POL={pol_a}选中{set_a} "
                      f"vs 快照POL={pol_b}选中{set_b}")
            assert False, "幂等性检查失败"
        else:
            print("  全部通过：同一批未discharge货物在不同导出快照里relabel结果完全一致。")


if __name__ == "__main__":
    main()
