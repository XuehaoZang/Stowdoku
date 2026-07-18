"""
debug/verify_n1_stack_quota.py - 独立校验脚本，不改动VesselClass.py。

任务5（n=1边界）：quota(n)公式在n==1时是quota(1)=1（单槽摞的HC配额就是它
自己，见Vessel._stack_hc_cap）。这个脚本先扫full_slot_table，找出所有
can_40ft槽位数n=1的摞(按bay_idx/row_idx/lr/hd分组)；如果船上根本没有这种
单槽摞，直接打印"无n=1摞，跳过"结束。如果有，跑一遍跟其它debug脚本同款的
seed=8245全流程求解，对每张POL快照的投影结果检查这些摞的is_hc标签数是否
符合quota(1)=1这个上限（即该摞is_hc标签数只能是0或1，不能出现>=2的情况——
对n=1的摞这本来就是物理不可能的，这里只是显式断言确认一下，顺带覆盖
"quota(n)公式在最小边界n=1时是否被proj_cell_to_vessel正确遵守"这个问题）。
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


def main():
    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()

    vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)

    # ── 先扫full_slot_table找n=1的摞，不涉及求解 ──
    full_table = vessel.full_slot_table
    can40 = full_table[full_table.can_40ft]
    stack_sizes = can40.groupby(["bay_idx", "row_idx", "lr", "hd"]).size()
    n1_stacks = stack_sizes[stack_sizes == 1]

    if n1_stacks.empty:
        print("无n=1摞，跳过")
        return

    n1_keys = set(n1_stacks.index)  # {(bay_idx, row_idx, lr, hd)}
    print(f"找到{len(n1_keys)}个n=1的摞:")
    for (bay_idx, row_idx, lr, hd) in sorted(n1_keys):
        print(f"  bay_idx={bay_idx}, row_idx={row_idx}, lr={lr}, hd={hd}")

    quota_1 = Vessel._stack_hc_cap(1, 0)
    print(f"\nquota(1) = {quota_1} (预期=1)")

    # ── 跑一遍跟其它debug脚本同款的seed=8245全流程求解 ──
    original_cbf = copy.deepcopy(vessel.cbf)
    random.seed(8245)
    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    if result_vessel is None:
        print("solve()失败：连一个箱子都没能装上，无法继续校验")
        return

    print(f"\nsolve success={success}, snapshots覆盖POL={sorted(snapshots.keys())}")

    b0_set = {b0 for (b0, b1) in STSE_BAY_PAIRS}
    b0_n1_keys = {k for k in n1_keys if k[0] in b0_set}
    if not b0_n1_keys:
        print("[WARN] 找到的n=1摞全部落在b1(镜像)侧，b0侧没有——理论上不该发生"
              "(b0/b1几何应该对称)，仍按n1_keys原样校验，不做b0过滤。")
        b0_n1_keys = n1_keys

    checked = 0
    failed = 0
    fail_msgs = []

    for pol in sorted(snapshots.keys()):
        df = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        for (bay_idx, row_idx, lr, hd) in sorted(b0_n1_keys):
            sub = df[
                (df.bay_idx == bay_idx) & (df.row_idx == row_idx)
                & (df.lr == lr) & (df.hd == hd) & (df.can_40ft)
            ]
            if sub.empty:
                continue
            checked += 1
            hc_count = int(sub["is_hc"].sum())
            if hc_count > quota_1:
                failed += 1
                fail_msgs.append(
                    f"POL={pol}, bay_idx={bay_idx}, row_idx={row_idx}, lr={lr}, hd={hd}: "
                    f"is_hc标签数={hc_count} > quota(1)={quota_1}"
                )

    print("\n" + "=" * 70)
    print("n=1摞quota校验 - 失败明细")
    print("=" * 70)
    if fail_msgs:
        for m in fail_msgs:
            print("  " + m)
    else:
        print("  (无)")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"检查的(快照, n=1摞)组合数 = {checked}")
    print(f"通过 = {checked - failed}, 失败 = {failed}")
    print(f"\n最终结论: {'[OK] 全部符合quota(1)=1' if failed == 0 else '[FAIL] 存在违规，见上方明细'}")

    assert failed == 0, f"存在{failed}处n=1摞quota违规，见上方打印"


if __name__ == "__main__":
    main()
