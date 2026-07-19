"""
debug/verify_hc_quota_and_conservation.py - 独立校验脚本，不改动VesselClass.py。

复用main.py的ensure_geometry/ensure_cbf，跑一遍跟debug/tail_real_run.py同款的
seed=8245全流程求解，拿到snapshots/original_cbf/result_vessel，对每个POL调用
proj_cell_to_vessel投影出slots DataFrame，做两组断言：

A. 摞级quota硬约束（按(bay_idx, row_idx, hd)分组统计每一摞的is_hc标签数，
   n=该摞can_40ft槽位数，quota(n)=Vessel._stack_hc_cap(n)，跟proj_cell_to_vessel
   内部用的同一条公式）：
   1. hold摞(hd=0)：is_hc标签数 <= quota(n)。
   2. deck摞(hd=1)：is_hc标签数只能是0或quota(n)，落在(0,quota(n))区间的
      "收尾摞混装"每个(POL,POD)分组最多允许1个。
   (bay_idx, row_idx, hd)已验证在can_40ft槽位里能唯一确定lr，不会把两个不同
   lr的摞混到一组。

B. 总量守恒（按(POL,POD)分组）：
   真实GP占用slot数 + 真实RF占用slot数 + is_hc=True总数 + 求解结束后
   self.cbf里该(POL,POD)剩余的(GP+HC+RF+HR) == original_cbf[POL][POD]的
   GP+HC+RF+HR原始总量。

C. 摞内occupied tier连续性（按(bay_idx, row_idx, hd)分组，即"摞"，
   (bay_idx, row_idx, hd)已验证在can_40ft槽位里能唯一确定lr）：
   实际占用(GP_count>0或RF_count>0)的tier_idx集合，必须是该摞全部
   can_40ft tier_idx(按tier_idx升序排序)的一个前缀——即从该摞最低tier
   开始连续占用，不能有空洞(例如低tier空、高tier反而占用)。这是
   _settle_leftover_gp/_settle_leftover_rf释放顺序(必须按tier_idx降序
   释放，先释放当前occupied里tier最高的)要保证的不变量。

同一个(POL,POD)分组会在它被discharge之前，原样出现在多张连续POL快照里
(proj_cell_to_vessel对同一分组的投影结果是幂等的，见其docstring)，B部分
只取每个分组第一次出现的那次投影结果计数，避免跨快照重复计入。

统计口径全部按full_slot_table的b0侧(STSE_BAY_PAIRS的第一个元素)去重，
不重复计入b1侧的镜像行——b1只是b0的物理镜像写回，不是独立的第二个箱子。
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
    original_cbf = copy.deepcopy(vessel.cbf)

    random.seed(8245)
    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    if result_vessel is None:
        print("solve()失败：连一个箱子都没能装上，无法继续校验")
        return

    print(f"solve success={success}, snapshots覆盖POL={sorted(snapshots.keys())}")

    b0_set = {b0 for (b0, b1) in STSE_BAY_PAIRS}

    a_checked = 0
    a_failed = 0
    a_hold_fail_msgs = []
    a_deck_fail_msgs = []

    c_checked = 0
    c_failed = 0
    c_fail_msgs = []

    # B部分：每个(POL,POD)分组只取第一次出现的投影结果计数
    b_seen = {}  # (pol, pod) -> {"real_gp":n, "real_rf":n, "hc":n}
    stacks_checked_keys = set()  # (pol_snapshot, bay_idx, row_idx, hd) 去重计数用

    for pol in sorted(snapshots.keys()):
        df = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        df_b0 = df[df.bay_idx.isin(b0_set)].copy()

        # ── A: 按(bay_idx, row_idx, hd)分组，只看can_40ft槽位 ──
        can40 = df_b0[df_b0.can_40ft]
        # 落在(0,quota)区间的deck摞，按(POL,POD)分组统计触发次数
        deck_mixed_by_group = {}

        for (bay_idx, row_idx, hd), sub in can40.groupby(["bay_idx", "row_idx", "hd"]):
            n = len(sub)
            quota = Vessel._stack_hc_cap(n, hd)
            hc_count = int(sub["is_hc"].sum())

            stack_key = (pol, bay_idx, row_idx, hd)
            if stack_key in stacks_checked_keys:
                continue
            stacks_checked_keys.add(stack_key)
            a_checked += 1

            # ── C: 摞内occupied tier必须从最低tier开始连续、无空洞 ──
            occupied_sub = sub[(sub["GP_count"] > 0) | (sub["RF_count"] > 0)]
            if not occupied_sub.empty:
                c_checked += 1
                full_tiers = sorted(sub["tier_idx"].tolist())
                occupied_tiers = sorted(occupied_sub["tier_idx"].tolist())
                expected_prefix = full_tiers[:len(occupied_tiers)]
                if occupied_tiers != expected_prefix:
                    c_failed += 1
                    lr = int(sub["lr"].iloc[0])
                    c_fail_msgs.append(
                        f"POL={pol}, bay_idx={bay_idx}, lr={lr}, hd={hd}, row_idx={row_idx}: "
                        f"占用tier集合={occupied_tiers}, 该摞全部tier(升序)={full_tiers}, "
                        f"应从最低tier连续占用(期望前缀={expected_prefix})"
                    )

            if hd == 0:
                # A.1 hold摞：is_hc标签数 <= quota(n)
                if hc_count > quota:
                    a_failed += 1
                    a_hold_fail_msgs.append(
                        f"POL={pol}, bay={bay_idx}, row_idx={row_idx}: "
                        f"is_hc标签数={hc_count} > quota({n})={quota}"
                    )
            else:
                # A.2 deck摞：is_hc标签数只能是0或quota(n)
                if hc_count == 0 or hc_count == quota:
                    continue
                if 0 < hc_count < quota:
                    # 收尾摞混装候选，记下来后面按(POL,POD)分组核查"最多1个"
                    pod_vals = set(sub.loc[sub["POD"] != -1, "POD"].unique())
                    pol_vals = set(sub.loc[sub["POL"] != -1, "POL"].unique())
                    if len(pod_vals) != 1 or len(pol_vals) != 1:
                        a_failed += 1
                        a_deck_fail_msgs.append(
                            f"POL={pol}, bay={bay_idx}, row_idx={row_idx}: "
                            f"deck摞落在(0,quota)区间但(POL,POD)不唯一 "
                            f"pol_vals={pol_vals}, pod_vals={pod_vals}"
                        )
                        continue
                    group_key = (next(iter(pol_vals)), next(iter(pod_vals)))
                    deck_mixed_by_group.setdefault(group_key, []).append(
                        (bay_idx, row_idx, n, quota, hc_count)
                    )
                else:
                    # hc_count > quota，理论上不该出现（deck摞quota硬上限）
                    a_failed += 1
                    a_deck_fail_msgs.append(
                        f"POL={pol}, bay={bay_idx}, row_idx={row_idx}: "
                        f"deck摞is_hc标签数={hc_count} > quota({n})={quota}"
                    )

        for group_key, stacks in deck_mixed_by_group.items():
            if len(stacks) > 1:
                a_failed += 1
                a_deck_fail_msgs.append(
                    f"POL={pol}, (POL,POD)={group_key}: 落在(0,quota)区间的"
                    f"deck摞有{len(stacks)}个(应<=1)，明细={stacks}"
                )

        # ── B: 按(POL,POD)分组，只取第一次出现的计数 ──
        occupied = df_b0[df_b0["POD"] != -1]
        for (p, d), sub in occupied.groupby(["POL", "POD"]):
            key = (p, d)
            if key in b_seen:
                continue
            real_gp = int(((sub["GP_count"] == 1) & (~sub["is_hc"])).sum())
            real_rf = int(((sub["RF_count"] == 1) & (~sub["is_hc"])).sum())
            hc_count = int(sub["is_hc"].sum())
            b_seen[key] = {"real_gp": real_gp, "real_rf": real_rf, "hc": hc_count}

    # 补上original_cbf里存在、但从没在任何投影里出现过的(POL,POD)（demand完全没装船）
    for p, pods in original_cbf.items():
        for d in pods.keys():
            b_seen.setdefault((p, d), {"real_gp": 0, "real_rf": 0, "hc": 0})

    b_checked = 0
    b_failed = 0
    b_fail_msgs = []

    for (p, d), counts in sorted(b_seen.items()):
        b_checked += 1
        remaining = result_vessel.cbf.get(p, {}).get(d, {})
        remaining_total = sum(remaining.get(k, 0) for k in ("GP", "HC", "RF", "HR"))
        orig = original_cbf.get(p, {}).get(d, {})
        orig_total = sum(orig.get(k, 0) for k in ("GP", "HC", "RF", "HR"))

        computed_total = counts["real_gp"] + counts["real_rf"] + counts["hc"] + remaining_total

        if computed_total != orig_total:
            b_failed += 1
            b_fail_msgs.append(
                f"(POL={p}, POD={d}): 算出总量={computed_total} "
                f"(真GP={counts['real_gp']}, 真RF={counts['real_rf']}, "
                f"HC标签={counts['hc']}, cbf剩余={remaining_total}) != "
                f"原始总量={orig_total}, 差值={computed_total - orig_total}"
            )

    print("\n" + "=" * 70)
    print("A. 摞级quota硬约束 - 失败明细")
    print("=" * 70)
    if a_hold_fail_msgs:
        print(f"hold摞违规({len(a_hold_fail_msgs)}条):")
        for m in a_hold_fail_msgs:
            print("  " + m)
    if a_deck_fail_msgs:
        print(f"deck摞违规({len(a_deck_fail_msgs)}条):")
        for m in a_deck_fail_msgs:
            print("  " + m)
    if not a_hold_fail_msgs and not a_deck_fail_msgs:
        print("  (无)")

    print("\n" + "=" * 70)
    print("B. 总量守恒 - 失败明细")
    print("=" * 70)
    if b_fail_msgs:
        for m in b_fail_msgs:
            print("  " + m)
    else:
        print("  (无)")

    print("\n" + "=" * 70)
    print("C. 摞内occupied tier连续性 - 违规明细")
    print("=" * 70)
    if c_fail_msgs:
        for m in c_fail_msgs:
            print("  " + m)
    else:
        print("  (无)")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"(POL,POD)分组数 = {len(b_seen)}")
    print(f"A部分检查的摞数 = {a_checked}, 通过 = {a_checked - a_failed}, 失败 = {a_failed}")
    print(f"B部分检查的分组数 = {b_checked}, 通过 = {b_checked - b_failed}, 失败 = {b_failed}")
    print(f"C部分检查的摞数(有占用的) = {c_checked}, 通过 = {c_checked - c_failed}, 失败 = {c_failed}")

    all_ok = (a_failed == 0) and (b_failed == 0) and (c_failed == 0)
    print(f"\n最终结论: {'[OK] 全部通过' if all_ok else '[FAIL] 存在违规，见上方明细'}")

    assert a_failed == 0, f"A部分存在{a_failed}处摞级quota违规，见上方打印"
    assert b_failed == 0, f"B部分存在{b_failed}处总量守恒违规，见上方打印"
    assert c_failed == 0, f"C部分存在{c_failed}处摞内tier连续性违规，见上方打印"


if __name__ == "__main__":
    main()
