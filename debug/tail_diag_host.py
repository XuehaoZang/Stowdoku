"""
debug/tail_diag_host.py - 诊断apply_tail_placements在真实数据下host=(1,0,1,3,2)
headroom校验失败的根因，只打印诊断信息，不改utils/tail.py里的任何逻辑。

复刻apply_tail_placements的处理流程（顺序、校验时机跟原函数完全一致，
逐字复制自utils/tail.py），但把校验从"对不上就AssertionError"改成
"对不上就打印+继续"，这样能看到目标host在全部相关placement处理完之后
的完整轨迹，而不是撞到第一条不一致就中断。

诊断目标（按用户要求逐条落实）：
    1. 打印所有命中host=(bay=1,lr=0,hd=1,host_POL=3,POD=2)的placement完整字段。
    2. 打印apply_tail_placements处理这些placement的顺序，以及每条处理前后，
       这个host涉及的每一张受影响POL快照上"预期已扣减量" vs "实际空槽位数"
       的对比（按快照拆开看，不只看全局累计数字）。
    3. 判断这个host是否横跨了存活区间不完全重叠的多条placement（不同POL但
       同一个host_key），如果是，说明当前实现按host_key（不区分具体快照）
       做全局累计扣减，是设计层面的问题。
    4. 如果不是3，再看校验时机是否发生在注入动作之前（读到的是还没被
       本次placement处理过的旧快照状态）。
"""
import copy
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ensure_geometry, ensure_cbf
from VesselClass import Vessel
from CSP_solver import solve
from utils.tail import (
    build_unified_tail_list, scan_host_candidates, match_tails_to_hosts,
    _tail_resource_kind, _select_empty_host_slots, _inject_tail_into_snapshot,
)
from utils.vessel_io import STSE_BAY_PAIRS

TARGET_HOST = (1, 0, 1, 3, 2)  # (bay, lr, hd, host_POL, POD)


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
    final_cbf = copy.deepcopy(result_vessel.cbf)

    unified_tail_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    host_pool = scan_host_candidates(result_vessel, snapshots)
    placements, unplaced = match_tails_to_hosts(
        unified_tail_list, host_pool, result_vessel.port_min, result_vessel.n_ports)

    # ── 1. 打印所有命中目标host的placement完整字段 ──
    print("=" * 70)
    print(f"目标host = {TARGET_HOST}  (bay, lr, hd, host_POL, POD)")
    print("=" * 70)
    target_placements = [
        p for p in placements
        if (p["host_bay"], p["host_lr"], p["host_hd"], p["host_POL"], p["POD"]) == TARGET_HOST
    ]
    print(f"\n命中目标host的placement共{len(target_placements)}条：")
    for p in target_placements:
        print(f"  {p}")

    host_entry = host_pool.get(TARGET_HOST)
    print(f"\nhost_pool中该host的静态entry: {host_entry}")

    snapshot_pols = sorted(snapshots.keys())
    print(f"\nsnapshots覆盖的POL: {snapshot_pols}")

    # ── 2+3+4. 复刻apply_tail_placements的处理流程，逐条打印诊断 ──
    print("\n" + "=" * 70)
    print("复刻apply_tail_placements处理流程（校验失败时打印+继续，不中断）")
    print("=" * 70)

    version2_dict = {}
    for pol in snapshot_pols:
        df = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        version2_dict[pol] = df.copy(deep=True)

    ordered_placements = sorted(placements, key=lambda p: p["POL"])
    print(f"\n处理顺序（全部placements按POL升序，共{len(ordered_placements)}条）:")
    for i, p in enumerate(ordered_placements):
        hk = (p["host_bay"], p["host_lr"], p["host_hd"], p["host_POL"], p["POD"])
        marker = "  <<< 目标host" if hk == TARGET_HOST else ""
        print(f"  [{i}] POL={p['POL']} POD={p['POD']} type={p['type']} count={p['count']} "
              f"host={hk} source={p.get('source')}{marker}")

    consumed_by_host = {}  # host_key -> {"GP": 已安置量, "RF": 已安置量}  (全局累计，跟原实现一致)
    consumed_by_host_per_pol = {}  # 诊断用：host_key -> {pol: {"GP":n,"RF":n}} 按快照拆开记录

    b0_target = STSE_BAY_PAIRS[TARGET_HOST[0]][0]

    for i, placement in enumerate(ordered_placements):
        host_key = (placement["host_bay"], placement["host_lr"], placement["host_hd"],
                    placement["host_POL"], placement["POD"])
        entry = host_pool.get(host_key)
        if entry is None:
            continue

        ctype = placement["type"]
        count = placement["count"]
        tail_pol = placement["POL"]
        pod = placement["POD"]
        resource = _tail_resource_kind(ctype)

        used = consumed_by_host.setdefault(host_key, {"GP": 0, "RF": 0})
        is_target = (host_key == TARGET_HOST)

        if is_target:
            print(f"\n---- 处理第[{i}]条 (目标host命中): "
                  f"tail_POL={tail_pol}, POD={pod}, type={ctype}, count={count} ----")

        # 原实现的前置校验：只在tail_pol那一张快照上检查
        check_df = version2_dict[tail_pol]
        b0 = STSE_BAY_PAIRS[placement["host_bay"]][0]
        actual_empty = len(_select_empty_host_slots(
            check_df, b0, placement["host_lr"], placement["host_hd"], resource))
        static_headroom = entry["rf_headroom"] if resource == "RF" else entry["gp_headroom"]
        expected_remaining = static_headroom - used[resource]

        if is_target:
            print(f"  [原实现校验] 快照POL={tail_pol}: 实际空槽位={actual_empty}, "
                  f"静态headroom={static_headroom}, 全局已安置(累计)={used[resource]}, "
                  f"预期剩余={expected_remaining}, "
                  f"{'一致' if actual_empty == expected_remaining else '!!不一致!!'}")

            # 诊断：按每张受影响快照，分别打印"预期已扣减量" vs "实际空槽位数"
            affected_pols_for_this = [p for p in snapshot_pols if tail_pol <= p < pod]
            print(f"  这条placement覆盖的快照区间 [{tail_pol}, {pod}) -> 受影响POL={affected_pols_for_this}")
            for chk_pol in snapshot_pols:
                df_chk = version2_dict[chk_pol]
                empty_now = len(_select_empty_host_slots(
                    df_chk, b0_target, TARGET_HOST[1], TARGET_HOST[2], resource))
                per_pol_used = consumed_by_host_per_pol.get(host_key, {}).get(chk_pol, {"GP": 0, "RF": 0})
                expected_by_pol = static_headroom - per_pol_used[resource]
                in_range = tail_pol <= chk_pol < pod
                print(f"    快照POL={chk_pol}: 实际空槽位={empty_now}, "
                      f"该host在此快照上按'按快照拆分累计'预期扣减={per_pol_used[resource]} -> "
                      f"按快照口径预期剩余={expected_by_pol}, "
                      f"{'一致' if empty_now == expected_by_pol else '!!不一致!!'}, "
                      f"是否在本条覆盖区间内={in_range}")

        if actual_empty != expected_remaining:
            print(f"  [原实现会在这里AssertionError] host={host_key} 资源={resource} "
                  f"实际空槽位({actual_empty}) != 预期剩余({expected_remaining})，"
                  f"（诊断脚本选择继续而非中断）")

        affected_pols = [p for p in snapshot_pols if tail_pol <= p < pod]
        for pol in affected_pols:
            _inject_tail_into_snapshot(
                version2_dict[pol], placement["host_bay"], placement["host_lr"], placement["host_hd"],
                tail_pol, pod, ctype, count, host_key,
            )
            per_pol_map = consumed_by_host_per_pol.setdefault(host_key, {})
            per_pol_entry = per_pol_map.setdefault(pol, {"GP": 0, "RF": 0})
            per_pol_entry[resource] += count

        used[resource] += count

        if is_target:
            print(f"  注入完成，全局累计consumed_by_host[{host_key}]={used}")

    # ── 3. 判断目标host是否横跨了存活区间不完全重叠的多条placement ──
    print("\n" + "=" * 70)
    print("结论判定")
    print("=" * 70)
    if len(target_placements) > 1:
        intervals = [(p["POL"], p["POD"]) for p in target_placements]
        print(f"目标host被{len(target_placements)}条placement共同命中，各自的存活区间[POL,POD): {intervals}")
        pol_sets = [set(pol for pol in snapshot_pols if lo <= pol < hi) for lo, hi in intervals]
        all_pols_union = set()
        for s in pol_sets:
            all_pols_union |= s
        overlap_fully = all(s == pol_sets[0] for s in pol_sets)
        print(f"各条placement覆盖的快照POL集合: {[sorted(s) for s in pol_sets]}")
        if overlap_fully:
            print("[判定] 所有placement的快照覆盖范围完全重叠 -> 不是'区间不完全重叠'的情形，"
                  "问题原因需要看第4点（校验时机）。")
        else:
            print("[判定] 存在存活区间不完全重叠的多条placement命中同一个host_key —— "
                  "当前实现的consumed_by_host是按host_key做全局累计扣减，不区分具体快照，"
                  "这正是设计层面的问题：某张快照可能只被其中一部分placement覆盖，"
                  "但全局计数器却把所有命中过这个host_key的placement的count都算进'已扣减量'，"
                  "导致校验用的'预期剩余'比这张快照上实际发生的扣减更小（或更大），跟"
                  "actual_empty对不上。")
    else:
        print(f"目标host只被{len(target_placements)}条placement命中，不是多条placement区间不重叠的情形，"
              "需要看上面打印的'原实现校验' vs '按快照拆分'的对比，判断是否是校验时机本身的问题"
              "（校验读到的是还没被本次placement处理过的旧快照状态，即时机在注入之前但预期计算口径有误）。")


if __name__ == "__main__":
    main()
