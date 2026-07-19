"""
debug/tail_diag_host_real_data.py - 诊断apply_tail_placements在真实STSE数据上
host=(bay=4, lr=1, hd=1, host_POL=6, POD=3)headroom校验失败的根因，只打印诊断
信息，不改utils/tail.py里apply_tail_placements的任何逻辑。

方法论完全复用debug/tail_diag_host.py（不重新设计，不改那个文件）：
    - 逐字复刻apply_tail_placements的处理流程（顺序、校验时机跟原函数完全
      一致——包括effective_start/effective_end的绕圈感知区间计算，这是
      utils/tail.py当前版本的逻辑，比debug/tail_diag_host.py里手抄的旧版本
      (直接用[tail_pol, pod)裸区间)要新，这里如实复刻当前版本）。
    - 唯一改动：把"对不上就AssertionError"改成"对不上就打印+继续"，这样能
      看到目标host在全部相关placement处理完之后的完整轨迹，而不是撞到第一
      条不一致就中断。
    - 只打印诊断信息，不改utils/tail.py/VesselClass.py/CSP_solver.py/main.py
      的任何逻辑，也不改debug/tail_diag_host.py本身。

数据源换成真实STSE数据，复用debug/tail_real_run_new_scheme.py现成的数据准备
(main.ensure_geometry/ensure_cbf) + solve() + build_tail_container_list链路
（当前口径，不是已经证伪的build_unified_tail_list旧口径）。

诊断目标（按用户要求逐条落实）：
    1. 打印所有命中host=(bay=4,lr=1,hd=1,host_POL=6,POD=3)的placement完整字段，
       以及apply_tail_placements处理这些placement的顺序。
    2. 打印每条处理前后，这个host涉及的每一张受影响POL快照上"预期已扣减量"
       vs "实际空槽位数"的对比（按快照拆开看，不只看全局累计数字）。
    3. 判断这个host是否横跨了存活区间(effective_start,effective_end)不完全
       重叠的多条placement（不同POL但同一个host_key）——如果是，说明按
       host_key做全局累计扣减是设计层面的问题，不是这一份真实数据偶然触发
       的边界case。
    4. 如果不是3，再看校验时机是否发生在注入动作之前（读到的是还没被本次
       placement处理过的旧快照状态）。
    5. 额外：统计全量数据里所有触发类似headroom不一致的host_key，看是个别
       host的孤立现象还是有规律的系统性偏差（数量、方向、差值分布）。
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
    build_tail_container_list, scan_host_candidates, match_tails_to_hosts,
    _tail_resource_kind, _select_empty_host_slots, _inject_tail_into_snapshot,
)
from utils.vessel_io import STSE_BAY_PAIRS

TARGET_HOST = (4, 1, 1, 6, 3)  # (bay, lr, hd, host_POL, POD)


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
        print("solve()失败：连一个箱子都没能装上，无法继续跑尾箱管线")
        return

    print(f"solve()完成: success={success}, snapshots覆盖的POL数={len(snapshots)}, "
          f"port_min={result_vessel.port_min}, port_max={result_vessel.port_max}, "
          f"n_ports={result_vessel.n_ports}")

    new_tail_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
    host_pool = scan_host_candidates(result_vessel, snapshots)
    placements, unplaced = match_tails_to_hosts(
        new_tail_list, host_pool, result_vessel.port_min, result_vessel.n_ports)
    print(f"new_tail_list总数={sum(r['count'] for r in new_tail_list)}, "
          f"host候选池条数={len(host_pool)}, placements条数={len(placements)}, "
          f"unplaced条数={len(unplaced)}")

    snapshot_pols = sorted(snapshots.keys())
    port_max = result_vessel.port_max

    # ── 1. 打印所有命中目标host的placement完整字段 ──
    print("=" * 78)
    print(f"目标host = {TARGET_HOST}  (bay, lr, hd, host_POL, POD)")
    print("=" * 78)
    target_placements = [
        p for p in placements
        if (p["host_bay"], p["host_lr"], p["host_hd"], p["host_POL"], p["POD"]) == TARGET_HOST
    ]
    print(f"\n命中目标host的placement共{len(target_placements)}条：")
    for p in target_placements:
        print(f"  {p}")

    host_entry = host_pool.get(TARGET_HOST)
    print(f"\nhost_pool中该host的静态entry: {host_entry}")
    print(f"snapshots覆盖的POL: {snapshot_pols}")

    b0_target = STSE_BAY_PAIRS[TARGET_HOST[0]][0]

    # ── 额外诊断：cell级聚合计数(GP_count/RF_count, scan_host_candidates口径)
    # vs slot级实际占用(proj_cell_to_vessel投影后的空槽位, apply_tail_placements
    # 校验口径)，在host自己的POL快照(6)上直接对比，不掺杂任何尾箱注入。 ──
    print("\n" + "=" * 78)
    print("额外诊断：cell级聚合计数 vs slot级实际占用（host自己的POL快照上，无尾箱干扰）")
    print("=" * 78)
    host_pol_snapshot = snapshots.get(TARGET_HOST[3])
    if host_pol_snapshot is not None:
        cell_record = host_pol_snapshot["cell"][TARGET_HOST[0], TARGET_HOST[1], TARGET_HOST[2]]
        print(f"snapshots[{TARGET_HOST[3]}]['cell'][{TARGET_HOST[0]},{TARGET_HOST[1]},{TARGET_HOST[2]}] = "
              f"{cell_record}")
        cap_total = host_entry["capacity_total"]
        cell_level_empty = cap_total - cell_record["GP_count"] - cell_record["RF_count"]
        print(f"cell级口径空槽位 = capacity_total({cap_total}) - GP_count({cell_record['GP_count']}) "
              f"- RF_count({cell_record['RF_count']}) = {cell_level_empty}  <- 这就是host_pool的"
              f"gp_headroom={host_entry['gp_headroom']}的来源")

        df_raw = result_vessel.proj_cell_to_vessel(cell_state=host_pol_snapshot, original_cbf=original_cbf)
        b0 = STSE_BAY_PAIRS[TARGET_HOST[0]][0]
        host_rows = df_raw[(df_raw["bay_idx"] == b0) & (df_raw["lr"] == TARGET_HOST[1])
                            & (df_raw["hd"] == TARGET_HOST[2])]
        occupied_rows = host_rows[host_rows["POD"] != -1]
        empty_rows = host_rows[host_rows["POD"] == -1]
        hc_rows = occupied_rows[occupied_rows["is_hc"] == True]
        print(f"slot级投影(未注入任何尾箱)：该host总槽位数={len(host_rows)}, "
              f"占用槽位数={len(occupied_rows)}, 空槽位数={len(empty_rows)}, "
              f"其中打了is_hc标签的占用槽位数={len(hc_rows)}")
        print(f"两种口径差值 = slot级空槽位({len(empty_rows)}) - cell级空槽位({cell_level_empty}) "
              f"= {len(empty_rows) - cell_level_empty}")
        if len(hc_rows) > 0:
            print("[发现] 该host存在is_hc标签的槽位——HC squeeze会把多个高柜集装箱的物理占用"
                  "压缩到更少的物理槽位（腾出空间），但cell级GP_count/RF_count是assign()阶段"
                  "记录的‘分配了几个箱子’，不会跟着squeeze同步减少，这正是两种口径产生差值的"
                  "来源：gp_headroom用的是‘箱子数’口径，_select_empty_host_slots用的是‘物理槽位’"
                  "口径，HC squeeze让二者出现落差。")
    else:
        print(f"snapshots里没有POL={TARGET_HOST[3]}这一张快照")

    # ── 2+3+4. 逐字复刻apply_tail_placements当前版本的处理流程 ──
    print("\n" + "=" * 78)
    print("复刻apply_tail_placements当前版本处理流程（校验失败时打印+继续，不中断）")
    print("=" * 78)

    version2_dict = {}
    for pol in snapshot_pols:
        df = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        version2_dict[pol] = df.copy(deep=True)

    ordered_placements = sorted(placements, key=lambda p: p["POL"])
    print(f"\n处理顺序（全部placements按tail.POL升序，共{len(ordered_placements)}条），"
          f"标出命中目标host的条目:")
    for i, p in enumerate(ordered_placements):
        hk = (p["host_bay"], p["host_lr"], p["host_hd"], p["host_POL"], p["POD"])
        marker = "  <<< 目标host" if hk == TARGET_HOST else ""
        print(f"  [{i}] POL={p['POL']} POD={p['POD']} type={p['type']} count={p['count']} "
              f"host={hk} source={p.get('source')}{marker}")

    consumed_by_host = {}  # host_key -> {"GP": 已安置量, "RF": 已安置量}  (全局累计，跟当前实现一致)
    consumed_by_host_per_pol = {}  # 诊断用：host_key -> {pol: {"GP":n,"RF":n}} 按快照拆开记录
    all_mismatches = []  # 诊断用：全量统计，每条(host_key, resource, actual, expected, diff, placement_idx)

    for i, placement in enumerate(ordered_placements):
        host_key = (placement["host_bay"], placement["host_lr"], placement["host_hd"],
                    placement["host_POL"], placement["POD"])
        entry = host_pool.get(host_key)
        if entry is None:
            print(f"  [{i}] host={host_key} 不在host_pool里，跳过（原实现会AssertionError）")
            continue

        ctype = placement["type"]
        count = placement["count"]
        tail_pol = placement["POL"]
        pod = placement["POD"]
        resource = _tail_resource_kind(ctype)

        used = consumed_by_host.setdefault(host_key, {"GP": 0, "RF": 0})
        is_target = (host_key == TARGET_HOST)

        # 当前版本的绕圈感知存活区间
        effective_start = max(placement["host_POL"], tail_pol)
        effective_end = pod if pod > effective_start else (port_max + 1)

        if is_target:
            print(f"\n---- 处理第[{i}]条 (目标host命中): "
                  f"tail_POL={tail_pol}, host_POL={placement['host_POL']}, POD={pod}, "
                  f"type={ctype}, count={count}, "
                  f"存活区间=[{effective_start},{effective_end}) ----")

        # 原实现的前置校验：只在effective_start那一张快照上检查
        check_df = version2_dict[effective_start]
        b0 = STSE_BAY_PAIRS[placement["host_bay"]][0]
        actual_empty = len(_select_empty_host_slots(
            check_df, b0, placement["host_lr"], placement["host_hd"], resource))
        static_headroom = entry["rf_headroom"] if resource == "RF" else entry["gp_headroom"]
        expected_remaining = static_headroom - used[resource]
        mismatch = (actual_empty != expected_remaining)

        if is_target:
            print(f"  [原实现校验] 快照POL={effective_start}: 实际空槽位={actual_empty}, "
                  f"静态headroom={static_headroom}, 全局已安置(累计)={used[resource]}, "
                  f"预期剩余={expected_remaining}, "
                  f"{'一致' if not mismatch else '!!不一致!!'}")

            affected_pols_for_this = [p for p in snapshot_pols if effective_start <= p < effective_end]
            print(f"  这条placement覆盖的快照区间[{effective_start},{effective_end}) -> "
                  f"受影响POL={affected_pols_for_this}")
            for chk_pol in snapshot_pols:
                df_chk = version2_dict[chk_pol]
                empty_now = len(_select_empty_host_slots(
                    df_chk, b0_target, TARGET_HOST[1], TARGET_HOST[2], resource))
                per_pol_used = consumed_by_host_per_pol.get(host_key, {}).get(chk_pol, {"GP": 0, "RF": 0})
                expected_by_pol = static_headroom - per_pol_used[resource]
                in_range = effective_start <= chk_pol < effective_end
                print(f"    快照POL={chk_pol}: 实际空槽位={empty_now}, "
                      f"该host在此快照上按'按快照拆分累计'预期扣减={per_pol_used[resource]} -> "
                      f"按快照口径预期剩余={expected_by_pol}, "
                      f"{'一致' if empty_now == expected_by_pol else '!!不一致!!'}, "
                      f"是否在本条覆盖区间内={in_range}")

        if mismatch:
            diff = actual_empty - expected_remaining
            all_mismatches.append({
                "idx": i, "host_key": host_key, "resource": resource,
                "actual": actual_empty, "expected": expected_remaining, "diff": diff,
                "tail_pol": tail_pol, "pod": pod, "host_POL": placement["host_POL"],
                "effective_start": effective_start, "effective_end": effective_end,
            })
            if is_target:
                print(f"  [原实现会在这里AssertionError] host={host_key} 资源={resource} "
                      f"实际空槽位({actual_empty}) != 预期剩余({expected_remaining}), diff={diff}"
                      f"（诊断脚本选择继续而非中断）")

        affected_pols = [p for p in snapshot_pols if effective_start <= p < effective_end]
        for pol in affected_pols:
            try:
                _inject_tail_into_snapshot(
                    version2_dict[pol], placement["host_bay"], placement["host_lr"], placement["host_hd"],
                    tail_pol, pod, ctype, count, host_key,
                )
            except AssertionError as exc:
                # 诊断脚本策略：注入本身也可能因为空槽位不足而失败（尤其是已经
                # 出现headroom不一致的host），打印+跳过这一张快照的注入，不中断
                # 整个统计流程，好让全量统计能跑完。
                print(f"  [注入失败，跳过] idx={i} host={host_key} pol快照={pol}: {exc}")
                continue
            per_pol_map = consumed_by_host_per_pol.setdefault(host_key, {})
            per_pol_entry = per_pol_map.setdefault(pol, {"GP": 0, "RF": 0})
            per_pol_entry[resource] += count

        used[resource] += count

        if is_target:
            print(f"  注入完成，全局累计consumed_by_host[{host_key}]={used}")

    # ── 3/4. 目标host的结论判定 ──
    print("\n" + "=" * 78)
    print("目标host结论判定")
    print("=" * 78)
    if len(target_placements) > 1:
        intervals = []
        for p in target_placements:
            es = max(p["host_POL"], p["POL"])
            ee = p["POD"] if p["POD"] > es else (port_max + 1)
            intervals.append((es, ee))
        print(f"目标host被{len(target_placements)}条placement共同命中，各自的存活区间"
              f"[effective_start,effective_end): {intervals}")
        pol_sets = [set(pol for pol in snapshot_pols if lo <= pol < hi) for lo, hi in intervals]
        overlap_fully = all(s == pol_sets[0] for s in pol_sets)
        print(f"各条placement覆盖的快照POL集合: {[sorted(s) for s in pol_sets]}")
        if overlap_fully:
            print("[判定] 所有placement的快照覆盖范围完全重叠 -> 不是'区间不完全重叠'的情形(c)，"
                  "问题原因需要看上面打印的'原实现校验' vs '按快照拆分'对比，判断是否是校验"
                  "时机本身的问题(d)。")
        else:
            print("[判定=c] 存在存活区间不完全重叠的多条placement命中同一个host_key —— "
                  "当前实现的consumed_by_host是按host_key做全局累计扣减，不区分具体快照，"
                  "这正是设计层面的问题：某张快照可能只被其中一部分placement覆盖，"
                  "但全局计数器却把所有命中过这个host_key的placement的count都算进'已扣减量'，"
                  "导致校验用的'预期剩余'比这张快照上实际发生的扣减更小（或更大），跟"
                  "actual_empty对不上。")
    else:
        print(f"目标host只被{len(target_placements)}条placement命中，不是多条placement区间不重叠"
              "的情形(c)，需要看上面打印的'原实现校验' vs '按快照拆分'对比，判断是否是校验"
              "时机本身的问题(d)（校验读到的是还没被本次placement处理过的旧快照状态，"
              "即时机在注入之前但预期计算口径有误）。")

    # ── 5. 全量统计：所有触发headroom不一致的host_key ──
    print("\n" + "=" * 78)
    print("全量统计：所有触发headroom不一致的host_key（不只看目标host）")
    print("=" * 78)
    mismatched_hosts = {}
    for m in all_mismatches:
        mismatched_hosts.setdefault(m["host_key"], []).append(m)

    print(f"\n处理过的placement总数={len(ordered_placements)}, "
          f"触发校验不一致的placement条数={len(all_mismatches)}, "
          f"涉及的不同host_key数={len(mismatched_hosts)}")

    if mismatched_hosts:
        diffs = [m["diff"] for m in all_mismatches]
        positive = sum(1 for d in diffs if d > 0)
        negative = sum(1 for d in diffs if d < 0)
        zero = sum(1 for d in diffs if d == 0)
        print(f"\ndiff=actual_empty-expected_remaining 方向统计: "
              f"actual>expected(diff>0)共{positive}条, actual<expected(diff<0)共{negative}条, "
              f"diff==0共{zero}条（理论上不应出现，因为mismatch已经排除了相等的情况）")
        print(f"diff分布: min={min(diffs)}, max={max(diffs)}, "
              f"去重后的diff取值集合={sorted(set(diffs))}")

        print(f"\n按host_key汇总（每个host_key下第一次出现mismatch时的详情，"
              f"以及该host_key总共触发了几次mismatch）:")
        for hk, ms in sorted(mismatched_hosts.items()):
            first = ms[0]
            hit_count = len([p for p in ordered_placements
                              if (p["host_bay"], p["host_lr"], p["host_hd"],
                                  p["host_POL"], p["POD"]) == hk])
            print(f"  host={hk}: 命中该host_key的placement总数={hit_count}, "
                  f"触发mismatch次数={len(ms)}, "
                  f"首次mismatch: idx={first['idx']}, resource={first['resource']}, "
                  f"actual={first['actual']}, expected={first['expected']}, diff={first['diff']}, "
                  f"tail_pol={first['tail_pol']}, pod={first['pod']}, "
                  f"存活区间=[{first['effective_start']},{first['effective_end']})")
    else:
        print("\n没有发现任何host_key触发headroom不一致（异常：说明目标host的不一致"
              "可能是本脚本诊断逻辑本身的问题，需要重新核对）")


if __name__ == "__main__":
    main()
