"""
debug/tail_real_run_new_scheme.py - 在真实STSE数据上验证build_tail_container_list，
并把新口径结果接到scan_host_candidates -> match_tails_to_hosts上做一次下游一致性验证。

跟debug/tail_real_run.py用同一份数据准备逻辑(main.ensure_geometry/ensure_cbf)和
同一个random.seed，但只读/只加验证代码，不改任何生产函数实现，也不改
debug/tail_real_run.py本身（那个继续跑旧口径全链路留作对照）。

校验标准统一用守恒不变量（P0阶段已证伪"新口径总数<=旧口径总数"这个假设，
不能再拿新旧口径大小关系当校验标准，见utils/tail.py._assert_tail_conservation
的说明）：
    对每个(POL, POD)：
        该(POL,POD)四类缺口之和
        == max(0, 该(POL,POD)总demand - 该(POL,POD)实际占用物理槽位数)
"实际占用物理槽位数"跟P0阶段口径一致，用utils/tail.py._physical_occupied_total。

下游一致性校验：
    sum(placements里的count) + sum(unplaced里的count) == sum(new_tail_list里的count)
    逐(POL,POD,type)也对一遍，不只对总数。
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
    _physical_occupied_total,
)


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

    print(f"solve()完成: success={success}, snapshots覆盖的POL数={len(snapshots)}")

    # 1. build_tail_container_list（新口径）
    new_tail_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
    new_tail_total = sum(rec["count"] for rec in new_tail_list)

    # original_cbf里出现过的每一个(POL,POD)
    pol_pod_pairs = sorted(
        (pol, pod)
        for pol, pod_dict in original_cbf.items()
        for pod in pod_dict.keys()
    )

    print("\n" + "=" * 78)
    print(f"──── 守恒不变量逐(POL,POD)校验（original_cbf覆盖{len(pol_pod_pairs)}个组合）────")
    print("=" * 78)

    conservation_ok = 0
    conservation_fail = []
    for pol, pod in pol_pod_pairs:
        demand = original_cbf.get(pol, {}).get(pod, {})
        gp_d, hc_d, rf_d, hr_d = (demand.get(k, 0) for k in ("GP", "HC", "RF", "HR"))
        demand_total = gp_d + hc_d + rf_d + hr_d

        gaps = {rec["type"]: rec["count"] for rec in new_tail_list
                if rec["POL"] == pol and rec["POD"] == pod}
        gp_g, hc_g, rf_g, hr_g = (gaps.get(k, 0) for k in ("GP", "HC", "RF", "HR"))
        actual_total_gap = gp_g + hc_g + rf_g + hr_g

        physical_occupied = _physical_occupied_total(result_vessel, snapshots, original_cbf, pol, pod)
        expected_total_gap = max(0, demand_total - physical_occupied)

        ok = (actual_total_gap == expected_total_gap)
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] POL={pol} POD={pod} | demand GP={gp_d} HC={hc_d} RF={rf_d} HR={hr_d} "
              f"(合计={demand_total}) | 新口径缺口 GP={gp_g} HC={hc_g} RF={rf_g} HR={hr_g} "
              f"(合计={actual_total_gap}) | 实际物理占用={physical_occupied} | "
              f"期望总缺口=max(0,{demand_total}-{physical_occupied})={expected_total_gap}")

        if ok:
            conservation_ok += 1
        else:
            conservation_fail.append({
                "POL": pol, "POD": pod, "demand_total": demand_total,
                "physical_occupied": physical_occupied,
                "expected_total_gap": expected_total_gap,
                "actual_total_gap": actual_total_gap,
                "gaps": gaps,
            })

    # 2. 下游一致性：scan_host_candidates -> match_tails_to_hosts
    print("\n" + "=" * 78)
    print("──── 下游一致性校验：placements+unplaced == new_tail_total ────")
    print("=" * 78)

    host_pool = scan_host_candidates(result_vessel, snapshots)
    print(f"host候选池条数 = {len(host_pool)}")

    placements, unplaced = match_tails_to_hosts(
        new_tail_list, host_pool, result_vessel.port_min, result_vessel.n_ports)
    placed_total = sum(p["count"] for p in placements)
    unplaced_total = sum(u["count"] for u in unplaced)

    downstream_total_ok = (placed_total + unplaced_total == new_tail_total)
    print(f"new_tail_list总数 = {new_tail_total}")
    print(f"placements总数 = {placed_total}, unplaced总数 = {unplaced_total}, "
          f"两者之和 = {placed_total + unplaced_total}")
    print(f"[{'OK' if downstream_total_ok else 'MISMATCH'}] 总数守恒: "
          f"{placed_total + unplaced_total} == {new_tail_total} ? {downstream_total_ok}")

    # 逐(POL,POD,type)对一遍
    def _by_key(records):
        agg = {}
        for r in records:
            key = (r["POL"], r["POD"], r["type"])
            agg[key] = agg.get(key, 0) + r["count"]
        return agg

    new_by_key = _by_key(new_tail_list)
    placed_by_key = _by_key(placements)
    unplaced_by_key = _by_key(unplaced)

    all_keys = set(new_by_key) | set(placed_by_key) | set(unplaced_by_key)
    downstream_key_mismatches = []
    for key in sorted(all_keys):
        n = new_by_key.get(key, 0)
        p = placed_by_key.get(key, 0)
        u = unplaced_by_key.get(key, 0)
        if p + u != n:
            downstream_key_mismatches.append((key, n, p, u))

    if downstream_key_mismatches:
        print(f"\n[MISMATCH] 逐(POL,POD,type)级别不一致，共{len(downstream_key_mismatches)}条：")
        for key, n, p, u in downstream_key_mismatches:
            pol, pod, ctype = key
            print(f"  POL={pol} POD={pod} type={ctype}: new_tail={n}, placed={p}, unplaced={u}, "
                  f"placed+unplaced={p + u}")
    else:
        print("\n[OK] 逐(POL,POD,type)级别全部一致，没有发现任何不一致")

    # 汇总
    print("\n" + "=" * 78)
    print("──── 汇总 ────")
    print("=" * 78)
    print(f"original_cbf覆盖的(POL,POD)组合总数 = {len(pol_pod_pairs)}")
    print(f"守恒不变量校验：通过 {conservation_ok}/{len(pol_pod_pairs)}, 失败 {len(conservation_fail)}")
    if conservation_fail:
        print("失败明细：")
        for f in conservation_fail:
            print(f"  POL={f['POL']} POD={f['POD']}: demand_total={f['demand_total']}, "
                  f"physical_occupied={f['physical_occupied']}, "
                  f"expected_total_gap={f['expected_total_gap']}, "
                  f"actual_total_gap={f['actual_total_gap']}, gaps={f['gaps']}")
    print(f"下游一致性校验：总数 {'OK' if downstream_total_ok else 'MISMATCH'} "
          f"({placed_total}+{unplaced_total} vs {new_tail_total}), "
          f"逐(POL,POD,type) {'全部一致' if not downstream_key_mismatches else f'{len(downstream_key_mismatches)}条不一致'}")


if __name__ == "__main__":
    main()
