"""
debug/tail_real_run.py - 在真实STSE数据上跑通尾箱处理管线，只统计数字，不出bayplan文件。

复用main.py现有的数据准备逻辑（ensure_geometry/ensure_cbf）加载真实Vessel+真实cbf，
跑一遍solve()拿到真实snapshots/original_cbf/final_cbf，依次跑：
    1. build_unified_tail_list  -> 注入前尾箱总数（按source分组）
    2. scan_host_candidates     -> host候选池条数
    3. match_tails_to_hosts     -> placements/unplaced总数
    4. apply_tail_placements    -> 只跑通，不落盘、不跑verify_cross_port_consistency

final_cbf必须在solve()刚结束、任何proj_cell_to_vessel调用之前深拷贝，
跟main.py/utils/tail.py里的同款时间点纪律保持一致。
"""
import copy
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ensure_geometry, ensure_cbf
from VesselClass import Vessel
from CSP_solver import solve
from utils.tail import build_unified_tail_list, scan_host_candidates, match_tails_to_hosts, apply_tail_placements


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

    # final_cbf必须在此处深拷贝——solve()刚结束、任何proj_cell_to_vessel调用之前。
    final_cbf = copy.deepcopy(result_vessel.cbf)

    print(f"solve()完成: success={success}, snapshots覆盖的POL数={len(snapshots)}")

    # 1. build_unified_tail_list
    unified_tail_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    tail_total = sum(rec["count"] for rec in unified_tail_list)
    tail_by_source = {}
    for rec in unified_tail_list:
        tail_by_source[rec["source"]] = tail_by_source.get(rec["source"], 0) + rec["count"]

    # 2. scan_host_candidates
    host_pool = scan_host_candidates(result_vessel, snapshots)
    print(f"host候选池条数 = {len(host_pool)}")

    # 3. match_tails_to_hosts
    placements, unplaced = match_tails_to_hosts(
        unified_tail_list, host_pool, result_vessel.port_min, result_vessel.n_ports)
    placed_total = sum(p["count"] for p in placements)
    unplaced_total = sum(u["count"] for u in unplaced)

    print("\n真实数据尾箱处理结果")
    print("─" * 20)
    print(f"处理前(尾箱逻辑接入前)：无法安置的箱子总数 = {tail_total}")
    src_str = ", ".join(f"来源{s}={tail_by_source.get(s, 0)}" for s in sorted(tail_by_source))
    print(f"  按来源: {src_str}")
    print("处理后(跑完match_tails_to_hosts)：")
    print(f"  成功安置 = {placed_total}")
    print(f"  仍未安置 = {unplaced_total}")
    rate = placed_total / tail_total if tail_total else float("nan")
    print(f"  安置率 = {rate:.2%}")

    # 4. apply_tail_placements（只跑通，不落盘、不跑verify_cross_port_consistency）
    try:
        version1_dict, version2_dict = apply_tail_placements(result_vessel, snapshots, original_cbf, placements)
        print(f"\napply_tail_placements跑通: version1覆盖POL数={len(version1_dict)}, "
              f"version2覆盖POL数={len(version2_dict)}")
    except AssertionError:
        print("\n[apply_tail_placements headroom前置校验失败，AssertionError完整贴出]")
        raise


if __name__ == "__main__":
    main()
