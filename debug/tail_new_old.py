"""
debug/tail_new_vs_old_real_run.py - 在真实STSE数据上对比新旧两版尾箱统计口径，
只统计数字、打印差异，不出bayplan文件。

跟debug/tail_real_run.py用完全同一份数据准备逻辑(main.ensure_geometry/ensure_cbf)
和同一个random.seed，保证新旧两版是在同一次solve()结果上对比，不是两次不同的求解
结果在打架：
    旧口径：build_unified_tail_list        （来源1+来源2+来源3独立相加，会重复计数）
    新口径：build_tail_container_list      （对每个(POL,POD)一次性比较"最终结果
                                              vs 原始demand"，count>0才出现）

然后把新口径的列表接到跟旧口径完全一样的下游接口上跑一遍
(scan_host_candidates -> match_tails_to_hosts -> apply_tail_placements)，
证明新格式("source": "final_vs_original")对这三个函数是即插即用的，不需要
改动它们的实现。

跑法：
    python debug/tail_new_vs_old_real_run.py
需要本机已有 data/STSE/... 真实数据（main.py的ensure_geometry/ensure_cbf会在
数据不存在时自动从data/STSE/raw构建；如果raw数据本身就没有，这一步会报错，
需要先把真实STSE数据放到data/STSE/下）。
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
    build_unified_tail_list, build_tail_container_list,
    scan_host_candidates, match_tails_to_hosts, apply_tail_placements,
)


def _summarize(label, records, count_types=("GP", "HC", "RF", "HR")):
    total = sum(rec["count"] for rec in records)
    by_type = {t: 0 for t in count_types}
    for rec in records:
        by_type[rec["type"]] = by_type.get(rec["type"], 0) + rec["count"]
    print(f"{label}: 总数={total}, 按type={by_type}")
    return total, by_type


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

    # final_cbf必须在此处深拷贝——solve()刚结束、任何proj_cell_to_vessel调用之前，
    # 旧口径(build_unified_tail_list)的来源1统计需要这个时间点。
    final_cbf = copy.deepcopy(result_vessel.cbf)

    print(f"solve()完成: success={success}, snapshots覆盖的POL数={len(snapshots)}")

    print("\n" + "=" * 60)
    print("──── 旧口径 vs 新口径 ────")
    print("=" * 60)

    old_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    old_total, old_by_type = _summarize("旧口径(来源1+2+3独立相加, build_unified_tail_list)", old_list)

    old_by_source = {}
    for rec in old_list:
        old_by_source[rec["source"]] = old_by_source.get(rec["source"], 0) + rec["count"]
    src_str = ", ".join(f"来源{s}={old_by_source.get(s, 0)}" for s in sorted(old_by_source))
    print(f"  旧口径按来源拆分: {src_str}")

    # build_tail_container_list内部会自己重新投影，不依赖build_unified_tail_list
    # 已经跑过的_tail_source2_log/_tail_source3_log状态，两者互不干扰，可以在
    # 同一个result_vessel上先后各跑一次。
    new_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
    new_total, new_by_type = _summarize("新口径(最终结果 vs 原始demand一次性比较, build_tail_container_list)", new_list)

    diff = old_total - new_total
    pct = (diff / old_total) if old_total else float("nan")
    print(f"\n差异：旧口径总尾箱数={old_total}, 新口径总尾箱数={new_total}, "
          f"减少={diff}（约{pct:.1%}）")
    print("按type对比(旧 -> 新):")
    for t in ("GP", "HC", "RF", "HR"):
        print(f"  {t}: {old_by_type.get(t, 0)} -> {new_by_type.get(t, 0)}")

    if new_total < old_total:
        print(f"[OK] 新口径总数({new_total}) < 旧口径总数({old_total})，"
              f"符合预期：新口径消除了来源1/来源3对同一批箱子的重复计数")
    else:
        print(f"[MISMATCH] 新口径总数({new_total}) 没有小于旧口径({old_total})，需要停下来查")

    # 用新口径的列表跑一遍完整下游管线，证明格式即插即用，不用改
    # scan_host_candidates/match_tails_to_hosts/apply_tail_placements的实现。
    print("\n" + "=" * 60)
    print("──── 新口径接入下游管线端到端验证 ────")
    print("=" * 60)

    host_pool = scan_host_candidates(result_vessel, snapshots)
    print(f"host候选池条数 = {len(host_pool)}")

    placements, unplaced = match_tails_to_hosts(
        new_list, host_pool, result_vessel.port_min, result_vessel.n_ports)
    placed_total = sum(p["count"] for p in placements)
    unplaced_total = sum(u["count"] for u in unplaced)

    print(f"新口径注入前尾箱总数 = {new_total}")
    print(f"  成功安置 = {placed_total}")
    print(f"  仍未安置 = {unplaced_total}")
    rate = placed_total / new_total if new_total else float("nan")
    print(f"  安置率 = {rate:.2%}")

    try:
        version1_dict, version2_dict = apply_tail_placements(result_vessel, snapshots, original_cbf, placements)
        print(f"\napply_tail_placements跑通: version1覆盖POL数={len(version1_dict)}, "
              f"version2覆盖POL数={len(version2_dict)}")
        print("[OK] 新口径列表格式与scan_host_candidates/match_tails_to_hosts/"
              "apply_tail_placements现有接口完全兼容，端到端跑通")
    except AssertionError:
        print("\n[apply_tail_placements headroom前置校验失败，AssertionError完整贴出]")
        raise


if __name__ == "__main__":
    main()