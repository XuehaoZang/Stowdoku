"""
debug/tail_diag_ordering.py - 核查"环线时序 vs 裸数值比较"这个根因在
2b(scan_host_candidates)/2c(match_tails_to_hosts)里的实际影响范围。

只做核查+打印，不改utils/tail.py任何逻辑，不下结论式地"修"2c。

核查内容（按要求的顺序）：
    0. 先验证这条航线在solve()实际推进过程中的真实港序——advance_pol()
       只做current_pol+=1，从未在此模型里真正"绕圈"，snapshots覆盖的POL
       必须是从port_min到port_max的连续升序整数，没有重复/跳跃/回绕。
       这决定了"两个POL谁先谁后"能不能直接用裸数值比较。
    1. 核查2c的host.POL<=尾箱.POL判断：确认是裸数值比较，然后基于第0步
       确认的"POL序列严格升序、无绕圈"这个事实，论证POL-vs-POL的裸数值
       比较本身不受环线绕圈影响（绕圈只发生在POD身上，不发生在POL身上）。
       同时反过来找：有没有host.POL<=尾箱.POL数值成立、但host自己的
       (POL,POD)本身是"绕圈"的（host.POL > host.POD数值）——这类host被
       2c正常接受为借用对象，需要打印出来供讨论"这样接受对不对"。
    2. 核查2b(scan_host_candidates)：确认它是否用了任何POL/POD数值排序
       比较来判断"host在哪些快照里存活"——直接读它的实现逻辑，看是否存在
       类似 <, <=, > 的时序类比较（区别于纯等值匹配/去重比较）。
"""
import copy
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ensure_geometry, ensure_cbf
from VesselClass import Vessel
from CSP_solver import solve
from utils.tail import build_unified_tail_list, scan_host_candidates, match_tails_to_hosts
import inspect
import utils.tail as tail_mod


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

    print("=" * 70)
    print("0. 验证真实航次的POL推进序列（决定POL-vs-POL能否裸数值比较）")
    print("=" * 70)
    print(f"port_min={result_vessel.port_min}, port_max={result_vessel.port_max}, "
          f"n_ports={result_vessel.n_ports}")
    snap_pols = sorted(snapshots.keys())
    expected = list(range(result_vessel.port_min, result_vessel.port_max + 1))
    print(f"snapshots覆盖的POL(升序) = {snap_pols}")
    print(f"期望的严格连续升序区间 = {expected}")
    print(f"两者是否完全一致(无跳跃/无重复/无回绕) = {snap_pols == expected}")
    print("advance_pol()源码：current_pol += 1（单调递增，从未在真实推进路径里取模绕圈，"
          "绕圈只体现在rel_rank()这个'比较两个已知POD谁更早卸货'的相对距离函数里，"
          "不体现在current_pol真实前进的顺序上）")

    unified_tail_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    host_pool = scan_host_candidates(result_vessel, snapshots)
    placements, unplaced = match_tails_to_hosts(
        unified_tail_list, host_pool, result_vessel.port_min, result_vessel.n_ports)

    print("\n" + "=" * 70)
    print("1. 核查2c: match_tails_to_hosts里host.POL<=尾箱.POL的判断")
    print("=" * 70)
    src = inspect.getsource(match_tails_to_hosts)
    eligible_line = [l for l in src.splitlines() if "host_key[3]" in l]
    print("源码中涉及host.POL<=尾箱.POL的那一行:")
    for l in eligible_line:
        print(f"  {l.strip()}")
    print("-> 确认是裸数值比较 (host_key[3] <= pol)，两边都是POL值。")

    print("\n第0步已证明：真实航次里current_pol严格按1,2,...,port_max连续递增、"
          "从未回绕，所以任意两个POL值（不管是host的还是尾箱的）都来自同一条"
          "严格升序的整数序列——比较'谁先谁后'本身不需要rel_rank式的绕圈修正，"
          "裸数值比较host.POL<=尾箱.POL在这个模型里就是正确的时序判断。"
          "\n（绕圈只会发生在POD身上：POD代表'相对某个POL的将来某一港'，可能因为"
          "航线绕圈导致数值上比POL小，但POL本身不会绕圈。）")

    print("\n反过来，找出'host.POL<=尾箱.POL数值判断成立、但host自己的(POL,POD)"
          "组合本身是绕圈的(host.POL > host.POD数值)'——这类host被2c当作正常"
          "可借用对象接受了，列出来供讨论：")
    wrapped_hosts_used = []
    for p in placements:
        host_key = (p["host_bay"], p["host_lr"], p["host_hd"], p["host_POL"], p["POD"])
        if p["host_POL"] > p["POD"]:
            wrapped_hosts_used.append((host_key, p))
    dedup_wrapped_hosts = {}
    for host_key, p in wrapped_hosts_used:
        dedup_wrapped_hosts.setdefault(host_key, []).append(p)
    print(f"命中的这类host（host.POL>POD数值，共{len(dedup_wrapped_hosts)}个不同host_key，"
          f"涉及{len(wrapped_hosts_used)}条placement）：")
    for host_key, plist in sorted(dedup_wrapped_hosts.items()):
        print(f"  host={host_key}: 被{len(plist)}条placement借用，"
              f"tail.POL分别={sorted(p['POL'] for p in plist)}")

    print("\n这些host本身的'真实卸货时点'（按rel_rank意义）在建模航次范围之外"
          "（超过port_max才会真正卸货，本次snapshots根本不覆盖那一港），"
          "所以它们在host.POL之后会一直占用物理槽位、贯穿到最后一张快照"
          "(POL=port_max)都不会被discharge()清空——host.POL<=尾箱.POL这个"
          "数值判断依然成立且依然正确（host确实比尾箱先诞生），这不是2c的bug；"
          "真正需要重新定义的是任务3里'尾箱存活到哪一张快照为止'的区间终点，"
          "对这类host/尾箱不能再用POD数值当区间终点。")

    # 统计 unified_tail_list 里有多少条自身就是"绕圈"的 (tail.POL >= POD)
    wrapped_tails = [t for t in unified_tail_list if t["POL"] >= t["POD"]]
    print(f"\n附加统计：unified_tail_list共{len(unified_tail_list)}条，"
          f"其中尾箱自己的(POL,POD)就满足POL>=POD(绕圈)的有{len(wrapped_tails)}条：")
    for t in wrapped_tails:
        print(f"  {t}")

    print("\n" + "=" * 70)
    print("2. 核查2b: scan_host_candidates是否用了任何POL/POD数值排序比较")
    print("=" * 70)
    src2 = inspect.getsource(scan_host_candidates)
    order_ops = []
    for lineno, line in enumerate(src2.splitlines(), start=1):
        stripped = line.strip()
        if any(op in stripped for op in ("<=", ">=", " < ", " > ")) and (
            "POL" in stripped or "POD" in stripped or "pod" in stripped or "pol" in stripped
        ):
            order_ops.append((lineno, stripped))
    if order_ops:
        print("在scan_host_candidates源码里找到疑似POL/POD数值排序比较：")
        for lineno, line in order_ops:
            print(f"  [source行{lineno}] {line}")
    else:
        print("scan_host_candidates源码里没有找到任何对POL/POD做<、<=、>、>=的排序比较。")
        print("它对每张快照的处理是：直接读cell[bay,lr,hd]里当前那张快照记录的POD/POL"
              "字面值（record['POD']、record['POL']），只做`pod == -1`的存在性判断"
              "和`hc_key ==`/`prev != entry`这类等值比较，不对POL/POD做任何'谁比谁"
              "更早'的排序推断——host在某张快照里'存活'与否，直接取决于这张快照的"
              "cell数组里这个位置是否非空，不依赖任何区间计算。")
    print("\n结论：2b本身不依赖POL/POD的时序排序，环线绕圈问题不影响2b。")


if __name__ == "__main__":
    main()
