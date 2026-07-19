"""
debug/verify_settle_row_ledger.py - 验证"总账公式"能否替代_settle_row+group-level
两条独立写回self.cbf HC/HR的路径，只打印诊断信息，不改VesselClass.py/utils/tail.py
任何逻辑。

背景：proj_cell_to_vessel按(POL,POD)分组处理时，self.cbf[pol][pod]["HC"/"HR"]会被
两处独立写回：
    1. _settle_row：每释放1个leftover slot（未被贴HC标签、又没被gp_true_budget/
       rf_true_budget核销留住的slot）就+=1，按释放前是RF还是GP分别计入HR/HC。
    2. 分组循环结束后的group-level leftover写回：gp_hc_budget/rf_hc_budget分不完
       的预算池余量一次性+=进HC/HR。
提议的重构方向：不再让_settle_row直接写self.cbf，只在内存里累计"这个分组释放了
多少个GP来源/RF来源的slot"，分组处理完再用一次性公式赋值。本脚本不改代码，只用
"前后快照做减法"的方式反推出settle_row这一步单独贡献了多少，从而验证：
    A. final_HC == baseline_HC + settle_gp_released + gp_hc_budget_leftover
       （对HR同理），即两条写回路径的加总跟观测到的最终值完全对得上（这一步只是
       确认"减法反推"这个验证方法本身没问题，不是新结论）。
    B. gp_hc_budget_leftover 是否精确等于 原始HC demand − 该分组成功tagged的GP
       来源slot总数（RF/HR同理）——这是用户想验证的核心恒等式，如果这个都不成立，
       说明问题在_tag_stack核销budget的逻辑本身有分支没覆盖，不是writeback公式
       的问题。

数据源：跟debug/tail_diag_host_real_data.py一样，复用main.ensure_geometry/
ensure_cbf + solve()真实STSE数据链路。
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

TARGET_CASES = [(2, 3), (0, 4), (5, 0), (6, 3)]
B0_BAYS = {b0 for b0, b1 in STSE_BAY_PAIRS}


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
        print("solve()失败：连一个箱子都没能装上，无法继续验证")
        return

    print(f"solve()完成: success={success}, snapshots覆盖的POL数={len(snapshots)}, "
          f"port_min={result_vessel.port_min}, port_max={result_vessel.port_max}")

    # baseline：proj_cell_to_vessel跑之前的self.cbf（"来源1"残量，_settle_row/
    # group-level写回都是在这个基础上做加法）
    baseline_cbf = copy.deepcopy(result_vessel.cbf)

    snapshot_pols = sorted(snapshots.keys())

    # 对每个POL快照都跑一遍proj_cell_to_vessel（已写回的分组会被already_written
    # 跳过真正的self.cbf写入，但_tail_source3_log每次都会重新append，且应该是
    # 幂等的同一批数字——用first_seen去重取第一次出现的记录）。
    source3_first_seen = {}  # (pol,pod) -> (gp_hc_budget_leftover, rf_hc_budget_leftover)
    source2_count = {}  # (pol,pod) -> deck-squeeze触发次数
    tagged_first_seen = {}  # (pol,pod) -> (gp_tagged, rf_tagged)

    for pol in snapshot_pols:
        before_s3_len = len(result_vessel._tail_source3_log)
        before_s2_len = len(result_vessel._tail_source2_log)
        df = result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)

        for (p, d, gp_leftover, rf_leftover) in result_vessel._tail_source3_log[before_s3_len:]:
            source3_first_seen.setdefault((p, d), (gp_leftover, rf_leftover))

        for (p, d) in result_vessel._tail_source2_log[before_s2_len:]:
            source2_count[(p, d)] = source2_count.get((p, d), 0) + 1

        # is_hc标签统计：只在这个分组第一次出现的快照上记录（settle_row不改动
        # 已贴标签slot的GP_count/RF_count，理论上后续快照重跑应给出相同数字，
        # 但只取第一次出现的，跟self.cbf真正写回的时机对齐）。
        # 只取b0侧：一个40ft cell的is_hc/GP_count/RF_count会镜像写到b0和b1两侧
        # 20ft物理行（VesselClass.proj_cell_to_vessel docstring里"b1侧镜像"那段），
        # 不过滤会把每个真实高箱数一数成2个。
        hc_rows = df[(df["is_hc"] == True) & (df["bay_idx"].isin(B0_BAYS))]
        for (p, d), sub in hc_rows.groupby(["POL", "POD"]):
            key = (int(p), int(d))
            if key not in tagged_first_seen:
                gp_tagged = int((sub["GP_count"] == 1).sum())
                rf_tagged = int((sub["RF_count"] == 1).sum())
                tagged_first_seen[key] = (gp_tagged, rf_tagged)

    final_cbf = copy.deepcopy(result_vessel.cbf)

    all_groups = sorted(set(source3_first_seen.keys()) | set(tagged_first_seen.keys()))
    print(f"\n共发现{len(all_groups)}个存在HC/HR相关处理的(POL,POD)分组")

    found_targets = [g for g in TARGET_CASES if g in all_groups]
    missing_targets = [g for g in TARGET_CASES if g not in all_groups]
    print(f"目标case命中: {found_targets}")
    if missing_targets:
        print(f"目标case未出现在本次运行结果里（种子8245下这些(POL,POD)分组没有"
              f"触发HC/HR贴标处理，跟上次对话中提到的四个case不是同一次运行——"
              f"下面对全部{len(all_groups)}个分组做同样的验证，结论具有一般性）: "
              f"{missing_targets}")

    a_pass = a_fail = 0
    b_pass = b_fail = 0
    b_fail_msgs = []

    print("\n" + "=" * 100)
    for pol, pod in all_groups:
        orig_hc = original_cbf.get(pol, {}).get(pod, {}).get("HC", 0)
        orig_hr = original_cbf.get(pol, {}).get(pod, {}).get("HR", 0)
        base_hc = baseline_cbf.get(pol, {}).get(pod, {}).get("HC", 0)
        base_hr = baseline_cbf.get(pol, {}).get(pod, {}).get("HR", 0)
        fin_hc = final_cbf.get(pol, {}).get(pod, {}).get("HC", 0)
        fin_hr = final_cbf.get(pol, {}).get(pod, {}).get("HR", 0)
        gp_leftover, rf_leftover = source3_first_seen.get((pol, pod), (0, 0))
        gp_tagged, rf_tagged = tagged_first_seen.get((pol, pod), (0, 0))
        squeeze_n = source2_count.get((pol, pod), 0)

        settle_gp_released = fin_hc - base_hc - gp_leftover
        settle_rf_released = fin_hr - base_hr - rf_leftover

        is_target = (pol, pod) in TARGET_CASES
        marker = "  <<< 目标case" if is_target else ""

        print(f"(POL={pol}, POD={pod}){marker}")
        print(f"  original_cbf: HC={orig_hc}, HR={orig_hr}")
        print(f"  baseline(solve后、proj_cell_to_vessel前): HC={base_hc}, HR={base_hr}")
        print(f"  gp_tagged(is_hc且GP来源)={gp_tagged}, rf_tagged(is_hc且RF来源)={rf_tagged}, "
              f"deck-squeeze触发次数={squeeze_n}")
        print(f"  source3 leftover: gp_hc_budget_leftover={gp_leftover}, rf_hc_budget_leftover={rf_leftover}")
        print(f"  最终self.cbf: HC={fin_hc}, HR={fin_hr}")
        print(f"  反推settle_row贡献: settle_gp_released(->HC)={settle_gp_released}, "
              f"settle_rf_released(->HR)={settle_rf_released}")

        # A. 减法反推法本身的自洽性：baseline + settle_row贡献 + leftover == 最终值
        # （这一步理论上必然成立，只是复核一下没有算错）
        a_ok = (base_hc + settle_gp_released + gp_leftover == fin_hc) and \
               (base_hr + settle_rf_released + rf_leftover == fin_hr)
        if a_ok:
            a_pass += 1
        else:
            a_fail += 1
        print(f"  [A: 减法反推自洽性] {'OK' if a_ok else 'FAIL'}")

        # B. 核心恒等式：leftover == 原始demand − tagged数
        b_hc_ok = (gp_leftover == orig_hc - gp_tagged)
        b_hr_ok = (rf_leftover == orig_hr - rf_tagged)
        b_ok = b_hc_ok and b_hr_ok
        if b_ok:
            b_pass += 1
        else:
            b_fail += 1
            b_fail_msgs.append(
                f"(POL={pol}, POD={pod}): "
                f"HC侧 gp_hc_budget_leftover({gp_leftover}) {'==' if b_hc_ok else '!='} "
                f"原始HC demand({orig_hc}) - gp_tagged({gp_tagged}) = {orig_hc - gp_tagged}; "
                f"HR侧 rf_hc_budget_leftover({rf_leftover}) {'==' if b_hr_ok else '!='} "
                f"原始HR demand({orig_hr}) - rf_tagged({rf_tagged}) = {orig_hr - rf_tagged}"
            )
        print(f"  [B: leftover==原始demand-tagged数] "
              f"HC侧{'OK' if b_hc_ok else 'FAIL'}, HR侧{'OK' if b_hr_ok else 'FAIL'}")
        print()

    print("=" * 100)
    print(f"汇总: A(减法反推自洽性) 通过={a_pass}, 失败={a_fail}")
    print(f"汇总: B(leftover==原始demand-tagged数恒等式) 通过={b_pass}, 失败={b_fail}")
    if b_fail_msgs:
        print("\nB部分失败明细:")
        for m in b_fail_msgs:
            print("  " + m)
        print("\n结论: 恒等式在部分分组不成立 -> 问题出在_tag_stack核销budget的逻辑"
              "本身有分支没覆盖到，需要先查那里，不能只调整writeback公式。")
    else:
        print("\n结论: 恒等式在全部分组都成立 -> gp_hc_budget_leftover/rf_hc_budget_leftover"
              "本身就是'原始demand-tagged数'，_tag_stack没有问题；总账公式方向可行，"
              "只需要把settle_row的逐次+=改成累计计数、分组结束后一次性写self.cbf。")


if __name__ == "__main__":
    main()
