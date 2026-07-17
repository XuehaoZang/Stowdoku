"""
debug/tail_diag_source3_double_count.py - 诊断来源3(HC/RF预算池分不完回退)
是否跟"已physically占用槽位"重复计数。只做核查+打印，不改任何现有逻辑。

背景：VesselClass.assign()里，HC需求和GP需求共享同一个物理footprint
（gp_used = gp_deduct_gp + gp_deduct_hc，demand["HC"]被当成demand["GP"]的
同类资源一起扣减capacity_total），RF/HR同理共享capacity_rf。也就是说，
一个POD的demand只要在assign()阶段被"装上船"，不管这份demand原本标的是
GP还是HC，占用的都是同一个物理槽位(cell["GP_count"])，没有区别。

而proj_cell_to_vessel里的is_hc贴标签是完全独立的第二步：按(POL,POD)共享
预算池(取original_cbf里的原始HC/HR总demand)，只在"这个槽位本来就已经被
占用(occupied_gp_in_stack/occupied_rf_in_stack)"的前提下才贴标签，贴不完
时(gp_hc_budget/rf_hc_budget剩余>0)记进_tail_source3_log——这个"贴不完"
只可能是"贴标签预算超过了每摞capacity_hc配额"，不代表这些箱子没有物理槽位。

诊断内容（按要求逐条落实）：
    对每个触发过来源3的(POL,POD)分组：
    1. 打印物理占用量：这个POD在所有cell上的GP_count+RF_count总和。
    2. 打印原始demand总量：original_cbf[POL][POD]里GP+HC+RF+HR之和。
    3. 打印来源1残量(final_cbf里该POD剩下的量)、来源3回退量
       (gp_hc_budget/rf_hc_budget剩余)。
    4. 核对等式：原始demand总量 == 物理占用量 + 来源1残量（不含来源3）。
    5. 如果等式成立，统计来源3的箱子里有多少对应的POD物理占用量上其实
       已经"装满"了(即这些箱子实际已在船上，只是没贴HC标签)。

物理占用量的取数口径：直接读snapshots[pol]["cell"]（assign()阶段的cell级
记账），不读proj_cell_to_vessel投影出的slot级DataFrame。squeeze(来源2)是
proj_cell_to_vessel内部的后置动作，会把某个槽位物理腾空、GP_count清零，
这个"腾空"要到下一次真正执行proj_cell_to_vessel时才会把+1写回self.cbf
（且final_cbf是在proj_cell_to_vessel被调用之前捕获的），如果用
proj_cell_to_vessel的slot输出算物理占用量，会把squeeze"挪走"的那个槽位
算漏，让等式对不上——这不是来源3的问题、是来源2的副作用混进来污染了这条
诊断。cell级数组是assign()阶段的真实记账，不受squeeze/贴标签这些
proj_cell_to_vessel内部后置逻辑影响，才是"这个箱子在CSP求解阶段有没有
拿到物理槽位"的准确来源。
"""
import copy
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ensure_geometry, ensure_cbf
from VesselClass import Vessel
from CSP_solver import solve
from utils.tail import print_source2_and_source3_tail, _dedup_tail_log_by_pol_pod

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
    final_cbf = copy.deepcopy(result_vessel.cbf)
    print(f"solve()完成: success={success}, snapshots覆盖POL={sorted(snapshots.keys())}")

    # 复刻build_unified_tail_list内部触发来源2/3日志的同一条路径（对每个POL
    # 快照真跑一次proj_cell_to_vessel），拿到去重后的来源3分组。
    print_source2_and_source3_tail(result_vessel, snapshots, original_cbf)
    dedup2 = _dedup_tail_log_by_pol_pod(result_vessel._tail_source2_log, key_len=2)
    dedup3 = _dedup_tail_log_by_pol_pod(result_vessel._tail_source3_log, key_len=2)

    print("\n" + "=" * 70)
    print(f"来源3触发的(POL,POD)分组数 = {len(dedup3)}")
    print("=" * 70)

    all_equal = True
    total_gp_leftover = 0
    total_rf_leftover = 0
    total_gp_leftover_onboard = 0
    total_rf_leftover_onboard = 0

    for (pol, pod), entry in sorted(dedup3.items()):
        _, _, gp_leftover, rf_leftover = entry
        total_gp_leftover += gp_leftover
        total_rf_leftover += rf_leftover

        # 1. 原始demand总量
        orig_demand = original_cbf.get(pol, {}).get(pod, {})
        orig_total = sum(orig_demand.get(k, 0) for k in ("GP", "HC", "RF", "HR"))

        # 2. 物理占用量：直接读这个组自己的POL快照的cell级数组(snapshots[pol]
        # ["cell"])，按(POL==pol, POD==pod)过滤求GP_count+RF_count之和。
        # 不用proj_cell_to_vessel的slot级DataFrame——squeeze(来源2)是
        # proj_cell_to_vessel内部的后置动作，只改slots局部变量和(仅在
        # already_written为False时)自身的self.cbf写回，不动snapshots/
        # self.cell，squeeze之后GP的+1回退要到下一次真正调用时才写进cbf，
        # 而final_cbf是在proj_cell_to_vessel被调用之前捕获的——如果用
        # proj_cell_to_vessel的slot输出算物理占用量，会把squeeze"挪走"的
        # 那1个槽位也算漏，让等式对不上，这不是来源3的问题、是来源2的
        # 副作用，混进来会污染这条诊断。cell级数组是assign()阶段的真实
        # 记账，不受squeeze/贴标签这些proj_cell_to_vessel内部后置逻辑影响，
        # 是"这个箱子在CSP求解阶段有没有拿到物理槽位"的准确来源。
        cell_arr = snapshots[pol]["cell"]
        physical_used = 0
        for bay in range(result_vessel.n_bay):
            for lr in range(2):
                for hd in range(2):
                    record = cell_arr[bay, lr, hd]
                    if record["POL"] == pol and record["POD"] == pod:
                        physical_used += record["GP_count"] + record["RF_count"]

        # 3. 来源1残量：final_cbf(solve()刚结束、proj_cell_to_vessel执行前)
        # 里这个(POL,POD)剩下的量
        leftover_counts = final_cbf.get(pol, {}).get(pod, {})
        source1_leftover = sum(leftover_counts.get(k, 0) for k in ("GP", "HC", "RF", "HR"))

        # 4. 核对等式
        rhs = physical_used + source1_leftover
        equal = (orig_total == rhs)
        all_equal = all_equal and equal

        print(f"\n(POL={pol}, POD={pod}):")
        print(f"  原始demand总量(original_cbf) = {orig_total} "
              f"(GP={orig_demand.get('GP',0)}, HC={orig_demand.get('HC',0)}, "
              f"RF={orig_demand.get('RF',0)}, HR={orig_demand.get('HR',0)})")
        print(f"  物理占用量(GP_count+RF_count, b0侧汇总) = {physical_used}")
        print(f"  来源1残量(final_cbf剩余) = {source1_leftover} "
              f"(GP={leftover_counts.get('GP',0)}, HC={leftover_counts.get('HC',0)}, "
              f"RF={leftover_counts.get('RF',0)}, HR={leftover_counts.get('HR',0)})")
        print(f"  来源3回退量 = gp_hc_budget剩{gp_leftover} + rf_hc_budget剩{rf_leftover} "
              f"= {gp_leftover + rf_leftover}")
        print(f"  等式核对: 原始demand({orig_total}) == 物理占用({physical_used}) + 来源1残量"
              f"({source1_leftover}) = {rhs}  -> {'成立' if equal else '不成立'}")

        if equal:
            # 5. 等式成立时，物理占用量已经完整解释了"原始demand-来源1残量"这部分，
            # 也就是来源3回退的这些箱子必然全部落在physical_used里面
            # （物理占用量本身就是"原始demand减去来源1残量"，跟来源3无关地
            # 独立算出来的，来源3的leftover是在同一批已占用槽位里"贴不完标签"
            # 的子集，不是额外多出来的物理需求）。
            gp_onboard = min(gp_leftover, physical_used)
            rf_onboard = min(rf_leftover, physical_used)
            total_gp_leftover_onboard += gp_onboard
            total_rf_leftover_onboard += rf_onboard
            print(f"  -> 等式成立：来源3回退的{gp_leftover + rf_leftover}箱(GP={gp_leftover},RF={rf_leftover})"
                  f"全部对应已physically占用的槽位(只是没贴HC/HR标签)，"
                  f"理论上不该再被当成需要额外安置的尾箱")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"来源3触发的{len(dedup3)}个(POL,POD)分组，全部等式成立 = {all_equal}")
    print(f"来源3回退总箱数 = GP合计{total_gp_leftover} + RF合计{total_rf_leftover} "
          f"= {total_gp_leftover + total_rf_leftover}")
    print(f"其中对应POD物理占用量已经'装满'(即已physically在船上，只是没贴标签)的箱数 = "
          f"GP合计{total_gp_leftover_onboard} + RF合计{total_rf_leftover_onboard} "
          f"= {total_gp_leftover_onboard + total_rf_leftover_onboard}")
    remaining = (total_gp_leftover + total_rf_leftover) - (total_gp_leftover_onboard + total_rf_leftover_onboard)
    print(f"来源3里剩下没被上面等式覆盖到的箱数 = {remaining} "
          f"({'0，全部来源3箱子都已physically在船上' if remaining == 0 else '非0，见上面逐组明细里等式不成立的分组'})")


if __name__ == "__main__":
    main()
