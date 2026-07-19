"""
main.py - 编排入口，负责数据准备 + 求解 + 导出的完整流水线

数据准备（若目标文件已存在则跳过，不重复构建）：
    1. geometry: 若 GEOMETRY_DIR/full_slot_table.csv 不存在，
       从 GEOMETRY_ALL_CSV 解析 + find_can_40ft/20ft/reefer 后落盘
    2. cbf:      若 CBF_JSON 不存在，从 CBF_RAW_DIR 批量解析 .cbf 文件后落盘

求解：
    3. Vessel.load_vessel() 读取上述两份数据构造Vessel
    4. solve() 跑CSP求解，snapshots记录每港departure状态

导出：
    5. vessel.export_bayplan() 把每港的cell级解投影回slot级，落盘bayplan csv
"""

import os
import json
import random
import copy
import time

import numpy as np
import pandas as pd

from utils.vessel_io import (
    build_vessel_geometry, find_can_40ft, find_can_20ft, find_can_reefer,
    batch_parse_cbf, batch_parse_cbf_with_20, STSE_PORT_MAP
)
from VesselClass import Vessel
from CSP_solver import solve, reset_small_pod_ci_stats, _small_pod_ci_stats
from utils.evaluate import evaluate_crane_time, evaluate_pod_discharge_spread

GEOMETRY_ALL_CSV = "data/STSE/geometry/all_slots.csv"
GEOMETRY_REEFER_CSV = "data/STSE/geometry/reefer_slots.csv"
GEOMETRY_DIR = "data/STSE/geometry"
CBF_RAW_DIR = "data/STSE/raw"
CBF_DIR = "data/STSE/cbf"
CBF_JSON = os.path.join(CBF_DIR, "cbf.json")
CBF_WITH_20_JSON = os.path.join(CBF_DIR, "cbf_with_20.json")
BAYPLAN_DIR = "data/STSE/bayplan"

PORT_NAMES = {v: k for k, v in STSE_PORT_MAP.items()}


def ensure_geometry() -> str:
    """确保full_slot_table.csv存在，不存在则从idx csv构建。返回geometry目录路径。"""
    out_path = os.path.join(GEOMETRY_DIR, "full_slot_table.csv")
    if os.path.exists(out_path):
        return GEOMETRY_DIR

    slots = build_vessel_geometry(GEOMETRY_ALL_CSV)
    slots = find_can_40ft(slots)
    slots = find_can_20ft(slots)
    slots = find_can_reefer(slots, GEOMETRY_REEFER_CSV)

    os.makedirs(GEOMETRY_DIR, exist_ok=True)
    slots.to_csv(out_path, index=False)
    print(f"[geometry] 已构建 {out_path}（{len(slots)}行）")
    return GEOMETRY_DIR


def ensure_cbf() -> str:
    """确保cbf.json存在，不存在则从raw .cbf文件批量解析（同时顺带生成不折算20ft的cbf_with_20.json，
    仅供核对用，不接入求解流程）。返回cbf.json路径。"""
    if not os.path.exists(CBF_JSON):
        batch_parse_cbf(CBF_RAW_DIR, CBF_DIR)
        print(f"[cbf] 已构建 {CBF_JSON}")

    if not os.path.exists(CBF_WITH_20_JSON):
        batch_parse_cbf_with_20(CBF_RAW_DIR, CBF_DIR)
        print(f"[cbf] 已构建 {CBF_WITH_20_JSON}")

    return CBF_JSON

def main():
    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()
    with open(CBF_WITH_20_JSON) as f:
        raw_cbf_with_20 = json.load(f)
    cbf_with_20 = {
        int(pol): {int(pod): counts for pod, counts in pod_counts.items()}
        for pol, pod_counts in raw_cbf_with_20.items()
    }

    # 初始化母本 Vessel
    base_vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)
    original_cbf = copy.deepcopy(base_vessel.cbf)

    # 实验参数设置
    seeds = [8245, 2817, 6490, 1757, 4073, 6612, 80, 3991, 6813, 7519]
    # seeds = [8245, 2817, 6490, 1757, 4073, 6612, 80, 3991, 6813, 7519,
    #      8607, 6950, 3128, 9071, 124, 8300, 1327, 9367, 3274, 7276,
    #      3066, 2166, 1464, 6874, 8985, 5833, 5750, 6326, 4255, 7361,
    #      1682, 7897, 8450, 5633, 1354, 2802, 3001, 9830, 6909, 6430,
    #      4428, 2576, 6710, 5293, 5169, 2301, 4915, 4578, 9232, 507,
    #      7925, 5761, 9663, 2689, 8645, 7896, 5785, 1965, 2731, 2232,
    #      8003, 6176, 3362, 4619, 4929, 3187, 1540, 898, 3273, 4265,
    #      4897, 3327, 858, 4889, 3839, 7382, 8850, 5670, 3775, 42,
    #      5602, 2395, 6084, 5245, 1095, 8587, 7321, 423, 9186, 8983,
    #      6692, 9527, 2063, 4235, 3320, 5648, 3081, 2277, 1899, 455]
    # 2x2消融：ci_pol_enabled(cell层CI，选格子阶段) × ci_pod_enabled(箱子层CI，选箱子阶段)
    groups = {
        "POL_ON_POD_ON":   (True,  True),
        # "POL_ON_POD_OFF":  (True,  False),
        # "POL_OFF_POD_ON":  (False, True),
        # "POL_OFF_POD_OFF": (False, False),
    }

    summary_data = []

    print("\n================ 开始跑批测试 ================\n")

    for group_name, (ci_pol_status, ci_pod_status) in groups.items():
        print(f"▶ 正在执行组别: {group_name} "
              f"(ci_pol_enabled={ci_pol_status}, ci_pod_enabled={ci_pod_status})")

        group_results = []
        best_record_for_group = None   # 存整条记录，不再分列各自求min
        best_vessel_for_group = None      # 存对象，用于导出bayplan
        best_snapshots = None

        # 组级discharge_spread汇总：跨这一组所有种子、所有POD拉平在一起算均值
        spread_variances = []
        spread_ranges = []
        spread_cis = []

        reset_small_pod_ci_stats()

        for seed in seeds:
            random.seed(seed)
            vessel = copy.deepcopy(base_vessel)
            snapshots = {}
            best = {"assigned": -1, "vessel": None}

            start_time = time.time()
            success = solve(vessel, is_debug=False, snapshots=snapshots, best=best,
                             ci_pol_enabled=ci_pol_status, ci_pod_enabled=ci_pod_status)
            exec_time = time.time() - start_time

            result_vessel = vessel if success else best["vessel"]
            if result_vessel is None:
                print(f"  [Seed {seed:>4}] 失败: 连一个箱子都没能装上")
                continue

            total_voyage_time = 0.0
            total_wait_time = 0.0
            voyage_utilization = None

            if snapshots:
                crane_res = evaluate_crane_time(result_vessel, snapshots, k=2, crane_rate=1.0,
                                                port_names=PORT_NAMES, if_debug=False)
                total_voyage_time = sum(r["time_port"] for r in crane_res)
                total_wait_time = sum(r["wait1"] + r.get("wait2", 0.0) for r in crane_res)
                total_work = sum(r["work1"] + r["work2"] for r in crane_res)
                total_capacity = sum(2 * r["time_port"] for r in crane_res)
                voyage_utilization = total_work / total_capacity if total_capacity > 0 else None

                spread_report = evaluate_pod_discharge_spread(result_vessel, snapshots,
                                                               port_names=PORT_NAMES, if_debug=False)
                for stats in spread_report.values():
                    spread_variances.append(stats["variance"])
                    spread_ranges.append(stats["range"])
                    if stats["ci"] is not None:
                        spread_cis.append(stats["ci"])

            record = {
                "seed": seed,
                "total_wait_time": total_wait_time,
                "total_voyage_time": total_voyage_time,
                "voyage_utilization": voyage_utilization,
                "exec_time": exec_time,
            }
            group_results.append(record)

            util_str = f"{voyage_utilization:.3f}" if voyage_utilization is not None else "N/A"
            print(f"  └─ 种子 {seed:>4}: 阻塞耗时={total_wait_time:>5.1f}, "
                f"全程耗时={total_voyage_time:>5.1f}, 利用率={util_str}, 求解耗时={exec_time:.2f}s")

            # 组最优：按全程耗时挑出"同一次运行"的完整记录，不再分列各自取min
            if best_record_for_group is None or total_voyage_time < best_record_for_group["total_voyage_time"]:
                best_record_for_group = record
                best_vessel_for_group = copy.deepcopy(result_vessel)
                best_snapshots = copy.deepcopy(snapshots)

        if group_results:
            waits = [r["total_wait_time"] for r in group_results]
            voyages = [r["total_voyage_time"] for r in group_results]
            utils = [r["voyage_utilization"] for r in group_results if r["voyage_utilization"] is not None]

            summary_data.append({
                "Group": group_name,
                "Wait (Mean/Var)": f"{np.mean(waits):.1f} / {np.var(waits):.1f}",
                "Voyage (Mean/Var)": f"{np.mean(voyages):.1f} / {np.var(voyages):.1f}",
                "Utilization (Mean/Var)": f"{np.mean(utils):.3f} / {np.var(utils):.3f}" if utils else "N/A",
            })

            b = best_record_for_group
            print(f"  🏆 {group_name} 组最优(种子{b['seed']}): "
                f"阻塞耗时={b['total_wait_time']:.1f}, "
                f"全程耗时={b['total_voyage_time']:.1f}, "
                f"利用率={b['voyage_utilization']:.3f}" if b['voyage_utilization'] is not None else "N/A")

        # discharge_spread组级汇总（所有POD×所有种子拉平算均值，不逐POD逐种子明细）
        if spread_variances:
            print(f"  📊 discharge_spread 组汇总(全部POD×全部种子, n={len(spread_variances)}): "
                  f"variance均值={np.mean(spread_variances):.3f}, "
                  f"range均值={np.mean(spread_ranges):.3f}, "
                  f"CI均值={np.mean(spread_cis):.3f}" if spread_cis else "CI均值=N/A")

        print("-" * 60)

    # 打印最终对比结果
    print("\n================ 实验结果汇总 ================\n")
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # 你可以在这里导出最优解的 bayplan:
    
    best_vessel_for_group.export_bayplan(best_snapshots, BAYPLAN_DIR, original_cbf, port_names=PORT_NAMES,
                                          if_csv=False, if_plot_phy=False, cbf_with_20=cbf_with_20)

if __name__ == "__main__":
    main()