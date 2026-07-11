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
import random
import copy
import time

import numpy as np
import pandas as pd

from utils.vessel_io import (
    build_vessel_geometry, find_can_40ft, find_can_20ft, find_can_reefer,
    batch_parse_cbf, STSE_PORT_MAP
)
from VesselClass import Vessel
from CSP_solver import solve
from utils.evaluate import evaluate_crane_time

GEOMETRY_ALL_CSV = "data/STSE/geometry/all_slots.csv"
GEOMETRY_REEFER_CSV = "data/STSE/geometry/reefer_slots.csv"
GEOMETRY_DIR = "data/STSE/geometry"
CBF_RAW_DIR = "data/STSE/raw"
CBF_DIR = "data/STSE/cbf"
CBF_JSON = os.path.join(CBF_DIR, "cbf.json")
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
    """确保cbf.json存在，不存在则从raw .cbf文件批量解析。返回cbf.json路径。"""
    if os.path.exists(CBF_JSON):
        return CBF_JSON

    batch_parse_cbf(CBF_RAW_DIR, CBF_DIR)
    print(f"[cbf] 已构建 {CBF_JSON}")
    return CBF_JSON

def main():
    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()

    # 初始化母本 Vessel
    base_vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)
    original_cbf = copy.deepcopy(base_vessel.cbf)

    # 实验参数设置
    seeds = [42, 54, 87, 100, 2026, 7, 999, 1234, 5555, 8888]
    groups = {"CI_OFF": False, "CI_ON": True}
    
    summary_data = []

    print("\n================ 开始跑批测试 ================\n")

    for group_name, ci_status in groups.items():
        print(f"▶ 正在执行组别: {group_name} (ci_enabled={ci_status})")
        
        group_results = []
        best_vessel_for_group = None
        best_metric_for_group = float('inf') 
        best_snapshots = None
        
        for seed in seeds:
            random.seed(seed)
            # 深拷贝确保每次搜索从初始状态开始
            vessel = copy.deepcopy(base_vessel)
            snapshots = {}
            best = {"assigned": -1, "vessel": None}
            
            start_time = time.time()
            success = solve(vessel, is_debug=False, snapshots=snapshots, best=best, ci_enabled=ci_status)
            exec_time = time.time() - start_time
            
            result_vessel = vessel if success else best["vessel"]
            
            if result_vessel is None:
                print(f"  [Seed {seed:>4}] 失败: 连一个箱子都没能装上")
                continue

            # 计算尾货总数
            tail_boxes = sum(
                counts.get("GP", 0) + counts.get("RF", 0) + counts.get("HC", 0) + counts.get("HR", 0)
                for pod_dict in result_vessel.cbf.values()
                for counts in pod_dict.values()
            )

            total_voyage_time = 0.0
            total_wait_time = 0.0
            
            if snapshots:
                crane_res = evaluate_crane_time(result_vessel, snapshots, k=2, crane_rate=1.0,
                                                 port_names=PORT_NAMES, if_debug=False)
                total_voyage_time = sum(r["time_port"] for r in crane_res)
                total_wait_time = sum(r["wait1"] + r.get("wait2", 0.0) for r in crane_res)

            group_results.append({
                "seed": seed,
                "tail_boxes": tail_boxes,
                "total_wait_time": total_wait_time,
                "total_voyage_time": total_voyage_time,
                "exec_time": exec_time
            })
            
            print(f"  └─ 种子 {seed:>4}: 尾箱={tail_boxes:>3}, 阻塞耗时={total_wait_time:>5.1f}, 全程耗时={total_voyage_time:>5.1f}, 求解耗时={exec_time:.2f}s")
            
            # 记录本组最优解: 优先全程耗时短
            sort_metric = total_voyage_time
            if sort_metric < best_metric_for_group:
                best_metric_for_group = sort_metric
                best_vessel_for_group = copy.deepcopy(result_vessel)
                best_snapshots = copy.deepcopy(snapshots)

        # 统计本组均值和方差
        if group_results:
            tails = [r["tail_boxes"] for r in group_results]
            waits = [r["total_wait_time"] for r in group_results]
            voyages = [r["total_voyage_time"] for r in group_results]
            
            summary_data.append({
                "Group": group_name,
                "Tails (Mean/Var)": f"{np.mean(tails):.2f} / {np.var(tails):.2f}",
                "Wait (Mean/Var)": f"{np.mean(waits):.1f} / {np.var(waits):.1f}",
                "Voyage (Mean/Var)": f"{np.mean(voyages):.1f} / {np.var(voyages):.1f}"
            })
            print(f"  🏆 {group_name} 组最优: 尾箱={min(tails)}, 阻塞耗时={min(waits):.1f}, 全程耗时={min(voyages):.1f}")
        print("-" * 60)

    # 打印最终对比结果
    print("\n================ 实验结果汇总 ================\n")
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # 你可以在这里导出最优解的 bayplan:
    best_vessel_for_group.export_bayplan(best_snapshots, BAYPLAN_DIR, original_cbf, port_names=PORT_NAMES, if_csv=False, if_plot_phy=False)

if __name__ == "__main__":
    main()