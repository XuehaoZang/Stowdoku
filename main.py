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

from utils.vessel_io import (
    build_vessel_geometry, find_can_40ft, find_can_20ft, find_can_reefer,
    batch_parse_cbf, STSE_PORT_MAP
)
from VesselClass import Vessel
from CSP_solver import solve
from utils.evaluate import evaluate_crane_intensity, evaluate_pod_leverage

CRANE_NUMBER = 3
# 港口吊车数量，作为evaluate_crane_intensity的参数，以后要是船公司变了配置，改这一个常量就够

GEOMETRY_ALL_CSV = "data/STSE/geometry/all_slots.csv"
GEOMETRY_REEFER_CSV = "data/STSE/geometry/reefer_slots.csv"
GEOMETRY_DIR = "data/STSE/geometry"
CBF_RAW_DIR = "data/STSE/raw"
CBF_DIR = "data/STSE/cbf"
CBF_JSON = os.path.join(CBF_DIR, "cbf.json")
BAYPLAN_DIR = "data/STSE/bayplan"

# TODO debug 现在缺少TXG的cbf数据
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
    import copy
    from CSP_solver import _total_assigned

    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()

    vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)
    original_cbf = copy.deepcopy(vessel.cbf)
    # solve()会原地扣减vessel.cbf，留一份原始计划量给evaluate_pod_leverage分析杠杆结构用

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)

    if success:
        result_vessel = vessel
        print("\n──── Solution Found ────")
    else:
        result_vessel = best["vessel"]
        print(
            f"\n──── No Full Solution — 输出搜索过程中最优的近似解 ────\n"
            f"求解卡在 current_pol={vessel.current_pol}, "
            f"total_remaining={vessel.total_remaining()}"
        )

    if result_vessel is None:
        print("连一个箱子都没能装上，没有可导出的近似解")
        return

    remaining_total = sum(
        counts.get("GP", 0) + counts.get("RF", 0)
        for pod_dict in result_vessel.cbf.values()
        for counts in pod_dict.values()
    )

    if remaining_total > 0:
        print("\n尾货cbf明细：")
        for pol, pod_dict in sorted(result_vessel.cbf.items()):
            for pod, counts in sorted(pod_dict.items()):
                if counts.get("GP", 0) > 0 or counts.get("RF", 0) > 0:
                    port_label = PORT_NAMES.get(pod, pod)
                    print(f"    POL={pol} POD={port_label}: {counts}")

    if snapshots:
        evaluate_crane_intensity(vessel, snapshots, crane_number=CRANE_NUMBER, port_names=PORT_NAMES)
    else:
        print("\n[evaluate] 没有完整的逐港snapshots（求解失败且未走到任何一港完成），跳过CI评估")
    evaluate_pod_leverage(original_cbf)

    paths = vessel.export_bayplan(snapshots, BAYPLAN_DIR, port_names=PORT_NAMES, if_plot_phy=False)
    print(f"Exported {len(paths)} bayplan files to {BAYPLAN_DIR}")
    

if __name__ == "__main__":
    main()