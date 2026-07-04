"""
main.py - 编排入口，负责数据准备 + 求解 + 导出的完整流水线

数据准备（若目标文件已存在则跳过，不重复构建）：
    1. geometry: 若 GEOMETRY_DIR/full_slot_table.csv 不存在，
       从 GEOMETRY_IDX_CSV 解析 + find_can_40ft/20ft/reefer 后落盘
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
    batch_parse_cbf,
)
from VesselClass import Vessel
from CSP_solver import solve

GEOMETRY_IDX_CSV = "data/STSE/geometry/STSE_slots_idx.csv"
GEOMETRY_DIR = "data/STSE/geometry"
CBF_RAW_DIR = "data/STSE/raw"
CBF_DIR = "data/STSE/cbf"
CBF_JSON = os.path.join(CBF_DIR, "cbf.json")
BAYPLAN_DIR = "data/STSE/bayplan"

# TODO: 反转utils.vessel_io.STSE_PORT_MAP填这里，比如 {0:"SHP",1:"TXG",...}
PORT_NAMES = None


def ensure_geometry() -> str:
    """确保full_slot_table.csv存在，不存在则从idx csv构建。返回geometry目录路径。"""
    out_path = os.path.join(GEOMETRY_DIR, "full_slot_table.csv")
    if os.path.exists(out_path):
        return GEOMETRY_DIR

    slots = build_vessel_geometry(GEOMETRY_IDX_CSV)
    slots = find_can_40ft(slots)
    slots = find_can_20ft(slots)
    slots = find_can_reefer(slots)

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

    vessel = Vessel.load_vessel(geometry_dir, cbf_json_path, current_pol=0)
    exit()
    snapshots = {}
    success = solve(vessel, is_debug=False, snapshots=snapshots)

    if not success:
        print(
            f"No solution found. current_pol={vessel.current_pol}, "
            f"total_remaining={vessel.total_remaining()}"
        )
        return

    paths = vessel.export_bayplan(snapshots, BAYPLAN_DIR, port_names=PORT_NAMES)
    print(f"Exported {len(paths)} bayplan files to {BAYPLAN_DIR}")


if __name__ == "__main__":
    main()