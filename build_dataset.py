"""
build_dataset.py

数据准备入口，跑一次、落盘、solver复用。
    1. geometry: STSE_slots_idx.csv -> is_valid/capacity_total/capacity_rf + 完整slot表
    2. init:     空船初始状态
    3. cbf:      批量解析.cbf原始文件 -> 按POL编号的汇总csv
ASC解析这次不跑（batch_parse_asc已经在utils/vessel_io.py里，需要时单独调用）。
"""

import os
import numpy as np

from utils.vessel_io import (
    build_vessel_geometry, find_can_40ft, find_can_20ft, find_can_reefer,
    build_vessel_cell, build_init_state, batch_parse_cbf, BAYPLAN_COLUMNS,
)

GEOMETRY_IDX_CSV = "data/STSE/geometry/STSE_slots_idx.csv"
GEOMETRY_DIR = "data/STSE/geometry"
INIT_DIR = "data/STSE/init"
CBF_RAW_DIR = "data/STSE/raw"
CBF_DIR = "data/STSE/cbf"


def build_geometry():
    slots = build_vessel_geometry(GEOMETRY_IDX_CSV)
    slots = find_can_40ft(slots)
    slots = find_can_20ft(slots)
    slots = find_can_reefer(slots)

    capacity_total = build_vessel_cell(slots, "can_40ft")
    is_valid = capacity_total > 0
    slots["can_reefer_40ft"] = slots["can_40ft"] & slots["can_reefer"]
    capacity_rf = build_vessel_cell(slots, "can_reefer_40ft")

    os.makedirs(GEOMETRY_DIR, exist_ok=True)
    slots.to_csv(os.path.join(GEOMETRY_DIR, "full_slot_table.csv"), index=False)
    np.savez(os.path.join(GEOMETRY_DIR, "vessel_geometry.npz"),
              is_valid=is_valid, capacity_total=capacity_total, capacity_rf=capacity_rf)

    print(f"[geometry] 完整slot表 {len(slots)}行, 40ft槽位{capacity_total.sum()}个 -> {GEOMETRY_DIR}")
    return slots


def build_init(slots):
    init = build_init_state(slots)
    os.makedirs(INIT_DIR, exist_ok=True)
    out_path = os.path.join(INIT_DIR, "init.csv")
    init.to_csv(out_path, index=False)
    print(f"[init] 空船状态({len(BAYPLAN_COLUMNS)}列, 0行) -> {out_path}")


def build_cbf():
    if not os.path.isdir(CBF_RAW_DIR):
        print(f"[cbf] 跳过: {CBF_RAW_DIR} 不存在")
        return
    batch_parse_cbf(CBF_RAW_DIR, CBF_DIR)
    print(f"[cbf] 已解析 {CBF_RAW_DIR} -> {CBF_DIR}")


if __name__ == "__main__":
    slots = build_geometry()
    build_init(slots)
    build_cbf()