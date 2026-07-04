"""
验证新版分层pipeline最终仍能喂给Vessel.__init__()，不调用solve()。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from VesselClass import Vessel
from utils.vessel_io import (
    build_vessel_geometry, find_can_40ft, find_can_20ft, find_can_reefer,
    build_vessel_cell, N_BAY,
)

IDX_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "STSE", "geometry", "STSE_slots_idx.csv")

# Layer1: 完整slot表，含所有bay_idx
slots = build_vessel_geometry(IDX_CSV)
assert set(slots.bay_idx) == {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

slots = find_can_40ft(slots)
slots = find_can_20ft(slots)
slots = find_can_reefer(slots)

# can_40ft/can_20ft互斥；bay_idx=0全can_20ft；
# 每对pair里b1侧属于交集的行，两者都是False（已被b0代表的40ft占用，非20ft-only）
assert not (slots.can_40ft & slots.can_20ft).any()
assert slots.loc[slots.bay_idx == 0, "can_20ft"].all()
assert not slots.loc[slots.bay_idx == 0, "can_40ft"].any()
neither = ~(slots.can_40ft | slots.can_20ft)
assert neither.sum() == slots.can_40ft.sum()  # 恰好是b0代表的40ft槽位数的b1镜像

slots.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "STSE", "geometry", "full_slot_table.csv"), index=False)

# Layer2: 聚合成(7,2,2) cell
capacity_total = build_vessel_cell(slots, "can_40ft")
is_valid = capacity_total > 0

slots["can_reefer_40ft"] = slots["can_40ft"] & slots["can_reefer"]
capacity_rf = build_vessel_cell(slots, "can_reefer_40ft")

assert capacity_total.shape == (N_BAY, 2, 2)
assert (capacity_rf == 0).all()

vessel = Vessel(is_valid, capacity_total, capacity_rf, cbf={0: {}})
assert vessel.n_bay == 7
assert not vessel.has_reefer.any()

print("capacity_total:\n", capacity_total)
print("总40ft槽位数:", capacity_total.sum())
print("总20ft-only槽位数:", int(slots.can_20ft.sum()))
print("Vessel构造成功，n_bay =", vessel.n_bay)