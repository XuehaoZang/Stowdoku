"""
测试数据说明：
4 bay x 2 lr x 2 hd，模拟从POL=0出发的航次

几何：
  bay0: 全部valid，bay1-left-hold有reefer能力
  bay2: 全部valid，无reefer
  bay3: 全部valid，无reefer
  bay4: 右侧全部invalid（模拟船首收窄）

capacity_total: 每个cell能放的40ft箱数（小船，每格容量小）
capacity_rf:    每个cell中有RF插座的槽数

init预装（vessel_pod初始值，模拟到港时船上已有货）：
  bay0-left-hold  = POD=1（已装，从上一港带来）
  bay2-right-hold = POD=2

cbf（全航次待装量）：
  POL=0: POD=1 GP=8, POD=2 GP=4, POD=2 RF=2
  POL=1: POD=3 GP=6
"""

import numpy as np
import sys
sys.path.insert(0, "D:/Stowage/Stowdoku")  # 用于在test目录下import上级目录的VesselClass.py
from VesselClass import Vessel

# ── 静态几何 ───────────────────────────────────────────────────────────
# shape: (n_bay=4, lr=2, hd=2)

is_valid = np.array([
    [[True,  True ],   # bay0-left:  hold valid, deck valid
     [True,  True ]],  # bay0-right: hold valid, deck valid
    [[True,  True ],   # bay1-left:  hold valid, deck valid
     [True,  True ]],  # bay1-right: hold valid, deck valid
    [[True,  True ],   # bay2-left:  hold valid, deck valid
     [True,  True ]],  # bay2-right: hold valid, deck valid
    [[True,  True ],   # bay3-left:  hold valid, deck valid
     [False, False]],  # bay3-right: invalid（船首收窄）
], dtype=bool)

capacity_total = np.array([
    [[4, 4], [4, 4]],  # bay0
    [[4, 4], [4, 4]],  # bay1
    [[4, 4], [4, 4]],  # bay2
    [[4, 4], [0, 0]],  # bay3, right=0因为invalid
], dtype=int)

capacity_rf = np.array([
    [[0, 0], [0, 0]],  # bay0: 无reefer
    [[2, 0], [0, 0]],  # bay1-left-hold: 2个RF插座
    [[0, 0], [0, 0]],  # bay2: 无reefer
    [[0, 0], [0, 0]],  # bay3: 无reefer
], dtype=int)

# ── 全航次cbf ──────────────────────────────────────────────────────────
cbf = {
    0: {
        1: {"GP": 8, "RF": 0},
        2: {"GP": 4, "RF": 2},  # POD=2有2个reefer
    },
    1: {
        3: {"GP": 6, "RF": 0},
    },
}

# ── 初始化Vessel ───────────────────────────────────────────────────────
vessel = Vessel(is_valid, capacity_total, capacity_rf, cbf, current_pol=0)

# 模拟预装货：从上港带来的箱子直接写入vessel_pod（不走assign，不动cbf）
vessel.vessel_pod[0, 0, 0] = 1   # bay0-left-hold = POD=1
vessel.vessel_type[0, 0, 0] = "GP"
vessel.vessel_pod[2, 1, 0] = 2   # bay2-right-hold = POD=2
vessel.vessel_type[2, 1, 0] = "GP"

# ══════════════════════════════════════════════════════════════════════
# TEST 1: get_candidates基础功能
# ══════════════════════════════════════════════════════════════════════
print("=== TEST 1: get_candidates ===")

# bay0-left-deck: hold已是POD=1，deck候选应该 <= 1，且cbf[0][1]["GP"]>0
c = vessel.get_candidates(0, 0, 1)
print(f"bay0-left-deck (hold=POD1): {c}")
assert (1, "GP") in c, "POD=1 GP应该在候选里"
assert all(pod <= 1 for pod, _ in c), "deck候选POD应该<=hold的POD=1"

# bay1-left-hold: has_reefer=True，cbf[0][2]["RF"]=2>0，应该有RF候选
c = vessel.get_candidates(1, 0, 0)
print(f"bay1-left-hold (has_reefer=True): {c}")
assert (2, "RF") in c, "reefer cell应该有RF候选"
assert (2, "GP") in c, "reefer cell也应该有GP候选"

# bay3-right-hold: is_valid=False，应该返回空集
c = vessel.get_candidates(3, 1, 0)
print(f"bay3-right-hold (invalid): {c}")
assert c == set(), "invalid cell应返回空集"

print("TEST 1 PASSED\n")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: assign / unassign
# ══════════════════════════════════════════════════════════════════════
print("=== TEST 2: assign / unassign ===")

gp_before = vessel.cbf[0][2]["GP"]
rf_before  = vessel.cbf[0][2]["RF"]

# assign GP
vessel.assign(2, 0, 0, 2, "GP")
assert vessel.vessel_pod[2, 0, 0] == 2
assert vessel.vessel_type[2, 0, 0] == "GP"
assert vessel.cbf[0][2]["GP"] == gp_before - vessel.capacity_total[2, 0, 0]
print(f"assign GP: cbf[0][2][GP] {gp_before} -> {vessel.cbf[0][2]['GP']}")

vessel.unassign(2, 0, 0, 2, "GP")
assert vessel.vessel_pod[2, 0, 0] == -1
assert vessel.cbf[0][2]["GP"] == gp_before
print(f"unassign GP: cbf[0][2][GP] restored to {vessel.cbf[0][2]['GP']}")

# assign RF
vessel.assign(1, 0, 0, 2, "RF")
assert vessel.cbf[0][2]["RF"] == rf_before - vessel.capacity_rf[1, 0, 0]
print(f"assign RF: cbf[0][2][RF] {rf_before} -> {vessel.cbf[0][2]['RF']}")

vessel.unassign(1, 0, 0, 2, "RF")
assert vessel.cbf[0][2]["RF"] == rf_before
print(f"unassign RF: cbf[0][2][RF] restored to {vessel.cbf[0][2]['RF']}")

print("TEST 2 PASSED\n")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: remaining_pods / port_complete / total_remaining
# ══════════════════════════════════════════════════════════════════════
print("=== TEST 3: remaining_pods / port_complete / total_remaining ===")

print(f"remaining_pods: {vessel.remaining_pods()}")
assert vessel.remaining_pods() == {1, 2}
assert not vessel.port_complete()
print(f"total_remaining: {vessel.total_remaining()}")
assert vessel.total_remaining() == 14  # GP:8+4 + RF:2 = 14

print("TEST 3 PASSED\n")

# ══════════════════════════════════════════════════════════════════════
# TEST 4: discharge / undischarge
# ══════════════════════════════════════════════════════════════════════
print("=== TEST 4: discharge / undischarge ===")

# bay0-left-hold已预装POD=1
records = vessel.discharge(1)
print(f"discharge POD=1: {records}")
assert vessel.vessel_pod[0, 0, 0] == -1
assert vessel.cbf[0][1]["GP"] == 8  # discharge不动cbf

vessel.undischarge(records)
assert vessel.vessel_pod[0, 0, 0] == 1
print("undischarge restored correctly")

print("TEST 4 PASSED\n")

# ══════════════════════════════════════════════════════════════════════
# TEST 5: snapshot / restore
# ══════════════════════════════════════════════════════════════════════
print("=== TEST 5: snapshot / restore ===")

snap = vessel.snapshot()
vessel.assign(3, 0, 0, 1, "GP")
assert vessel.vessel_pod[3, 0, 0] == 1

vessel.restore(snap)
assert vessel.vessel_pod[3, 0, 0] == -1
assert vessel.cbf[0][1]["GP"] == 8
print("snapshot / restore correct")

print("TEST 5 PASSED\n")

# ══════════════════════════════════════════════════════════════════════
# TEST 6: advance_pol + export_cell_state
# ══════════════════════════════════════════════════════════════════════
print("=== TEST 6: advance_pol / export_cell_state ===")

vessel.advance_pol()
assert vessel.current_pol == 1
assert vessel.remaining_pods() == {3}
print(f"advance_pol: current_pol={vessel.current_pol}, remaining={vessel.remaining_pods()}")

state = vessel.export_cell_state()
print(f"export_cell_state: {state}")
# bay0-left-hold=POD1, bay2-right-hold=POD2 是预装货，应该都在
assert (0, 0, 0) in state
assert (2, 1, 0) in state

print("TEST 6 PASSED\n")

print("ALL TESTS PASSED")