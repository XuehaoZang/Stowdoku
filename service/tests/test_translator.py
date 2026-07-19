"""
translator.to_bay_plan_result 的真实场景验证：跑一次真实STSE求解（种子8245，
与 upstream-contract-snapshot.md / debug/tail_real_run.py 同款调用方式），
取 snapshots[0]（起运港SHP）喂给 to_bay_plan_result，核对输出结构与数量。

不是单元测试框架，是与仓库既有 debug/test_VesselCass.py 同风格的内联assert脚本，
直接 `python service/tests/test_translator.py` 运行。
"""

import copy
import json
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from main import ensure_geometry, ensure_cbf, PORT_NAMES
from VesselClass import Vessel
from CSP_solver import solve
from utils.evaluate import (
    evaluate_crane_time,
    evaluate_crane_intensity,
    evaluate_pod_discharge_spread,
    evaluate_ci_theoretical_ceiling,
)
from service.translator import to_bay_plan_result
from service.codes import is_b0_bay_idx, bay_idx_to_bay_id


def main():
    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()

    vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)
    original_cbf = copy.deepcopy(vessel.cbf)

    random.seed(8245)
    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    assert success, "solve()应当一次成功（seed=8245，全船真实STSE数据）"
    assert sorted(snapshots.keys()) == list(range(7)), f"应产出7港快照，实际={sorted(snapshots.keys())}"

    crane_time_res = evaluate_crane_time(vessel, snapshots, k=2, crane_rate=1.0, port_names=PORT_NAMES)
    crane_intensity_res = evaluate_crane_intensity(vessel, snapshots, port_names=PORT_NAMES)
    spread_res = evaluate_pod_discharge_spread(vessel, snapshots, port_names=PORT_NAMES)
    ci_ceiling = evaluate_ci_theoretical_ceiling(vessel)

    eval_results = {
        "crane_time": crane_time_res,
        "crane_intensity": crane_intensity_res,
        "pod_discharge_spread": spread_res,
        "ci_theoretical_ceiling": ci_ceiling,
    }

    voyage_leg_port_codes = [PORT_NAMES[i] for i in range(7)]

    result = to_bay_plan_result(
        vessel=vessel,
        snapshot=snapshots[0],
        original_cbf=original_cbf,
        eval_results=eval_results,
        run_id="RUN-TEST-0001",
        voyage_id="VOY-TEST-0001",
        port_code=PORT_NAMES[0],
        voyage_leg_port_codes=voyage_leg_port_codes,
    )

    # 1. slots[]总数=1003，且逐行跟full_slot_table.csv对得上
    full_slot_table = pd.read_csv(os.path.join(geometry_dir, "full_slot_table.csv"))
    total_slots = sum(len(bay["slots"]) for bay in result["bays"])
    assert total_slots == 1003, f"slots总数应为1003，实际={total_slots}"
    assert total_slots == len(full_slot_table)

    slot_index = {}
    for bay in result["bays"]:
        for slot in bay["slots"]:
            slot_index[(bay["bayId"], slot["row"], slot["tier"])] = slot
    for _, row in full_slot_table.iterrows():
        bay_id = bay_idx_to_bay_id(int(row["bay_idx"]))
        key = (bay_id, int(row["row_idx"]), int(row["tier_idx"]))
        assert key in slot_index, f"缺少slot: {key}"
        slot = slot_index[key]
        assert slot["capability"]["can20ft"] == bool(row["can_20ft"])
        assert slot["capability"]["can40ft"] == bool(row["can_40ft"])

    # 2. UNAVAILABLE/STRUCTURAL行数与can_20ft/can_40ft均False的真实行数一致
    expected_unavailable = int(((~full_slot_table["can_20ft"]) & (~full_slot_table["can_40ft"])).sum())
    actual_unavailable = sum(
        1 for bay in result["bays"] for slot in bay["slots"]
        if slot["status"] == "UNAVAILABLE" and slot["unavailableReason"] == "STRUCTURAL"
    )
    assert actual_unavailable == expected_unavailable == 478, (
        f"UNAVAILABLE/STRUCTURAL行数应为478，实际={actual_unavailable}（预期同源值={expected_unavailable}）"
    )

    # 3. is_hc=True的行都映射出了HIGH_CUBE
    #    注意：真实数据里，proj_cell_to_vessel()对is_hc=True的68行中有34行落在
    #    can_20ft/can_40ft均False（STRUCTURAL不可用）的物理槽位上——这是upstream
    #    投影启发式与full_slot_table物理能力表之间的既有不一致（详见本文件末尾说明），
    #    不是translator的bug。按service_api_design.md §1 Slot.status的显式优先级
    #    （STRUCTURAL判定先于OCCUPIED判定），这些行的container会是None、不产出
    #    HIGH_CUBE标记，因此这里按“is_hc=True 且 非STRUCTURAL”行数核对，而不是68。
    df = vessel.proj_cell_to_vessel(cell_state=snapshots[0], original_cbf=original_cbf)
    hc_total = int(df["is_hc"].sum())
    assert hc_total == 68, f"is_hc=True行数预期68，实际={hc_total}"
    structural_mask = (~df["can_20ft"]) & (~df["can_40ft"])
    expected_hc_count = int((df["is_hc"] & ~structural_mask).sum())
    hc_on_structural = int((df["is_hc"] & structural_mask).sum())
    actual_hc_count = sum(
        1 for bay in result["bays"] for slot in bay["slots"]
        if slot["container"] is not None and "HIGH_CUBE" in slot["container"]["attributeFlags"]
    )
    assert actual_hc_count == expected_hc_count, (
        f"HIGH_CUBE标记数应为{expected_hc_count}（68行is_hc中刨除{hc_on_structural}行落在STRUCTURAL槽位），"
        f"实际={actual_hc_count}"
    )

    # 4. metrics.byBay的key数量=7
    assert len(result["metrics"]["byBay"]) == 7, (
        f"metrics.byBay应有7个key，实际={len(result['metrics']['byBay'])}: "
        f"{list(result['metrics']['byBay'].keys())}"
    )

    # 5. 整份输出能被json.dumps无异常序列化
    serialized = json.dumps(result, ensure_ascii=False)
    assert isinstance(serialized, str) and len(serialized) > 0

    print("全部断言通过。")
    print(f"  slots总数={total_slots}")
    print(f"  UNAVAILABLE/STRUCTURAL={actual_unavailable}")
    print(f"  HIGH_CUBE标记数={actual_hc_count}")
    print(f"  metrics.byBay key数={len(result['metrics']['byBay'])}: {sorted(result['metrics']['byBay'].keys())}")
    print(f"  metrics.global 条数={len(result['metrics']['global'])}")
    print(f"  json序列化长度={len(serialized)}字节")


if __name__ == "__main__":
    main()
