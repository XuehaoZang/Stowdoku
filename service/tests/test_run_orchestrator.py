"""
run_orchestrator.execute_run 的真实场景验证：跑一次真实STSE求解（种子8245，同款
调用方式见 service/tests/test_translator.py），断言7港落盘 + index.json + round-trip，
并统计全量数据里 dataIntegrityFlag 命中的行数与分布。

内联assert脚本，直接 `python service/tests/test_run_orchestrator.py` 运行。
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from main import ensure_geometry, ensure_cbf, PORT_NAMES
from VesselClass import Vessel
from service.run_orchestrator import execute_run
from service import storage

RUN_ID = "RUN-TEST-ORCH-8245"
VOYAGE_ID = "VOY-TEST-ORCH-0001"


def main():
    geometry_dir = ensure_geometry()
    cbf_json_path = ensure_cbf()

    vessel = Vessel.load_vessel(geometry_dir, cbf_json_path)
    voyage_leg_port_codes = [PORT_NAMES[i] for i in range(7)]

    random.seed(8245)
    index = execute_run(vessel, run_id=RUN_ID, voyage_id=VOYAGE_ID,
                         voyage_leg_port_codes=voyage_leg_port_codes)

    # 1. index.json 结构 + 7港覆盖、互不重复
    assert index["runId"] == RUN_ID
    assert index["status"] == "COMPLETED"
    assert len(index["plans"]) == 7, f"应产出7个Plan，实际={len(index['plans'])}"
    port_codes = [p["portCode"] for p in index["plans"]]
    assert set(port_codes) == set(voyage_leg_port_codes), (
        f"portCode覆盖应=全部7港，实际={sorted(port_codes)}"
    )
    assert len(set(port_codes)) == 7, f"portCode不应重复，实际={port_codes}"

    # 2. 磁盘上确实生成了7个result.json + 1个index.json
    run_dir = os.path.join(storage.STORAGE_ROOT, "runs", RUN_ID)
    index_path = os.path.join(run_dir, "index.json")
    assert os.path.exists(index_path), f"缺少index.json: {index_path}"

    result_json_count = 0
    for port_code in port_codes:
        result_path = os.path.join(run_dir, "plans", port_code, "result.json")
        assert os.path.exists(result_path), f"缺少result.json: {result_path}"
        result_json_count += 1
    assert result_json_count == 7

    # 3. load_run_index round-trip
    reloaded_index = storage.load_run_index(RUN_ID)
    assert reloaded_index == index, "load_run_index读回内容应与execute_run返回一致"

    # 4. load_plan round-trip：读回内容与落盘前一致
    all_results = {}
    integrity_hits = []  # (portCode, bayId, slotId, dataIntegrityFlag)
    for plan_entry in index["plans"]:
        plan_id = plan_entry["planId"]
        assert plan_id == f"{RUN_ID}-{plan_entry['portCode']}"
        loaded = storage.load_plan(plan_id)
        assert loaded["planId"] == plan_id
        assert loaded["portCode"] == plan_entry["portCode"]
        assert loaded["runId"] == RUN_ID
        assert loaded["voyageId"] == VOYAGE_ID
        all_results[plan_entry["portCode"]] = loaded

        for bay in loaded["bays"]:
            for slot in bay["slots"]:
                flag = slot.get("dataIntegrityFlag")
                if flag:
                    integrity_hits.append((plan_entry["portCode"], bay["bayId"], slot["slotId"], flag))

    # round-trip: 直接对比同一份result内容与磁盘文件内容（不经过execute_run的返回值，
    # 而是重新读一次磁盘文件，验证save_plan落盘的内容和load_plan读回的内容一致）
    import json
    for port_code in port_codes:
        result_path = os.path.join(run_dir, "plans", port_code, "result.json")
        with open(result_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assert raw == all_results[port_code], f"落盘前后内容不一致: {port_code}"

    print("全部断言通过。")
    print(f"  runId={RUN_ID}")
    print(f"  落盘: 7×result.json + 1×index.json，路径={run_dir}")
    print(f"  portCode覆盖: {sorted(port_codes)}")

    # 5. dataIntegrityFlag 统计
    print(f"\ndataIntegrityFlag 命中总行数（7港合计，不限于HC） = {len(integrity_hits)}")
    by_port = {}
    by_flag = {}
    for port_code, bay_id, slot_id, flag in integrity_hits:
        by_port[port_code] = by_port.get(port_code, 0) + 1
        by_flag[flag] = by_flag.get(flag, 0) + 1
    print(f"  按港口分布: { {k: by_port[k] for k in sorted(by_port)} }")
    print(f"  按flag取值分布: {by_flag}")
    if integrity_hits:
        print("  样例(最多5条): ")
        for row in integrity_hits[:5]:
            print(f"    {row}")


if __name__ == "__main__":
    main()
