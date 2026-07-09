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

TARGET_CI = 2.0
# CI目标基准，对应总作业量500以内、完全均匀分布下相邻bay对占比2/n_bay的理论值
# 不是硬约束，只是打印时用来标注"低于目标"，不同总量级别可能需要另外校准

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

def tag_hicube_allocation(self, snapshots: dict, original_cbf: dict, hc_order="cap_desc"):
    grand_total = {"hc_need": 0, "hc_gap": 0, "hr_need": 0, "hr_gap": 0,
                   "slots_short": 0, "n_pod_with_gap": 0}

    for pol, snap in sorted(snapshots.items()):
        cell = snap["cell"]
        pod_cells = {}
        for bay in range(self.n_bay):
            for lr in range(2):
                for hd in range(2):
                    record = cell[bay, lr, hd]
                    if record["POD"] != -1:
                        pod_cells.setdefault(record["POD"], []).append((bay, lr, hd, record))

        pol_totals = {"hc_need": 0, "hc_tagged": 0, "hr_need": 0, "hr_tagged": 0,
                      "total_slots": 0, "total_demand": 0}
        gap_lines = []  # 只收集有缺口的POD，逐条打印，避免刷屏

        for pod, cells in sorted(pod_cells.items()):
            demand = original_cbf.get(pol, {}).get(pod, {})
            hc_remaining = demand.get("HC", 0)
            rf_hc_remaining = demand.get("HR", 0)
            hc_need, rf_hc_need = hc_remaining, rf_hc_remaining

            if hc_order == "cap_desc":
                cells = sorted(cells, key=lambda c: self.capacity_hc[c[0], c[1], c[2]], reverse=True)

            for bay, lr, hd, record in cells:
                cap_hc = self.capacity_hc[bay, lr, hd]
                rf_hc_used = min(rf_hc_remaining, record["RF_count"], cap_hc)
                gp_hc_used = min(hc_remaining, record["GP_count"], cap_hc - rf_hc_used)
                record["RF_HC_count"] = rf_hc_used
                record["GP_HC_count"] = gp_hc_used
                rf_hc_remaining -= rf_hc_used
                hc_remaining -= gp_hc_used

            total_slots = sum(r["GP_count"] + r["RF_count"] for _, _, _, r in cells)
            total_demand = sum(demand.get(k, 0) for k in ("GP", "HC", "RF", "HR"))

            pol_totals["hc_need"] += hc_need
            pol_totals["hc_tagged"] += hc_need - hc_remaining
            pol_totals["hr_need"] += rf_hc_need
            pol_totals["hr_tagged"] += rf_hc_need - rf_hc_remaining
            pol_totals["total_slots"] += total_slots
            pol_totals["total_demand"] += total_demand

            gap = hc_remaining + rf_hc_remaining
            if gap > 0:
                slots_short = max(total_demand - total_slots, 0)
                gap_lines.append(
                    f"    POD={pod}: HC缺口={hc_remaining}(需{hc_need}) "
                    f"HR缺口={rf_hc_remaining}(需{rf_hc_need}) "
                    f"槽位未分够={slots_short}"
                )

        hc_gap = pol_totals["hc_need"] - pol_totals["hc_tagged"]
        hr_gap = pol_totals["hr_need"] - pol_totals["hr_tagged"]
        slots_short_total = max(pol_totals["total_demand"] - pol_totals["total_slots"], 0)
        print(f"POL={pol}: HC需求={pol_totals['hc_need']} 已贴标={pol_totals['hc_tagged']} 缺口={hc_gap}  |  "
              f"HR需求={pol_totals['hr_need']} 已贴标={pol_totals['hr_tagged']} 缺口={hr_gap}  |  "
              f"本港总槽位未分够={slots_short_total}  |  有缺口的POD数={len(gap_lines)}")
        for line in gap_lines:
            print(line)

        grand_total["hc_need"] += pol_totals["hc_need"]
        grand_total["hc_gap"] += hc_gap
        grand_total["hr_need"] += pol_totals["hr_need"]
        grand_total["hr_gap"] += hr_gap
        grand_total["slots_short"] += slots_short_total
        grand_total["n_pod_with_gap"] += len(gap_lines)

    print(f"\n[全航次汇总] HC需求={grand_total['hc_need']} 总缺口={grand_total['hc_gap']}  |  "
          f"HR需求={grand_total['hr_need']} 总缺口={grand_total['hr_gap']}  |  "
          f"总槽位未分够={grand_total['slots_short']}  |  有缺口的POD-港口对数={grand_total['n_pod_with_gap']}")

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
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best, if_match_HC=False)

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
        counts.get("GP", 0) + counts.get("RF", 0) + counts.get("HC", 0) + counts.get("HR", 0)
        for pod_dict in result_vessel.cbf.values()
        for counts in pod_dict.values()
    )

    if remaining_total > 0:
        print("\n尾货cbf明细：")
        for pol, pod_dict in sorted(result_vessel.cbf.items()):
            for pod, counts in sorted(pod_dict.items()):
                if counts.get("GP", 0) > 0 or counts.get("RF", 0) > 0 or counts.get("HC", 0) > 0 or counts.get("HR", 0) > 0:
                    port_label = PORT_NAMES.get(pod, pod)
                    print(f"    POL={pol} POD={port_label}: {counts}")

    # if snapshots:
    #     evaluate_crane_intensity(vessel, snapshots, target_ci=TARGET_CI, port_names=PORT_NAMES)
    # else:
    #     print("\n[evaluate] 没有完整的逐港snapshots（求解失败且未走到任何一港完成），跳过CI评估")
    # evaluate_pod_leverage(original_cbf)
    
    tag_hicube_allocation(vessel, snapshots, original_cbf)

    # paths = vessel.export_bayplan(snapshots, BAYPLAN_DIR, port_names=PORT_NAMES, if_plot_phy=False)
    # print(f"Exported {len(paths)} bayplan files to {BAYPLAN_DIR}")
    

if __name__ == "__main__":
    main()