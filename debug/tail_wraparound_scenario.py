"""
debug/tail_wraparound_scenario.py - 最小合成场景，专门验证apply_tail_placements
绕圈感知修复：尾箱POL=5、POD=1（数值绕圈，5>=1），host也是绕圈存活
（host.POL=3、POD=1，3>=1）。

几何：1个hold cell（bay pair(2,3), lr=0, hd=0），capacity_total=20，无reefer。

cbf设计成"host在POL=3真实装船、留有物理headroom，尾箱残量在POL=5才出现，
目的港相同"，POL=6/7留空只是为了让port_max=7，让"存活到最后一张快照"这句话
在这个场景里覆盖到不止一张快照（POL=5,6,7三张），不是退化成单张快照：
    POL=3: {POD=1: GP=12}   demand=12>tail_threshold(5)，会被真实assign()，
                             cap_total=20，全部12个都装得下，demand->0，
                             port3立即complete，host诞生，headroom=20-12=8
    POL=5: {POD=1: GP=3}    demand=3<=tail_threshold(5)，是尾货，不会被
                             assign()碰，原样留在cbf里，供build_unified_tail_list
                             的来源1捡到

期望：
    - host_key=(bay,lr,hd,3,1)的host.POL(3)本身就>POD(1)，是"host自己绕圈"
      的样例。
    - 这条尾箱记录tail.POL=5，effective_start=max(3,5)=5；POD(1)<=5，判定
      绕圈，effective_end=port_max+1=8。
    - apply_tail_placements应该把这3个GP尾箱注入进POL=5,6,7这三张快照，
      POL=0..4的快照不受影响（这个host在POL<3根本不存在，POL=3,4也不该被
      这条尾箱记录污染——它自己要到POL=5才登船）。
    - verify_cross_port_consistency用同一套绕圈感知判据独立复算，应该返回
      True（不报MISMATCH）。
"""
import copy
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from VesselClass import Vessel
from CSP_solver import solve
from utils.tail import (
    build_unified_tail_list, scan_host_candidates, match_tails_to_hosts,
    apply_tail_placements, verify_cross_port_consistency,
)


def build_scenario():
    rows = []
    for bay_idx in (2, 3):
        for row_idx in range(20):
            rows.append({
                "bay_idx": bay_idx, "row_idx": row_idx, "tier_idx": 0,
                "lr": 0, "hd": 0,
                "can_40ft": True, "can_20ft": False, "can_reefer": False,
            })
    full_slot_table = pd.DataFrame(rows)

    cbf = {
        0: {}, 1: {}, 2: {},
        3: {1: {"GP": 12}},
        4: {},
        5: {1: {"GP": 3}},
        6: {}, 7: {},
    }
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def main():
    random.seed(0)
    vessel = build_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)
    print(f"port_min={vessel.port_min}, port_max={vessel.port_max}, n_ports={vessel.n_ports}")

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    final_cbf = copy.deepcopy(result_vessel.cbf)
    print(f"solve()完成: success={success}, snapshots覆盖POL={sorted(snapshots.keys())}")

    host_key_expected = (0, 0, 0, 3, 1)
    host_cell = result_vessel.cell[0, 0, 0] if 0 < result_vessel.n_bay else None
    print(f"host在最终态self.cell[0,0,0] = {result_vessel.cell[0,0,0]} (预期POD=1,POL=3,GP_count=12)")

    unified_tail_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    print(f"\nunified_tail_list = {unified_tail_list}")

    host_pool = scan_host_candidates(result_vessel, snapshots)
    print(f"host_pool = {host_pool}")
    print(f"host自身是否绕圈(host.POL>POD): {host_key_expected in host_pool and host_key_expected[3] > host_key_expected[4]}")

    placements, unplaced = match_tails_to_hosts(
        unified_tail_list, host_pool, result_vessel.port_min, result_vessel.n_ports)
    print(f"\nplacements = {placements}")
    print(f"unplaced = {unplaced}")

    version1_dict, version2_dict = apply_tail_placements(result_vessel, snapshots, original_cbf, placements)
    print(f"\napply_tail_placements完成，version2覆盖POL={sorted(version2_dict.keys())}")

    b0 = 2  # bay pair(2,3)的b0侧
    print("\n逐快照核对host物理占用(bay_idx==2,lr==0,hd==0)的POL/POD/GP_count：")
    for pol in sorted(version2_dict.keys()):
        df = version2_dict[pol]
        mask = (df["bay_idx"] == b0) & (df["lr"] == 0) & (df["hd"] == 0) & (df["POD"] != -1)
        rows_at_pol = df[mask][["POL", "POD", "GP_count"]].to_dict("records")
        tail_rows = [r for r in rows_at_pol if r["POL"] == 5]
        print(f"  POL快照={pol}: 全部非空记录={rows_at_pol}, "
              f"其中属于尾箱记录(POL=5)的={tail_rows} "
              f"({'预期非空(在[5,8)区间内)' if 5 <= pol < 8 else '预期为空(在区间外)'})")

    expected_in_range = set(range(5, 8))
    all_snapshot_pols = set(version2_dict.keys())
    ok = True
    for pol in sorted(all_snapshot_pols):
        df = version2_dict[pol]
        mask = (df["bay_idx"] == b0) & (df["lr"] == 0) & (df["hd"] == 0) & (df["POL"] == 5) & (df["POD"] == 1)
        count_here = int(mask.sum())
        if pol in expected_in_range:
            if count_here != 3:
                ok = False
                print(f"  [MISMATCH] POL={pol} 应该有3条尾箱记录，实际={count_here}")
        else:
            if count_here != 0:
                ok = False
                print(f"  [MISMATCH] POL={pol} 不应该有尾箱记录，实际={count_here}")
    print(f"\n手工核对结果: {'[OK] 尾箱正确注入POL=5,6,7，POL<5不受影响' if ok else '[FAIL] 见上面MISMATCH'}")

    print("\n---- verify_cross_port_consistency（绕圈感知版）独立复算 ----")
    cross_ok = verify_cross_port_consistency(version2_dict, placements, result_vessel.port_max)
    print(f"verify_cross_port_consistency结果: {'PASS' if cross_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
