"""
debug/verify_tail_retrofit_e2e.py - 尾箱retrofit链路端到端集成验证脚本

目的：在真实STSE数据接入之前，用一个比4-bay demo更接近真实规模的合成场景，
把"solve() -> 尾箱缺口统计 -> host扫描+retrofit分配 -> 缺口重新核算 -> export_bayplan"
这条完整链路串起来跑一遍，专门找集成层面的bug（cross-module的接口对不上、
字段矛盾、负数缺口这类），不是新增功能，也不改VesselClass.py/utils/tail.py现有逻辑。

合成场景规模（用真实STSE 7个大bay几何 STSE_BAY_PAIRS，覆盖全部7对）：
    - 4个POL（0,1,2,3），每个POL 2-3个POD，POD最远到6（n_ports=7）
    - GP/HC/RF混合demand，多处reefer槽位
    - POL=2的POD=6、POL=3的POD=6故意给了远超物理容量的GP/HC demand，
      确保tail_threshold之上会产生真实的HC/GP尾箱缺口

跑通顺序（对应任务要求的1-6步）：
    1. Vessel初始化 + solve()
    2. build_tail_container_list(original_cbf) -> retrofit前缺口
    3. scan_host_candidates + retrofit_tail_placements -> 现场摆尾箱
    4. build_tail_container_list(proj_override=retrofit结果) -> retrofit后缺口
    5. 对比前后缺口：GP/HC只能变少或不变，RF/HR必须完全不变，且用未clamp的
       原始gap（demand-final，不经过build_tail_container_list内部的max(0,..)）
       额外核对没有出现负数（负数=过量分配，是build_tail_container_list的
       max(0,..)会悄悄吞掉的那类bug，所以这里单独算一遍不clamp的版本）
    6. retrofit_slots（叠加尾箱后的slot级DataFrame）直接喂给plot_bayplan验证
       不报错，并核对is_hc/GP_count/RF_count之间没有矛盾状态；另外用原始
       snapshots跑一次真正的export_bayplan()做一次不改动调用方式的冒烟测试。

所有assert失败时都会先打印具体触发条件（哪个(POL,POD,type)/哪个host/哪一步）
再抛出，不吞掉。
"""
import copy
import os
import random
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VesselClass import Vessel
from CSP_solver import solve
from utils.tail import (
    build_tail_container_list,
    scan_host_candidates,
    retrofit_tail_placements,
    print_tail_by_port,
)
from utils.vessel_io import STSE_BAY_PAIRS, _BIG_BAY_OF_B0
from utils.viz import plot_bayplan

SEED = 42

PORT_NAMES = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5", 6: "P6"}

# ── 合成场景几何：7个大bay，每个4个cell(holdL/holdR/deckL/deckR)。
#    值=(该cell的row(摞)数, 其中前n_reefer个物理槽位带reefer=capacity_rf)。
#    每摞固定4层(tier)，capacity_total=row数*4——用多层摞而不是单层摞，
#    是为了让row_idx(<=10)/tier_idx(<=9)落在utils.viz._render_bayplan依赖的
#    STSE_ROW_LABELS(11档)/STSE_TIER_LABELS(10档)范围内，否则画图会KeyError。
#    hold用tier_idx 0-3，deck用tier_idx 4-7，都在10档以内，也让hold摞n_eff=4
#    正好落在proj_to_slot的HOLD_HC_QUOTA_TABLE({0:0,1:1,2:2,3:2,4:2})覆盖范围内，
#    不触发n_eff>4的占位兜底分支。 ──
N_TIER_PER_ROW = 4
BAY_PLAN = {
    0: {"holdL": (5, 3), "holdR": (4, 2), "deckL": (3, 0), "deckR": (3, 0)},
    1: {"holdL": (3, 0), "holdR": (3, 0), "deckL": (3, 1), "deckR": (3, 1)},
    2: {"holdL": (4, 0), "holdR": (4, 0), "deckL": (2, 0), "deckR": (2, 0)},
    3: {"holdL": (2, 0), "holdR": (2, 0), "deckL": (4, 0), "deckR": (4, 0)},
    4: {"holdL": (3, 1), "holdR": (0, 0), "deckL": (3, 0), "deckR": (3, 0)},  # holdR=0 顺带覆盖invalid cell
    5: {"holdL": (3, 2), "holdR": (3, 2), "deckL": (3, 2), "deckR": (3, 2)},
    6: {"holdL": (3, 0), "holdR": (3, 0), "deckL": (3, 0), "deckR": (3, 0)},
}
_CELL_LR_HD = {"holdL": (0, 0), "holdR": (1, 0), "deckL": (0, 1), "deckR": (1, 1)}

# ── 合成场景cbf：4个POL，混合GP/HC/RF。刻意让同一个POD在早港demand不大
#    （早港这个cell大概率留有物理headroom）、晚港demand远超剩余物理空间
#    （POL=2/3的POD=6、POL=2的POD=4、POL=3的POD=5），既保证retrofit有真实
#    headroom可用，也保证tail_threshold之上会产生真实缺口 ──
CBF = {
    0: {2: {"GP": 25, "HC": 6, "RF": 3}, 4: {"GP": 10, "HC": 3, "RF": 2}, 6: {"GP": 8, "HC": 2}},
    1: {3: {"GP": 22, "HC": 6, "RF": 2}, 5: {"GP": 10, "HC": 3}, 6: {"GP": 2, "HC": 15}},
    2: {4: {"GP": 30, "HC": 12, "RF": 3}, 6: {"GP": 35, "HC": 12, "RF": 2}},
    3: {5: {"GP": 28, "HC": 10, "RF": 2}, 6: {"GP": 30, "HC": 20, "RF": 2}},
}


def build_synthetic_scenario():
    rows = []
    for big_bay, cells in BAY_PLAN.items():
        b0, b1 = STSE_BAY_PAIRS[big_bay]
        for cell_name, (n_rows, n_reefer) in cells.items():
            lr, hd = _CELL_LR_HD[cell_name]
            tier_base = 0 if hd == 0 else N_TIER_PER_ROW
            reefer_left = n_reefer
            for row_idx in range(n_rows):
                for tier_off in range(N_TIER_PER_ROW):
                    can_reefer = reefer_left > 0
                    if can_reefer:
                        reefer_left -= 1
                    for bay_idx in (b0, b1):
                        rows.append({
                            "bay_idx": bay_idx, "row_idx": row_idx, "tier_idx": tier_base + tier_off,
                            "lr": lr, "hd": hd,
                            "can_40ft": True, "can_20ft": False, "can_reefer": can_reefer,
                        })
    full_slot_table = pd.DataFrame(rows)
    cbf = copy.deepcopy(CBF)
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=min(cbf.keys()))
    return vessel


def _raw_gap_records(original_cbf: dict, slot_dict: dict) -> list:
    """跟build_tail_container_list内部同一套final_gp/hc/rf/hr统计口径，
    但不做max(0,..)截断——专门用来抓"负数缺口"(=过量分配/多算)这类
    会被build_tail_container_list的clamp悄悄吞掉的bug。"""
    records = []
    pol_pod_pairs = sorted(
        (pol, pod) for pol, pod_dict in original_cbf.items() for pod in pod_dict.keys()
    )
    for pol, pod in pol_pod_pairs:
        demand = original_cbf.get(pol, {}).get(pod, {})
        df = slot_dict.get(pol)
        if df is None:
            final = {"GP": 0, "HC": 0, "RF": 0, "HR": 0}
        else:
            mask = (
                (df["POL"] == pol) & (df["POD"] == pod)
                & df["bay_idx"].isin(_BIG_BAY_OF_B0.keys())
            )
            sub = df.loc[mask]
            final = {
                "GP": int(((sub["GP_count"] == 1) & (~sub["is_hc"])).sum()),
                "HC": int(((sub["GP_count"] == 1) & (sub["is_hc"])).sum()),
                "RF": int(((sub["RF_count"] == 1) & (~sub["is_hc"])).sum()),
                "HR": int(((sub["RF_count"] == 1) & (sub["is_hc"])).sum()),
            }
        for ctype in ("GP", "HC", "RF", "HR"):
            raw_gap = demand.get(ctype, 0) - final[ctype]
            records.append({"POL": pol, "POD": pod, "type": ctype, "raw_gap": raw_gap, "final": final[ctype]})
    return records


def _assert_no_negative_raw_gap(raw_records: list, label: str):
    negatives = [r for r in raw_records if r["raw_gap"] < 0]
    if negatives:
        print(f"[FAIL][{label}] 发现{len(negatives)}条负数缺口(过量分配):")
        for r in negatives:
            print(f"    POL={r['POL']} POD={r['POD']} type={r['type']} "
                  f"raw_gap={r['raw_gap']} (final={r['final']})")
        raise AssertionError(f"[{label}] 存在负数缺口，说明retrofit/diff逻辑过量分配，见上方明细")
    print(f"[OK][{label}] 全部(POL,POD,type)的原始缺口(未clamp)均>=0，无过量分配")


def _check_slot_consistency(df: pd.DataFrame, label: str):
    """核对proj_cell_to_vessel/retrofit产出的slot级DataFrame里，
    GP_count/RF_count/is_hc三者之间没有互斥矛盾。"""
    problems = []

    both = (df["GP_count"] == 1) & (df["RF_count"] == 1)
    if both.any():
        problems.append(("GP_count与RF_count同时为1", df.loc[both]))

    bad_gp = ~df["GP_count"].isin([0, 1])
    if bad_gp.any():
        problems.append(("GP_count不是0/1", df.loc[bad_gp]))

    bad_rf = ~df["RF_count"].isin([0, 1])
    if bad_rf.any():
        problems.append(("RF_count不是0/1", df.loc[bad_rf]))

    hc_without_gp = df["is_hc"] & (df["GP_count"] == 0)
    if hc_without_gp.any():
        problems.append(("is_hc=True但GP_count=0", df.loc[hc_without_gp]))

    hc_with_rf = df["is_hc"] & (df["RF_count"] == 1)
    if hc_with_rf.any():
        problems.append(("is_hc=True但RF_count=1(HR未实现，不应出现)", df.loc[hc_with_rf]))

    unassigned_with_count = (df["POD"] == -1) & ((df["GP_count"] != 0) | (df["RF_count"] != 0))
    if unassigned_with_count.any():
        problems.append(("POD=-1但GP_count/RF_count非0", df.loc[unassigned_with_count]))

    if problems:
        print(f"[FAIL][{label}] 发现{len(problems)}类矛盾:")
        for desc, sub in problems:
            print(f"  - {desc}，共{len(sub)}行，示例:")
            print(sub.head(5).to_string())
        raise AssertionError(f"[{label}] slot级DataFrame存在GP_count/RF_count/is_hc矛盾，见上方明细")
    print(f"[OK][{label}] is_hc/GP_count/RF_count之间无矛盾状态")


def main():
    print("=" * 70)
    print("尾箱retrofit链路端到端集成验证")
    print("=" * 70)

    random.seed(SEED)
    vessel = build_synthetic_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    print(f"\n场景规模: {vessel.n_bay}个big_bay, 物理总容量capacity_total={int(vessel.capacity_total.sum())}, "
          f"RF槽位capacity_rf={int(vessel.capacity_rf.sum())}")
    total_demand = sum(
        c.get(k, 0) for pod_dict in original_cbf.values() for c in pod_dict.values() for k in ("GP", "HC", "RF", "HR")
    )
    print(f"cbf总demand={total_demand}, POL span={sorted(original_cbf.keys())}")

    # ── 1. solve() ──
    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    if result_vessel is None:
        raise AssertionError("solve()连一个箱子都没能装上，场景需要重新设计")
    print(f"\n[1] solve()完成: success={success}, snapshots覆盖POL={sorted(snapshots.keys())}")

    # ── 2. retrofit前缺口 ──
    tail_before = build_tail_container_list(result_vessel, snapshots, original_cbf)
    print(f"\n[2] retrofit前尾箱缺口 (共{len(tail_before)}条记录):")
    for rec in sorted(tail_before, key=lambda r: (r["POL"], r["POD"], r["type"])):
        print(f"    POL={rec['POL']} POD={rec['POD']} type={rec['type']} count={rec['count']}")
    print_tail_by_port(tail_before, original_cbf, port_names=PORT_NAMES, label="retrofit前")

    if not tail_before:
        raise AssertionError("场景没有产出任何尾箱缺口，无法验证retrofit链路，需要加大demand或缩小capacity重新设计场景")

    has_gp_hc_gap_before = any(r["type"] in ("GP", "HC") for r in tail_before)
    if not has_gp_hc_gap_before:
        raise AssertionError("场景没有产出GP/HC类型的尾箱缺口(只有RF/HR)，retrofit逻辑只处理GP/HC，"
                              "这个场景无法验证retrofit的核心行为，需要重新设计场景")
    print(f"\n[OK] 场景确实产出了GP/HC尾箱缺口，可以继续验证retrofit")

    # before这一步顺带自己算一遍proj slot_dict，供第5步做未clamp的原始缺口核对用
    before_slot_dict = {
        pol: result_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        for pol in snapshots
    }
    _assert_no_negative_raw_gap(_raw_gap_records(original_cbf, before_slot_dict), "retrofit前")

    # ── 3. host扫描 + retrofit分配 ──
    host_pool = scan_host_candidates(result_vessel, snapshots)
    print(f"\n[3] scan_host_candidates: host候选池共{len(host_pool)}条")

    retrofit_slots = retrofit_tail_placements(result_vessel, snapshots, tail_before, original_cbf)
    print(f"    retrofit_tail_placements完成，覆盖POL={sorted(retrofit_slots.keys())}")

    # ── 4. retrofit后缺口 ──
    tail_after = build_tail_container_list(
        result_vessel, snapshots, original_cbf, proj_override=retrofit_slots)
    print(f"\n[4] retrofit后尾箱缺口 (共{len(tail_after)}条记录):")
    for rec in sorted(tail_after, key=lambda r: (r["POL"], r["POD"], r["type"])):
        print(f"    POL={rec['POL']} POD={rec['POD']} type={rec['type']} count={rec['count']}")
    print_tail_by_port(tail_after, original_cbf, port_names=PORT_NAMES, label="retrofit后")

    _assert_no_negative_raw_gap(_raw_gap_records(original_cbf, retrofit_slots), "retrofit后")

    # ── 5. 对比前后缺口 ──
    print("\n[5] retrofit前后缺口对比")
    before_map = {(r["POL"], r["POD"], r["type"]): r["count"] for r in tail_before}
    after_map = {(r["POL"], r["POD"], r["type"]): r["count"] for r in tail_after}
    all_keys = set(before_map) | set(after_map)

    gp_hc_regressions = []
    rf_hr_changed = []
    for key in sorted(all_keys):
        pol, pod, ctype = key
        b = before_map.get(key, 0)
        a = after_map.get(key, 0)
        if ctype in ("GP", "HC"):
            if a > b:
                gp_hc_regressions.append((key, b, a))
        else:  # RF/HR
            if a != b:
                rf_hr_changed.append((key, b, a))

    if gp_hc_regressions:
        print(f"  [FAIL] {len(gp_hc_regressions)}条GP/HC缺口在retrofit后反而变多:")
        for (pol, pod, ctype), b, a in gp_hc_regressions:
            print(f"    POL={pol} POD={pod} type={ctype}: before={b} -> after={a}")
        raise AssertionError("存在GP/HC缺口retrofit后变多的情况，见上方明细")
    print(f"  [OK] 所有GP/HC缺口retrofit后都 <= retrofit前 (共核对{sum(1 for k in all_keys if k[2] in ('GP','HC'))}条)")

    if rf_hr_changed:
        print(f"  [FAIL] {len(rf_hr_changed)}条RF/HR缺口在retrofit前后发生了变化(retrofit不该碰RF/HR):")
        for (pol, pod, ctype), b, a in rf_hr_changed:
            print(f"    POL={pol} POD={pod} type={ctype}: before={b} -> after={a}")
        raise AssertionError("RF/HR缺口被retrofit意外改动，见上方明细——retrofit逻辑碰到了不该碰的类型")
    print(f"  [OK] 所有RF/HR缺口retrofit前后完全一致 (共核对{sum(1 for k in all_keys if k[2] in ('RF','HR'))}条)")

    total_before = sum(before_map.values())
    total_after = sum(after_map.values())
    print(f"  总缺口: retrofit前={total_before}, retrofit后={total_after} "
          f"({'减少' if total_after < total_before else '未变化' if total_after == total_before else '增加(异常)'})")
    if total_after > total_before:
        raise AssertionError(f"总缺口retrofit后不降反升: {total_before} -> {total_after}")

    # ── 6a. retrofit后的slot级DataFrame：结构一致性 + 直接喂给plot_bayplan验证不报错 ──
    print("\n[6a] retrofit后slot级DataFrame一致性核对 + plot_bayplan渲染冒烟测试")
    scratch_dir = os.path.join(
        "/private/tmp/claude-501/-Users-hi-babe-Documents-Stowage-Stowdoku/030fac59-4664-4549-81f6-a293db2f777a/scratchpad",
        "tail_retrofit_e2e_bayplan",
    )
    os.makedirs(scratch_dir, exist_ok=True)

    all_pods = set()
    for df in retrofit_slots.values():
        all_pods.update(int(p) for p in df.loc[df["POD"] != -1, "POD"].unique())
    from utils.viz import _default_port_colors
    port_colors = _default_port_colors(all_pods)

    for pol in sorted(retrofit_slots.keys()):
        df = retrofit_slots[pol]
        _check_slot_consistency(df, label=f"retrofit后 POL={pol}")
        paths = plot_bayplan(
            df, title=f"POL={pol} ({PORT_NAMES.get(pol, pol)}) departure (post-retrofit)",
            filename=f"{pol}_{PORT_NAMES.get(pol, pol)}_DEP_bayplan_retrofit.png",
            save_dir=scratch_dir, port_colors=port_colors, port_names=PORT_NAMES,
            if_plot_phy=False,
        )
        print(f"    [OK] POL={pol}: plot_bayplan未报错, 输出: {paths}")

    # ── 6b. 原始snapshots（未叠加retrofit的cell级状态）跑一次真正的export_bayplan，
    #        确认这条改动没有把export_bayplan本身跑挂（retrofit不写回snapshots，
    #        所以export_bayplan看到的仍是retrofit前的cell级状态，是预期行为，
    #        这里只做"接口没坏"的冒烟测试） ──
    print("\n[6b] export_bayplan(原始snapshots) 冒烟测试")
    export_paths = result_vessel.export_bayplan(
        snapshots, scratch_dir, original_cbf, port_names=PORT_NAMES,
        if_csv=False, if_plot_phy=False,
    )
    print(f"    [OK] export_bayplan未报错，产出{len(export_paths)}个文件")

    print("\n" + "=" * 70)
    print("[PASS] 尾箱retrofit链路端到端集成验证全部通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
