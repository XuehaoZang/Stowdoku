import numpy as np
import copy
from VesselClass import Vessel
from utils.viz import print_vessel


def cal_candidates(vessel: Vessel) -> dict:
    """
    计算所有valid且未赋值的cell的候选集。
    跳过cbf已空导致候选为空的cell（不视为dead cell）。
    返回 {(bay, lr, hd): set of (POD, ctype)}，若某cell候选为空且cbf仍有余量则返回None。
    """
    choices = {}
    has_remaining = vessel.total_remaining() > 0
    for bay in range(vessel.n_bay):
        for lr in range(2):
            for hd in range(2):
                if not vessel.is_valid[bay, lr, hd]:
                    continue
                if vessel.cell[bay, lr, hd]["POD"] != -1:
                    continue
                cands = vessel.get_candidates(bay, lr, hd)
                if not cands:
                    if has_remaining:
                        return None  # 真正的dead cell：cbf有货但此处放不下
                    # cbf已空，跳过此cell
                    continue
                choices[(bay, lr, hd)] = cands
    return choices


def mrv_select(choices: dict, vessel: Vessel):
    """
    MRV选择：优先has_reefer且含RF候选的cell，组内按候选集大小升序。
    返回 (bay, lr, hd)
    """
    def priority(item):
        (bay, lr, hd), cands = item
        has_rf_cand = vessel.has_reefer[bay, lr, hd] and any(t == "RF" for _, t in cands)
        return (0 if has_rf_cand else 1, len(cands))

    return min(choices.items(), key=priority)[0]


def solve(vessel: Vessel, is_debug=False, snapshots=None) -> bool:
    """
    统一大递归：装载 + 换港，discharge作为递归中的特殊节点。
    vessel内部维护current_pol和cbf状态。
    """
    if snapshots is None:
        snapshots = {}

    # 终止条件：已超过最后一个港口
    if vessel.current_pol > max(vessel.cbf.keys()):
        return True

    # 当前港装完 → 换港
    if vessel.port_complete():
        snapshots[vessel.current_pol] = vessel.snapshot()

        vessel.advance_pol()
        discharged = vessel.discharge(vessel.current_pol)

        if is_debug:
            print(f"[Departure] POL={vessel.current_pol - 1}")
            print(f"[Arrive] POL={vessel.current_pol}, 卸了{len(discharged)}个cell")

        if solve(vessel, is_debug, snapshots):
            return True

        # 下一港失败，回溯
        vessel.undischarge(discharged)
        vessel.current_pol -= 1
        del snapshots[vessel.current_pol]

        if is_debug:
            print(f"[Backtrack] 回溯到POL={vessel.current_pol}")
        return False

    # 计算候选集
    choices = cal_candidates(vessel)
    if choices is None:
        return False  # dead cell，cbf有余量但某cell放不下

    if not choices:
        # 所有空cell的cbf候选都已耗尽，等价于port_complete
        # 下一次递归开头的port_complete()会处理换港
        return solve(vessel, is_debug, snapshots)

    # MRV选位置
    pos = mrv_select(choices, vessel)
    bay, lr, hd = pos

    for pod, ctype in sorted(choices[pos]):
        vessel.assign(bay, lr, hd, pod, ctype)

        if is_debug:
            print(f"  assign ({bay},{lr},{hd}) POD={pod} {ctype}")

        if solve(vessel, is_debug, snapshots):
            return True

        vessel.unassign(bay, lr, hd, pod, ctype)

    return False


if __name__ == "__main__":
    import pandas as pd

    def _make_pair_rows(b0, b1, cells):
        """cells: {(lr,hd): {"capacity": 0或1, "reefer": bool}}。
        capacity=1时给这对bay各生成一行占位slot，capacity=0则不生成（对应is_valid=False）。"""
        rows = []
        for (lr, hd), spec in cells.items():
            if spec["capacity"] == 0:
                continue
            row_idx = 0 if lr == 0 else 5
            tier_idx = 0 if hd == 0 else 4
            for bay_idx in (b0, b1):
                rows.append({
                    "bay_idx": bay_idx, "row_idx": row_idx, "tier_idx": tier_idx,
                    "lr": lr, "hd": hd,
                    "can_40ft": True, "can_20ft": False, "can_reefer": spec["reefer"],
                })
        return rows

    # 复现原先4-bay测试场景的is_valid/capacity/reefer分布，
    # 分别对应STSE_BAY_PAIRS的前4对(2,3)/(4,5)/(6,7)/(8,9)（big_bay 0-3）
    # 后3对(big_bay 4-6)不给数据，capacity_total=0，永远is_valid=False，不参与搜索
    rows = []
    rows += _make_pair_rows(2, 3, {
        (0, 0): {"capacity": 0, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 1, "reefer": False},
        (1, 1): {"capacity": 1, "reefer": True},
    })
    rows += _make_pair_rows(4, 5, {
        (0, 0): {"capacity": 1, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 1, "reefer": False},
        (1, 1): {"capacity": 1, "reefer": False},
    })
    rows += _make_pair_rows(6, 7, {
        (0, 0): {"capacity": 1, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 1, "reefer": True},
        (1, 1): {"capacity": 1, "reefer": False},
    })
    rows += _make_pair_rows(8, 9, {
        (0, 0): {"capacity": 1, "reefer": False},
        (0, 1): {"capacity": 1, "reefer": False},
        (1, 0): {"capacity": 0, "reefer": False},
        (1, 1): {"capacity": 1, "reefer": False},
    })
    full_slot_table = pd.DataFrame(rows)

    cbf = {
        0: {
            1: {"GP": 7, "RF": 1},
            2: {"GP": 5, "RF": 1},
        },
        1: {
            3: {"GP": 7, "RF": 1},
        },
    }

    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    vessel_init = copy.deepcopy(vessel)

    snapshots = {}
    if solve(vessel, is_debug=False, snapshots=snapshots):
        print("\n──── Solution Found ────")
        print("[init]")
        print_vessel(vessel_init)  # 初始状态（已是final，仅做参考）
 
        for pol in sorted(snapshots.keys()):
            print(f"[departure] POL={pol} 出发状态:")
            print_vessel(snapshots[pol])
 
        print("[final state]")
        print_vessel(vessel)
    else:
        print("No solution found")