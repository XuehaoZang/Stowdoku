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
                if vessel.vessel_pod[bay, lr, hd] != -1:
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
    # 复用test_vessel.py里的测试数据
    is_valid = np.array([
        [[False,  True], [True,  True]],
        [[True,  True], [True,  True]],
        [[True,  True], [True,  True]],
        [[True,  True], [False, True]],
    ], dtype=bool)

    capacity_total = np.array([
        [[0, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [0, 1]],
    ], dtype=int)

    capacity_rf = np.array([
        [[0, 0], [0, 1]],
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 0]],
    ], dtype=int)

    cbf = {
        0: {
            1: {"GP": 7, "RF": 1},
            2: {"GP": 5, "RF": 1},
        },
        1: {
            3: {"GP": 7, "RF": 1},
        },
    }

    vessel = Vessel(is_valid, capacity_total, capacity_rf, cbf, current_pol=0)
    vessel_init = copy.deepcopy(vessel)

    # 预装货
    # vessel.vessel_pod[0, 0, 0] = 1
    # vessel.vessel_type[0, 0, 0] = "GP"
    # vessel.vessel_pod[2, 1, 0] = 2
    # vessel.vessel_type[2, 1, 0] = "GP"

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