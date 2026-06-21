import numpy as np
import json
from utils.viz import print_vessel
from utils.vessel import total_containers, remaining_PODs, if_discharge, discharge, undischarge

# 装入箱子, TODO 从 current port = 0 开始放, current port 逐步增加
def CSP(vessel, container_queue):
    # 这里应该套一个大循环POL，相当于模拟遍历各个港，还要加上discharge的逻辑
    
    num_container_queue = total_containers(container_queue)
    if num_container_queue == 0:
        return True # 安全放完, 没有dead slot
    # 1. 根据现在的bay, 计算每个位置的候选(是一个集合比如{1,2,3})
    current_candidates = cal_candidates(vessel)
    # 缩小搜索空间 - 空位 且 候选与剩余箱子有交集
    avail_PODs = remaining_PODs(container_queue)
    choices = {}
    for idx in np.ndindex(vessel.shape):
        if vessel[idx] != -1:
            continue
        valid = current_candidates[idx] & avail_PODs
        if not valid:
            return False                         # dead slot → 回溯
        choices[idx] = valid
    
    if not choices:
        if container_queue:
            return False # 无空位但是有箱子没放 - 失败
        return True
    
    # MRV: 选候选最少的位置
    pos = min(choices, key=lambda p: len(choices[p]))
    
    for POD in sorted(choices[pos]): # 遍历这个位置所有可放的目的港, 比如能放{1,2}, 依次试

        # TODO 每次只能考虑当前origin, 这里逻辑要改, 只能从当前origin选
        POL = None
        for o, ds in container_queue.items():
            if ds.get(POD, 0) > 0:       # 这个 origin 还有去 dest 的箱子, default_value=0
                POL = o
                break      
        if POL is None:
            continue

        # 放箱子
        vessel[pos] = POD
        container_queue[POL][POD] -= 1

        if CSP(vessel, container_queue):
            return True

        # 回溯
        vessel[pos] = -1
        container_queue[POL][POD] += 1

def cal_candidates(vessel):
    '''
    如果tier上面有值，则candidate 必须是>=上面最大值；
    如果tier下面有值，则candidates 必须 <=上面最小值
    '''
    n_bay, n_row, n_tier = vessel.shape
    cands = np.empty(vessel.shape, dtype=object)

    for b in range(n_bay):
        for r in range(n_row):
            for t in range(n_tier):
                if vessel[b][r][t] != -1:
                    cands[b][r][t] = set()
                    continue

                lo = 0                          # 候选下界
                hi = NUM_PORT - 1               # 候选上界

                above = [vessel[b][r][k] for k in range(t+1, n_tier) if vessel[b][r][k] != -1]
                below = [vessel[b][r][k] for k in range(0, t)       if vessel[b][r][k] != -1]

                if above: lo = max(above)       # >= 上方最大值
                if below: hi = min(below)       # <= 下方最小值

                cands[b][r][t] = set(range(lo, hi + 1)) if lo <= hi else set()

    return cands

def solve(vessel, current_POL, container_queue, is_debug=False, snapshots=None):
    """
    统一大递归，管装载和换港
    - 装载：调用 CSP 在当前港放箱子
    - 换港：discharge 当前港，递归进入 current_POL + 1
    """
    if snapshots is None:
        snapshots = {}
        
    # 终止条件：所有港口都过完了
    if current_POL > max(container_queue.keys()):
        return total_containers(container_queue) == 0

    # 触发 discharge：当前港装完，推进到下一港
    if if_discharge(current_POL, container_queue):
        
        snapshots[current_POL] = vessel.copy()  # departure 快照
        
        next_POL = current_POL + 1
        discharged = discharge(vessel, next_POL)
        
        if is_debug:
            print(f"[Departure] 从 POL={current_POL} 出发状态:")  # 改这里
            print_vessel(vessel)
            print("=" * 30)
            print(f"[Arrive] 到达 POL={next_POL}")
            print(f"[Discharge] 卸了 {len(discharged)} 个 POD={next_POL} 的箱子")
            print_vessel(vessel)
            print(f"[Loading] 开始装 POL={next_POL} 的箱子")

        if solve(vessel, next_POL, container_queue, is_debug, snapshots):
            return True

        # 下一港失败，还原 discharge，回到当前港继续枚举
        undischarge(vessel, discharged)
        del snapshots[current_POL]
        
        if is_debug:
            print(f"[Undo discharge] POL={next_POL} 失败，回溯还原到 POL={current_POL}")
        return False

    # 当前港还有箱子，执行一步装载决策
    current_candidates = cal_candidates(vessel)
    avail_PODs = remaining_PODs(container_queue, current_POL)  # 只看当前港剩余
    
    choices = {}
    for idx in np.ndindex(vessel.shape):
        if vessel[idx] != -1:
            continue
        valid = current_candidates[idx] & avail_PODs
        if not valid:
            return False  # dead slot → 回溯
        choices[idx] = valid

    if not choices:
        return False  # 无空位但当前港还有箱子

    # MRV
    pos = min(choices, key=lambda p: len(choices[p]))

    for POD in sorted(choices[pos]):
        if container_queue.get(current_POL, {}).get(POD, 0) == 0:
            continue

        vessel[pos] = POD
        container_queue[current_POL][POD] -= 1

        if solve(vessel, current_POL, container_queue, is_debug, snapshots):
            return True

        vessel[pos] = -1
        container_queue[current_POL][POD] += 1

    return False

if __name__ == "__main__":
    NUM_PORT = 5 # 0,1,2,3,4港口
    with open("data/test_data_3.json", encoding="utf-8") as f:
        data = json.load(f)

    vessel = np.array(data["init"], dtype=int)    # 直接就是 4×2×2 的嵌套 list
    
    # print("Candidates of init:")
    # print_vessel(cal_candidates(vessel))

    # POL = Port of Loading 出发，POD = Port of Discharge 终点
    cbf = {
        int(POL): {int(d): n for d, n in POD.items()}
        for POL, POD in data["cbf"].items()
    }
    # print(cbf)
    # print(total_containers(cbf))

    snapshots={}
    if solve(vessel, 0, cbf, is_debug=False, snapshots=snapshots):
        print("\n-------- [Final Solution] --------")
        print("[init] 从 POL=0  初始化状态:")
        print_vessel(np.array(data["init"], dtype=int))
        for POL in sorted(snapshots.keys()):
            print(f"[departure] POL={POL} 出发状态:")
            print_vessel(snapshots[POL])
        print(f"[final] 最终到达状态:")
        print_vessel(vessel)
    else:
        print("\n-------- [No Solution Found] --------")