import numpy as np
import json
from utils.viz import print_bay
from utils.dataset import total_containers, remaining_PODs

# 装入箱子, TODO 从origin 0 开始放
def CSP(bay, left_containers):
    num_left_containers = total_containers(left_containers)
    if num_left_containers == 0:
        return True # 安全放完, 没有dead slot
    # 1. 根据现在的bay, 计算每个位置的候选(是一个集合比如{1,2,3})
    current_candidates = cal_candidates(bay)
    # 缩小搜索空间 - 空位 且 候选与剩余箱子有交集
    avail_PODs = remaining_PODs(left_containers)
    choices = {}
    for idx in np.ndindex(bay.shape):
        if bay[idx] != -1:
            continue
        valid = current_candidates[idx] & avail_PODs
        if not valid:
            return False                         # dead slot → 回溯
        choices[idx] = valid
    
    if not choices:
        if left_containers:
            return False # 无空位但是有箱子没放 - 失败
        return True
    
    # MRV: 选候选最少的位置
    pos = min(choices, key=lambda p: len(choices[p]))

    for POD in sorted(choices[pos]): # 遍历这个位置所有可放的目的港, 比如能放{1,2}, 依次试
        # 比如现在在试dest = 0, 从没放的箱子里找一个dest = 0的放入

        # TODO 每次只能考虑当前origin, 这里逻辑要改, 只能从当前origin选
        POL = None
        for o, ds in left_containers.items():
            if ds.get(POD, 0) > 0:       # 这个 origin 还有去 dest 的箱子, default_value=0
                POL = o
                break      
        if POL is None:
            continue

        # 放箱子
        bay[pos] = POD
        left_containers[POL][POD] -= 1

        if CSP(bay, left_containers):
            return True

        # 回溯
        bay[pos] = -1
        left_containers[POL][POD] += 1

def cal_candidates(bay):
    '''
    如果tier上面有值，则candidate 必须是>=上面最大值；
    如果tier下面有值，则candidates 必须 <=上面最小值
    '''
    n_bay, n_row, n_tier = bay.shape
    cands = np.empty(bay.shape, dtype=object)

    for b in range(n_bay):
        for r in range(n_row):
            for t in range(n_tier):
                if bay[b][r][t] != -1:
                    cands[b][r][t] = set()
                    continue

                lo = 0                          # 候选下界
                hi = NUM_PORT - 1               # 候选上界

                above = [bay[b][r][k] for k in range(t+1, n_tier) if bay[b][r][k] != -1]
                below = [bay[b][r][k] for k in range(0, t)       if bay[b][r][k] != -1]

                if above: lo = max(above)       # >= 上方最大值
                if below: hi = min(below)       # <= 下方最小值

                cands[b][r][t] = set(range(lo, hi + 1)) if lo <= hi else set()

    return cands

if __name__ == "__main__":
    NUM_PORT = 5 # 0,1,2,3,4港口
    with open("test.json", encoding="utf-8") as f:
        data = json.load(f)

    init = data["init"]   
    init = np.array(data["init"], dtype=int)    # 直接就是 4×2×2 的嵌套 list
    print("Init:")
    print_bay(init)
    print("Candidates of Init:")
    print_bay(cal_candidates(init))

    # POL = Port of Loading 出发，POD = Port of Discharge 终点
    cbf = {
        int(POL): {int(d): n for d, n in POD.items()}
        for POL, POD in data["cbf"].items()
    }
    # print(cbf)
    left_containers = cbf
    # print(total_containers(left_containers))

    if CSP(init, left_containers):
        print_bay(init)
    else:
        print("didn't find a solution!")