import numpy as np
import json

def total_containers(cbf):
    return sum(n for POD in cbf.values() for n in POD.values())

def remaining_PODs(container_queue, current_POL):
    """只返回当前港 current_POL 剩余的 POD 集合"""
    return {POD for POD, n in container_queue.get(current_POL, {}).items() if n > 0}

def if_discharge(current_POL, container_queue):
    """
    判断是否该推进到下一港 = 触发 discharge 
    条件：container_queue[current_POL] 全部装完（count 全为 0）
    """
    current_POL_queue = container_queue.get(current_POL, {})
    return all(n == 0 for n in current_POL_queue.values())

def discharge(vessel, current_POL):
    """
    卸货：把 vessel 里所有 POD == current_POL 的箱子移走（置为 -1）
    返回卸货记录，用于回溯还原
    记录格式：[(idx, POD), ...]
    """
    discharged = []
    for idx in np.ndindex(vessel.shape):
        if vessel[idx] == current_POL:
            discharged.append((idx, vessel[idx]))
            vessel[idx] = -1
    return discharged

def undischarge(vessel, discharged):
    """
    回溯还原 discharge，把 discharged 记录里的箱子放回 vessel
    """
    for idx, POD in discharged:
        vessel[idx] = POD