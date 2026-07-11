"""
utils/evaluate.py - 配载方案评估工具集

给CSP求解器出的每一版bay plan算一组量化指标，作为后续"引导搜索"或者
"事后打分挑解"的依据。所有函数都是只读分析，不改动VesselClass/CSP_solver
的任何状态，可以在main.py求解完成后直接调用。

命名约定：
- 对外的主评估函数一律 evaluate_xxx 命名（比如 evaluate_crane_intensity），
  每个对应一类独立的评估维度，互不依赖，可以按需只调用其中几个。
- 仅供内部复用的小工具函数一律加下划线前缀（比如 _make_zones），不对外暴露。

目前已有：
- evaluate_crane_intensity:      吊车负荷强度(CI)评估
- evaluate_ci_theoretical_ceiling: 给定船体bay容量分布的理论CI上限
- evaluate_pod_leverage:         POD杠杆(demand leverage)分析

后续预留（还没实现，先占位注释，方便按同样的命名习惯继续加）：
- evaluate_weight_distribution: 重量分布/重心评估
- evaluate_short_landing_rate:  亏仓率评估（尾货/未装箱比例）
"""
import numpy as np
from VesselClass import Vessel


# ── 内部小工具，不对外暴露 ───────────────────────────────────────────

def _bay_totals_from_cell(cell: np.ndarray, n_bay: int, filter_fn) -> np.ndarray:
    """按bay汇总cell里满足filter_fn(record)的GP_count+RF_count。"""
    totals = np.zeros(n_bay, dtype=int)
    for bay in range(n_bay):
        for lr in range(2):
            for hd in range(2):
                rec = cell[bay, lr, hd]
                if rec["POD"] != -1 and filter_fn(rec):
                    totals[bay] += rec["GP_count"] + rec["RF_count"]
    return totals


def _max_adjacent_pair_sum(bay_totals: np.ndarray) -> int:
    """相邻两个大bay作业量之和的最大值——两台真实吊车不能挨得太近同时作业，
    所以真正的瓶颈是"最挤的那一对相邻bay"，不是任何人为划出来的zone。
    bay index的相邻关系默认对应船体前后方向上的物理相邻（标准做法），
    n_bay<2时没有"相邻对"这个概念，返回总量本身，交给上层按total==0的逻辑处理。
    """
    if len(bay_totals) < 2:
        return int(bay_totals.sum())
    return int(max(bay_totals[i] + bay_totals[i + 1] for i in range(len(bay_totals) - 1)))


def _ci_from_bay_totals(bay_totals: np.ndarray):
    """CI = 总作业量 / 相邻两个大bay作业量之和的最大值。
    完全均匀分布时，任意相邻两bay应占总量 2/n_bay，这就是CI的理论上限
    """
    total = bay_totals.sum()
    if total == 0:
        return None
    max_pair = _max_adjacent_pair_sum(bay_totals)
    if max_pair == 0:
        return None
    return total / max_pair


def _port_sequence(port: int, port_min: int, n_ports: int) -> int:
    """航次内的相对顺序位置，0起点，允许绕圈。"""
    return (port - port_min) % n_ports


# ── 对外评估函数 ─────────────────────────────────────────────────────

def evaluate_crane_intensity(vessel: Vessel, snapshots: dict, target_ci: float = 2,
                              port_names: dict = None) -> list:
    """
    对solve()跑出来的snapshots逐港口计算实际CI值，打印一张表并返回明细。

    discharge_tally: 到达port X时船上POD==X的箱量(按bay)
                      = 紧邻X之前的那个departure快照里POD==X的记录
                      （航次起点没有"到达"这个动作，记为全0，是合理的边界情况）
    loading_tally:    port X自己departure快照里POL==X的记录(按bay)
    两者相加得到这一港每个bay的作业量，CI = 总量 / 最挤的相邻bay对之和。

    target_ci是参考基准，不是硬约束——按目前拿到的运营经验，
    总作业量500以内目标CI在3.5左右（对应n_bay=7下完全均匀分布的理论值），
    不同总量级别下这个目标可能要相应调整，这里先留一个可传参的默认值，
    不同吨位/箱量的港口不一定该用同一个数字比较。
    """
    pols_in_order = sorted(snapshots.keys())
    prev_snap = None
    results = []

    for pol in pols_in_order:
        snap = snapshots[pol]
        cell = snap["cell"]

        if prev_snap is not None:
            discharge_tally = _bay_totals_from_cell(
                prev_snap["cell"], vessel.n_bay,
                filter_fn=lambda rec, _pol=pol: rec["POD"] == _pol,
            )
        else:
            discharge_tally = np.zeros(vessel.n_bay, dtype=int)

        loading_tally = _bay_totals_from_cell(
            cell, vessel.n_bay,
            filter_fn=lambda rec, _pol=pol: rec["POL"] == _pol,
        )

        bay_total = discharge_tally + loading_tally
        ci = _ci_from_bay_totals(bay_total)

        label = port_names.get(pol, pol) if port_names else pol
        results.append({
            "pol": pol, "label": label,
            "discharge_tally": discharge_tally, "loading_tally": loading_tally,
            "bay_total": bay_total, "ci": ci,
        })
        prev_snap = snap

    print(f"\n──── CI评估（相邻bay对滑窗定义, target_ci={target_ci}）────")
    for r in results:
        bay_str = " ".join(str(int(x)) for x in r["bay_total"])
        if r["ci"] is None:
            ci_str = "N/A(本港无吊车动作)"
        else:
            flag = "  ⚠️ 低于目标" if r["ci"] < target_ci else ""
            ci_str = f"{r['ci']:.3f}{flag}"
        print(f"  POL={r['pol']}({r['label']}): 各bay作业量=[{bay_str}]  CI={ci_str}")

    return results


def evaluate_ci_theoretical_ceiling(vessel: Vessel) -> float:
    """
    这艘船在当前bay容量分布下能达到的理论CI上限，跟具体装了多少箱、哪个港口
    无关，只取决于船体几何形状——可以当作"最好情况"的基准去对比实际CI值。

    推导：假设总箱量W能按bay_capacity_share连续、无颗粒度限制地分配到每个bay
    （load[bay] = W * share[bay]，share来自VesselClass.__init__里已经算好的
    bay_capacity_share = capacity_total[bay]/总容量），代入CI定义：
        peak_ideal = W * max_i(share[i] + share[i+1])   # 只看相邻pair
        CI_ideal   = W / peak_ideal = 1 / max_i(share[i] + share[i+1])
    W被约掉，这个上限只取决于share本身。
    """
    share = vessel.bay_capacity_share
    if len(share) < 2:
        return float("inf")
    max_pair_share = max(share[i] + share[i + 1] for i in range(len(share) - 1))
    return 1.0 / max_pair_share


def evaluate_pod_leverage(cbf: dict) -> dict:
    """
    纯粹基于原始cbf需求表分析每个POD的杠杆分布，不依赖任何一次具体的solve()结果，
    分析的是"设计上的杠杆结构"——cbf本身的需求形状决定了哪些POD天生就没有回旋余地。

    ⚠️ 调用方必须传入solve()之前的cbf（原始计划量），不能传求解后被原地扣减过的
    vessel.cbf（那是剩余/尾货量）。main.py里要在solve()之前先深拷贝一份。

    对每个POD，把所有给它供货的POL按航次先后排序，从"最后一个能装它的港口"往前
    做后缀和：leverage(POL) = 这一港的量 / (这一港 + 它之后所有港口原计划要给这个
    POD的量之和)。离POD越近的港口leverage天然趋近1(没有回头路了)，
    这个函数关心的是离POD比较远的港口leverage是不是也已经很高——
    如果是，说明这个POD的需求本来就集中在少数几个早期港口，没法指望后面兜底。
    """
    all_pols = sorted(cbf.keys())
    all_pods = set()
    for pod_counts in cbf.values():
        all_pods.update(pod_counts.keys())
    port_min = min(set(all_pols) | all_pods)
    port_max = max(set(all_pols) | all_pods)
    n_ports = port_max - port_min + 1

    report = {}
    print("\n──── POD杠杆分析（基于原始cbf需求表，与crane_number无关）────")
    for pod in sorted(all_pods):
        contrib = []
        for pol in all_pols:
            qty = cbf.get(pol, {}).get(pod, {}).get("GP", 0) + \
                  cbf.get(pol, {}).get(pod, {}).get("RF", 0)
            if qty > 0 and _port_sequence(pol, port_min, n_ports) < _port_sequence(pod, port_min, n_ports):
                contrib.append((pol, qty))

        total = sum(q for _, q in contrib)
        if total == 0:
            continue

        contrib.sort(key=lambda x: _port_sequence(x[0], port_min, n_ports))  # 按航次先后排序

        # 从最后一个(离POD最近)往前做后缀和，得到每个POL的leverage
        leverages = {}
        running = 0
        for pol, qty in reversed(contrib):
            running += qty
            leverages[pol] = qty / running

        print(f"\n  POD={pod}: 总需求={total}")
        for pol, qty in contrib:
            print(f"    POL={pol}: 量={qty:>4}  leverage={leverages[pol]:.2f}"
                  f"{'  <- 几乎是唯一机会' if leverages[pol] > 0.8 else ''}")

        report[pod] = {"total": total, "contrib": contrib, "leverages": leverages}

    return report