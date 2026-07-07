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
- evaluate_crane_intensity: 吊车负荷强度(CI)评估
- evaluate_pod_leverage:    POD杠杆(demand leverage)分析

后续预留（还没实现，先占位注释，方便按同样的命名习惯继续加）：
- evaluate_weight_distribution: 重量分布/重心评估
- evaluate_short_landing_rate:  亏仓率评估（尾货/未装箱比例）
"""
import numpy as np
from VesselClass import Vessel


# ── 内部小工具，不对外暴露 ───────────────────────────────────────────

def _make_zones(n_bay: int, crane_number: int = 3):
    """把n_bay个big_bay连续切成crane_number组，尽量均匀（前面的组多分一个）。
    n_bay=7, crane_number=3 -> [(0,1,2), (3,4), (5,6)]
    """
    base, rem = divmod(n_bay, crane_number)
    zones, start = [], 0
    for i in range(crane_number):
        size = base + (1 if i < rem else 0)
        zones.append(tuple(range(start, start + size)))
        start += size
    return zones


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


def _ci_from_zone_loads(zone_loads: np.ndarray, crane_number: int):
    """CI = 总作业量 / 最大吊车作业量，越接近crane_number越均衡。
    本港完全没有吊车动作(total=0)时返回None，不参与打分——不是不均衡，是没有可比性。
    """
    total = zone_loads.sum()
    if total == 0:
        return None
    return total / zone_loads.max()


def _port_sequence(port: int, port_min: int, n_ports: int) -> int:
    """航次内的相对顺序位置，0起点，允许绕圈。"""
    return (port - port_min) % n_ports


# ── 对外评估函数 ─────────────────────────────────────────────────────

def evaluate_crane_intensity(vessel: Vessel, snapshots: dict, crane_number: int = 3,
                              zones=None, port_names: dict = None) -> list:
    """
    对solve()跑出来的snapshots逐港口计算实际CI值，打印一张表并返回明细。

    discharge_tally: 到达port X时船上POD==X的箱量(按bay)
                      = 紧邻X之前的那个departure快照里POD==X的记录
                      （航次起点没有"到达"这个动作，记为全0，是合理的边界情况）
    loading_tally:    port X自己departure快照里POL==X的记录(按bay)
    两者相加、按zones分组求和，代入CI公式。
    """
    if zones is None:
        zones = _make_zones(vessel.n_bay, crane_number)

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
        zone_loads = np.array([bay_total[list(z)].sum() for z in zones])
        ci = _ci_from_zone_loads(zone_loads, crane_number)

        label = port_names.get(pol, pol) if port_names else pol
        results.append({
            "pol": pol, "label": label,
            "discharge_tally": discharge_tally, "loading_tally": loading_tally,
            "zone_loads": zone_loads, "ci": ci,
        })
        prev_snap = snap

    print(f"\n──── CI评估（crane_number={crane_number}, zones={zones}）────")
    for r in results:
        zone_str = " / ".join(str(int(x)) for x in r["zone_loads"])
        ci_str = f"{r['ci']:.3f}" if r["ci"] is not None else "N/A(本港无吊车动作)"
        print(f"  POL={r['pol']}({r['label']}): zone作业量=[{zone_str}]  CI={ci_str}")

    return results


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