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
- evaluate_crane_time:           k=2吊车排班下的实际作业耗时(Time_port/Total_voyage_time)评估
- evaluate_pod_leverage:         POD杠杆(demand leverage)分析
- evaluate_pod_discharge_spread: 逐POD到港卸货分散度(variance/range/CI)评估

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


def _port_bay_totals(vessel: Vessel, snapshots: dict, port_names: dict = None) -> list:
    """
    按航次顺序，对每港算出discharge_tally/loading_tally/bay_total(按bay)。
    evaluate_crane_intensity和evaluate_crane_time共用这份计算，不用各自重新扫
    一遍snapshots。

    discharge_tally: 到达port X时船上POD==X的箱量(按bay)
                      = 紧邻X之前的那个departure快照里POD==X的记录
                      （航次起点没有"到达"这个动作，记为全0，是合理的边界情况）
    loading_tally:    port X自己departure快照里POL==X的记录(按bay)

    返回按POL升序的list[dict]：{"pol","label","discharge_tally","loading_tally","bay_total"}。
    """
    pols_in_order = sorted(snapshots.keys())
    prev_snap = None
    out = []

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
        label = port_names.get(pol, pol) if port_names else pol
        out.append({
            "pol": pol, "label": label,
            "discharge_tally": discharge_tally, "loading_tally": loading_tally,
            "bay_total": bay_total,
        })
        prev_snap = snap

    return out


def _two_crane_port_schedule(bay_total: np.ndarray, crane_rate: float = 1.0) -> dict:
    """
    单港k=2排班：crane1拿[0..i]、crane2拿[i+1..n_bay-1]，两台都按bay升序顺序
    作业。边界(bay_i, bay_{i+1})相邻，物理上不能被两台吊车同时占用——crane2
    永远t=0就先手拿下bay_{i+1}，所以中途(mid)阶段只有crane1可能要在赶到bay_i时
    等crane2让出边界，这是"都按bay升序处理"这个顺序假设下的结构性事实，crane2
    永远不会中途被crane1挡住。

    但"中途被挡"只是阻塞的一种；另一种是"自己活干完了，等对面/等这港结束"——
    这种收尾空等谁都可能摊上（先干完的那台就在原地等，不分crane1/crane2）。
    两种情况本质上都是"这港的总时长(makespan)里，这台吊车没有在真正作业的时间"，
    所以统一用 wait_i = makespan - work_i 计算，对两台吊车完全对称，不需要
    分别处理中途/收尾两段（work_i+mid_wait_i<=makespan，两者的差自动就是收尾空等）。

    对每个切分点i(0<=i<n_bay-1)：
        S1 = sum(load[0..i-1])                     crane1顺序干完前面, 到达bay_i的时刻
        boundary = load[i+1]                        crane2占着bay_{i+1}到这个时刻才让开
        work1 = S1 + load[i]                        crane1实际作业耗时(不含等待)
        mid_wait1 = max(0, boundary - S1)           crane1中途被crane2挡在边界的等待
        time1 = work1 + mid_wait1                   crane1干完自己那部分的时刻
        work2 = time2 = sum(load[i+1:])             crane2中途不会被挡，正常干完
        makespan(i) = max(time1, time2)             整港真正结束的时刻(船离港前两台都要等到这一刻)
        wait1 = makespan - work1                    crane1在整港时长里没有真正作业的时间(中途+收尾)
        wait2 = makespan - work2                    crane2在整港时长里没有真正作业的时间(中途+收尾)
    取makespan最小的切分点，其makespan即这港的Time_port。

    n_bay<2时没有"两台吊车分工"的意义，全部量算在crane1头上，crane2完全闲置
    (work2=0)，wait2按同一套定义等于makespan(全程都没活干)。

    utilization: (work1+work2) / (2*makespan) —— 两台吊车在这港实际占用的
    总时长(2*makespan，每台都经历了makespan这么久，不管是在干活还是在等)里，
    有多大比例真正花在搬箱子上(work1+work2)。等价地
    utilization = 1 - (wait1+wait2)/(2*makespan)，因为work_i+wait_i=makespan
    对两台吊车都成立，两种算法互为验证。
    """
    time = np.asarray(bay_total, dtype=float) / crane_rate
    n_bay = len(time)

    if n_bay < 2:
        total = float(time.sum())
        utilization = 0.5 if total > 0 else None
        return {"split": None, "work1": total, "wait1": 0.0, "time1": total,
                "work2": 0.0, "wait2": total, "time2": 0.0, "makespan": total,
                "utilization": utilization}

    prefix = np.concatenate(([0.0], np.cumsum(time)))  # prefix[i] = sum(time[0:i])
    total_time = float(time.sum())

    best = None
    for i in range(n_bay - 1):
        S1 = prefix[i]
        boundary = time[i + 1]
        work1 = S1 + time[i]
        mid_wait1 = max(0.0, boundary - S1)
        time1 = work1 + mid_wait1
        work2 = total_time - prefix[i + 1]
        time2 = work2

        makespan = max(time1, time2)
        wait1 = makespan - work1
        wait2 = makespan - work2

        if best is None or makespan < best["makespan"]:
            utilization = (work1 + work2) / (2 * makespan) if makespan > 0 else None
            best = {"split": i, "work1": work1, "wait1": wait1, "time1": time1,
                    "work2": work2, "wait2": wait2, "time2": time2, "makespan": makespan,
                    "utilization": utilization}

    return best


# ── 对外评估函数 ─────────────────────────────────────────────────────

def evaluate_crane_intensity(vessel: Vessel, snapshots: dict, target_ci: float = 2,
                              port_names: dict = None, if_debug: bool = True) -> list:
    """
    对solve()跑出来的snapshots逐港口计算实际CI值，返回明细；if_debug=True时
    额外打印一张逐港明细表（批量跑消融实验时可以传if_debug=False只取返回值、
    不刷屏）。

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
    port_totals = _port_bay_totals(vessel, snapshots, port_names)
    results = [{**r, "ci": _ci_from_bay_totals(r["bay_total"])} for r in port_totals]

    if if_debug:
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


def evaluate_crane_time(vessel: Vessel, snapshots: dict, k: int = 2, crane_rate: float = 1.0,
                         port_names: dict = None, if_debug: bool = True) -> list:
    """
    在evaluate_crane_intensity同一份bay_total基础上，模拟k台吊车按bay升序顺序
    作业、遇到相邻bay冲突就等待的排班过程，估算每港实际耗时Time_port，以及
    全航次总耗时Total_voyage_time = sum(Time_port for all ports)。

    目前只实现k=2：两台吊车按总量最优二分（crane1拿[0..i]、crane2拿[i+1..n-1]，
    都从自己区间里最小的bay开始按升序作业），遍历切分点i取makespan最小的方案
    （见_two_crane_port_schedule）。crane_rate：单位作业量对应的耗时倍率的倒数
    （crane_rate=1.0时，作业量数字本身就是耗时单位）。

    if_debug=True时逐港打印吊车1/2的作业时间(work，不含等待)、阻塞/空闲时间
    (wait=makespan-work，中途被对方挡在边界，或自己先干完在原地等对方/等这港
    结束，两台吊车对称统计，谁都可能非0)、这一港总耗时(Time_port=makespan)，
    以及最后的Total_voyage_time；批量跑消融实验时可以传if_debug=False只取
    返回值、不刷屏。
    
    每港结果新增utilization，按work工时加权：
    voyage_utilization = sum(work1+work2 各港) / sum(2*Time_port 各港)。
    """
    if k != 2:
        raise NotImplementedError("evaluate_crane_time目前只实现k=2的排班逻辑")

    port_totals = _port_bay_totals(vessel, snapshots, port_names)

    results = []
    if if_debug:
        print(f"\n──── 吊车作业耗时评估（k={k}台吊车, crane_rate={crane_rate}）────")
    for r in port_totals:
        sched = _two_crane_port_schedule(r["bay_total"], crane_rate=crane_rate)
        time_port = sched["makespan"]
        results.append({**r, **sched, "time_port": time_port})

        if if_debug:
            util_str = f"{sched['utilization']:.3f}" if sched["utilization"] is not None else "N/A"
            print(f"  POL={r['pol']}({r['label']}): 切分点={sched['split']}  "
                  f"吊车1[作业={sched['work1']:.1f} 阻塞={sched['wait1']:.1f}]  "
                  f"吊车2[作业={sched['work2']:.1f} 阻塞={sched['wait2']:.1f}]  "
                  f"Time_port={time_port:.1f}  利用率={util_str}")

    total_voyage_time = sum(r["time_port"] for r in results)
    total_work = sum(r["work1"] + r["work2"] for r in results)
    total_capacity = sum(2 * r["time_port"] for r in results)
    voyage_utilization = total_work / total_capacity if total_capacity > 0 else None

    if if_debug:
        util_str = f"{voyage_utilization:.3f}" if voyage_utilization is not None else "N/A"
        print(f"\n  Total_voyage_time(全程总时间) = {total_voyage_time:.1f}"
              f"  全航次利用率(work工时加权) = {util_str}")

    return results


def evaluate_pod_discharge_spread(vessel: Vessel, snapshots: dict, port_names: dict = None,
                                   if_debug: bool = True) -> dict:
    """
    逐POD评估到港卸货量在各bay间的分散度，直接复用_port_bay_totals算好的
    discharge_tally，不重新扫snapshots。

    POD到港时(即vessel.current_pol推进到==该POD编号那一港)，discharge_tally
    就是这个POD卸货量按bay的分布——_port_bay_totals里discharge_tally定义为
    "紧邻这一港之前的departure快照里POD==这一港编号的记录"，两者是同一件事，
    这里只是换个角度按POD而不是按港口去解读同一份数据。
    跳过discharge_tally全零的港（航次起点没有"到达"动作，不是一个真实的POD）。

    三个分散度指标：
    - variance: np.var(discharge_tally)，越大说明卸货量在bay间越不均匀
    - range:    max-min，最挤和最空的bay之间的差距
    - ci:       复用_ci_from_bay_totals同款定义(总量/最挤相邻bay对之和)，
                越低说明越集中在少数相邻bay(对吊车越不友好)

    返回 {pod: {"variance":.., "range":.., "ci":..}}。
    """
    port_totals = _port_bay_totals(vessel, snapshots, port_names)

    report = {}
    if if_debug:
        print("\n──── POD到港卸货分散度评估 ────")
    for r in port_totals:
        discharge_tally = r["discharge_tally"]
        if discharge_tally.sum() == 0:
            continue

        pod = r["pol"]  # POD到港时，港口编号本身就是这个POD
        variance = float(np.var(discharge_tally))
        rng = int(discharge_tally.max() - discharge_tally.min())
        ci = _ci_from_bay_totals(discharge_tally)
        report[pod] = {"variance": variance, "range": rng, "ci": ci}

        if if_debug:
            bay_str = " ".join(str(int(x)) for x in discharge_tally)
            ci_str = f"{ci:.3f}" if ci is not None else "N/A"
            print(f"  POD={pod}({r['label']}): 各bay卸货量=[{bay_str}]  "
                  f"variance={variance:.3f}  range={rng}  CI={ci_str}")

    return report


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