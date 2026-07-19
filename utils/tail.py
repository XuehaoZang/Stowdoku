"""
utils/tail.py - 尾箱处理后处理管线的测试fixture + 发现阶段调试脚本

跑一遍CSP_solver.py __main__里那个4-bay测试场景的solve() + export_bayplan()
（不改求解器逻辑本身），把snapshots/original_cbf/最终vessel.cbf落盘成pickle，
供后续"尾箱安置"任务复用，避免每次都重新跑一遍搜索。

同时手动核对proj_cell_to_vessel里三条独立的cbf写回路径：
    来源1（tail_threshold小额尾货）：get_candidates()里从未参与搜索、原样
        留在最终vessel.cbf里的残量。在solve()刚结束、任何proj_cell_to_vessel
        调用之前读取（本脚本final_cbf = deepcopy(result_vessel.cbf)那一行
        就在print_source2/3_tail之前），跟来源2/3互不重叠。
    来源2（deck-squeeze）：HC标签把deck摞摆满后触发的物理腾空，固定回退
        1个GP名额/次。对每个POL重跑一次proj_cell_to_vessel，靠
        VesselClass._tail_source2_log统计触发次数。
    来源3（HC/RF预算池分不完回退）：跟来源2是proj_cell_to_vessel里两条
        独立的写回路径（不是同一段代码的两个分支），(POL,POD)分组的HC/RF
        预算池分不完时触发，回退量=分不完的余量(可以>1)。靠
        VesselClass._tail_source3_log统计。

本脚本只做“发现”，不做“安置”：不修改cbf、不重新分配槽位，只打印/落盘诊断信息。
"""
import copy
import os
import pickle
import random

import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VesselClass import Vessel
from CSP_solver import solve
from utils.vessel_io import _BIG_BAY_OF_B0, STSE_BAY_PAIRS


def _make_pair_rows(b0, b1, cells):
    """复刻CSP_solver.py __main__里的同名辅助函数，构造一对bay的slot行。"""
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


def build_test_scenario():
    """复刻CSP_solver.py __main__里的4-bay测试场景，返回(vessel, vessel_init)。"""
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
    return vessel, vessel_init


def print_source1_tail(final_cbf: dict):
    """[legacy] 仅用于回归对比和调试，新代码请使用build_tail_container_list，不要在新功能里依赖本函数。

    来源1：遍历最终vessel.cbf，打印每个非零的(POL, POD, GP/HC/RF/HR, 数量)。"""
    print("\n──── 尾箱来源1（tail_threshold小额尾货）────")
    total = 0
    for pol, pod_dict in sorted(final_cbf.items()):
        for pod, counts in sorted(pod_dict.items()):
            for ctype in ("GP", "HC", "RF", "HR"):
                n = counts.get(ctype, 0)
                if n:
                    print(f"  POL={pol} POD={pod} {ctype}={n}")
                    total += n
    print(f"来源1总箱数: {total}")
    return total


def _dedup_tail_log_by_pol_pod(log: list, key_len: int = 2) -> dict:
    """按记录的前key_len个字段(POL,POD)对_tail_source2/3_log分组去重。

    同一个(POL,POD)分组在被discharge之前会原样出现在它存活期内的每一张
    POL快照里，export_bayplan对每个POL都调用一次proj_cell_to_vessel，
    这些记录因此会重复出现——但重复只是"同一件事重算了几次"，不是"发生了
    几次"，按(POL,POD)去重后每组只应保留1条，代表真正发生过1次的事件。

    去重的同时做幂等性健康检查：断言同一个(POL,POD)分组下所有raw记录
    完全相同（squeeze/预算池回退的计算结果不该随着reruns漂移）。这个检查
    以后要留在主报告路径里常驻——万一某次改动意外破坏了这份幂等性
    （比如proj_cell_to_vessel开始依赖某个会变化的状态），应该在正常跑
    pipeline时就直接AssertionError报错，而不是要等专门写合成场景才能发现。

    返回{(pol, pod): 代表记录(取组内任意一条，因为已经断言过组内一致)}。
    """
    groups = {}
    for entry in log:
        key = entry[:key_len]
        groups.setdefault(key, []).append(entry)

    deduped = {}
    for key, entries in groups.items():
        distinct = set(entries)
        if len(distinct) != 1:
            raise AssertionError(
                f"[尾箱日志幂等性校验失败] (POL,POD)={key} 在多次POL快照replay里"
                f"记录不一致，squeeze/预算池回退的计算结果不是幂等的！"
                f"该分组下的原始记录: {entries}"
            )
        deduped[key] = entries[0]

    return deduped


def print_source2_and_source3_tail(vessel: Vessel, snapshots: dict, original_cbf: dict):
    """[legacy] 仅用于回归对比和调试，新代码请使用build_tail_container_list，不要在新功能里依赖本函数。

    来源2+来源3：对每个POL只重跑一次proj_cell_to_vessel（两条日志在同一次调用里
    各自累积，分开跑两遍会让不受already_written去重保护的来源3被重复计入两次），
    分别靠_tail_source2_log/_tail_source3_log统计触发次数和回退量。

    同一个(POL,POD)分组在被discharge之前会跨越多张POL快照，每张快照replay
    一次proj_cell_to_vessel就会重复记一条日志，所以raw log条数会系统性地
    比"实际发生过的事件数"多——这里按(POL,POD)去重后再统计，去重时顺带
    校验幂等性（_dedup_tail_log_by_pol_pod），保证这个健康检查是主报告路径
    的常驻部分，不是只在临时验证脚本里跑一次。

    返回(来源2去重后箱数, 来源3去重后箱数)。
    """
    print("\n──── 尾箱来源2（HC降级挤出触发deck腾空）────")
    for pol in sorted(snapshots.keys()):
        before2 = len(vessel._tail_source2_log)
        before3 = len(vessel._tail_source3_log)
        vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        triggered2 = vessel._tail_source2_log[before2:]
        triggered3 = vessel._tail_source3_log[before3:]
        for pol_hit, pod_hit in triggered2:
            print(f"  [POL快照={pol}] 触发腾空: POL={pol_hit}, POD={pod_hit}, 数量=1")
        for pol_hit, pod_hit, gp_leftover, rf_leftover in triggered3:
            print(f"  [POL快照={pol}] 触发预算池回退: POL={pol_hit}, POD={pod_hit}, "
                  f"gp_hc_budget剩={gp_leftover}, rf_hc_budget剩={rf_leftover}")

    log2 = vessel._tail_source2_log
    log3 = vessel._tail_source3_log

    dedup2 = _dedup_tail_log_by_pol_pod(log2, key_len=2)
    dedup3 = _dedup_tail_log_by_pol_pod(log3, key_len=2)

    raw2_count = len(log2)
    dedup2_count = len(dedup2)  # 每个(POL,POD)分组固定贡献1个GP名额
    print(f"来源2: 去重前raw条数={raw2_count}, 去重后事件数/箱数={dedup2_count}"
          + ("（去重前后相等，这个场景没有跨POL重复replay）" if raw2_count == dedup2_count else ""))

    raw3_gp_total = sum(gp for (_, _, gp, _) in log3)
    raw3_rf_total = sum(rf for (_, _, _, rf) in log3)
    dedup3_gp_total = sum(gp for (_, _, gp, _) in dedup3.values())
    dedup3_rf_total = sum(rf for (_, _, _, rf) in dedup3.values())
    dedup3_total = dedup3_gp_total + dedup3_rf_total
    print(f"\n──── 尾箱来源3（HC/RF预算池分不完回退）────")
    print(f"来源3: 去重前raw条数={len(log3)} (GP合计{raw3_gp_total}, RF合计{raw3_rf_total}), "
          f"去重后分组数={len(dedup3)} (GP合计{dedup3_gp_total}, RF合计{dedup3_rf_total})"
          + ("（去重前后相等，这个场景没有跨POL重复replay）" if len(log3) == len(dedup3) else ""))

    return dedup2_count, dedup3_total


def build_unified_tail_list(vessel: Vessel, final_cbf: dict, snapshots: dict, original_cbf: dict,
                             dedup2: dict = None, dedup3: dict = None) -> list:
    """[legacy] 仅用于回归对比和调试，新代码请使用build_tail_container_list，不要在新功能里依赖本函数。

    合并三个尾箱来源为统一列表，供任务2b/2c的host匹配消费。

    来源1：final_cbf（solve()刚结束、任何proj_cell_to_vessel调用之前的
        vessel.cbf快照）里每个非零的(POL, POD, 类型)残量。
    来源2：dedup2（按(POL,POD)去重后的_tail_source2_log），每条记录固定
        回退1个GP。
    来源3：dedup3（按(POL,POD)去重后的_tail_source3_log），每条记录按
        (gp_hc_budget剩, rf_hc_budget剩)回退进HC/HR。

    dedup2/dedup3可由调用方直接传入（复用已经跑过一次
    print_source2_and_source3_tail后的日志去重结果，避免重复触发
    proj_cell_to_vessel、污染vessel._tail_source2_log/_tail_source3_log
    供后续诊断使用）；不传时本函数自己调用一次
    print_source2_and_source3_tail并对其产生的日志去重。

    返回list[dict]，每条{"POL","POD","type","count","source"}，count>0。

    注意：同一个(POL,POD,type)完全可能同时被多个来源命中，且这是合理的、
    预期内的情况——例如同一个GP bucket，既可能在tail_threshold下留有
    未参与搜索的残量（来源1），也可能同时因为deck-squeeze触发过回退
    （来源2），这是两套独立的触发机制在统计同一个bucket，不是重复计数，
    不应该合并/去重/求和成一条。本函数因此不对跨来源重叠做任何校验或
    合并，每个来源各自贡献的记录都原样保留在返回列表里；调用方如果需要
    某个(POL,POD,type)的合计数量，需要自己对返回列表按需sum。
    """
    if dedup2 is None or dedup3 is None:
        print_source2_and_source3_tail(vessel, snapshots, original_cbf)
        dedup2 = _dedup_tail_log_by_pol_pod(vessel._tail_source2_log, key_len=2)
        dedup3 = _dedup_tail_log_by_pol_pod(vessel._tail_source3_log, key_len=2)

    records = []

    # 来源1：final_cbf里每个非零(POL, POD, 类型)残量
    for pol, pod_dict in sorted(final_cbf.items()):
        for pod, counts in sorted(pod_dict.items()):
            for ctype in ("GP", "HC", "RF", "HR"):
                n = counts.get(ctype, 0)
                if n:
                    records.append({"POL": pol, "POD": pod, "type": ctype, "count": n, "source": 1})

    # 来源2：每个去重后的(POL,POD)分组固定回退1个GP名额
    for pol, pod in sorted(dedup2.keys()):
        records.append({"POL": pol, "POD": pod, "type": "GP", "count": 1, "source": 2})

    # 来源3：按(gp_hc_budget剩, rf_hc_budget剩)回退进HC/HR
    for (pol, pod), entry in sorted(dedup3.items()):
        _, _, gp_leftover, rf_leftover = entry
        if gp_leftover:
            records.append({"POL": pol, "POD": pod, "type": "HC", "count": gp_leftover, "source": 3})
        if rf_leftover:
            records.append({"POL": pol, "POD": pod, "type": "HR", "count": rf_leftover, "source": 3})

    return records


def build_tail_container_list(vessel: Vessel, snapshots: dict, original_cbf: dict) -> list:
    """新口径尾箱统计：不再把三条独立触发路径(来源1/2/3)的残量相加，而是对
    每个(POL,POD)做一次性的"最终结果 vs 原始demand"比较，直接算出真实缺口。

    背景：来源1(assign()按总量分配、不知道cap_hc配额)和来源3(贴标签阶段
    才第一次知道cap_hc，把预算池分不完的部分回退)统计的是同一批箱子在两个
    不同信息状态下的两次观察，不是两件独立发生的事——旧的build_unified_tail_list
    把它们当独立来源相加会重复计数。真实缺口只取决于"最终船上实际贴出了
    多少标签"和"最初demand要多少"这两个端点，中间assign()阶段猜错、
    贴标签阶段又回退的过程量不重要。

    公式（对original_cbf里出现过的每个(POL,POD)，不依赖旧的按日志分组的
    判断）：
        最终HC标签数 = 该(POL,POD)自己这港新装的slot里，is_hc=True 且
            GP_count=1（GP物理占用，非RF占用）的slot数
        最终HR标签数 = 同理，is_hc=True 且 RF_count=1（RF物理占用）的slot数
        最终GP数 = 该(POL,POD)的GP_count总和 − 最终HC标签数
        最终RF数 = 该(POL,POD)的RF_count总和 − 最终HR标签数

        GP缺口 = max(0, original GP demand − 最终GP数)
        HC缺口 = max(0, original HC demand − 最终HC标签数)
        RF缺口 = max(0, original RF demand − 最终RF数)
        HR缺口 = max(0, original HR demand − 最终HR标签数)

    "自己这港新装的部分"通过两点保证，不会牵连进船上其他更早港口已经在船、
    但同一次投影里恰好也被算到的货：
        1. 每个(POL,POD)固定用snapshots[POL]（这个POL自己的离港快照）做
           投影，而不是这批货存活期内路过的所有后续POL快照——proj_cell_to_vessel
           对同一份未discharge的货是幂等的，snapshots[POL]已经是这批货刚装船
           那一刻的完整状态，没必要也不应该再看后面的快照。
        2. 即便snapshots[POL]里同时混有更早港口还没卸的货，投影结果的每个
           slot都带着自己的POL字段（record["POL"]在assign()时写死，不会被
           后续港口的装货动作覆盖），所以按(POL==本港, POD==目标港)双重
           过滤就能精确切出"这港新装的部分"，不会把别的港口装的同POD货
           算进来。

    每个(POL)只投影一次（不管这一港有多少个POD），投影用vessel的deepcopy
    跑（proj_cell_to_vessel会写self.cbf/_tail_source2_log/_tail_source3_log
    等副作用，不能污染调用方传入的vessel实例，做法跟scan_host_candidates
    一致）。original_cbf里出现的POL如果压根没进snapshots（比如这一港demand
    全部是0，或者这一港所有货都变成了完全没上船的尾货），视为这港最终
    HC/HR/GP/RF全部是0，对应缺口=完整的原始demand。

    返回list[dict]，每条{"POL","POD","type","count","source"}，count>0，
    source统一标注"final_vs_original"，跟build_unified_tail_list产出的
    (POL,POD,type,count,source)格式完全一致，可以直接接到
    scan_host_candidates/match_tails_to_hosts/apply_tail_placements的
    现有接口上跑。
    """
    proj_vessel = copy.deepcopy(vessel)
    proj_cache = {}
    for snap_pol in sorted(snapshots.keys()):
        proj_cache[snap_pol] = proj_vessel.proj_cell_to_vessel(
            cell_state=snapshots[snap_pol], original_cbf=original_cbf
        )

    pol_pod_pairs = sorted(
        (pol, pod)
        for pol, pod_dict in original_cbf.items()
        for pod in pod_dict.keys()
    )

    records = []
    for pol, pod in pol_pod_pairs:
        demand = original_cbf.get(pol, {}).get(pod, {})
        gp_demand = demand.get("GP", 0)
        hc_demand = demand.get("HC", 0)
        rf_demand = demand.get("RF", 0)
        hr_demand = demand.get("HR", 0)

        df = proj_cache.get(pol)
        if df is None:
            # 这个POL压根没有离港快照（这一港没有任何货真正上船），
            # 最终结果全是0，缺口=完整的原始demand。
            final_hc = final_hr = final_gp = final_rf = 0
        else:
            # 只认_BIG_BAY_OF_B0能映射到的b0侧行——proj_cell_to_vessel会把
            # 每个cell的标签原样镜像写到b1侧(bay_idx+1)，b1侧是重复的镜像
            # 数据，不是额外的物理槽位，跟capacity_hc/scan_host_candidates
            # 的统计口径保持一致（否则每个slot会被算两遍，final_gp/final_hc
            # 变成两倍，缺口算出来会全部被冲成0）。
            mask = (
                (df["POL"] == pol) & (df["POD"] == pod)
                & df["bay_idx"].isin(_BIG_BAY_OF_B0.keys())
            )
            sub = df.loc[mask]
            final_hc = int(((sub["is_hc"]) & (sub["GP_count"] == 1)).sum())
            final_hr = int(((sub["is_hc"]) & (sub["RF_count"] == 1)).sum())
            gp_total = int(sub["GP_count"].sum())
            rf_total = int(sub["RF_count"].sum())
            final_gp = gp_total - final_hc
            final_rf = rf_total - final_hr

        gaps = {
            "GP": max(0, gp_demand - final_gp),
            "HC": max(0, hc_demand - final_hc),
            "RF": max(0, rf_demand - final_rf),
            "HR": max(0, hr_demand - final_hr),
        }
        for ctype in ("GP", "HC", "RF", "HR"):
            n = gaps[ctype]
            if n:
                records.append({"POL": pol, "POD": pod, "type": ctype, "count": n, "source": "final_vs_original"})

    return records


def scan_host_candidates(vessel: Vessel, snapshots: dict) -> dict:
    """
    任务2b：遍历snapshots所有POL快照的cell，按(bay,lr,hd,POL,POD)去重收集
    host候选池，并计算每个host的静态headroom（GP/RF/HC三种名额还能再放多少）。

    与vessel.cbf_original无关——不看demand侧，只看已经装到船上的host cell
    还剩多少物理空间，供2c阶段做尾箱-host匹配用。

    hc_headroom修正（原实现的bug）：不能用_tail_source2_log/_tail_source3_log
    的(POL,POD)分组命中与否一刀切——那两份日志的key只到(POL,POD)，不含
    (bay,lr,hd)，同一个(POL,POD)完全可能占了不止一个host cell，只有其中
    真正被贴过HC标签、触发squeeze的那个cell该扣headroom，同组内其它未被
    动过的cell不该被连坐清零。改为对每张快照真正调用一次
    proj_cell_to_vessel（拿到slot级is_hc标签），按(big_bay,lr,hd,POL,POD)
    精确统计这个host实际用掉了几个HC名额（hc_used），hc_headroom=
    capacity_hc-hc_used，是host cell级的精确值。

    proj_cell_to_vessel会真实写self.cbf（受_hc_cbf_writeback_seen去重保护）、
    追加_tail_source2_log/_tail_source3_log——这些副作用不该污染调用方传入
    的vessel实例，所以每张快照都在vessel的一份deepcopy上调用，原vessel和
    传入的snapshots全程只读。用的是vessel.cbf_original（航次开始前的原始
    cbf快照，Vessel.__init__已经存了一份，内容等价于其它地方手动deepcopy
    出来的original_cbf）作为HC贴标签预算池的来源，跟proj_cell_to_vessel
    在别处的调用口径一致。每张快照只投影一次，同一张快照里的多个host共享
    这一次调用的结果，不逐host重复投影。

    只读vessel.capacity_total/capacity_rf/capacity_hc/cbf_original和
    snapshots里的cell记录，不修改传入的self.cell/self.cbf。

    返回dict，key=(bay,lr,hd,POL,POD)，value={"gp_headroom","rf_headroom",
    "hc_headroom","hd","capacity_total","capacity_rf","capacity_hc"}。

    同一host在不同快照里重复出现时，headroom必须是静态值（HC贴标签逻辑
    对同一份未discharge的货是幂等的）——这里显式比对，不一致就
    AssertionError，不静默取任意一份。
    """
    candidates = {}
    for snap_pol in sorted(snapshots.keys()):
        snap = snapshots[snap_pol]
        cell = snap["cell"]

        # 只读投影：在vessel的deepcopy上跑，避免proj_cell_to_vessel的写回
        # 副作用（self.cbf/_hc_cbf_writeback_seen/_tail_source2_log/
        # _tail_source3_log）污染调用方传入的vessel实例。
        proj_vessel = copy.deepcopy(vessel)
        slots_df = proj_vessel.proj_cell_to_vessel(cell_state=snap, original_cbf=vessel.cbf_original)

        # 按(big_bay,lr,hd,POL,POD)精确统计这张快照里每个host实际贴了几个
        # is_hc标签。只认_BIG_BAY_OF_B0能映射到的b0侧行——跟capacity_hc
        # 本身的统计口径一致（_derive_capacity_hc同样只数b0侧），b1侧是
        # proj_cell_to_vessel镜像写出来的重复标签，不能重复计数。
        hc_used_by_host = {}
        hc_rows = slots_df[(slots_df["POD"] != -1) & slots_df["is_hc"]]
        for row in hc_rows.itertuples(index=False):
            big_bay = _BIG_BAY_OF_B0.get(row.bay_idx)
            if big_bay is None:
                continue
            hc_key = (big_bay, row.lr, row.hd, row.POL, row.POD)
            hc_used_by_host[hc_key] = hc_used_by_host.get(hc_key, 0) + 1

        for bay in range(vessel.n_bay):
            for lr in range(2):
                for hd in range(2):
                    record = cell[bay, lr, hd]
                    pod = record["POD"]
                    if pod == -1:
                        continue
                    pol = record["POL"]
                    key = (bay, lr, hd, pol, pod)

                    cap_total = int(vessel.capacity_total[bay, lr, hd])
                    cap_rf = int(vessel.capacity_rf[bay, lr, hd])
                    cap_hc = int(vessel.capacity_hc[bay, lr, hd])

                    gp_headroom = cap_total - record["GP_count"] - record["RF_count"]
                    rf_headroom = cap_rf - record["RF_count"]
                    hc_used = hc_used_by_host.get(key, 0)
                    hc_headroom = cap_hc - hc_used

                    entry = {
                        "gp_headroom": gp_headroom,
                        "rf_headroom": rf_headroom,
                        "hc_headroom": hc_headroom,
                        "hd": hd,
                        "capacity_total": cap_total,
                        "capacity_rf": cap_rf,
                        "capacity_hc": cap_hc,
                    }

                    if key in candidates:
                        prev = candidates[key]
                        if prev != entry:
                            raise AssertionError(
                                f"[host候选池 静态headroom不一致] host={key} 在不同快照里算出"
                                f"不同的headroom！之前={prev}, 现在={entry}（同一host的headroom"
                                f"必须是静态值，出现分歧说明capacity数组或cell记录有问题）"
                            )
                    else:
                        candidates[key] = entry

    return candidates


def build_host_discharged_scenario():
    """
    最小合成场景，专门验证scan_host_candidates任务要求1：某个host在最终
    state里已被discharge、不在self.cell里，但在某张早期快照里存在，仍应被
    正确收进候选池。

    几何：只给1个valid cell——bay pair(2,3)、lr=0、hd=0(hold)，只1行slot
    (bay_idx=2, row=0, tier=0)。capacity_total=1、capacity_rf=0、
    capacity_hc=_stack_hc_cap(n=1,hd=0)=min(1,2)=1（bay_idx=3那一行只用于
    proj_cell_to_vessel的b1侧镜像，不计入capacity——跟build_test_scenario里
    "capacity=1"的含义一致，见Vessel.build_vessel_cell只认_BIG_BAY_OF_B0
    映射到的b0侧行）。

    cbf设计成"这个host会在最后一港被discharge掉，中途还路过一个空港口"：
        POL=0: {POD=2: GP=6}   货在港口0装船，GP=6>tail_threshold(5)能进候选集，
                                cap_total=1，assign()只填满1个槽位，剩GP=5
                                (<=5)判定这个POD"完成"(退化尾货)，港口0立即complete
        POL=1: {}               港口1空需求，纯路过，立即complete——此时host仍
                                原样留在snapshots[1]里(还没到POD=2真正卸货)
    all_ports={0,1,2}(POL键0,1 + POD值2)，port_max=2。换到POL=2时
    discharge(arriving_pod=2)才真正卸掉这个host，随后current_pol(2)>
    max(cbf.keys())=1，solve()成功返回——此时vessel.cell[0,0,0]已被清空，
    但snapshots[0]和snapshots[1]里都还留着这个host，正好验证目标场景。

    返回vessel（未跑solve()，调用方自己跑）。
    """
    rows = [
        {"bay_idx": 2, "row_idx": 0, "tier_idx": 0, "lr": 0, "hd": 0,
         "can_40ft": True, "can_20ft": False, "can_reefer": False},
        {"bay_idx": 3, "row_idx": 0, "tier_idx": 0, "lr": 0, "hd": 0,
         "can_40ft": True, "can_20ft": False, "can_reefer": False},
    ]
    full_slot_table = pd.DataFrame(rows)
    cbf = {0: {2: {"GP": 6}}, 1: {}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def build_partial_hc_squeeze_scenario():
    """
    最小合成场景，专门验证scan_host_candidates修正后的hc_headroom是host
    cell级精确值，不是被(POL,POD)分组日志一刀切：构造同一个(POL,POD)占了
    2个host cell、其中只有1个被HC squeeze动过的情形。

    几何：2个独立的deck host cell，分别在big_bay=0(bay pair(2,3))和
    big_bay=1(bay pair(4,5))，都是lr=0,hd=1，每个host只有1个"摞"
    (row_idx=0)、2个tier(4/8)——跟build_deck_squeeze_scenario同款几何，
    capacity_total=2、capacity_hc=_stack_hc_cap(n=2,hd=1)=max(2-1,0)=1，
    两个host完全对称。

    cbf只有一条：POL=0, POD=1: GP=7, HC=1。GP+HC=8>tail_threshold(5)能进
    候选集。求解过程：
        - solver先填满某一个cell(cap_total=2)：gp_used=2(全部来自GP，因为
          demand.get("GP")>=gp_used，assign()按gp优先扣减)，demand变为
          GP=5,HC=1(合计6，仍>5，未完成)
        - 再填满另一个cell(cap_total=2)：gp_used=2，demand变为GP=3,HC=1
          (合计4<=5)，判定完成，port_complete
    两个cell最终都是GP_count=2、RF_count=0(纯GP占满，HC字段在solve()内部
    从未真正被扣减)。

    HC标签怎么分：proj_cell_to_vessel按(POL,POD)共享预算池，budget取自
    original_cbf（航次开始前的HC=1），按cell的capacity_hc降序分配、
    tie时按cell_infos的构建顺序(big_bay升序)——两个cell的capacity_hc相同
    (都是1)，big_bay=0的host永远排在前面，budget=1会被它一次性拿走，
    big_bay=1的host分不到任何HC预算：
        - big_bay=0的host：拿到1个HC标签，且这一摞是满摞(occupied==n=2)
          触发deck-squeeze——releaes顶层tier(不是被标HC的那格)，squeeze后
          物理占用降到1格(POD=1,is_hc=True)，hc_used=1，
          hc_headroom=capacity_hc(1)-1=0
        - big_bay=1的host：budget耗尽，没有任何is_hc标签，两格都原样保留
          (POD=1,GP_count各1)，hc_used=0，hc_headroom=capacity_hc(1)-0=1

    两个host共享同一个(POL=0,POD=1)分组，_tail_source2_log只会记录
    big_bay=0那一次squeeze（big_bay=1没有squeeze，不会追加记录）——这正是
    要验证的场景：旧实现按(POL,POD)命中日志与否一刀切hc_headroom，会让
    big_bay=1也被错误地清零成0；新实现应该精确区分出big_bay=1的host
    hc_headroom=1（未受squeeze影响），跟big_bay=0的0不同。

    返回vessel（未跑solve()，调用方自己跑）。
    """
    rows = []
    for bay_idx in (2, 3):
        rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": 4, "lr": 0, "hd": 1,
                      "can_40ft": True, "can_20ft": False, "can_reefer": False})
        rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": 8, "lr": 0, "hd": 1,
                      "can_40ft": True, "can_20ft": False, "can_reefer": False})
    for bay_idx in (4, 5):
        rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": 4, "lr": 0, "hd": 1,
                      "can_40ft": True, "can_20ft": False, "can_reefer": False})
        rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": 8, "lr": 0, "hd": 1,
                      "can_40ft": True, "can_20ft": False, "can_reefer": False})
    full_slot_table = pd.DataFrame(rows)
    cbf = {0: {1: {"GP": 7, "HC": 1}}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_scan_host_candidates():
    """
    验证scan_host_candidates：
    1. build_test_scenario()跑一遍，打印全部host候选池内容，人工核对
       capacity/headroom是否手算一致。
    2. build_host_discharged_scenario()验证"host在最终态已discharge、不在
       self.cell里，但某张早期快照里存在，仍能被正确收进候选池"。
    3. 跨快照headroom一致性检查——scan_host_candidates内部扫描时已经对每个
       host做过这个断言，这里额外挑目标host手工跨快照复算，独立验证scan函数
       本身没有静默吃掉不一致。
    4. build_partial_hc_squeeze_scenario()验证hc_headroom修正为host cell级
       精确值：同一个(POL,POD)命中过_tail_source2_log的组内，2个host cell
       的hc_headroom应该不同（被squeeze的那个=0，未被动过的那个=capacity_hc
       全额），而不是被(POL,POD)分组日志一刀切成一样的0。
    只做核对+打印，不做host匹配（那是2c的事），不改vessel状态。
    """
    print("\n" + "=" * 60)
    print("──── scan_host_candidates 验证 ────")
    print("=" * 60)

    # ── 1. build_test_scenario：打印全部host候选池 ──
    print("\n---- 场景1: build_test_scenario ----")
    vessel1, _ = build_test_scenario()
    snapshots1 = {}
    best1 = {"assigned": -1, "vessel": None}
    success1 = solve(vessel1, is_debug=False, snapshots=snapshots1, best=best1)
    result_vessel1 = vessel1 if success1 else best1["vessel"]
    print(f"solve()完成: success={success1}")

    pool1 = scan_host_candidates(result_vessel1, snapshots1)
    print(f"host候选池共{len(pool1)}条，全部内容：")
    for key, entry in sorted(pool1.items()):
        print(f"  host={key} -> {entry}")

    if pool1:
        (bay, lr, hd, pol, pod), entry = sorted(pool1.items())[0]
        print("\n人工核对示例（挑第一条）：")
        print(f"  host=(bay={bay}, lr={lr}, hd={hd}, POL={pol}, POD={pod})")
        print(f"  capacity_total={result_vessel1.capacity_total[bay, lr, hd]}, "
              f"capacity_rf={result_vessel1.capacity_rf[bay, lr, hd]}, "
              f"capacity_hc={result_vessel1.capacity_hc[bay, lr, hd]}")
        print(f"  返回的entry={entry}")
        print("  手算gp_headroom = capacity_total - GP_count - RF_count、"
              "rf_headroom = capacity_rf - RF_count，应与上面entry一致（自行核对）")

    # ── 2. build_host_discharged_scenario：验证discharge后仍能扫到早期host ──
    # TODO(已知问题，本次尾箱统计口径修复不处理): 这个场景在_stack_hc_cap公式
    # 改成n-1后solve()返回success=False（原本假设的最小demand不再能让port
    # 顺利complete），导致snapshots2为空，下面snapshots2[0]/[1]直接KeyError。
    # 跟本文件里build_tail_container_list的3组fixture修复无关，需要单独排查
    # build_host_discharged_scenario的demand/geometry是否也要跟着新公式调整。
    print("\n---- 场景2: build_host_discharged_scenario (验证discharge后仍可扫到) ----")
    vessel2 = build_host_discharged_scenario()
    snapshots2 = {}
    best2 = {"assigned": -1, "vessel": None}
    success2 = solve(vessel2, is_debug=False, snapshots=snapshots2, best=best2)
    result_vessel2 = vessel2 if success2 else best2["vessel"]
    print(f"solve()完成: success={success2}")
    print(f"snapshots覆盖的POL: {sorted(snapshots2.keys())} (预期含0和1)")

    target_host = (0, 0, 0, 0, 2)  # (bay,lr,hd,POL,POD)
    final_cell_record = result_vessel2.cell[0, 0, 0]
    print(f"最终态 self.cell[0,0,0] = {final_cell_record} (预期POD=-1，已被discharge)")
    discharged_confirmed = final_cell_record["POD"] == -1

    pod_in_snap0 = snapshots2[0]["cell"][0, 0, 0]["POD"]
    pod_in_snap1 = snapshots2[1]["cell"][0, 0, 0]["POD"]
    print(f"snapshots[0]里该cell POD={pod_in_snap0} (预期2)")
    print(f"snapshots[1]里该cell POD={pod_in_snap1} (预期2，路过港口仍未卸货)")
    in_snapshot0 = pod_in_snap0 != -1
    in_snapshot1 = pod_in_snap1 != -1

    pool2 = scan_host_candidates(result_vessel2, snapshots2)
    print(f"scan_host_candidates返回的候选池: {pool2}")

    host_found = target_host in pool2
    print(f"目标host={target_host} 是否在候选池里: {host_found} (预期True)")

    if discharged_confirmed and in_snapshot0 and in_snapshot1 and host_found:
        print("[OK] 场景2验证通过：host在最终态已discharge(不在self.cell里)，"
              "但在早期快照(snapshots[0]/[1])里存在，仍被正确收进候选池")
    else:
        print(f"[MISMATCH] 场景2验证失败: discharged_confirmed={discharged_confirmed}, "
              f"in_snapshot0={in_snapshot0}, in_snapshot1={in_snapshot1}, host_found={host_found}")

    if host_found:
        entry = pool2[target_host]
        cap_total = int(result_vessel2.capacity_total[0, 0, 0])
        cap_rf = int(result_vessel2.capacity_rf[0, 0, 0])
        cap_hc = int(result_vessel2.capacity_hc[0, 0, 0])
        print(f"目标host的capacity: capacity_total={cap_total}, capacity_rf={cap_rf}, capacity_hc={cap_hc}")
        print(f"目标host的entry: {entry}")
        expected_gp_headroom = cap_total - 1 - 0  # GP_count=1(assign装了1个GP), RF_count=0
        expected_rf_headroom = cap_rf - 0
        expected_hc_headroom = cap_hc  # 这个场景没有触发过_tail_source2_log/_tail_source3_log
        headroom_ok = (entry["gp_headroom"] == expected_gp_headroom
                       and entry["rf_headroom"] == expected_rf_headroom
                       and entry["hc_headroom"] == expected_hc_headroom)
        print(f"手算预期: gp_headroom={expected_gp_headroom}, rf_headroom={expected_rf_headroom}, "
              f"hc_headroom={expected_hc_headroom}")
        print(f"[{'OK' if headroom_ok else 'MISMATCH'}] 目标host的headroom与手算一致")

    # ── 3. 跨快照headroom一致性检查 ──
    print("\n---- 3. 跨快照headroom一致性检查 ----")
    print("scan_host_candidates内部对每个host在多张快照里重复出现时都会比对"
          "entry是否完全相同，不一致会直接AssertionError（见函数实现）。这里"
          "额外用snapshots2独立复算一遍target_host在snapshots[0]和snapshots[1]"
          "里的headroom，验证两份手算结果彼此一致、且都等于scan_host_candidates"
          "的返回值——不是只信任函数内部断言通过了就算数。")

    def _manual_headroom(vessel, cell_record, bay, lr, hd):
        cap_total = int(vessel.capacity_total[bay, lr, hd])
        cap_rf = int(vessel.capacity_rf[bay, lr, hd])
        gp_headroom = cap_total - cell_record["GP_count"] - cell_record["RF_count"]
        rf_headroom = cap_rf - cell_record["RF_count"]
        return gp_headroom, rf_headroom

    gp0, rf0 = _manual_headroom(result_vessel2, snapshots2[0]["cell"][0, 0, 0], 0, 0, 0)
    gp1, rf1 = _manual_headroom(result_vessel2, snapshots2[1]["cell"][0, 0, 0], 0, 0, 0)
    print(f"snapshots[0]手算: gp_headroom={gp0}, rf_headroom={rf0}")
    print(f"snapshots[1]手算: gp_headroom={gp1}, rf_headroom={rf1}")

    cross_snapshot_ok = (gp0 == gp1 == pool2[target_host]["gp_headroom"]
                          and rf0 == rf1 == pool2[target_host]["rf_headroom"])
    if cross_snapshot_ok:
        print("[OK] 跨快照headroom完全一致，且与scan_host_candidates返回值一致")
    else:
        print(f"[MISMATCH] 跨快照headroom不一致: snapshot0=({gp0},{rf0}), "
              f"snapshot1=({gp1},{rf1}), 函数返回={pool2[target_host]}")

    # ── 4. build_partial_hc_squeeze_scenario：验证hc_headroom是host cell级精确值 ──
    print("\n---- 场景4: build_partial_hc_squeeze_scenario "
          "(验证同一(POL,POD)组内多个host的hc_headroom不被一刀切) ----")
    vessel4 = build_partial_hc_squeeze_scenario()
    snapshots4 = {}
    best4 = {"assigned": -1, "vessel": None}
    success4 = solve(vessel4, is_debug=False, snapshots=snapshots4, best=best4)
    result_vessel4 = vessel4 if success4 else best4["vessel"]
    print(f"solve()完成: success={success4}")

    host_bigbay0 = (0, 0, 1, 0, 1)  # (bay,lr,hd,POL,POD)：big_bay=0的deck host
    host_bigbay1 = (1, 0, 1, 0, 1)  # big_bay=1的deck host，同一个(POL,POD)组

    pool4 = scan_host_candidates(result_vessel4, snapshots4)
    print(f"host候选池: {pool4}")

    both_found = host_bigbay0 in pool4 and host_bigbay1 in pool4
    print(f"两个host是否都在候选池里: {both_found} (预期True)")

    if both_found:
        entry0 = pool4[host_bigbay0]
        entry1 = pool4[host_bigbay1]
        print(f"big_bay=0 host entry: {entry0}")
        print(f"big_bay=1 host entry: {entry1}")

        # 独立复算：在这份vessel4的deepcopy上重跑一次proj_cell_to_vessel，
        # 直接数is_hc标签，不复用scan_host_candidates内部的计算路径，
        # 避免"函数自己骗自己"。
        proj_check = copy.deepcopy(result_vessel4)
        snap0_pol = sorted(snapshots4.keys())[0]
        df_check = proj_check.proj_cell_to_vessel(
            cell_state=snapshots4[snap0_pol], original_cbf=result_vessel4.cbf_original)
        hc_count_bigbay0 = int((
            (df_check["bay_idx"] == 2) & df_check["is_hc"] & (df_check["POD"] != -1)
        ).sum())
        hc_count_bigbay1 = int((
            (df_check["bay_idx"] == 4) & df_check["is_hc"] & (df_check["POD"] != -1)
        ).sum())
        print(f"独立复算: big_bay=0(bay_idx=2侧) is_hc占用数={hc_count_bigbay0} (预期1), "
              f"big_bay=1(bay_idx=4侧) is_hc占用数={hc_count_bigbay1} (预期0)")

        source2_log = result_vessel4._tail_source2_log if success4 else []
        print(f"这次真实solve()跑出的_tail_source2_log(诊断,非本次投影产生): "
              f"{getattr(result_vessel4, '_tail_source2_log', 'N/A')}")

        expected_entry0 = {"gp_headroom": 0, "rf_headroom": 0, "hc_headroom": 0,
                            "hd": 1, "capacity_total": 2, "capacity_rf": 0, "capacity_hc": 1}
        expected_entry1 = {"gp_headroom": 0, "rf_headroom": 0, "hc_headroom": 1,
                            "hd": 1, "capacity_total": 2, "capacity_rf": 0, "capacity_hc": 1}
        print(f"手动预期: big_bay=0 entry={expected_entry0}")
        print(f"手动预期: big_bay=1 entry={expected_entry1}")

        entries_ok = (entry0 == expected_entry0 and entry1 == expected_entry1)
        manual_ok = (hc_count_bigbay0 == 1 and hc_count_bigbay1 == 0)
        distinct_ok = entry0["hc_headroom"] != entry1["hc_headroom"]

        if entries_ok and manual_ok and distinct_ok:
            print("[OK] 场景4验证通过：同一(POL,POD)组内2个host cell的hc_headroom"
                  "各自独立(0 vs 1)，不是被分组日志一刀切成一样的值")
        else:
            print(f"[MISMATCH] 场景4验证失败: entries_ok={entries_ok}, "
                  f"manual_ok={manual_ok}, distinct_ok={distinct_ok}")
    else:
        print(f"[MISMATCH] 场景4验证失败：预期的2个host不全在候选池里，pool4={pool4}")

    print("\nvessel/vessel4本身在scan_host_candidates调用前后是否被污染的检查："
          f"result_vessel4.cbf={result_vessel4.cbf}, "
          f"_hc_cbf_writeback_seen={result_vessel4._hc_cbf_writeback_seen}, "
          f"_tail_source2_log长度={len(result_vessel4._tail_source2_log)}"
          "（scan_host_candidates调用只应读这些值，不应该让它们在调用后发生"
          "跟'真实solve()流程本身'无关的额外变化——scan_host_candidates内部"
          "全程操作的是deepcopy，这里能看到的变化只可能来自solve()本身）")


def _dist(pol_from: int, pod: int, port_min: int, n_ports: int) -> int:
    """跟Vessel.rel_rank同一个公式体，只是把self.current_pol换成任意传入的
    pol_from——"从pol_from出发，绕圈到达pod要经过多少港"的相对距离，
    允许绕圈（环线航次里pod数值可能比pol_from还小）。不重新定义一套数值，
    port_min/n_ports必须从调用方的vessel上取，跟Vessel.rel_rank口径一致。
    """
    c = (pol_from - port_min) % n_ports
    p = (pod - port_min) % n_ports
    return (p - c) if p >= c else (p - c + n_ports)


def match_tails_to_hosts(unified_tail_list, host_pool, port_min: int, n_ports: int):
    """
    任务2c：把unified_tail_list里的尾箱记录逐条匹配进host_pool，产出安置台账。

    按交接摘要"已确定的设计原则"逐条实现：
    - 只匹配同POD的host（尾箱POD必须与host POD完全一致）。
    - 用_dist()（跟Vessel.rel_rank同一套绕圈感知的相对距离公式）比较host跟
      尾箱谁离POD更近：要求_dist(host.POL, POD) >= _dist(尾箱.POL, POD)
      （允许相等）。这条距离规则完全替代了原先"host.POL <= 尾箱.POL"的裸
      数值比较——host.POL<=尾箱.POL只在航线不绕圈时等价于"host比尾箱先诞生"，
      在真实环线航次里会既漏判（host自己绕圈、host.POL数值上比尾箱大，但
      host其实早就存在）又错判（host.POL数值上更小、但它离POD更近，物理上
      会先于尾箱被discharge、届时尾箱根本借不到它）。距离越大代表离真正
      discharge越远（还能撑得住更久），所以host的距离必须>=尾箱自己的距离，
      host才"活得够久"、扛得到尾箱登船那一刻还没被卸货。
      任务3(apply_tail_placements)里host_life_end要处理的是"这个host在实际
      快照序列里哪一港真的discharge、注入终点该摆到哪"，是另一个独立问题，
      这里的距离规则只负责"这次匹配在物理上站不站得住脚"，两者不合并。
    - GP/RF类型尾箱只消耗对应的gp_headroom/rf_headroom；HC/HR类型尾箱
      需要同时满足hc_headroom>0且对应gp_headroom/rf_headroom>0（HC看
      gp_headroom，HR看rf_headroom），取min作为这次能塞的量——HC/HR本身
      也要占用一个物理槽位，不能只看hc_headroom而忽视host cell根本没有
      物理空位。
    - 同一host的headroom被消耗后跨多条尾箱记录累减：在host_pool的本地
      可变副本（state）上原地扣减，同一个host_key在后续尾箱记录里读到
      的是上一条记录扣减后的余量，不是host_pool的原始值。
    - host候选排序：同POD/POL条件满足的host里，优先选headroom更小的
      （先塞满小空位，把大空位留给后续可能出现的大尾箱）。排序键与该
      尾箱类型实际消耗的额度口径一致（GP用gp_headroom，RF用rf_headroom，
      HC用min(hc_headroom,gp_headroom)，HR用min(hc_headroom,rf_headroom)）。

    只产出台账，不做二次投影：不改vessel.cell/vessel.cbf，host_pool本身
    也不被就地修改（本地deepcopy一份headroom状态操作）。

    返回(placements, unplaced)：
        placements: list[dict]，每条{"POL","POD","type","count","source",
            "host_bay","host_lr","host_hd","host_POL"}
        unplaced: list[dict]，匹配不完的尾箱残量，格式同unified_tail_list
            的条目（count为未安置的剩余量）。
    """
    state = {host_key: dict(entry) for host_key, entry in host_pool.items()}

    def _avail(host_state, ctype):
        if ctype == "GP":
            return host_state["gp_headroom"]
        if ctype == "RF":
            return host_state["rf_headroom"]
        if ctype == "HC":
            return min(host_state["hc_headroom"], host_state["gp_headroom"])
        if ctype == "HR":
            return min(host_state["hc_headroom"], host_state["rf_headroom"])
        raise ValueError(f"未知尾箱类型: {ctype}")

    def _deduct(host_state, ctype, take):
        if ctype == "GP":
            host_state["gp_headroom"] -= take
        elif ctype == "RF":
            host_state["rf_headroom"] -= take
        elif ctype == "HC":
            host_state["hc_headroom"] -= take
            host_state["gp_headroom"] -= take
        elif ctype == "HR":
            host_state["hc_headroom"] -= take
            host_state["rf_headroom"] -= take

    placements = []
    unplaced = []

    for tail in unified_tail_list:
        pol, pod, ctype = tail["POL"], tail["POD"], tail["type"]
        remaining = tail["count"]
        tail_dist = _dist(pol, pod, port_min, n_ports)

        eligible = [
            (host_key, host_state) for host_key, host_state in state.items()
            if host_key[4] == pod and _dist(host_key[3], pod, port_min, n_ports) >= tail_dist
        ]
        eligible.sort(key=lambda kv: _avail(kv[1], ctype))

        for host_key, host_state in eligible:
            if remaining <= 0:
                break
            avail = _avail(host_state, ctype)
            if avail <= 0:
                continue
            take = min(avail, remaining)
            _deduct(host_state, ctype, take)
            placements.append({
                "POL": pol, "POD": pod, "type": ctype, "count": take,
                "source": tail.get("source"),
                "host_bay": host_key[0], "host_lr": host_key[1], "host_hd": host_key[2],
                "host_POL": host_key[3],
            })
            remaining -= take

        if remaining > 0:
            leftover = dict(tail)
            leftover["count"] = remaining
            unplaced.append(leftover)

    return placements, unplaced


def verify_match_tails_to_hosts():
    """5个最小合成场景验证match_tails_to_hosts，每个场景各自独立构造
    tail_list/host_pool（不复用之前的求解场景），跑完打印placements+unplaced，
    人工核对数字，并assert：
    - 所有placements的count之和 + 所有unplaced的count之和 == 输入
      unified_tail_list的count之和（不多不少）。
    - 每个host被消耗的headroom总量不超过它的初始headroom（不能超装），
      通过比对消耗前后的state验证。
    """
    print("\n" + "=" * 60)
    print("──── match_tails_to_hosts 验证 ────")
    print("=" * 60)

    def _check(label, tail_list, host_pool, port_min=0, n_ports=10):
        placements, unplaced = match_tails_to_hosts(tail_list, host_pool, port_min, n_ports)
        print(f"\n---- 场景: {label} ----")
        print(f"输入 unified_tail_list: {tail_list}")
        print(f"输入 host_pool: {host_pool}")
        print(f"placements: {placements}")
        print(f"unplaced: {unplaced}")

        input_total = sum(t["count"] for t in tail_list)
        placed_total = sum(p["count"] for p in placements)
        unplaced_total = sum(u["count"] for u in unplaced)
        print(f"输入总数={input_total}, placements总数={placed_total}, unplaced总数={unplaced_total}")
        assert placed_total + unplaced_total == input_total, \
            f"[{label}] 箱数对不上账: {placed_total}+{unplaced_total} != {input_total}"

        # 每个host消耗量不超过初始headroom：按host_key+字段重算消耗量，
        # 跟host_pool原始值比对（host_pool本身不应被就地修改）。
        consumed = {}
        for p in placements:
            hk = (p["host_bay"], p["host_lr"], p["host_hd"], p["host_POL"], p["POD"])
            consumed.setdefault(hk, {"GP": 0, "RF": 0, "HC": 0, "HR": 0})
            consumed[hk][p["type"]] += p["count"]

        for hk, used in consumed.items():
            entry = host_pool[hk]
            gp_used = used["GP"] + used["HC"]
            rf_used = used["RF"] + used["HR"]
            hc_used = used["HC"] + used["HR"]
            assert gp_used <= entry["gp_headroom"], f"[{label}] host={hk} gp超装: {gp_used} > {entry['gp_headroom']}"
            assert rf_used <= entry["rf_headroom"], f"[{label}] host={hk} rf超装: {rf_used} > {entry['rf_headroom']}"
            assert hc_used <= entry["hc_headroom"], f"[{label}] host={hk} hc超装: {hc_used} > {entry['hc_headroom']}"

        print(f"[OK] {label} 对账通过，且未发现超装")
        return placements, unplaced

    # 1. 完美匹配
    _check(
        "1-完美匹配",
        [{"POL": 0, "POD": 1, "type": "GP", "count": 3, "source": 1}],
        {(0, 0, 0, 0, 1): {"gp_headroom": 3, "rf_headroom": 0, "hc_headroom": 0,
                            "hd": 0, "capacity_total": 3, "capacity_rf": 0, "capacity_hc": 0}},
    )

    # 2. headroom不够，多条尾箱记录分摊同一个host，最后一条部分进unplaced
    _check(
        "2-分摊同一host_部分unplaced",
        [
            {"POL": 0, "POD": 2, "type": "GP", "count": 3, "source": 1},
            {"POL": 0, "POD": 2, "type": "GP", "count": 4, "source": 2},
        ],
        {(0, 0, 0, 0, 2): {"gp_headroom": 5, "rf_headroom": 0, "hc_headroom": 0,
                            "hd": 0, "capacity_total": 5, "capacity_rf": 0, "capacity_hc": 0}},
    )

    # 3. host_POL在裸数值上晚于尾箱POL——在distance规则替换裸数值比较之前，
    # 这种情形恒被拒绝、整条进unplaced；换成_dist()之后，是否匹配取决于
    # port_min/n_ports（环线绕圈可能让数值更大的host.POL其实离POD更远、
    # 依然合法）。这里port_min=0,n_ports=10：dist(host.POL=5,POD=2)=7，
    # dist(tail.POL=0,POD=2)=2，7>=2，现在反而会匹配成功——这不是回归，
    # 是"距离规则完全替代裸数值比较"这个设计变化的直接后果，旧的"晚于就拒绝"
    # 结论不再普遍成立，只在不绕圈(或未绕圈到覆盖这对POL的程度)时才成立。
    _check(
        "3-host_POL数值晚于尾箱POL(distance规则下不再必然拒绝)",
        [{"POL": 0, "POD": 2, "type": "GP", "count": 4, "source": 1}],
        {(0, 0, 0, 5, 2): {"gp_headroom": 10, "rf_headroom": 0, "hc_headroom": 0,
                            "hd": 0, "capacity_total": 10, "capacity_rf": 0, "capacity_hc": 0}},
    )

    # 4. RF类型尾箱只匹配rf_headroom>0的host，不能被GP-only host吃掉
    _check(
        "4-RF只匹配rf_headroom_host",
        [{"POL": 0, "POD": 3, "type": "RF", "count": 2, "source": 1}],
        {
            (0, 0, 0, 0, 3): {"gp_headroom": 5, "rf_headroom": 0, "hc_headroom": 0,
                              "hd": 0, "capacity_total": 5, "capacity_rf": 0, "capacity_hc": 0},
            (1, 0, 0, 0, 3): {"gp_headroom": 2, "rf_headroom": 2, "hc_headroom": 0,
                              "hd": 0, "capacity_total": 2, "capacity_rf": 2, "capacity_hc": 0},
        },
    )

    # 5. HC类型：gp_headroom=0但hc_headroom>0的host应被跳过（物理槽位满了）
    _check(
        "5-HC跳过物理槽位已满的host",
        [{"POL": 0, "POD": 4, "type": "HC", "count": 2, "source": 1}],
        {
            (0, 0, 0, 0, 4): {"gp_headroom": 0, "rf_headroom": 0, "hc_headroom": 3,
                              "hd": 1, "capacity_total": 3, "capacity_rf": 0, "capacity_hc": 3},
            (1, 0, 0, 0, 4): {"gp_headroom": 2, "rf_headroom": 0, "hc_headroom": 2,
                              "hd": 1, "capacity_total": 2, "capacity_rf": 0, "capacity_hc": 2},
        },
    )

    # 6. 手算例子：tail POL=2 -> POD=5 (port_min=0, n_ports=7)。
    # dist(tail.POL=2, POD=5) = 5-2 = 3。候选host的POL分别是6,1,2,3,4，
    # 每个host只给headroom=1（跟count错开，方便按"有没有出现在placements"
    # 直接判定这个host有没有被匹配到，不需要另外核对余量）：
    #   dist(6,5)=5-6+7=6 >=3 -> 应匹配
    #   dist(1,5)=5-1=4   >=3 -> 应匹配
    #   dist(2,5)=5-2=3   >=3 -> 应匹配（等于也算，边界情形）
    #   dist(3,5)=5-3=2   <3  -> 应拒绝
    #   dist(4,5)=5-4=1   <3  -> 应拒绝
    print("\n---- 手算场景6: tail POL=2 -> POD=5 (port_min=0, n_ports=7) ----")
    tail_list_6 = [{"POL": 2, "POD": 5, "type": "GP", "count": 10, "source": 1}]
    host_pool_6 = {
        (0, 0, 0, host_pol, 5): {"gp_headroom": 1, "rf_headroom": 0, "hc_headroom": 0,
                                  "hd": 0, "capacity_total": 1, "capacity_rf": 0, "capacity_hc": 0}
        for host_pol in (6, 1, 2, 3, 4)
    }
    placements_6, unplaced_6 = _check(
        "6-手算distance例子(2→5)", tail_list_6, host_pool_6, port_min=0, n_ports=7)
    matched_pols_6 = {p["host_POL"] for p in placements_6}
    expected_matched_6 = {6, 1, 2}
    print(f"实际匹配到的host.POL集合={matched_pols_6}, 手算预期={expected_matched_6}")
    assert matched_pols_6 == expected_matched_6, \
        f"[手算场景6] 匹配到的host.POL集合跟手算不一致: {matched_pols_6} != {expected_matched_6}"
    print("[OK] 手算场景6：匹配结果与手算完全一致")

    # 7. 手算例子：tail POL=5 -> POD=2 (port_min=0, n_ports=7)。
    # dist(tail.POL=5, POD=2) = 2-5+7 = 4。候选host的POL分别是3,4,5,6,0,1：
    #   dist(3,2)=2-3+7=6 >=4 -> 应匹配
    #   dist(4,2)=2-4+7=5 >=4 -> 应匹配
    #   dist(5,2)=2-5+7=4 >=4 -> 应匹配（边界相等）
    #   dist(6,2)=2-6+7=3 <4  -> 应拒绝
    #   dist(0,2)=2-0=2   <4  -> 应拒绝
    #   dist(1,2)=2-1=1   <4  -> 应拒绝
    print("\n---- 手算场景7: tail POL=5 -> POD=2 (port_min=0, n_ports=7) ----")
    tail_list_7 = [{"POL": 5, "POD": 2, "type": "GP", "count": 10, "source": 1}]
    host_pool_7 = {
        (0, 0, 0, host_pol, 2): {"gp_headroom": 1, "rf_headroom": 0, "hc_headroom": 0,
                                  "hd": 0, "capacity_total": 1, "capacity_rf": 0, "capacity_hc": 0}
        for host_pol in (3, 4, 5, 6, 0, 1)
    }
    placements_7, unplaced_7 = _check(
        "7-手算distance例子(5→2)", tail_list_7, host_pool_7, port_min=0, n_ports=7)
    matched_pols_7 = {p["host_POL"] for p in placements_7}
    expected_matched_7 = {3, 4, 5}
    print(f"实际匹配到的host.POL集合={matched_pols_7}, 手算预期={expected_matched_7}")
    assert matched_pols_7 == expected_matched_7, \
        f"[手算场景7] 匹配到的host.POL集合跟手算不一致: {matched_pols_7} != {expected_matched_7}"
    print("[OK] 手算场景7：匹配结果与手算完全一致")

    print("\n[OK] 全部7个场景验证通过")


def _tail_resource_kind(ctype: str) -> str:
    """GP/HC占用同一种物理槽位资源(gp_headroom口径)，RF/HR占用另一种
    (rf_headroom口径，限can_reefer槽位)。"""
    if ctype in ("GP", "HC"):
        return "GP"
    if ctype in ("RF", "HR"):
        return "RF"
    raise ValueError(f"未知尾箱类型: {ctype}")


def _host_slot_mask(df: pd.DataFrame, bay_idx: int, lr: int, hd: int, resource: str) -> pd.Series:
    """host cell在b0侧对应的槽位行mask，resource='RF'时只认can_reefer槽位
    （RF/HR类型只能占用具备reefer能力的物理槽位）。"""
    mask = (df["bay_idx"] == bay_idx) & (df["lr"] == lr) & (df["hd"] == hd) & df["can_40ft"]
    if resource == "RF":
        mask = mask & df["can_reefer"]
    return mask


def _select_empty_host_slots(df: pd.DataFrame, bay_idx: int, lr: int, hd: int, resource: str) -> list:
    """在host cell里，按proj_cell_to_vessel同款槽位选择顺序（tier_idx升序、
    从中间到两边的row_idx顺序），挑出当前POD==-1的空槽位index，供尾箱摊入。
    不重新发明摆放顺序，直接复刻proj_cell_to_vessel里那段排序逻辑。"""
    mask = _host_slot_mask(df, bay_idx, lr, hd, resource)
    idx_list = list(df.index[mask])
    row_reverse = (lr == 0)
    idx_list.sort(key=lambda idx: (
        df.at[idx, "tier_idx"],
        -df.at[idx, "row_idx"] if row_reverse else df.at[idx, "row_idx"],
    ))
    return [idx for idx in idx_list if df.at[idx, "POD"] == -1]


def _inject_tail_into_snapshot(df: pd.DataFrame, big_bay: int, lr: int, hd: int,
                                tail_pol: int, pod: int, ctype: str, count: int, host_key) -> None:
    """把这条尾箱记录摊进df（version2的某一张POL快照）里，占用host cell当前
    空着的槽位，b0/b1两侧同步写回，跟proj_cell_to_vessel的镜像写回口径一致。

    写回的POL标记用尾箱记录自己的POL（tail_pol，这批箱子实际的登船港），
    不是host的POL——host只是"借用"的那个物理cell，尾箱本身是另一趟单独的
    booking，物理上是从tail_pol这一港才装船的，标记成host_POL会让这批箱子
    看起来在host诞生的那一港就已经在船上，跟事实不符。

    只在这里做"物理槽位是否真的够"的最后一道防线校验——真正的静态headroom
    对账在apply_tail_placements里injection之前就做过一次，这里如果还是不够，
    说明状态在两次检查之间被意外改变了，直接报错而不是摊出界。
    """
    b0, b1 = STSE_BAY_PAIRS[big_bay]
    resource = _tail_resource_kind(ctype)
    empty_idx = _select_empty_host_slots(df, b0, lr, hd, resource)
    if len(empty_idx) < count:
        raise AssertionError(
            f"[apply_tail_placements] host={host_key} 实际空槽位({len(empty_idx)}, "
            f"资源类型={resource})不足以安置{count}个{ctype}尾箱——注入过程中状态被意外改变了"
        )
    target_idx = empty_idx[:count]
    for idx in target_idx:
        row_idx = df.at[idx, "row_idx"]
        tier_idx = df.at[idx, "tier_idx"]

        df.at[idx, "POL"] = tail_pol
        df.at[idx, "POD"] = pod
        if resource == "GP":
            df.at[idx, "GP_count"] = 1
        else:
            df.at[idx, "RF_count"] = 1
        if ctype in ("HC", "HR"):
            df.at[idx, "is_hc"] = True

        b1_mask = (df["bay_idx"] == b1) & (df["row_idx"] == row_idx) & (df["tier_idx"] == tier_idx)
        for b1_idx in df.index[b1_mask]:
            df.at[b1_idx, "POL"] = tail_pol
            df.at[b1_idx, "POD"] = pod
            if resource == "GP":
                df.at[b1_idx, "GP_count"] = 1
            else:
                df.at[b1_idx, "RF_count"] = 1
            if ctype in ("HC", "HR"):
                df.at[b1_idx, "is_hc"] = True


def apply_tail_placements(vessel: Vessel, snapshots: dict, original_cbf: dict, placements: list):
    """
    任务3：把match_tails_to_hosts产出的placements二次投影进slot级DataFrame，
    产出版本1（原始投影，未受尾箱影响）和版本2（叠加尾箱后的投影），供人工/
    自动核对尾箱摆放是否合理、跨港是否一致。

    不改vessel.cell/vessel.cbf/snapshots本身：版本1/版本2都是各POL快照
    proj_cell_to_vessel输出的DataFrame的独立副本。

    对每条placement记录，存活区间是[effective_start, effective_end)——不是
    单纯的[tail.POL, POD)裸数值区间。区间起点effective_start=
    max(host.POL, tail.POL)（原则上恒等于tail.POL，因为match_tails_to_hosts
    已经保证host.POL<=tail.POL，这里仍显式取max是为了不偷偷依赖那个前提）：
    host.POL只是这个物理cell本身第一次被装货的港口，跟这条尾箱记录自己的
    POL(它真正登船的港口)是两回事，尾箱在自己的POL之前根本没上船，不能出现
    在更早港口的departure快照里——覆盖的POL快照范围必须以尾箱自己的POL为起点。

    区间终点effective_end是"绕圈感知"的：这条船的POL推进在真实航次里严格
    从port_min升到port_max、从不回绕（Vessel.advance_pol()只是
    current_pol+=1），但POD是"相对某个POL的将来某一港"，可能因为航线本身是
    环线，数值上比它自己的POL还小（例如host.POL=3却POD=2）——这种情况下这批
    货真正被discharge的那一港落在本次建模航次范围之外（超过port_max才会
    真正卸货，根本不会出现在snapshots里），如果还照字面数值算[POL,POD)，
    会因为POD数值<=起点而得到一个空区间，导致注入被静默跳过，但headroom
    计数器仍会正常累加——这正是真实数据里62.5%尾箱记录(POL>=POD)会触发
    headroom前置校验误报AssertionError的根因。修正为：
        effective_end = POD if POD > effective_start else (vessel.port_max + 1)
    即POD数值大于起点时按原样处理（正常区间，未绕圈）；POD<=起点时视为
    "绕圈，这一港在本次建模航次里追不上"，改为一直存活到最后一张快照
    （vessel.port_max，区间右开，所以传port_max+1）。

    在这个区间覆盖的每一张POL快照上，把这条尾箱摊进host对应的物理槽位
    （摊入顺序复用_select_empty_host_slots，就是proj_cell_to_vessel本身的
    槽位选择顺序，不重新发明）。

    注入前的静态headroom前置校验：用scan_host_candidates重新算一遍host候选池
    （2c阶段host_pool的headroom就是这么算出来的，这里没有单独的口径），
    在每个host第一次被注入前，比对它在尾箱自己POL那张快照里的实际物理空槽位数
    跟host_pool记录的headroom是否一致（累减掉本次调用里已经安置过的量）。
    对不上就说明状态在2b算完之后被什么东西改变了，直接AssertionError，
    不静默注入导致箱子摆进本来有货的slot。

    同一个host被多条placement共享时，按各自的tail.POL升序处理——因为
    Interval(tail.POL)=[effective_start,effective_end)在同一个host(同一个POD)
    下随tail.POL增大而单调收缩（不管POD是否绕圈：未绕圈时effective_end=POD
    固定，起点变大区间变小；绕圈时effective_end=port_max+1固定，同理），
    升序处理能保证轮到某条记录做headroom前置校验时，所有tail.POL更早（区间
    更大、必然覆盖当前这张检查快照）的记录都已经真实注入过了，检查用的
    "实际空槽位数"才跟累计扣减的"预期剩余"对得上，不会因为乱序而误报。

    返回(version1_dict, version2_dict)，key都是POL，value是slot级DataFrame。
    """
    host_pool = scan_host_candidates(vessel, snapshots)

    version1_dict = {}
    version2_dict = {}
    for pol in sorted(snapshots.keys()):
        df = vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
        version1_dict[pol] = df
        version2_dict[pol] = df.copy(deep=True)

    consumed_by_host = {}  # host_key -> {"GP": 已安置量, "RF": 已安置量}

    for placement in sorted(placements, key=lambda p: p["POL"]):
        host_key = (placement["host_bay"], placement["host_lr"], placement["host_hd"],
                    placement["host_POL"], placement["POD"])
        host_entry = host_pool.get(host_key)
        if host_entry is None:
            raise AssertionError(
                f"[apply_tail_placements] placement引用的host={host_key} 不在"
                f"scan_host_candidates重算出的host候选池里，2b/2c之间的状态已经不一致了"
            )

        ctype = placement["type"]
        count = placement["count"]
        tail_pol = placement["POL"]
        pod = placement["POD"]
        resource = _tail_resource_kind(ctype)

        used = consumed_by_host.setdefault(host_key, {"GP": 0, "RF": 0})

        # 绕圈感知的存活区间：起点恒等于tail_pol（match_tails_to_hosts已保证
        # host.POL<=tail.POL，这里显式取max不偷偷依赖那个前提）；终点在
        # POD>起点时按原样处理，POD<=起点（绕圈，这一港在本次建模航次里追不
        # 上）时改为一直存活到最后一张快照(vessel.port_max)。
        effective_start = max(placement["host_POL"], tail_pol)
        effective_end = pod if pod > effective_start else (vessel.port_max + 1)

        # 前置校验：用effective_start那张快照(必然在snapshots范围内)上的
        # 实际空槽位数，比对host_pool记录的静态headroom(扣掉这次调用里已经
        # 安置过的量)是否吻合。
        check_df = version2_dict[effective_start]
        b0 = STSE_BAY_PAIRS[placement["host_bay"]][0]
        actual_empty = len(_select_empty_host_slots(
            check_df, b0, placement["host_lr"], placement["host_hd"], resource))
        static_headroom = host_entry["rf_headroom"] if resource == "RF" else host_entry["gp_headroom"]
        expected_remaining = static_headroom - used[resource]
        if actual_empty != expected_remaining:
            raise AssertionError(
                f"[apply_tail_placements] host={host_key} 资源类型={resource} 的实际空槽位"
                f"({actual_empty})跟host_pool记录的headroom推算值({expected_remaining}, "
                f"静态headroom={static_headroom}，本次调用已安置={used[resource]})对不上，"
                f"说明2b算完之后状态被意外改变了，拒绝静默注入"
            )

        affected_pols = [p for p in sorted(snapshots.keys()) if effective_start <= p < effective_end]
        for pol in affected_pols:
            _inject_tail_into_snapshot(
                version2_dict[pol], placement["host_bay"], placement["host_lr"], placement["host_hd"],
                tail_pol, pod, ctype, count, host_key,
            )

        used[resource] += count

    return version1_dict, version2_dict


def verify_cross_port_consistency(version2_dict: dict, placements: list, port_max: int) -> bool:
    """
    跨港一致性回归检查（黑盒，不依赖apply_tail_placements内部实现）：
    对placements每条记录，确认其host坐标(big_bay, lr, hd)：
    - 在[effective_start, effective_end)覆盖的每一张POL快照的version2
      DataFrame里，都能查到>=count条(bay_idx==b0, lr, hd, POL==tail.POL,
      POD==POD)的匹配记录；
    - 在这个区间之外的POL快照里，这个host坐标不应该出现任何
      (POL==tail.POL, POD==POD)的匹配记录（防止箱子凭空出现/消失）。

    区间起点用max(host.POL, tail记录自己的POL)：这批箱子在自己的POL之前
    根本没上船，不能出现在更早港口的departure快照里；host.POL只是host这个
    物理cell自己诞生的港口，跟尾箱是两个独立的量，不能替代——两者取max只是
    不偷偷依赖"host.POL<=tail.POL"这个由match_tails_to_hosts保证的前提。

    区间终点是"绕圈感知"的，跟apply_tail_placements用的是同一个判据（但
    port_max由调用方显式传入，不读取vessel/host内部状态，仍然是黑盒重新
    推导，不共享apply_tail_placements内部计算出的effective_end）：这条船的
    POL推进严格从port_min升到port_max、从不回绕，但POD代表"相对某个POL的
    将来某一港"，可能因为航线本身是环线而数值上比起点还小——这种情况下
    真正的discharge发生在本次建模航次范围之外，判定为一直存活到最后一张
    快照(port_max)。POD>起点时按原样处理（未绕圈，区间就是[起点,POD)）。
    这里跟apply_tail_placements各自独立按同一份placements重新推导判据，
    不共享内部状态，避免两边用同一个错误前提互相掩盖问题——如果只改了
    apply_tail_placements而不同步这里，就会退回到"验证函数和被验证函数
    共享同一个（错误）前提"的老问题，绕圈场景会被误判为一致。

    不一致的记录逐条打印细节（不只给pass/fail）。返回是否全部一致。
    """
    all_ok = True
    for placement in placements:
        big_bay = placement["host_bay"]
        lr = placement["host_lr"]
        hd = placement["host_hd"]
        host_pol = placement["host_POL"]
        tail_pol = placement["POL"]
        pod = placement["POD"]
        count = placement["count"]
        b0 = STSE_BAY_PAIRS[big_bay][0]

        effective_start = max(host_pol, tail_pol)
        effective_end = pod if pod > effective_start else (port_max + 1)

        for pol in sorted(version2_dict.keys()):
            df = version2_dict[pol]
            mask = (
                (df["bay_idx"] == b0) & (df["lr"] == lr) & (df["hd"] == hd)
                & (df["POL"] == tail_pol) & (df["POD"] == pod)
            )
            matched = int(mask.sum())
            in_interval = effective_start <= pol < effective_end

            if in_interval and matched < count:
                all_ok = False
                print(f"[MISMATCH-区间内缺失] placement={placement}: POL快照={pol} 在存活区间"
                      f"[{effective_start},{effective_end})内，只找到{matched}条匹配记录(预期>={count})")
            if not in_interval and matched > 0:
                all_ok = False
                print(f"[MISMATCH-区间外出现] placement={placement}: POL快照={pol} 在存活区间"
                      f"[{effective_start},{effective_end})之外，却出现了{matched}条匹配记录(预期0)")

    if all_ok:
        print("[OK] 跨港一致性回归检查通过：所有placements在存活区间内外都符合预期")
    return all_ok


def _slots_bay_totals(df: pd.DataFrame, n_bay: int, filter_col: str, filter_val: int) -> np.ndarray:
    """按big_bay汇总slot级DataFrame里满足(filter_col==filter_val)的GP_count+
    RF_count，只读b0侧行——跟Vessel.build_vessel_cell/capacity_hc的统计口径
    一致，b1侧是镜像，不重复计数。

    直接在slot级别按POL/POD过滤求和，不经过cell级(n_bay,2,2)单record重建：
    Vessel.cell的"一个cell只认一个POL"是求解阶段的真实不变量（assign()整格
    赋值），但apply_tail_placements之后的version2里，同一个host物理cell完全
    可能同时装着host自己的原始货(host.POL)和不同POL的尾箱(tail.POL)——这是
    尾箱安置故意引入的、模型里此前不会出现的情形，重建单POL的cell record
    在这种场景下要么丢箱、要么断言失败，所以CI对比改成直接在slot粒度上按
    POL/POD过滤求和，天然兼容一个物理cell里混着多个POL的情况。
    """
    totals = np.zeros(n_bay, dtype=int)
    for big_bay in range(n_bay):
        b0 = STSE_BAY_PAIRS[big_bay][0]
        rows = df[(df["bay_idx"] == b0) & df["can_40ft"] & (df["POD"] != -1) & (df[filter_col] == filter_val)]
        totals[big_bay] = int((rows["GP_count"] + rows["RF_count"]).sum())
    return totals


def _ci_from_slot_version_dict(version_dict: dict, n_bay: int) -> list:
    """跟utils.evaluate._port_bay_totals+evaluate_crane_intensity同一套定义
    （discharge_tally=上一张快照里POD==本港的箱量，loading_tally=本快照里
    POL==本港的箱量，CI=总量/最挤相邻bay对之和），只是直接在slot级
    DataFrame上算，不经过cell级单POL假设（见_slots_bay_totals）。

    返回list[dict]，跟evaluate_crane_intensity()的返回格式对齐：
    {"pol","bay_total","ci"}，供跟真正的evaluate_crane_intensity结果对比打印。
    """
    from utils.evaluate import _ci_from_bay_totals

    results = []
    prev_df = None
    for pol in sorted(version_dict.keys()):
        df = version_dict[pol]
        if prev_df is not None:
            discharge_tally = _slots_bay_totals(prev_df, n_bay, "POD", pol)
        else:
            discharge_tally = np.zeros(n_bay, dtype=int)
        loading_tally = _slots_bay_totals(df, n_bay, "POL", pol)
        bay_total = discharge_tally + loading_tally
        results.append({"pol": pol, "bay_total": bay_total, "ci": _ci_from_bay_totals(bay_total)})
        prev_df = df
    return results


def build_tail_placement_demo_scenario():
    """
    专门构造的最小场景，保证match_tails_to_hosts至少产出1条placement，且这条
    placement的host跨越discharge边界——build_test_scenario/build_multi_pol_replay_
    scenario在验证环节实测下来匹配数都是0（要么每个host的headroom都被榨干成0，
    要么根本没触发尾箱来源2/3），不满足任务3验证要求，所以单独设计这一个。

    几何：2个独立的hold cell，都是lr=0,hd=0，互不影响封舱约束：
        big_bay=0(bay pair(2,3))：8个row(row_idx=0..7,tier_idx=0)，capacity_total=8
        big_bay=1(bay pair(4,5))：10个row，capacity_total=10
    capacity_rf=0(不含reefer槽位)，只用GP，不涉及HC/RF，尽量简化。

    cbf设计成"host在port0诞生、留有物理headroom，尾箱残量在port1才出现，
    河对岸destination相同"：
        POL=0: {POD=2: GP=6}   demand=6>tail_threshold(5)，进入候选集
        POL=1: {POD=2: GP=12}

    实测(mrv_select的CI评分在这两个capacity下稳定地)：
        - port0只有big_bay=0(cap=8)被选中：gp_used=min(8,6)=6(demand<=cap，
          没触顶)，demand->0，port0立即complete。这个host的headroom=8-6=2>0，
          且这个POD自己的demand残量=0(不产生来源1尾货)。
        - port1时big_bay=0已被占用(要等POD=2到港才discharge)，只有big_bay=1
          (cap=10)可选：gp_used=min(10,12)=10(demand>cap，顶满，headroom=0)，
          demand->12-10=2(<=5，port1完成)，这2个GP成为来源1尾货，
          归属(POL=1,POD=2)。

    尾箱残量(POL=1,POD=2,GP=2)跟host(big_bay=0,POL=0,POD=2,headroom=2)完全匹配
    (host.POL=0 <= 尾箱.POL=1，同POD=2)，产出恰好1条placement，count=2。

    host(big_bay=0)从POL=0装船，到POD=2才discharge，中途完整经过POL=1这一港的
    departure快照(还没被discharge)，也就是它的存活区间[0,2)跨越了2张POL快照，
    正好覆盖"跨discharge边界"这个验证要求。

    solve()内部mrv_select用到random.random()做兜底排序，为了让上面这套实测行为
    可复现，调用方需要在跑solve()之前固定random.seed(0)(或其它同样验证过的种子)。

    返回vessel（未跑solve()，调用方自己跑，并自己控制random.seed）。
    """
    rows = []
    for row_idx in range(8):
        for bay_idx in (2, 3):
            rows.append({"bay_idx": bay_idx, "row_idx": row_idx, "tier_idx": 0, "lr": 0, "hd": 0,
                         "can_40ft": True, "can_20ft": False, "can_reefer": False})
    for row_idx in range(10):
        for bay_idx in (4, 5):
            rows.append({"bay_idx": bay_idx, "row_idx": row_idx, "tier_idx": 0, "lr": 0, "hd": 0,
                         "can_40ft": True, "can_20ft": False, "can_reefer": False})
    full_slot_table = pd.DataFrame(rows)

    cbf = {0: {2: {"GP": 6}}, 1: {2: {"GP": 12}}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_apply_tail_placements():
    """
    验证apply_tail_placements + verify_cross_port_consistency：
    1. 优先复用build_test_scenario()跑出的unified_tail_list/host_pool/placements，
       其次尝试build_multi_pol_replay_scenario()；这两个场景实测匹配数都是0
       （host的headroom要么被榨干成0，要么没触发尾箱来源2/3），最终用专门
       构造的build_tail_placement_demo_scenario()兜底，保证至少有1条placement、
       且host跨越discharge边界（存活区间跨2张POL快照）。
    2. 对版本1/版本2各跑一遍evaluate_crane_intensity(如果utils/evaluate.py里有)，
       打印两者CI数值对比，只做观察不做断言。
    3. 手动打印几个受影响host的version1 vs version2槽位记录，供人工核对。
    """
    print("\n" + "=" * 60)
    print("──── apply_tail_placements + 跨港一致性回归检查 验证 ────")
    print("=" * 60)

    def _build_placements_for(vessel_builder, label, seed=None):
        if seed is not None:
            random.seed(seed)
        vessel = vessel_builder()
        original_cbf = copy.deepcopy(vessel.cbf)
        snapshots = {}
        best = {"assigned": -1, "vessel": None}
        success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
        result_vessel = vessel if success else best["vessel"]
        final_cbf = copy.deepcopy(result_vessel.cbf)

        source2_total, source3_total = print_source2_and_source3_tail(result_vessel, snapshots, original_cbf)
        dedup2 = _dedup_tail_log_by_pol_pod(result_vessel._tail_source2_log, key_len=2)
        dedup3 = _dedup_tail_log_by_pol_pod(result_vessel._tail_source3_log, key_len=2)
        unified = build_unified_tail_list(
            result_vessel, final_cbf, snapshots, original_cbf, dedup2=dedup2, dedup3=dedup3)
        host_pool = scan_host_candidates(result_vessel, snapshots)
        placements, unplaced = match_tails_to_hosts(
            unified, host_pool, result_vessel.port_min, result_vessel.n_ports)
        print(f"\n---- 场景: {label} ----")
        print(f"solve()完成: success={success}, snapshots覆盖POL={sorted(snapshots.keys())}")
        print(f"unified_tail_list={unified}")
        print(f"host_pool={host_pool}")
        print(f"placements({len(placements)}条)={placements}")
        print(f"unplaced={unplaced}")
        return result_vessel, snapshots, original_cbf, placements

    result_vessel, snapshots, original_cbf, placements = _build_placements_for(
        lambda: build_test_scenario()[0], "build_test_scenario")

    if not placements:
        print("\nbuild_test_scenario()匹配数为0，改用build_multi_pol_replay_scenario()"
              "（尾箱来源2/3天然会产生跨discharge边界存活的host）")
        result_vessel, snapshots, original_cbf, placements = _build_placements_for(
            build_multi_pol_replay_scenario, "build_multi_pol_replay_scenario")

    if not placements:
        print("\nbuild_multi_pol_replay_scenario()匹配数仍为0，改用专门构造的"
              "build_tail_placement_demo_scenario()（seed固定为0，见其docstring里"
              "记录的实测行为），保证至少产出1条跨discharge边界的placement")
        result_vessel, snapshots, original_cbf, placements = _build_placements_for(
            build_tail_placement_demo_scenario, "build_tail_placement_demo_scenario", seed=0)

    if not placements:
        print("[MISMATCH] 三个场景placements都是0条，无法验证apply_tail_placements，需要重新设计场景")
        return

    # 至少验证一条host确实跨越了discharge边界(host_POL < POD，且中间横跨了
    # >=1张POL快照的discharge动作，即[host_POL,POD)区间长度>=2)
    cross_boundary = [p for p in placements if (p["POD"] - p["host_POL"]) >= 2]
    print(f"\n跨discharge边界的placements(POD-host_POL>=2){'找到' if cross_boundary else '未找到'}: "
          f"{cross_boundary if cross_boundary else '(本次场景里全部host都在单一港口内消化，未跨边界)'}")

    version1_dict, version2_dict = apply_tail_placements(result_vessel, snapshots, original_cbf, placements)
    print(f"\napply_tail_placements完成，version1覆盖POL={sorted(version1_dict.keys())}，"
          f"version2覆盖POL={sorted(version2_dict.keys())}")

    ok = verify_cross_port_consistency(version2_dict, placements, result_vessel.port_max)
    print(f"跨港一致性回归检查结果: {'PASS' if ok else 'FAIL'}")

    # ── 2. CI版本1/版本2对比(观察，不断言)：直接在slot级上按同一套CI定义算，
    # 不经过evaluate_crane_intensity要求的cell级单POL快照重建——host cell混装
    # 多个POL的尾箱后，那套重建在这里已经不适用了(见_ci_from_slot_version_dict
    # 的说明)，但CI公式本身仍然复用utils.evaluate._ci_from_bay_totals ──
    try:
        from utils.evaluate import _ci_from_bay_totals as _ci_probe  # noqa: F401
        ci_available = True
    except ImportError:
        ci_available = False

    if ci_available:
        print("\n---- CI(crane intensity) 版本1 vs 版本2 对比(仅观察，不做断言) ----")
        results_v1 = _ci_from_slot_version_dict(version1_dict, result_vessel.n_bay)
        results_v2 = _ci_from_slot_version_dict(version2_dict, result_vessel.n_bay)
        print(f"版本1(原始投影): {[(r['pol'], list(r['bay_total']), r['ci']) for r in results_v1]}")
        print(f"版本2(叠加尾箱后): {[(r['pol'], list(r['bay_total']), r['ci']) for r in results_v2]}")
        for r1, r2 in zip(results_v1, results_v2):
            ci1, ci2 = r1["ci"], r2["ci"]
            print(f"  POL={r1['pol']}: CI版本1={ci1}, CI版本2={ci2}, "
                  f"差异={None if (ci1 is None or ci2 is None) else round(ci2 - ci1, 4)}")
    else:
        print("\nutils.evaluate里没有CI相关函数，跳过CI对比")

    # ── 3. 手动打印受影响host的version1 vs version2槽位记录，供人工核对 ──
    # 打印范围用host的完整存活区间[host.POL, POD)（不是注入区间[tail.POL,POD)），
    # 这样才能对比出"host.POL到tail.POL之前应该保持原样为空、tail.POL开始才
    # 出现尾箱"这个具体要求，而不是只看注入发生的那几张快照。
    print("\n---- 受影响host的version1 vs version2槽位记录(人工核对) ----")
    for placement in placements:
        big_bay = placement["host_bay"]
        lr, hd = placement["host_lr"], placement["host_hd"]
        host_pol, tail_pol, pod = placement["host_POL"], placement["POL"], placement["POD"]
        b0 = STSE_BAY_PAIRS[big_bay][0]
        print(f"\nplacement={placement} (host存活区间=[{host_pol},{pod})，尾箱注入区间=[{tail_pol},{pod}))")
        for pol in sorted(version2_dict.keys()):
            if not (host_pol <= pol < pod):
                continue
            df1 = version1_dict[pol]
            df2 = version2_dict[pol]
            mask = (df1["bay_idx"] == b0) & (df1["lr"] == lr) & (df1["hd"] == hd)
            print(f"  POL快照={pol}:")
            print(f"    version1: {df1[mask][['bay_idx','row_idx','tier_idx','POL','POD','GP_count','RF_count','is_hc']].to_dict('records')}")
            print(f"    version2: {df2[mask][['bay_idx','row_idx','tier_idx','POL','POD','GP_count','RF_count','is_hc']].to_dict('records')}")


def build_deck_squeeze_scenario():
    """
    最小合成场景，专门验证来源2(HC降级挤出)的_tail_source2_log记账是否正确
    （不是求真实解，只是构造一个结构上必然触发deck腾空的畸形船/cbf）。

    只有1个valid cell：big_bay=0（对应STSE_BAY_PAIRS[0]=(2,3)）的lr=0,hd=1
    (deck)，这一摞给2个tier(row_idx=0, tier_idx=4/8)，n=2 -> 按
    Vessel._stack_hc_cap(n=2, hd=1)=max(2-1,0)=1，即这一摞最多贴1个HC标签。
    其余(lr,hd)组合没有任何行 -> capacity_total=0 -> is_valid=False，不参与搜索，
    保证solve()只有这一个cell可用、demand只够刚好摆满它。

    cbf只有POL=0, POD=1一条：GP=5, HC=2。
    GP+HC=7 > tail_threshold(默认5)，能进入候选集；这个cell capacity_total=2，
    assign()只会填满这2个槽位(gp_deduct_gp=2，不动HC)，恰好把这一摞摆满
    (occupied=n=2)。填完后剩余GP=3+HC=2=5，5不大于tail_threshold(5)，
    remaining_pods()判定这个POD已"完成"(退化成尾货)，port_complete()=True，
    换港后current_pol超出cbf.keys()，solve()成功返回——这样才有departure
    snapshot可以喂给proj_cell_to_vessel验证，不是靠best-effort近似解。
    original_cbf里HC=2作为贴标签预算池，但这一摞cap_hc只有1，故只贴1个HC标签，
    触发"摆满摞+出现HC标签"的腾空分支。

    手动预期：
        - _tail_source2_log 应恰好新增1条 (POL=0, POD=1)：squeeze回退1个GP名额
        - _tail_source3_log 应恰好新增1条 (POL=0, POD=1, gp_hc_budget剩=1, rf_hc_budget剩=0)：
          原始HC预算池=2，squeeze那一步只消耗了1(cap_hc=1)，剩下1回退进HC
        - 两条日志合起来应完整解释cbf[0][1]从{GP:3,HC:2}变成{GP:4,HC:3}的变化：
          GP的+1由来源2解释，HC的+1由来源3解释，不再有"数字变了但没有日志能解释"的部分
    """
    rows = []
    for bay_idx in (2, 3):
        rows.append({
            "bay_idx": bay_idx, "row_idx": 0, "tier_idx": 4, "lr": 0, "hd": 1,
            "can_40ft": True, "can_20ft": False, "can_reefer": False,
        })
        rows.append({
            "bay_idx": bay_idx, "row_idx": 0, "tier_idx": 8, "lr": 0, "hd": 1,
            "can_40ft": True, "can_20ft": False, "can_reefer": False,
        })
    full_slot_table = pd.DataFrame(rows)

    cbf = {0: {1: {"GP": 5, "HC": 2}}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_deck_squeeze_scenario():
    """跑一遍build_deck_squeeze_scenario()，核对_tail_source2_log是否恰好
    记录了手动预期的(POL=0, POD=1)一条腾空记录。只做核对+打印，不做安置。"""
    print("\n" + "=" * 60)
    print("──── 来源2(deck-squeeze)最小合成场景验证 ────")
    print("=" * 60)

    vessel = build_deck_squeeze_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    print(f"solve()完成: success={success}, 剩余cbf={result_vessel.cbf}")

    # success=True时current_pol已推进、discharge()会把这个cell清空，
    # 这里打印的是清空后的状态，仅供参照——真正验证用的是换港前snapshots[0]里的cell。
    cell_rec = result_vessel.cell[0, 0, 1]
    print(f"合成cell(big_bay=0, lr=0, hd=1) 当前(discharge后)状态: {cell_rec}")

    cbf_before = copy.deepcopy(result_vessel.cbf[0][1])

    source2_total, source3_total = print_source2_and_source3_tail(result_vessel, snapshots, original_cbf)

    cbf_after = result_vessel.cbf[0][1]

    log2 = result_vessel._tail_source2_log
    log3 = result_vessel._tail_source3_log
    expected2 = [(0, 1)]
    expected3 = [(0, 1, 1, 0)]
    print(f"\n_tail_source2_log 实际记录: {log2}")
    print(f"手动预期记录:              {expected2}")
    print(f"_tail_source3_log 实际记录: {log3}")
    print(f"手动预期记录:              {expected3}")

    ok2 = log2 == expected2
    ok3 = log3 == expected3
    if ok2 and ok3:
        print("[OK] 来源2、来源3记账逻辑核对均一致")
    else:
        if not ok2:
            print(f"[MISMATCH] 来源2与预期不符！实际={log2}, 预期={expected2}")
        if not ok3:
            print(f"[MISMATCH] 来源3与预期不符！实际={log3}, 预期={expected3}")

    # 核对print_source2_and_source3_tail(去重+幂等性校验后)的结果是否完整解释
    # cbf[0][1]的变化，不留"数字变了但没有日志能解释"的空白
    dedup3 = _dedup_tail_log_by_pol_pod(log3, key_len=2)
    gp_delta_from_log2 = source2_total  # 每个去重后的来源2事件固定回退1个GP名额
    hc_delta_from_log3 = sum(gp_leftover for (_, _, gp_leftover, _) in dedup3.values())
    rf_delta_from_log3 = sum(rf_leftover for (_, _, _, rf_leftover) in dedup3.values())

    gp_delta_actual = cbf_after.get("GP", 0) - cbf_before.get("GP", 0)
    hc_delta_actual = cbf_after.get("HC", 0) - cbf_before.get("HC", 0)
    rf_delta_actual = cbf_after.get("RF", 0) - cbf_before.get("RF", 0)

    print(f"\ncbf[0][1] 变化前: {cbf_before}")
    print(f"cbf[0][1] 变化后: {cbf_after}")
    print(f"GP变化: 实际={gp_delta_actual}, 来源2日志能解释={gp_delta_from_log2}")
    print(f"HC变化: 实际={hc_delta_actual}, 来源3日志能解释={hc_delta_from_log3}")
    print(f"RF变化: 实际={rf_delta_actual}, 来源3日志能解释={rf_delta_from_log3}")

    if gp_delta_actual == gp_delta_from_log2 and hc_delta_actual == hc_delta_from_log3 \
            and rf_delta_actual == rf_delta_from_log3:
        print("[OK] 来源2+来源3两条日志完整解释了这次cbf变化，没有遗漏")
    else:
        print("[MISMATCH] 两条日志加起来对不上实际cbf变化，还有没被记录到的写回路径！")


def build_multi_pol_replay_scenario():
    """
    最小合成场景，专门验证同一个(POL,POD)分组在被真正discharge之前，
    跨越多个POL快照replay时_tail_source2_log/_tail_source3_log会不会重复记录。

    跟build_deck_squeeze_scenario同一套船体几何(big_bay=0, lr=0,hd=1,
    2-tier deck摞)，唯一区别是cbf设计成"这批货中途要经过一个空港口才卸货"：
        POL=0: {POD=2: GP=5, HC=2}    货在港口0装船，目的港POD=2
        POL=1: {}                     港口1没有新增需求，纯粹"路过"
    货物在POL=0装船后，要先经过POL=1这一港（POL=1没有该货的discharge动作，
    arriving_pod=1 != POD=2），POL=1的cbf又是空字典 -> port_complete()对
    POL=1瞬间成立(remaining_pods()==set()) -> solve()会在snapshots里同时
    留下POL=0和POL=1两份departure快照，且两份快照里这个cell的状态完全相同
    (还没被discharge)。直到换到POL=2时才discharge()真正卸货。

    export_bayplan会对snapshots里每个POL都调用一次proj_cell_to_vessel，
    也就是对这同一个(POL=0,POD=2)分组重复投影2次——这正是本函数要验证的
    "跨POL快照replay"场景。

    手动预期：
        - 原始（未去重）日志里，(POL=0,POD=2)应该恰好出现2条来源2记录、
          2条来源3记录（每个POL快照replay一次，共2次）
        - 这2条来源2记录彼此完全相同(都是(0,2))；2条来源3记录彼此完全
          相同(都是(0,2,1,0))——验证"同一分组重算是幂等的"这个描述本身站得住脚
        - 按(POL,POD)去重后，来源2应只剩1条(回退1个GP名额)，来源3也只剩1条
          (回退1个HC名额)，不是2倍——这批货实际只发生过1次squeeze，不是2次
    """
    rows = []
    for bay_idx in (2, 3):
        rows.append({
            "bay_idx": bay_idx, "row_idx": 0, "tier_idx": 4, "lr": 0, "hd": 1,
            "can_40ft": True, "can_20ft": False, "can_reefer": False,
        })
        rows.append({
            "bay_idx": bay_idx, "row_idx": 0, "tier_idx": 8, "lr": 0, "hd": 1,
            "can_40ft": True, "can_20ft": False, "can_reefer": False,
        })
    full_slot_table = pd.DataFrame(rows)

    cbf = {
        0: {2: {"GP": 5, "HC": 2}},
        1: {},
    }
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_multi_pol_replay_dedup():
    """跑一遍build_multi_pol_replay_scenario()，核对：
    1) 同一(POL,POD)分组确实在原始日志里出现了2条记录（对应2次POL快照replay）；
    2) 这2条记录的数量/budget值彼此完全一致（验证幂等性）；
    3) 按(POL,POD)去重后的总数等于这批货实际只发生一次的真实箱数，不是2倍。
    """
    print("\n" + "=" * 60)
    print("──── 跨POL快照replay去重验证 ────")
    print("=" * 60)

    vessel = build_multi_pol_replay_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    print(f"solve()完成: success={success}, 剩余cbf={result_vessel.cbf}")
    print(f"snapshots覆盖的POL: {sorted(snapshots.keys())}"
          f"（应该同时含POL=0和POL=1，同一批货中途路过POL=1还没被discharge）")

    # 走主报告路径(print_source2_and_source3_tail)，而不是自己手写一遍投影循环——
    # 这样才是真正验证"固化进主路径的去重+幂等性校验"本身工作正常，
    # 不是又搭了一套只在这个合成场景里跑的平行逻辑。
    source2_total, source3_total = print_source2_and_source3_tail(result_vessel, snapshots, original_cbf)

    log2 = result_vessel._tail_source2_log
    log3 = result_vessel._tail_source3_log
    print(f"\n原始(未去重) _tail_source2_log: {log2}")
    print(f"原始(未去重) _tail_source3_log: {log3}")
    print(f"print_source2_and_source3_tail返回的去重后总数: 来源2={source2_total}, 来源3={source3_total}")

    # 1) 确认(POL=0,POD=2)确实出现了2条记录
    target_pol, target_pod = 0, 2
    hits2 = [e for e in log2 if e == (target_pol, target_pod)]
    hits3 = [e for e in log3 if e[:2] == (target_pol, target_pod)]
    print(f"\n(POL={target_pol}, POD={target_pod}) 在来源2里出现次数: {len(hits2)} (预期2)")
    print(f"(POL={target_pol}, POD={target_pod}) 在来源3里出现次数: {len(hits3)} (预期2)")
    check1_ok = len(hits2) == 2 and len(hits3) == 2

    # 2) 确认这2条记录彼此完全一致（幂等性）
    check2_ok = (len(set(hits2)) == 1) and (len(set(hits3)) == 1)
    print(f"来源2这2条记录是否彼此完全一致: {check2_ok if len(hits2) == 2 else 'N/A(条数不对)'} "
          f"(内容: {hits2})")
    print(f"来源3这2条记录是否彼此完全一致: {check2_ok if len(hits3) == 2 else 'N/A(条数不对)'} "
          f"(内容: {hits3})")

    if check1_ok and check2_ok:
        print("[OK] 验证1+2通过：确实重复记录了2次，且2次的数值完全一致（幂等性成立）")
    else:
        print("[MISMATCH] 验证1或2失败，幂等性假设可能不成立，需要停下来查")

    # 3) 按(POL,POD)去重，去重后的总数应等于真实只发生一次的箱数
    dedup2 = {}
    for pol_hit, pod_hit in log2:
        dedup2[(pol_hit, pod_hit)] = dedup2.get((pol_hit, pod_hit), 0) + 1  # 去重后每组只应计1次
    dedup2_total = len(dedup2)  # 按去重后的分组数计数，每组固定回退1个GP名额

    dedup3 = {}
    for pol_hit, pod_hit, gp_leftover, rf_leftover in log3:
        dedup3[(pol_hit, pod_hit)] = (gp_leftover, rf_leftover)  # 同一分组的值应完全一致，取哪条都一样
    dedup3_gp_total = sum(gp for (gp, rf) in dedup3.values())
    dedup3_rf_total = sum(rf for (gp, rf) in dedup3.values())

    raw2_total = len(log2)
    raw3_gp_total = sum(gp for (_, _, gp, _) in log3)

    print(f"\n未去重来源2总数: {raw2_total} vs 去重后来源2总数: {dedup2_total}")
    print(f"未去重来源3(GP)总数: {raw3_gp_total} vs 去重后来源3(GP)总数: {dedup3_gp_total}")
    print(f"去重后来源2总数(预期1，不是2): {dedup2_total}")
    print(f"去重后来源3 GP总数(预期1，不是2): {dedup3_gp_total}, RF总数(预期0): {dedup3_rf_total}")

    if dedup2_total == 1 and dedup3_gp_total == 1 and dedup3_rf_total == 0:
        print("[OK] 验证3通过：去重后总数=1，对应这批货实际只发生过1次squeeze，"
              "不是被2次POL快照replay放大成2")
    else:
        print("[MISMATCH] 去重后总数不等于1，去重逻辑或场景构造有问题")

    # 交叉核对：这里手算的去重结果应该跟print_source2_and_source3_tail(主报告路径)
    # 自己返回的去重后总数完全一致——证明"主路径的去重逻辑"和"这里手算验证的去重逻辑"
    # 是同一件事，不是两套各自为政的实现。
    manual_source3_total = dedup3_gp_total + dedup3_rf_total
    if dedup2_total == source2_total and manual_source3_total == source3_total:
        print(f"[OK] 主报告路径返回值(来源2={source2_total}, 来源3={source3_total})"
              f"与手算去重结果完全一致")
    else:
        print(f"[MISMATCH] 主报告路径返回值(来源2={source2_total}, 来源3={source3_total})"
              f"与手算去重结果(来源2={dedup2_total}, 来源3={manual_source3_total})不一致！")


def _physical_occupied_total(vessel: Vessel, snapshots: dict, original_cbf: dict, pol: int, pod: int) -> int:
    """(POL,POD)自己这港新装、实际占用的物理槽位总数——不分是否is_hc，纯计
    GP_count+RF_count的物理占用，只认_BIG_BAY_OF_B0能映射到的b0侧行（b1侧
    是proj_cell_to_vessel镜像写出来的重复行，不能重复计数，口径同
    build_tail_container_list）。

    在vessel的deepcopy上跑一次proj_cell_to_vessel（避免污染调用方传入的
    vessel实例），只用于测试fixture里的守恒不变量断言，不是生产路径。
    """
    proj_vessel = copy.deepcopy(vessel)
    df = proj_vessel.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)
    mask = (
        (df["POL"] == pol) & (df["POD"] == pod)
        & df["bay_idx"].isin(_BIG_BAY_OF_B0.keys())
    )
    sub = df.loc[mask]
    return int(sub["GP_count"].sum() + sub["RF_count"].sum())


def _assert_tail_conservation(result_vessel: Vessel, snapshots: dict, original_cbf: dict,
                               pol: int, pod: int, new_list: list) -> tuple:
    """守恒不变量：不管cap_hc具体公式怎么分配预算，对每个(POL,POD)都必须满足
        GP缺口+HC缺口+RF缺口+HR缺口 == max(0, 原始demand总量 - 实际物理占用槽位总数)
    这条不变量只依赖"最终占了多少物理槽位"和"最初要多少"这两个端点，不依赖
    cap_hc配额怎么在多个摞/多个cell之间分配——所以不会随着_stack_hc_cap公式
    调整就跟着碎。

    返回(demand_total, physical_occupied, expected_total_gap, actual_total_gap)
    供调用方打印/进一步断言。
    """
    demand = original_cbf.get(pol, {}).get(pod, {})
    demand_total = sum(demand.get(k, 0) for k in ("GP", "HC", "RF", "HR"))

    physical_occupied = _physical_occupied_total(result_vessel, snapshots, original_cbf, pol, pod)
    expected_total_gap = max(0, demand_total - physical_occupied)

    actual_total_gap = sum(rec["count"] for rec in new_list if rec["POL"] == pol and rec["POD"] == pod)

    assert actual_total_gap == expected_total_gap, (
        f"[守恒不变量被打破] (POL={pol},POD={pod}) 原始demand总量={demand_total}, "
        f"实际物理占用槽位总数={physical_occupied}, 期望总缺口=max(0,{demand_total}-{physical_occupied})="
        f"{expected_total_gap}, build_tail_container_list算出的总缺口={actual_total_gap}"
    )
    return demand_total, physical_occupied, expected_total_gap, actual_total_gap


def build_hold_hc_budget_scenario():
    """最小合成场景，专门用来验证build_tail_container_list：1个hold cell，
    10个物理槽位(cap_total=10)，demand是GP=8、HC=6(合计14)。

    几何：只给1个valid cell——big_bay=0（对应STSE_BAY_PAIRS[0]=(2,3)）的
    lr=0,hd=0(hold)。2个"摞"(row_idx)：
        row_idx=0: 8个tier(0..7)的can_40ft槽位 -> 这一摞n=8
        row_idx=1: 2个tier(0..1)的can_40ft槽位 -> 这一摞n=2
    capacity_total=8+2=10。capacity_hc取决于Vessel._stack_hc_cap(n, hd)
    的具体公式——具体数值随公式调整会变，不在这里手算写死；调用方如果需要
    校验，应直接读result_vessel.capacity_hc[0, 0, 0]，而不是假设某个固定值。
    bay_idx=3是b1侧镜像行，只用于proj_cell_to_vessel写回镜像标签，不计入
    capacity（跟build_test_scenario同款做法，capacity只认_BIG_BAY_OF_B0能
    映射到的b0侧行）。

    cbf: POL=0, POD=1: GP=8, HC=6，合计14远超tail_threshold(默认4)能进
    候选集；这个vessel只有1个valid cell、1个POD候选，solve()会把它直接
    分进这个cell，assign()按cap_total=10把GP+HC demand扣掉10个（优先扣
    GP），剩余4个(<=tail_threshold)判定port立即complete。

    真实缺口不再靠手算cap_hc公式反推，而是靠一条不依赖cap_hc具体怎么分配
    的守恒不变量校验（见_assert_tail_conservation）：
        GP缺口+HC缺口+RF缺口+HR缺口 == max(0, 原始demand总量(14) - 实际
        物理占用槽位总数(<=capacity_total=10))
    不管_stack_hc_cap以后怎么调，这条不变量都成立，不需要再手算一遍
    cap_hc具体数字。

    返回vessel（未跑solve()，调用方自己跑）。
    """
    rows = []
    for bay_idx in (2, 3):
        for tier_idx in range(8):
            rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": tier_idx, "lr": 0, "hd": 0,
                         "can_40ft": True, "can_20ft": False, "can_reefer": False})
        for tier_idx in range(2):
            rows.append({"bay_idx": bay_idx, "row_idx": 1, "tier_idx": tier_idx, "lr": 0, "hd": 0,
                         "can_40ft": True, "can_20ft": False, "can_reefer": False})
    full_slot_table = pd.DataFrame(rows)
    cbf = {0: {1: {"GP": 8, "HC": 6}}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_build_tail_container_list_hold_example():
    """跑build_hold_hc_budget_scenario()，用跟cap_hc具体公式无关的守恒不变量
    （_assert_tail_conservation）校验build_tail_container_list算出的缺口，
    并对比旧口径(build_unified_tail_list，来源1+来源3独立相加)的总数，定性
    确认新口径没有比旧口径算出更大的总数（新口径修的是重复计数，不会让总数
    变大——这个场景不是build_multi_cell_squeeze_scenario那种反直觉的例外）。
    """
    print("\n" + "=" * 60)
    print("──── build_tail_container_list hold例子验证（守恒不变量） ────")
    print("=" * 60)

    vessel = build_hold_hc_budget_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    print(f"solve()完成: success={success}, 剩余cbf={result_vessel.cbf}")

    final_cbf = copy.deepcopy(result_vessel.cbf)

    new_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
    print(f"\n新口径(final_vs_original)结果: {new_list}")

    new_by_type = {}
    for rec in new_list:
        new_by_type[rec["type"]] = new_by_type.get(rec["type"], 0) + rec["count"]
    gp_gap = new_by_type.get("GP", 0)
    hc_gap = new_by_type.get("HC", 0)
    rf_gap = new_by_type.get("RF", 0)
    hr_gap = new_by_type.get("HR", 0)
    print(f"新口径按type求和: GP缺口={gp_gap}, HC缺口={hc_gap}, RF缺口={rf_gap}, HR缺口={hr_gap}")

    demand_total, physical_occupied, expected_total_gap, actual_total_gap = _assert_tail_conservation(
        result_vessel, snapshots, original_cbf, pol=0, pod=1, new_list=new_list
    )
    print(f"[OK] 守恒不变量成立: demand总量={demand_total}, 物理占用总数={physical_occupied}, "
          f"总缺口=max(0,{demand_total}-{physical_occupied})={expected_total_gap}"
          f"（与build_tail_container_list算出的总缺口{actual_total_gap}一致）")
    assert rf_gap == 0 and hr_gap == 0, f"这个场景没有RF/HR demand，缺口应全为0，实际RF={rf_gap} HR={hr_gap}"

    # 对比旧口径：来源1(assign()后残量) + 来源3(HC预算池分不完回退)独立相加
    old_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    print(f"\n旧口径(来源1+2+3独立相加)结果: {old_list}")
    old_by_type = {}
    for rec in old_list:
        old_by_type[rec["type"]] = old_by_type.get(rec["type"], 0) + rec["count"]
    old_gp_gap = old_by_type.get("GP", 0)
    old_hc_gap = old_by_type.get("HC", 0)
    print(f"旧口径按type求和: GP缺口={old_gp_gap}, HC缺口={old_hc_gap}")

    old_total = sum(old_by_type.values())
    new_total = sum(new_by_type.values())
    print(f"\n旧口径总尾箱数={old_total}, 新口径总尾箱数={new_total}")

    assert new_total <= old_total, (
        f"新口径总尾箱数({new_total}) 不应该大于旧口径({old_total})——新口径修的是"
        f"重复计数，不会让总数变大（这个场景不是build_multi_cell_squeeze_scenario"
        f"那种反直觉的例外场景）"
    )
    print(f"[OK] 新口径总尾箱数({new_total}) <= 旧口径总尾箱数({old_total})，符合预期")


def build_hold_hr_budget_scenario():
    """build_hold_hc_budget_scenario的RF/HR镜像版：同样1个hold cell，
    cap_total=10，demand换成RF/HR(而不是GP/HC)，槽位全部can_reefer=True。
    用来验证build_tail_container_list的RF/HR分支(proj_cell_to_vessel第一步
    "摊RF需求"+labeling阶段的rf_hc_budget循环)跟GP/HC分支是完全对称的一套
    代码路径，不是只测了GP/HC那一半。

    注意：不能直接照搬build_hold_hc_budget_scenario的demand数字。
    VesselClass.remaining_pods()对GP/HC和RF/HR的"可接受尾货"判断不对称：
        GP/HC: (GP+HC) > tail_threshold(默认4) 才算"还有余量待装"，
               <=tail_threshold的小额残量会被直接容忍、当场port_complete。
        RF/HR: (RF+HR) > 0 就算"还有余量待装"，一点残量都不容忍。
    这是VesselClass本身的业务规则(reefer货不允许被静默地当小额尾货放着，
    要么真放上船要么明确处理)，不是_stack_hc_cap公式的事。GP/HC例子能同时
    做出GP缺口、HC缺口两个都非零，靠的正是assign()后允许留一点HC残量仍
    判定port_complete；RF/HR不允许这个残量存在，assign()后必须
    D_rf+D_hr恰好等于occupied(不能有任何残留)否则solve()会在"没有更多
    cell可用但仍有RF/HR需求"时判定dead cell直接搜索失败(不会推进到
    port_complete，snapshots也不会正确写入)。

    所以这个场景选RF=1,HR=9(合计10，恰好用满cap_total=10，无残留)：
    assign()会把RF+HR demand全部扣到0(rf_used=min(cap_rf,10)=10，
    rf_deduct_rf=min(1,10)=1，rf_deduct_hr=9)，port立即complete；HR demand
    (9)刻意选得比这个cell两个摞的HC配额总和(quota(n=8)+quota(n=2)，具体数值
    随_stack_hc_cap公式而定)更大，保证labeling阶段的rf_hc_budget分不完
    整个HR demand，从而必然留下一个非零的HR缺口——不管_stack_hc_cap公式
    怎么调，quota之和不可能超过cap_total(10)，HR=9只要quota之和<9就会触发
    缺口，这条设计对公式细节不敏感。

    真实缺口同样靠_assert_tail_conservation的守恒不变量校验，不在这里手算
    写死cap_hc相关的具体数字。

    返回vessel（未跑solve()，调用方自己跑）。
    """
    rows = []
    for bay_idx in (2, 3):
        for tier_idx in range(8):
            rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": tier_idx, "lr": 0, "hd": 0,
                         "can_40ft": True, "can_20ft": False, "can_reefer": True})
        for tier_idx in range(2):
            rows.append({"bay_idx": bay_idx, "row_idx": 1, "tier_idx": tier_idx, "lr": 0, "hd": 0,
                         "can_40ft": True, "can_20ft": False, "can_reefer": True})
    full_slot_table = pd.DataFrame(rows)
    cbf = {0: {1: {"RF": 1, "HR": 9}}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_build_tail_container_list_rf_mirror():
    """跑build_hold_hr_budget_scenario()，用跟cap_hc具体公式无关的守恒不变量
    （_assert_tail_conservation）校验build_tail_container_list算出的缺口，
    证明RF/HR分支跟GP/HC分支是同一套逻辑的镜像、独立正确，不是只测过GP/HC
    那一半就假定RF/HR"应该也对"。这个场景demand无残留(RF+HR恰好用满
    cap_total)，所以RF缺口和HR缺口不会同时非零，原因见
    build_hold_hr_budget_scenario docstring——这是VesselClass业务规则本身
    的限制，不是_stack_hc_cap公式的事。
    """
    print("\n" + "=" * 60)
    print("──── build_tail_container_list RF/HR镜像场景验证（守恒不变量） ────")
    print("=" * 60)

    vessel = build_hold_hr_budget_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    print(f"solve()完成: success={success}, 剩余cbf={result_vessel.cbf}")
    assert success, "这个场景demand恰好用满capacity(无残留)，应该能成功solve()"

    new_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
    print(f"新口径结果: {new_list}")

    by_type = {}
    for rec in new_list:
        by_type[rec["type"]] = by_type.get(rec["type"], 0) + rec["count"]
    rf_gap = by_type.get("RF", 0)
    hr_gap = by_type.get("HR", 0)
    gp_gap = by_type.get("GP", 0)
    hc_gap = by_type.get("HC", 0)
    print(f"按type求和: GP缺口={gp_gap}, HC缺口={hc_gap}, RF缺口={rf_gap}, HR缺口={hr_gap}")

    demand_total, physical_occupied, expected_total_gap, actual_total_gap = _assert_tail_conservation(
        result_vessel, snapshots, original_cbf, pol=0, pod=1, new_list=new_list
    )
    print(f"[OK] 守恒不变量成立: demand总量={demand_total}, 物理占用总数={physical_occupied}, "
          f"总缺口=max(0,{demand_total}-{physical_occupied})={expected_total_gap}"
          f"（与build_tail_container_list算出的总缺口{actual_total_gap}一致）")

    assert gp_gap == 0 and hc_gap == 0, f"这个场景没有GP/HC demand，缺口应全为0，实际GP={gp_gap} HC={hc_gap}"
    assert not (rf_gap > 0 and hr_gap > 0), (
        f"这个场景demand无残留(RF+HR恰好用满cap_total)，代数上RF缺口和HR缺口"
        f"不该同时非零，实际RF缺口={rf_gap}, HR缺口={hr_gap}"
    )
    assert hr_gap > 0, (
        f"HR demand(9)刻意设计得比这个cell的HC配额总量大，无论_stack_hc_cap"
        f"公式怎么调都应该留下非零HR缺口，实际HR缺口={hr_gap}（说明这个场景"
        f"没能触发它想验证的'HR demand超出HC配额'路径，需要调整demand数字）"
    )
    print(f"[OK] RF/HR镜像场景与守恒不变量一致：GP缺口=0, HC缺口=0, RF缺口={rf_gap}, HR缺口={hr_gap}"
          "（RF/HC分支的proj_cell_to_vessel第一步摊RF需求+labeling阶段"
          "rf_hc_budget贴标循环，跟GP/HC分支是完全对称、独立正确的代码路径）")


def build_multi_cell_squeeze_scenario():
    """最小合成场景，永久固化"旧口径_tail_source2_log的隐藏欠计数缺陷"这个发现：
    同一个(POL,POD)的货跨3个独立deck cell装，每个cell各自独立触发一次
    deck-squeeze——3个真实发生的不同物理事件，理应各自算1个GP损失，合计3个。

    但_tail_source2_log的去重key只到(POL,POD)，不含(bay,lr,hd)，没法区分
    "同一个cell的squeeze被多张POL快照重复replay观察了几次"(这种情况多条
    记录内容相同，理应去重成1)和"同一个POD跨多个cell各自独立触发squeeze"
    (这种情况多条记录内容也相同——都只有(POL,POD)两个字段——但代表的是
    N个不同物理事件，不该被压缩成1)。旧代码把两种情况混为一谈，一律按
    "内容相同就只算1条"处理，导致后一种情况被系统性地压缩成固定1个GP，
    不管真实触发了几次。

    这正是真实STSE数据上旧口径(108) vs 新口径(155)出现"新口径更大"这个
    反直觉结果的根因：真实数据里大量POD的货跨许多deck cell装载，每个cell
    各自触发squeeze，旧口径永远只记1个GP/组，新口径通过直接测量最终物理
    占用状态，如实揭示了被低估的真实GP损失。

    几何：3个独立deck cell(big_bay=0/1/2，对应STSE_BAY_PAIRS[0..2])，
    都是lr=0,hd=1，各自1个摞(row_idx=0)、2个tier(4/8)，capacity_total=2，
    capacity_hc取决于Vessel._stack_hc_cap(n=2, hd=1)的具体公式——不在这里
    手算写死，3个cell完全对称、互不影响封舱约束。

    cbf: POL=0, POD=1: GP=7, HC=3。demand设计成"3个cell都被这个POD填满，
    且都要经历完整的HC贴标+squeeze"：
        - assign()按顺序把这个POD填满3个cell(每个cap=2，共6)，GP demand(7)
          在前2个cell都远超单个cell的capacity，第3个cell上GP demand降到3
          (>2)仍够填满整个cell，gp_deduct_hc全程为0，HC demand(3)原样留在
          residual里没被assign()动过。3个cell用完后，demand=GP:7-6=1，
          HC:3(合计4<=tail_threshold默认值4，port complete)——这个残量
          门槛依赖的是tail_threshold这个业务常量，不是cap_hc公式。
        - labeling: budget=original HC demand=3。这个场景专门选HC demand
          恰好等于3个cell的capacity_hc之和，使budget刚好能覆盖所有3个
          cell各自的整摞quota——不管_stack_hc_cap公式怎么定义quota(n)，
          只要3个对称cell的quota之和被HC demand精确覆盖，3个cell就都会
          触发"整摞转HC+满摞腾空"的squeeze路径，這是这个场景设计要保留
          的关键效果(3次独立squeeze，不是1次)，不依赖quota的具体数值。

    真实缺口靠_assert_tail_conservation的守恒不变量校验，不在这里手算
    写死GP/HC缺口的具体数字——不管_stack_hc_cap公式如何变化，这条不变量
    都成立。

    这个场景的关键点仍然是：旧口径(build_unified_tail_list)靠
    _tail_source2_log按(POL,POD)去重，把3个cell各自独立触发的3次真实
    squeeze事件压缩成1个GP，系统性欠计数；新口径(build_tail_container_list)
    不依赖这份日志，直接测最终物理占用状态，如实测出这3次squeeze的完整
    影响。旧口径因此漏记了squeeze多扣的GP，新口径的总数在这个场景下反而
    会比旧口径更大——这是本文件里目前唯一一个新口径总数超过旧口径的场景，
    专门用来固化这个反直觉但正确的发现，防止以后有人看到"新口径应该更小"
    的直觉就误改代码（verify_multi_cell_squeeze_undercounting里用
    `new_total > old_total`定性断言，不写死具体数字）。

    返回vessel（未跑solve()，调用方自己跑）。
    """
    rows = []
    for b0, b1 in STSE_BAY_PAIRS[:3]:
        for bay_idx in (b0, b1):
            rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": 4, "lr": 0, "hd": 1,
                         "can_40ft": True, "can_20ft": False, "can_reefer": False})
            rows.append({"bay_idx": bay_idx, "row_idx": 0, "tier_idx": 8, "lr": 0, "hd": 1,
                         "can_40ft": True, "can_20ft": False, "can_reefer": False})
    full_slot_table = pd.DataFrame(rows)
    cbf = {0: {1: {"GP": 7, "HC": 3}}}
    vessel = Vessel(full_slot_table=full_slot_table, cbf=cbf, current_pol=0)
    return vessel


def verify_multi_cell_squeeze_undercounting():
    """跑build_multi_cell_squeeze_scenario()，固化验证：
    1) 旧口径_tail_source2_log确实把3个不同cell的3次真实squeeze事件去重成1个
       （证明这是旧代码的真实缺陷，不是我瞎猜的）
    2) 新口径(build_tail_container_list)不依赖_tail_source2_log，直接测最终
       物理占用状态，用跟cap_hc具体公式无关的守恒不变量校验缺口
       （_assert_tail_conservation），如实测出这3次squeeze的完整影响
    3) 这个场景下新口径总数 > 旧口径总数——反直觉但正确，用定性断言（不写死
       具体数字）永久留在回归测试里防止未来被误判成bug改回去
    """
    print("\n" + "=" * 60)
    print("──── 旧口径_tail_source2_log多cell squeeze欠计数场景验证 ────")
    print("=" * 60)

    vessel = build_multi_cell_squeeze_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    result_vessel = vessel if success else best["vessel"]
    print(f"solve()完成: success={success}, 剩余cbf={result_vessel.cbf}")
    assert success, "这个场景demand设计成能让port在3个cell都填满后立即complete，应该能成功solve()"
    final_cbf = copy.deepcopy(result_vessel.cbf)

    old_list = build_unified_tail_list(result_vessel, final_cbf, snapshots, original_cbf)
    raw_source2 = result_vessel._tail_source2_log
    dedup2 = _dedup_tail_log_by_pol_pod(raw_source2, key_len=2) if raw_source2 else {}
    print(f"\n_tail_source2_log原始条数(真实物理squeeze事件数)={len(raw_source2)}: {raw_source2}")
    print(f"去重后(旧口径实际采用的计数)={len(dedup2)}: {dedup2}")
    assert len(raw_source2) == 3, f"这个场景应该触发3次真实squeeze，实际raw条数={len(raw_source2)}"
    assert len(dedup2) == 1, f"旧去重逻辑应该把3次真实事件压缩成1，实际去重后={len(dedup2)}"
    print("[OK] 确认旧_tail_source2_log的欠计数缺陷：3次真实物理squeeze事件被去重成1个GP")

    new_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
    old_total = sum(r["count"] for r in old_list)
    new_total = sum(r["count"] for r in new_list)
    print(f"\n旧口径结果: {old_list} (总数={old_total})")
    print(f"新口径结果: {new_list} (总数={new_total})")

    demand_total, physical_occupied, expected_total_gap, actual_total_gap = _assert_tail_conservation(
        result_vessel, snapshots, original_cbf, pol=0, pod=1, new_list=new_list
    )
    print(f"[OK] 守恒不变量成立: demand总量={demand_total}, 物理占用总数={physical_occupied}, "
          f"总缺口=max(0,{demand_total}-{physical_occupied})={expected_total_gap}"
          f"（与build_tail_container_list算出的总缺口{actual_total_gap}一致）")

    assert new_total > old_total, (
        f"这个场景应该复现'新口径总数({new_total})>旧口径总数({old_total})'的反直觉但正确的结果——"
        f"旧口径的_tail_source2_log把3次独立squeeze去重成1次，systematically欠计了真实GP损失"
    )
    print(f"[OK] 新口径总数({new_total}) > 旧口径总数({old_total})，符合预期："
          f"新口径如实测出了旧_tail_source2_log漏计的{new_total - old_total}个真实GP损失")


def verify_build_tail_container_list_cross_check():
    """独立交叉验证：build_tail_container_list算出的final_hc/final_hr
    (通过gap反推：final=demand-gap，仅在demand>=final即gap未被clip时精确成立，
    这里选用的场景demand都覆盖了最终标签数，反推是精确的)，应该跟
    scan_host_candidates(一个已经在更早任务里独立写好、独立验证过的函数，
    用完全不同的代码结构——按host聚合而不是按(POL,POD)聚合——重新算了一遍
    is_hc标签数)算出的hc_used(=capacity_hc-hc_headroom)按(POL,POD)求和后
    完全一致。两个函数从两条独立代码路径算出同一个物理量，如果结果一致，
    说明build_tail_container_list没有沿着某条隐藏的bug路径算出"看似合理
    但恰好一致"的错误答案。

    在build_hold_hc_budget_scenario/build_hold_hr_budget_scenario/
    build_multi_cell_squeeze_scenario三个场景上跑一遍这个交叉验证。
    """
    print("\n" + "=" * 60)
    print("──── build_tail_container_list × scan_host_candidates 交叉验证 ────")
    print("=" * 60)

    scenarios = [
        ("build_hold_hc_budget_scenario", build_hold_hc_budget_scenario, {"GP": 8, "HC": 6}),
        ("build_hold_hr_budget_scenario", build_hold_hr_budget_scenario, {"RF": 1, "HR": 9}),
        ("build_multi_cell_squeeze_scenario", build_multi_cell_squeeze_scenario, {"GP": 7, "HC": 3}),
    ]

    all_ok = True
    for label, builder, demand in scenarios:
        print(f"\n---- 场景: {label} ----")
        vessel = builder()
        original_cbf = copy.deepcopy(vessel.cbf)
        snapshots = {}
        best = {"assigned": -1, "vessel": None}
        success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
        result_vessel = vessel if success else best["vessel"]

        new_list = build_tail_container_list(result_vessel, snapshots, original_cbf)
        gap_by_type = {rec["type"]: rec["count"] for rec in new_list}

        # 反推final_hc/final_hr：final = demand - gap（这几个场景demand >= final，
        # gap没有被max(0,.)截断，反推精确）
        final_hc = demand.get("HC", 0) - gap_by_type.get("HC", 0)
        final_hr = demand.get("HR", 0) - gap_by_type.get("HR", 0)
        my_hc_combined = final_hc + final_hr
        print(f"build_tail_container_list反推: final_hc={final_hc}, final_hr={final_hr}, "
              f"合计={my_hc_combined}")

        # 独立路径：scan_host_candidates按host聚合，hc_used=capacity_hc-hc_headroom，
        # 按(POL,POD)=(0,1)对应的所有host求和。
        host_pool = scan_host_candidates(result_vessel, snapshots)
        pod_hosts = {k: v for k, v in host_pool.items() if k[3] == 0 and k[4] == 1}
        host_hc_used_total = sum(v["capacity_hc"] - v["hc_headroom"] for v in pod_hosts.values())
        print(f"scan_host_candidates独立路径: 匹配的host={pod_hosts}, "
              f"hc_used合计={host_hc_used_total}")

        ok = (my_hc_combined == host_hc_used_total)
        print(f"[{'OK' if ok else 'MISMATCH'}] 两条独立路径算出的HC/HR标签总数"
              f"{'一致' if ok else '不一致'}: build_tail_container_list反推={my_hc_combined}, "
              f"scan_host_candidates独立算出={host_hc_used_total}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n[OK] 三个场景的交叉验证全部通过：build_tail_container_list与"
              "scan_host_candidates(独立代码路径)算出的HC/HR标签总数完全一致")
    else:
        print("\n[MISMATCH] 至少一个场景交叉验证失败，需要停下来查")
    return all_ok


def summarize_tail_by_port(tail_list: list, original_cbf: dict, port_names: dict = None) -> list:
    """把build_tail_container_list(或格式兼容的build_unified_tail_list)的flat list
    按POL汇总成"这港demand多少、装了多少、甩了多少"的每港报表，供main.py/调试脚本
    打印用，不用每次现场手写聚合逻辑。

    demand按original_cbf逐港求和(GP+HC+RF+HR四个字段加总)；甩货按tail_list里
    这个POL的count求和；已装=demand-甩货(下限截0，正常情况下gap<=demand不会触发
    截断，除非tail_list传入了不满足这个约束的自定义数据)。

    返回list[dict]，按POL升序排列，每条:
        {"POL", "port_name", "demand", "placed", "tail", "tail_rate"}
    port_names可选{POL: 名字}，不传则port_name=str(POL)。
    """
    demand_by_pol = {}
    for pol, pod_dict in original_cbf.items():
        total = sum(
            counts.get(k, 0)
            for counts in pod_dict.values()
            for k in ("GP", "HC", "RF", "HR")
        )
        demand_by_pol[pol] = demand_by_pol.get(pol, 0) + total

    tail_by_pol = {}
    for rec in tail_list:
        tail_by_pol[rec["POL"]] = tail_by_pol.get(rec["POL"], 0) + rec["count"]

    rows = []
    for pol in sorted(demand_by_pol.keys()):
        demand = demand_by_pol[pol]
        tail = tail_by_pol.get(pol, 0)
        placed = max(0, demand - tail)
        rate = (tail / demand) if demand else 0.0
        rows.append({
            "POL": pol,
            "port_name": port_names.get(pol, str(pol)) if port_names else str(pol),
            "demand": demand,
            "placed": placed,
            "tail": tail,
            "tail_rate": rate,
        })
    return rows


def print_tail_by_port(tail_list: list, original_cbf: dict, port_names: dict = None, label: str = ""):
    """打印summarize_tail_by_port()的结果，格式：
        POL=0(SHA): demand=120, 已装=115, 甩货=5 (甩货率=4.2%)
    末尾追加一行全船合计。label用于区分"新口径"/"旧口径"等标注。
    """
    rows = summarize_tail_by_port(tail_list, original_cbf, port_names)
    title = f"每港装/甩货明细{f'({label})' if label else ''}"
    print(f"\n{title}")
    print("─" * len(title) * 2)
    total_demand = total_placed = total_tail = 0
    for row in rows:
        print(f"  POL={row['POL']}({row['port_name']}): demand={row['demand']:>4}, "
              f"已装={row['placed']:>4}, 甩货={row['tail']:>4} (甩货率={row['tail_rate']:.1%})")
        total_demand += row["demand"]
        total_placed += row["placed"]
        total_tail += row["tail"]
    total_rate = (total_tail / total_demand) if total_demand else 0.0
    print(f"  {'─' * 40}")
    print(f"  全船合计: demand={total_demand:>4}, 已装={total_placed:>4}, "
          f"甩货={total_tail:>4} (甩货率={total_rate:.1%})")
    return rows


def verify_unified_tail_list():
    """跑一遍三个场景，核对build_unified_tail_list按source分组求和的结果
    是否跟已有验证函数/打印函数报告的三个来源总数完全对账，并人工打印完整
    列表供肉眼检查字段格式。只做核对，不做host匹配。"""
    print("\n" + "=" * 60)
    print("──── build_unified_tail_list 合并验证 ────")
    print("=" * 60)

    def _run_and_check(label, vessel_builder):
        print(f"\n---- 场景: {label} ----")
        vessel = vessel_builder()
        original_cbf = copy.deepcopy(vessel.cbf)

        snapshots = {}
        best = {"assigned": -1, "vessel": None}
        success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
        result_vessel = vessel if success else best["vessel"]
        print(f"solve()完成: success={success}")

        # final_cbf必须在print_source2_and_source3_tail(触发proj_cell_to_vessel
        # 写回)之前深拷贝，跟__main__块里的注释是同一个道理。
        final_cbf = copy.deepcopy(result_vessel.cbf)

        source1_total = print_source1_tail(final_cbf)
        source2_total, source3_total = print_source2_and_source3_tail(result_vessel, snapshots, original_cbf)

        dedup2 = _dedup_tail_log_by_pol_pod(result_vessel._tail_source2_log, key_len=2)
        dedup3 = _dedup_tail_log_by_pol_pod(result_vessel._tail_source3_log, key_len=2)

        unified = build_unified_tail_list(
            result_vessel, final_cbf, snapshots, original_cbf, dedup2=dedup2, dedup3=dedup3
        )

        by_source_sum = {1: 0, 2: 0, 3: 0}
        for rec in unified:
            by_source_sum[rec["source"]] += rec["count"]

        print(f"\n[{label}] build_unified_tail_list按source求和: {by_source_sum}")
        print(f"[{label}] 已打印/已验证的三个来源总数: 来源1={source1_total}, "
              f"来源2={source2_total}, 来源3={source3_total}")

        ok = (by_source_sum[1] == source1_total
              and by_source_sum[2] == source2_total
              and by_source_sum[3] == source3_total)
        print(f"[{'OK' if ok else 'MISMATCH'}] {label} 合并后按source求和对账")

        print(f"[{label}] build_unified_tail_list完整返回列表:")
        for rec in unified:
            print(f"  {rec}")

        # 诊断打印（非assert）：找出被多个来源同时命中的(POL,POD,type)分组。
        # 这种重叠是预期内的（例如来源1的tail_threshold残量和来源2的
        # deck-squeeze回退，统计的是同一个GP bucket的两套独立触发机制，
        # 不代表重复计数），所以这里只打印供人工确认"重叠看起来是否合理"，
        # 不做校验、不合并、不报错。特别标出"来源2/3重叠"，因为来源2固定
        # 映射到GP、来源3固定映射到HC/HR，正常情况下二者类型不交叉，
        # 如果观察到来源2和来源3同时命中同一个(POL,POD,type)，那才是真正
        # 需要停下来查的可疑信号，跟"来源1 vs 来源2/3重叠"的性质不同。
        groups = {}
        for rec in unified:
            key = (rec["POL"], rec["POD"], rec["type"])
            groups.setdefault(key, set()).add(rec["source"])
        overlaps = {key: sources for key, sources in groups.items() if len(sources) > 1}

        print(f"[{label}] 跨来源重叠诊断（同一(POL,POD,type)被多个source命中）:")
        if not overlaps:
            print("  （无重叠）")
        else:
            for key, sources in sorted(overlaps.items()):
                tag = "⚠ 来源2/3重叠（可疑，需要停下来查）" if sources >= {2, 3} else "来源1 vs 来源2/3重叠（预期内，独立触发机制统计同一bucket）"
                print(f"  (POL,POD,type)={key} 来源={sorted(sources)} —— {tag}")

        return ok

    ok1 = _run_and_check("build_test_scenario", lambda: build_test_scenario()[0])
    ok2 = _run_and_check("build_deck_squeeze_scenario", build_deck_squeeze_scenario)
    ok3 = _run_and_check("build_multi_pol_replay_scenario", build_multi_pol_replay_scenario)

    if ok1 and ok2 and ok3:
        print("\n[OK] 三个场景的build_unified_tail_list合并结果均与已有验证数字完全对账")
    else:
        print("\n[MISMATCH] 至少一个场景的合并结果对账失败，见上方逐场景输出")


if __name__ == "__main__":
    vessel, vessel_init = build_test_scenario()
    original_cbf = copy.deepcopy(vessel.cbf)

    snapshots = {}
    best = {"assigned": -1, "vessel": None}
    success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)

    result_vessel = vessel if success else best["vessel"]
    print(f"\nsolve()完成: success={success}")

    # 注意：final_cbf在这里(solve()刚结束、下面任何proj_cell_to_vessel调用之前)
    # 深拷贝读取，这样来源1读到的残量和来源2/3(靠重跑proj_cell_to_vessel触发)
    # 是同一个时间点的两个互斥切面——来源1读的是"proj_cell_to_vessel执行前"的cbf，
    # 来源2/3记录的是"执行proj_cell_to_vessel期间"新触发的写回，不会重叠也不会遗漏。
    final_cbf = copy.deepcopy(result_vessel.cbf)

    source1_total = print_source1_tail(final_cbf)
    source2_total, source3_total = print_source2_and_source3_tail(result_vessel, snapshots, original_cbf)

    print(f"\n──── 汇总 ────")
    print(f"[确认] 来源1读取时间点 = solve()刚结束、proj_cell_to_vessel执行之前"
          f"（final_cbf在调用print_source2_and_source3_tail之前已deepcopy完成）")
    print(f"来源1(小额尾货)总箱数: {source1_total}")
    print(f"来源2(deck-squeeze)总箱数: {source2_total}")
    print(f"来源3(HC/RF预算池分不完回退)总箱数: {source3_total}")
    print(f"合计尾箱: {source1_total + source2_total + source3_total}")

    fixture_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug", "fixtures")
    os.makedirs(fixture_dir, exist_ok=True)
    fixture_path = os.path.join(fixture_dir, "tail_fixture.pkl")
    with open(fixture_path, "wb") as f:
        pickle.dump({
            "snapshots": snapshots,
            "original_cbf": original_cbf,
            "final_cbf": final_cbf,
        }, f)
    print(f"\nfixture已落盘: {fixture_path}")

    # verify_deck_squeeze_scenario()
    # verify_multi_pol_replay_dedup()
    # verify_unified_tail_list()
    verify_build_tail_container_list_hold_example()
    verify_build_tail_container_list_rf_mirror()
    verify_multi_cell_squeeze_undercounting()
    verify_build_tail_container_list_cross_check()
    try:
        verify_scan_host_candidates()
    except Exception:
        # 已知问题（跟本次尾箱统计口径修复无关，见build_host_discharged_scenario
        # 调用处的TODO注释）：不在这里修，只是不让它中断脚本、挡住后面几个函数。
        import traceback
        traceback.print_exc()
        print("\n[跳过] verify_scan_host_candidates出现已知问题(见TODO注释)，"
              "继续跑后面的verify函数")
    verify_match_tails_to_hosts()
    verify_apply_tail_placements()