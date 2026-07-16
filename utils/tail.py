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

import pandas as pd

from VesselClass import Vessel
from CSP_solver import solve


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
    """来源1：遍历最终vessel.cbf，打印每个非零的(POL, POD, GP/HC/RF/HR, 数量)。"""
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
    """来源2+来源3：对每个POL只重跑一次proj_cell_to_vessel（两条日志在同一次调用里
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

    verify_deck_squeeze_scenario()
    verify_multi_pol_replay_dedup()
