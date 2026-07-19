"""
反腐层本体：单港snapshot + eval_results + Vessel静态几何 -> 单个BayPlanResult。

纯函数，不做7港循环、不落盘、不接HTTP（那些是上层业务服务层的职责，
见 service_api_design.md §3）。字段级转换规则逐条对应 service_api_design.md §1
的 TypeScript schema，具体依据见各函数内注释引用的 schema 行号语义。
"""

import logging
from datetime import datetime, timezone

import numpy as np

from utils.vessel_io import STSE_BAY_PAIRS, STSE_DECK_TIER
from service.codes import (
    port_code_to_num,
    port_num_to_code,
    bay_idx_to_bay_id,
    is_b0_bay_idx,
    big_bay_of_bay_idx,
    bay_number_physical,
)
from service.errorcodes import ServiceError, ErrorCode


def _to_native(value):
    """把numpy标量/数组/嵌套dict-list递归转换成原生Python类型，确保能json.dumps。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def _metric(key, label, value, unit=None):
    """构造一个MetricEntry。value为None时返回None（调用方负责跳过），
    因为schema里MetricEntry.value是非可选number，本版选择不产出该条目而不是塞None。
    """
    value = _to_native(value)
    if value is None:
        return None
    entry = {"key": key, "label": label, "value": value}
    if unit is not None:
        entry["unit"] = unit
    entry["status"] = None
    entry["thresholds"] = None
    return entry


def _build_slot(row, bay_id):
    can_20ft = bool(row["can_20ft"])
    can_40ft = bool(row["can_40ft"])
    pod = int(row["POD"])
    pol = int(row["POL"])

    data_integrity_flag = None
    if not can_20ft and not can_40ft:
        status = "UNAVAILABLE"
        unavailable_reason = "STRUCTURAL"
        # can_20ft/can_40ft均False的槽位本不该有货，但proj_cell_to_vessel的HC/HR
        # 二次分配启发式与full_slot_table物理能力表之间存在既有不一致（见
        # service/tests/test_translator.py 里hc_on_structural的记录），出现过POD/POL
        # 已赋值却落在STRUCTURAL槽位上的行。status仍按STRUCTURAL优先级处理不变，
        # 只是把这个信号打上标记，不静默丢弃。
        if pod != -1 or pol != -1:
            data_integrity_flag = "STRUCTURAL_OCCUPIED_CONFLICT"
    elif pod == -1:
        status = "EMPTY"
        unavailable_reason = None
    else:
        status = "OCCUPIED"
        unavailable_reason = None

    container = None
    if status == "OCCUPIED":
        is_20ft = bool(row["is_20ft"])
        is_hc = bool(row["is_hc"])
        size_prefix = "20" if is_20ft else "40"
        size_suffix = "HC" if is_hc else "GP"
        size_type = f"{size_prefix}{size_suffix}"

        attribute_flags = []
        if is_hc:
            attribute_flags.append("HIGH_CUBE")
        if int(row["RF_count"]) == 1:
            attribute_flags.append("REEFER")

        container = {
            "pod": port_num_to_code(pod),
            "pol": port_num_to_code(pol),
            "sizeType": size_type,
            "attributeFlags": attribute_flags,
        }

    row_idx = int(row["row_idx"])
    tier_idx = int(row["tier_idx"])
    return {
        "slotId": f"{bay_id}-R{row_idx:02d}-T{tier_idx:02d}",
        "row": row_idx,
        "tier": tier_idx,
        "status": status,
        "unavailableReason": unavailable_reason,
        "dataIntegrityFlag": data_integrity_flag,
        "capability": {"can20ft": can_20ft, "can40ft": can_40ft},
        "container": container,
    }


def _build_bays(df):
    bays = []
    for bay_idx, group in df.groupby("bay_idx"):
        bay_idx = int(bay_idx)
        bay_id = bay_idx_to_bay_id(bay_idx)
        slots = [_build_slot(row, bay_id) for _, row in group.iterrows()]
        bays.append({
            "bayId": bay_id,
            "bayNumber": bay_number_physical(bay_idx),
            "gridBounds": {
                "minRow": int(group["row_idx"].min()),
                "maxRow": int(group["row_idx"].max()),
                "minTier": int(group["tier_idx"].min()),
                "maxTier": int(group["tier_idx"].max()),
            },
            "deckBoundaryTier": STSE_DECK_TIER,
            "slots": slots,
            "_bay_idx": bay_idx,  # 内部排序用，最后剥离
        })
    bays.sort(key=lambda b: b["_bay_idx"])
    for b in bays:
        del b["_bay_idx"]
    return bays


def _find_port_entry(entries, port_num, metric_name):
    for entry in entries:
        if int(entry["pol"]) == port_num:
            return entry
    raise ServiceError(
        ErrorCode.PLAN_TRANSLATION_FAILED,
        f"eval_results['{metric_name}'] 中找不到 pol={port_num} 对应的条目",
    )


def _build_global_metrics(eval_results, port_num):
    metrics = []

    if "crane_time" in eval_results:
        ct = _find_port_entry(eval_results["crane_time"], port_num, "crane_time")
        for key, label, unit in [
            ("split", "吊车作业切分点(bay index)", None),
            ("work1", "吊车1作业量", None),
            ("wait1", "吊车1等待时长", "h"),
            ("time1", "吊车1作业耗时", "h"),
            ("work2", "吊车2作业量", None),
            ("wait2", "吊车2等待时长", "h"),
            ("time2", "吊车2作业耗时", "h"),
            ("makespan", "本港作业总耗时", "h"),
            ("utilization", "吊车利用率", None),
            ("time_port", "在港时长", "h"),
        ]:
            m = _metric(f"CRANE_TIME_{key.upper()}", label, ct.get(key), unit)
            if m is not None:
                metrics.append(m)

    if "crane_intensity" in eval_results:
        ci_entry = _find_port_entry(eval_results["crane_intensity"], port_num, "crane_intensity")
        m = _metric("CRANE_INTENSITY_CI", "吊车强度指数(CI)", ci_entry.get("ci"))
        if m is not None:
            metrics.append(m)

    if "ci_theoretical_ceiling" in eval_results:
        m = _metric(
            "CI_THEORETICAL_CEILING", "CI理论上限（仅取决于船体几何）",
            eval_results["ci_theoretical_ceiling"],
        )
        if m is not None:
            metrics.append(m)

    if "pod_discharge_spread" in eval_results:
        for pod, stats in eval_results["pod_discharge_spread"].items():
            pod_code = port_num_to_code(int(pod))
            for key, label in [
                ("variance", "到港分散度(方差)"),
                ("range", "到港分散度(极差)"),
                ("ci", "到港分散度(CI)"),
            ]:
                m = _metric(
                    f"POD_SPREAD_{key.upper()}_{pod_code}",
                    f"{pod_code} {label}",
                    stats.get(key),
                )
                if m is not None:
                    metrics.append(m)

    return metrics


def _build_by_bay_metrics(eval_results, port_num):
    by_bay = {}
    if "crane_time" not in eval_results:
        return by_bay

    ct = _find_port_entry(eval_results["crane_time"], port_num, "crane_time")
    discharge_tally = ct.get("discharge_tally")
    loading_tally = ct.get("loading_tally")
    bay_total = ct.get("bay_total")

    for big_bay_idx, (b0, _b1) in enumerate(STSE_BAY_PAIRS):
        bay_id = bay_idx_to_bay_id(b0)
        entries = []
        for key, label, arr in [
            ("DISCHARGE_TALLY", "本bay卸货量", discharge_tally),
            ("LOADING_TALLY", "本bay装货量", loading_tally),
            ("BAY_TOTAL", "本bay总作业量", bay_total),
        ]:
            if arr is None:
                continue
            m = _metric(key, label, arr[big_bay_idx])
            if m is not None:
                entries.append(m)
        if entries:
            by_bay[bay_id] = entries

    return by_bay


def _build_legend(voyage_leg_port_codes):
    legend = []
    for i, port_code in enumerate(voyage_leg_port_codes):
        legend.append({
            "groupKey": port_code,
            "label": port_code,
            "displayOrder": i,
            "kind": "POD",
        })
    return legend


def _build_voyage_legs(voyage_leg_port_codes):
    return [
        {"sequence": i, "portCode": pc, "portName": pc}
        for i, pc in enumerate(voyage_leg_port_codes)
    ]


def to_bay_plan_result(
    vessel,
    snapshot: dict,
    original_cbf: dict,
    eval_results: dict,
    run_id: str,
    voyage_id: str,
    port_code: str,
    voyage_leg_port_codes: list,
) -> dict:
    """单港快照 -> BayPlanResult（可直接json.dumps的dict）。纯函数，不做IO。"""
    try:
        port_num = port_code_to_num(port_code)

        df = vessel.proj_cell_to_vessel(cell_state=snapshot, original_cbf=original_cbf)

        bays = _build_bays(df)
        bay_ids_in_order = [bay_idx_to_bay_id(idx) for idx in sorted(df["bay_idx"].unique())]

        result = {
            "planId": f"{run_id}-{port_code}",
            "runId": run_id,
            "voyageId": voyage_id,
            "portCode": port_code,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "vessel": {
                "vesselId": voyage_id,
                "vesselName": voyage_id,
                "bayOrder": bay_ids_in_order,
            },
            "voyageLegs": _build_voyage_legs(voyage_leg_port_codes),
            "legend": _build_legend(voyage_leg_port_codes),
            "bays": bays,
            "metrics": {
                "global": _build_global_metrics(eval_results, port_num),
                "byBay": _build_by_bay_metrics(eval_results, port_num),
            },
        }
        return result
    except ServiceError:
        raise
    except Exception as exc:
        logging.exception("to_bay_plan_result 转换失败：port_code=%s run_id=%s", port_code, run_id)
        raise ServiceError(
            ErrorCode.PLAN_TRANSLATION_FAILED,
            f"port_code={port_code} 转换失败: {exc}",
        ) from exc
