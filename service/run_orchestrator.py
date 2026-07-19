"""
业务服务层：一次Run的编排 —— 调用solve() -> 对7港循环调用translator -> 落盘。

不改 solve()/VesselClass.py/utils/evaluate.py 的行为，只编排调用顺序和异常转译。
vessel 由调用方构造好传入（Vessel.load_vessel() 之后、solve() 之前的状态），
本函数内部会对其原地调用 solve()，调用方不应在调用后复用同一个 vessel 实例。
"""

import copy
import logging

from CSP_solver import solve
from utils.evaluate import (
    evaluate_crane_time,
    evaluate_crane_intensity,
    evaluate_pod_discharge_spread,
    evaluate_ci_theoretical_ceiling,
)
from service.translator import to_bay_plan_result
from service.codes import port_num_to_code
from service.errorcodes import ServiceError, ErrorCode
from service import storage


def execute_run(vessel, run_id: str, voyage_id: str, voyage_leg_port_codes: list) -> dict:
    original_cbf = copy.deepcopy(vessel.cbf)
    snapshots = {}
    best = {"assigned": -1, "vessel": None}

    try:
        success = solve(vessel, is_debug=False, snapshots=snapshots, best=best)
    except Exception as exc:
        logging.exception(
            "execute_run: solve()异常 run_id=%s voyage_id=%s", run_id, voyage_id
        )
        raise ServiceError(ErrorCode.PLAN_INFEASIBLE, f"solve()异常: {exc}") from exc

    result_vessel = vessel if success else best["vessel"]
    if result_vessel is None or not snapshots:
        logging.error(
            "execute_run: solve()未产出可行解 run_id=%s voyage_id=%s success=%s",
            run_id, voyage_id, success,
        )
        raise ServiceError(ErrorCode.PLAN_INFEASIBLE, "solve()未能产出任何可行解")

    port_names = {pol: port_num_to_code(pol) for pol in snapshots.keys()}

    try:
        eval_results = {
            "crane_time": evaluate_crane_time(
                result_vessel, snapshots, k=2, crane_rate=1.0, port_names=port_names
            ),
            "crane_intensity": evaluate_crane_intensity(
                result_vessel, snapshots, port_names=port_names
            ),
            "pod_discharge_spread": evaluate_pod_discharge_spread(
                result_vessel, snapshots, port_names=port_names
            ),
            "ci_theoretical_ceiling": evaluate_ci_theoretical_ceiling(result_vessel),
        }
    except Exception as exc:
        logging.exception(
            "execute_run: evaluate_xxx异常 run_id=%s voyage_id=%s", run_id, voyage_id
        )
        raise ServiceError(ErrorCode.PLAN_INFEASIBLE, f"evaluate_xxx异常: {exc}") from exc

    plans = []
    for pol in sorted(snapshots.keys()):
        port_code = port_num_to_code(pol)
        try:
            bay_plan_result = to_bay_plan_result(
                vessel=result_vessel,
                snapshot=snapshots[pol],
                original_cbf=original_cbf,
                eval_results=eval_results,
                run_id=run_id,
                voyage_id=voyage_id,
                port_code=port_code,
                voyage_leg_port_codes=voyage_leg_port_codes,
            )
        except ServiceError:
            logging.exception(
                "execute_run: translator失败 run_id=%s port_code=%s", run_id, port_code
            )
            raise
        except Exception as exc:
            logging.exception(
                "execute_run: translator未预期异常 run_id=%s port_code=%s", run_id, port_code
            )
            raise ServiceError(
                ErrorCode.PLAN_TRANSLATION_FAILED,
                f"port_code={port_code} 转换失败: {exc}",
            ) from exc

        plan_id = storage.save_plan(run_id, port_code, bay_plan_result)
        plans.append({"portCode": port_code, "planId": plan_id, "status": "COMPLETED"})

    storage.save_run_index(run_id, plans)
    return storage.load_run_index(run_id)
