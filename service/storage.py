"""
文件存储：result.json / index.json 的落盘与读取。

路径规则见 service_api_design.md §4：
    storage/runs/{runId}/plans/{portCode}/result.json
    storage/runs/{runId}/index.json

只做文件IO，不调用 translator、不做业务校验；calls 传入的 dict 需自行保证
能被 json.dumps 序列化（translator.to_bay_plan_result 的输出已满足这点）。
"""

import json
import os

STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage"
)


def _run_dir(run_id: str) -> str:
    return os.path.join(STORAGE_ROOT, "runs", run_id)


def _plan_dir(run_id: str, port_code: str) -> str:
    return os.path.join(_run_dir(run_id), "plans", port_code)


def _plan_path(run_id: str, port_code: str) -> str:
    return os.path.join(_plan_dir(run_id, port_code), "result.json")


def _index_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "index.json")


def save_plan(run_id: str, port_code: str, bay_plan_result: dict) -> str:
    """落盘单港 BayPlanResult，返回 planId（{runId}-{portCode}）。"""
    plan_dir = _plan_dir(run_id, port_code)
    os.makedirs(plan_dir, exist_ok=True)
    with open(_plan_path(run_id, port_code), "w", encoding="utf-8") as f:
        json.dump(bay_plan_result, f, ensure_ascii=False, indent=2)
    return f"{run_id}-{port_code}"


def load_plan(plan_id: str) -> dict:
    """读回单个 planId 对应的 result.json。planId 格式固定为 {runId}-{portCode}，
    portCode 恒为不含'-'的三字码，故按最后一个'-'切分即可还原 runId/portCode。"""
    run_id, port_code = plan_id.rsplit("-", 1)
    with open(_plan_path(run_id, port_code), "r", encoding="utf-8") as f:
        return json.load(f)


def save_run_index(run_id: str, plans: list) -> None:
    """落盘该次Run的港口目录（portCode/planId/status列表），对应GET /runs/{runId}的响应内容。"""
    run_dir = _run_dir(run_id)
    os.makedirs(run_dir, exist_ok=True)
    index = {"runId": run_id, "status": "COMPLETED", "plans": plans}
    with open(_index_path(run_id), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_run_index(run_id: str) -> dict:
    with open(_index_path(run_id), "r", encoding="utf-8") as f:
        return json.load(f)
