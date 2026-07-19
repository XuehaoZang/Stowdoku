"""
业务错误枚举 + 统一业务异常基类。

给 translator / API层 raise 用，只覆盖本阶段实际用到的错误码，
对应 service_api_design.md 里出现的 errorCode 值。
"""


class ErrorCode:
    # POST /voyages：portCode 不在 STSE_PORT_MAP 中
    VOYAGE_PORT_NOT_SUPPORTED = "VOYAGE_PORT_NOT_SUPPORTED"
    # POST /voyages/{voyageId}/cbf：解析结果为空/关键字段缺失
    CBF_FORMAT_INVALID = "CBF_FORMAT_INVALID"
    # GET /voyages/{voyageId}/runs/{runId}：solve() 失败，无可行解
    PLAN_INFEASIBLE = "PLAN_INFEASIBLE"
    # translator.to_bay_plan_result 内部遇到未预期的输入/结构问题
    PLAN_TRANSLATION_FAILED = "PLAN_TRANSLATION_FAILED"


class ServiceError(Exception):
    """统一业务异常基类。code 取值应为 ErrorCode 中定义的枚举字符串。"""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
