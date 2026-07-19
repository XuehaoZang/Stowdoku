"""
service/errorcodes.py 单测：异常类携带code/message，且错误码与 service_api_design.md 对得上。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from service.errorcodes import ServiceError, ErrorCode

# ── ServiceError 正确携带 code/message ──
err = ServiceError(ErrorCode.CBF_FORMAT_INVALID, "解析结果为空")
assert err.code == "CBF_FORMAT_INVALID"
assert err.message == "解析结果为空"
assert isinstance(err, Exception)
assert "CBF_FORMAT_INVALID" in str(err)
assert "解析结果为空" in str(err)

# ── 三个错误码与 service_api_design.md 里出现的字符串完全一致 ──
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "service_api_design.md")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc_text = f.read()

for code in (
    ErrorCode.VOYAGE_PORT_NOT_SUPPORTED,
    ErrorCode.CBF_FORMAT_INVALID,
    ErrorCode.PLAN_INFEASIBLE,
):
    assert code in doc_text, f"{code} 未在 service_api_design.md 中出现"

print("service/tests/test_errorcodes.py: 全部通过")
