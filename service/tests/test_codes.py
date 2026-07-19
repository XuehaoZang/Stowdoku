"""
service/codes.py 单测：对 STSE_PORT_MAP / STSE_BAY_PAIRS 全量做round-trip和边界验证。
用assert而非pytest，风格对齐 debug/test_vessel_geometry.py，可直接 python 运行。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.vessel_io import STSE_PORT_MAP, STSE_BAY_PAIRS
from service.codes import (
    port_code_to_num,
    port_num_to_code,
    bay_idx_to_bay_id,
    is_b0_bay_idx,
    big_bay_of_bay_idx,
    bay_number_physical,
)
from service.errorcodes import ServiceError, ErrorCode

# ── port_code_to_num / port_num_to_code round-trip（全部真实港口码） ──
for code, num in STSE_PORT_MAP.items():
    assert port_code_to_num(code) == num, f"port_code_to_num({code}) != {num}"
    assert port_num_to_code(num) == code, f"port_num_to_code({num}) != {code}"

# ── 未知港口码/编号：抛 ServiceError(VOYAGE_PORT_NOT_SUPPORTED) ──
try:
    port_code_to_num("ZZZ")
    assert False, "应抛出 ServiceError"
except ServiceError as e:
    assert e.code == ErrorCode.VOYAGE_PORT_NOT_SUPPORTED

try:
    port_num_to_code(999)
    assert False, "应抛出 ServiceError"
except ServiceError as e:
    assert e.code == ErrorCode.VOYAGE_PORT_NOT_SUPPORTED

# ── STSE_BAY_PAIRS 全部真实pair：is_b0_bay_idx / big_bay_of_bay_idx ──
for i, (b0, b1) in enumerate(STSE_BAY_PAIRS):
    assert is_b0_bay_idx(b0) is True, f"b0={b0} 应为 True"
    assert is_b0_bay_idx(b1) is False, f"b1={b1} 应为 False"
    assert big_bay_of_bay_idx(b0) == i, f"big_bay_of_bay_idx({b0}) != {i}"

# b1侧不在 _BIG_BAY_OF_B0 里，big_bay_of_bay_idx应返回None
for b0, b1 in STSE_BAY_PAIRS:
    assert big_bay_of_bay_idx(b1) is None, f"b1={b1} 的big_bay应为None"

# ── bay_idx=0（纯20ft bay，非配对）── big_bay_of_bay_idx返回None，is_b0_bay_idx为False
assert big_bay_of_bay_idx(0) is None
assert is_b0_bay_idx(0) is False

# ── bay_idx_to_bay_id：格式稳定可读 ──
assert bay_idx_to_bay_id(0) == "BAY-00"
assert bay_idx_to_bay_id(15) == "BAY-15"

# ── bay_number_physical：直接透传 idx_to_phy_bay ──
assert bay_number_physical(0) == "01"
assert bay_number_physical(2) == "03"
assert bay_number_physical(15) == "29"

print("service/tests/test_codes.py: 全部通过")
