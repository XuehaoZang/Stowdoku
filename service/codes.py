"""
映射封装层：收拢 utils/vessel_io.py 里已有的港口/bay映射引用，
不重新实现任何映射规则本身，只做函数化封装供 translator/API层调用。

- port_code_to_num / port_num_to_code：三字码 <-> STSE_PORT_MAP 数字编号
- bay_idx_to_bay_id：bay_idx -> schema里的 Bay.bayId
- is_b0_bay_idx / big_bay_of_bay_idx：STSE_BAY_PAIRS 相关判定，复用 _BIG_BAY_OF_B0
- bay_number_physical：bay_idx -> schema里的 Bay.bayNumber（直接调用 idx_to_phy_bay）
"""

from utils.vessel_io import (
    STSE_PORT_MAP,
    STSE_BAY_PAIRS,
    _BIG_BAY_OF_B0,
    idx_to_phy_bay,
)
from service.errorcodes import ServiceError, ErrorCode

_PORT_NAMES = {v: k for k, v in STSE_PORT_MAP.items()}
_B0_BAY_IDXS = {b0 for b0, b1 in STSE_BAY_PAIRS}


def port_code_to_num(port_code: str) -> int:
    """三字码 -> STSE_PORT_MAP 数字编号。查不到抛 ServiceError(VOYAGE_PORT_NOT_SUPPORTED)。"""
    if port_code not in STSE_PORT_MAP:
        raise ServiceError(
            ErrorCode.VOYAGE_PORT_NOT_SUPPORTED,
            f"port code '{port_code}' 不在 STSE_PORT_MAP 中",
        )
    return STSE_PORT_MAP[port_code]


def port_num_to_code(port_num: int) -> str:
    """数字编号 -> 三字码。查不到抛 ServiceError(VOYAGE_PORT_NOT_SUPPORTED)。"""
    if port_num not in _PORT_NAMES:
        raise ServiceError(
            ErrorCode.VOYAGE_PORT_NOT_SUPPORTED,
            f"port num '{port_num}' 不在 STSE_PORT_MAP 中",
        )
    return _PORT_NAMES[port_num]


def bay_idx_to_bay_id(bay_idx: int) -> str:
    """bay_idx -> schema里的 Bay.bayId，格式稳定可读。"""
    return f"BAY-{int(bay_idx):02d}"


def is_b0_bay_idx(bay_idx: int) -> bool:
    """判断 bay_idx 是否是 STSE_BAY_PAIRS 里每对的 b0（用于过滤 metrics.byBay 只保留b0侧）。"""
    return bay_idx in _B0_BAY_IDXS


def big_bay_of_bay_idx(bay_idx: int):
    """bay_idx -> big_bay(0-6)。非配对bay（如bay_idx=0）返回 None。直接复用 _BIG_BAY_OF_B0。"""
    return _BIG_BAY_OF_B0.get(bay_idx)


def bay_number_physical(bay_idx: int) -> str:
    """bay_idx -> 真实物理Bay码，对应 schema 里的 Bay.bayNumber。直接调用 idx_to_phy_bay。"""
    return idx_to_phy_bay(bay_idx)
