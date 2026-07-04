"""
验证ASC解析：parse_iso_code / parse_asc_header / parse_asc_lines /
parse_asc_file / batch_parse_asc 跑一遍真实样本，不调用solve()。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.vessel_io import (
    parse_iso_code, parse_asc_header, parse_asc_file, batch_parse_asc, STSE_PORT_MAP,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "STSE", "raw")
BAYPLAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "STSE", "bayplan")

# ── parse_iso_code ───────────────────────────────────────────────────────
assert parse_iso_code("40HC") == (40, 9.5, "HC")
assert parse_iso_code("PF20") == (20, None, "PF")  # 尺寸在后的倒序写法，height未登记
assert parse_iso_code("20G0") == (20, 8.5, "GP")  # G0归一化成GP
assert parse_iso_code("") == (None, None, None)
print("parse_iso_code 通过")

# ── parse_asc_header ──────────────────────────────────────────────────────
sample_file = os.path.join(RAW_DIR, "STSE_0_SHP_DEP.ASC")
with open(sample_file, "r", encoding="latin-1") as f:
    lines = [l.rstrip("\r\n") for l in f if l.strip()]
header = parse_asc_header(lines)
print("header:", header)
assert header["pol"] == "SHP"
assert header["record_count"] == 178

# ── parse_asc_file（单文件） ───────────────────────────────────────────────
df, pol_code = parse_asc_file(sample_file)
print(df.head())
assert list(df.columns) == [
    "bay_idx", "row_idx", "tier_idx", "POL", "POD",
    "length", "height", "type", "weight", "status", "is_IMDG",
]
assert pol_code == "SHP"
assert df["weight"].dtype.kind in "iu"  # 全部是整数，缺失/非数字已归零
assert set(df["status"].unique()) <= {"F", "E", ""}
print(f"单文件解析通过，{len(df)}行，weight范围[{df.weight.min()},{df.weight.max()}]")

# ── batch_parse_asc（全量，含RECORD行数校验、trailer过滤、文件名覆盖警告） ──
batch_parse_asc(RAW_DIR, BAYPLAN_DIR)
out_files = sorted(os.listdir(BAYPLAN_DIR))
print("bayplan输出:", out_files)

print("全部通过")