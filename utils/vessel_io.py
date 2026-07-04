"""
STSE船型几何提取。

两层结构：
    1. build_vessel_geometry(): 从idx版槽位csv读出完整slot表，加lr/hd两列坐标派生信息。
    2. find_can_40ft / find_can_20ft / find_can_reefer(): 在完整slot表上各自打一个bool子集标签列。
    3. build_vessel_cell(): 聚合成(7,2,2)的cell田字格，给solver

坐标系见 VesselClass.py 顶部：
    lr: 0=left(row_idx 0-4), 1=right(row_idx 5-10)
    hd: 0=hold(tier_idx 0-3), 1=deck(tier_idx 4-9)

bay映射（真实物理 -> 0-base idx）：
    phy Bay01 -> idx0（只放20ft，不参与大箱pair）
    phy Bay03 -> idx2, Bay05 -> idx3, ... Bay29 -> idx15
    40ft大箱占用相邻一对(idx偶,idx偶+1)，即STSE_BAY_PAIRS里的7对，
    重新编号为big_bay 0..6（用pair里的第一个bay_idx，即b0，作为代表）。
"""

import os
import re
import json
import numpy as np
import pandas as pd

# ── STSE专属硬编码常量 ──────────────────────────────────────────────────

# 有7个大Bay，能放40ft大箱，每个大箱占两个Bay编号，记为B0/B1
STSE_BAY_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
_BIG_BAY_OF_B0 = {b0: i for i, (b0, b1) in enumerate(STSE_BAY_PAIRS)}
_PAIRED_BAYS = {b for pair in STSE_BAY_PAIRS for b in pair}
N_BAY = len(STSE_BAY_PAIRS)

STSE_ROW_LABELS = ["10", "08", "06", "04", "02", "00", "01", "03", "05", "07", "09"]
STSE_TIER_LABELS = ["02", "04", "06", "08", "82", "84", "86", "88", "90", "92"]

STSE_HATCH_SPLIT = 5   # row_idx < 5 -> left, >=5 -> right
STSE_DECK_TIER = 4     # tier_idx < 4 -> hold, >=4 -> deck

STSE_PORT_MAP = {"SHP": 0, "TXG": 1, "DLC": 2, "YKK": 3, "NGO": 4, "TYO": 5, "YOK": 6, "LYG": 7}
 
_ISO_TYPE_HEIGHT = {"GP": 8.5, "HC": 9.5, "RF": 9.5, "HR": 9.5, "FR": 8.5, "FP": 8.5, "OT": 8.5, "TK": 8.5, "TG": 8.5}
 
BAYPLAN_COLUMNS = [
    "bay_idx", "row_idx", "tier_idx", "POL", "POD",
    "length", "height", "type", "weight", "status", "is_IMDG",
]

# ── 解析船只几何信息 ──────────────────────────────────────────────────

def phy_to_idx(bay_p, row_p, tier_p):
    """真实物理坐标(Bay奇数/Row/Tier) -> 0-base索引(bay_idx,row_idx,tier_idx)。"""
    bay_p = int(bay_p)
    b_idx = (bay_p - 1) // 2
    if bay_p > 1:
        b_idx += 1
    r_idx = STSE_ROW_LABELS.index(str(int(row_p)).zfill(2))
    t_idx = STSE_TIER_LABELS.index(str(int(tier_p)).zfill(2))
    return b_idx, r_idx, t_idx


def build_vessel_geometry(idx_csv_path) -> pd.DataFrame:
    """读取idx版槽位csv，返回完整slot表，不做任何bay过滤。
    列：bay_idx, row_idx, tier_idx, lr, hd"""
    df = pd.read_csv(idx_csv_path).rename(
        columns={"Bay": "bay_idx", "Row": "row_idx", "Tier": "tier_idx"}
    )
    df["lr"] = (df["row_idx"] >= STSE_HATCH_SPLIT).astype(int)
    df["hd"] = (df["tier_idx"] >= STSE_DECK_TIER).astype(int)
    return df


def find_can_40ft(slots: pd.DataFrame) -> pd.DataFrame:
    """加can_40ft列：True仅标在每对STSE_BAY_PAIRS的b0一侧，且(row,tier)在两侧交集内
    （只标b0一侧是为了不把同一个40ft槽位重复计两遍）。"""
    slots = slots.copy()
    can_40ft_keys = set()
    for b0, b1 in STSE_BAY_PAIRS:
        rt_0 = set(map(tuple, slots.loc[slots.bay_idx == b0, ["row_idx", "tier_idx"]].values))
        rt_1 = set(map(tuple, slots.loc[slots.bay_idx == b1, ["row_idx", "tier_idx"]].values))
        for r, t in rt_0 & rt_1:
            can_40ft_keys.add((b0, r, t))
    keys = list(zip(slots.bay_idx, slots.row_idx, slots.tier_idx))
    slots["can_40ft"] = [k in can_40ft_keys for k in keys]
    return slots


def find_can_20ft(slots: pd.DataFrame) -> pd.DataFrame:
    """加can_20ft列：只放不了大箱的位置才算can_20ft。
    - bay_idx=0（无配对）：全部can_20ft=True
    - 配对bay（b0/b1两侧都算）：(row,tier)不在两侧交集内的，各自那一侧can_20ft=True
    can_40ft和can_20ft在完整slot表上互斥且覆盖全部有效槽位。
    """
    slots = slots.copy()
    in_intersection = set()  # (bay_idx,row,tier)：属于某个pair交集的所有位置，两侧都记
    for b0, b1 in STSE_BAY_PAIRS:
        rt_0 = set(map(tuple, slots.loc[slots.bay_idx == b0, ["row_idx", "tier_idx"]].values))
        rt_1 = set(map(tuple, slots.loc[slots.bay_idx == b1, ["row_idx", "tier_idx"]].values))
        for r, t in rt_0 & rt_1:
            in_intersection.add((b0, r, t))
            in_intersection.add((b1, r, t))

    def _can_20(bay_idx, row_idx, tier_idx):
        if bay_idx not in _PAIRED_BAYS:
            return True
        return (bay_idx, row_idx, tier_idx) not in in_intersection

    slots["can_20ft"] = [
        _can_20(b, r, t) for b, r, t in zip(slots.bay_idx, slots.row_idx, slots.tier_idx)
    ]
    return slots


def find_can_reefer(slots: pd.DataFrame) -> pd.DataFrame:
    """占位：reefer位置数据未接入，先全部标False。等人工reefer位置数据就绪后重写。"""
    slots = slots.copy()
    slots["can_reefer"] = False
    return slots


def build_vessel_cell(slots: pd.DataFrame, flag_col: str) -> np.ndarray:
    """把slots按flag_col筛选后，聚合成(7,2,2)的cell田字格。
    """
    cell = np.zeros((N_BAY, 2, 2), dtype=int)
    hits = slots[slots[flag_col]]
    for bay_idx, lr, hd in zip(hits.bay_idx, hits.lr, hits.hd):
        big_bay = _BIG_BAY_OF_B0.get(bay_idx)
        if big_bay is None:
            continue
        cell[big_bay, lr, hd] += 1
    return cell

def build_init_state(slots: pd.DataFrame) -> pd.DataFrame:
    """空船初始状态：船上还没有任何箱子"""
    return pd.DataFrame(columns=BAYPLAN_COLUMNS)


def proj_vessel_to_cell(slots: pd.DataFrame):
    """
    slot级 -> cell级(N_BAY,2,2)投影。
    要求slots已含can_40ft/can_reefer列（build_vessel_geometry + find_can_40ft +
    find_can_reefer之后的产物），聚合出Vessel构造需要的三个静态几何数组。
    返回 (is_valid, capacity_total, capacity_rf)。
    """
    capacity_total = build_vessel_cell(slots, "can_40ft")
    is_valid = capacity_total > 0
    slots = slots.copy()
    slots["can_reefer_40ft"] = slots["can_40ft"] & slots["can_reefer"]
    capacity_rf = build_vessel_cell(slots, "can_reefer_40ft")
    return is_valid, capacity_total, capacity_rf


# ── CBF解析（.cbf原始货量文件 -> 汇总csv） ───────────────────────────────
 
_CBF_DATA_LINE_RE = re.compile(r"^46\s+([A-Z]{2}[A-Z]{3})\s+([A-Z0-9]{4})\s+(\d+)")
 
def parse_cbf_file(cbf_path) -> pd.DataFrame:
    """解析单个.cbf文件(CASP导出的货量汇总)，返回列: POD, length, type, count。"""
    counts = {}
    with open(cbf_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.startswith("46"):
                continue
            m = _CBF_DATA_LINE_RE.match(line)
            if not m:
                print(f"警告: {cbf_path} 无法解析的46行: {line!r}")
                continue
            pod_raw, iso_raw, count_str = m.groups()
            pod_code = pod_raw[2:]
            length, _, box_type = parse_iso_code(iso_raw)
            key = (pod_code, length if length is not None else "UNKNOWN", box_type)
            counts[key] = counts.get(key, 0) + int(count_str)
 
    rows = [
        {"POD": STSE_PORT_MAP.get(pod_code, -1), "length": length, "type": box_type, "count": count}
        for (pod_code, length, box_type), count in counts.items()
    ]
    return pd.DataFrame(rows, columns=["POD", "length", "type", "count"])

def _bucket_type(box_type):
    """把箱型归入solver认识的GP/RF两类。RF/HR是冷藏箱归RF，其余(GP/HC/PF/TK/OT/未识别)一律归GP。"""
    return "RF" if box_type in ("RF", "HR") else "GP"

def cbf_df_to_dict(df: pd.DataFrame) -> dict:
    """把parse_cbf_file的输出(POD,length,type,count)转成{POD:{"GP":n,"RF":n}}。
    20ft和40ft按 20ft//2 + 40ft 合并成40ft槽位数(2个20ft算1个大箱)；
    length='UNKNOWN'(ISO无法识别)的行没法换算，跳过并警告。"""
    result = {}
    for pod, group in df.groupby("POD"):
        totals = {"GP": 0, "RF": 0}
        for _, row in group.iterrows():
            if row.length == "UNKNOWN":
                print(f"警告: POD={pod} type={row.type} length无法识别，count={row['count']}已跳过")
                continue
            slots = row["count"] // 2 if row.length == 20 else row["count"]
            totals[_bucket_type(row.type)] += int(slots)
        result[int(pod)] = totals
    return result

def batch_parse_cbf(raw_dir, cbf_dir):
    """遍历raw_dir下所有.cbf文件，从文件名取最后一段(空格/下划线分隔)作为POL三字码，
    查STSE_PORT_MAP编号，汇总成{POL:{POD:{"GP":n,"RF":n}}}，存一份cbf.json。"""
    os.makedirs(cbf_dir, exist_ok=True)
    cbf = {}
    written = {}
    for fname in os.listdir(raw_dir):
        if not fname.upper().endswith(".CBF"):
            continue
        pol_code = re.split(r"[ _]+", os.path.splitext(fname)[0])[-1].upper()
        if pol_code not in STSE_PORT_MAP:
            print(f"警告: {fname} 文件名末段POL='{pol_code}' 不在STSE_PORT_MAP里，跳过")
            continue
 
        df = parse_cbf_file(os.path.join(raw_dir, fname))
        pol_num = STSE_PORT_MAP[pol_code]
        if pol_num in written:
            print(f"警告: POL={pol_num}({pol_code}) 被 {fname} 覆盖（之前来自 {written[pol_num]}）")
        written[pol_num] = fname
        cbf[pol_num] = cbf_df_to_dict(df)
 
    cbf = dict(sorted(cbf.items()))
    out_path = os.path.join(cbf_dir, "cbf.json")
    with open(out_path, "w") as f:
        json.dump(cbf, f, indent=2)
    return cbf
        
# ── ASC解析（.ASC原始配载文件 -> bayplan csv） ──────────────────────────
 
def parse_iso_code(iso_raw):
    """ISO代码 -> (length, height, type)。正则拆解，不查精确字典。"""
    iso_raw = (iso_raw or "").strip()
    if not iso_raw:
        return None, None, None
 
    m = re.match(r"^(\d{2})([A-Z0-9]{2})$", iso_raw) or re.match(r"^([A-Z]{2})(\d{2})$", iso_raw)
    if not m:
        print(f"警告: 无法识别的ISO代码 '{iso_raw}'")
        return None, None, None
 
    g1, g2 = m.groups()
    length_str, type_str = (g1, g2) if g1.isdigit() else (g2, g1)
    length = int(length_str)
    box_type = "GP" if type_str == "G0" else type_str
    height = _ISO_TYPE_HEIGHT.get(box_type)
    if height is None:
        print(f"警告: 未登记的箱型 '{box_type}' (来自ISO '{iso_raw}')")
    return length, height, box_type
 
 
def parse_asc_header(lines) -> dict:
    """解析.ASC前2行header，返回{'pol': str, 'record_count': int}。"""
    fields = lines[0].strip().split("/")
    pol = None
    record_count = None
    for f in fields:
        if f.startswith("POL:"):
            pol = f[4:].strip()
        elif f.startswith("RECORD="):
            record_count = int(f[len("RECORD="):])
    return {"pol": pol, "record_count": record_count}
 
 
def parse_asc_lines(data_lines) -> pd.DataFrame:
    rows = []
    for l in data_lines:
        weight_raw = l[48:51].strip()
        rows.append({
            "bay_p": int(l[0:2]), "row_p": int(l[2:4]), "tier_p": int(l[4:6]),
            "pol_raw": l[27:30].strip(), "pod_raw": l[30:33].strip(),
            "iso_raw": l[44:48].strip(),
            "weight": int(weight_raw) if weight_raw.isdigit() else 0,
            "status": l[51:52].strip(),
            "imdg_raw": l[60:64].strip(),
        })
    return pd.DataFrame(rows)
 
 
def parse_asc_file(asc_path) -> pd.DataFrame:
    """解析单个.ASC文件，返回列: bay_idx,row_idx,tier_idx,POL,POD,length,height,type,weight,status,is_IMDG。"""
    with open(asc_path, "r", encoding="latin-1") as f:
        lines = [line.rstrip("\r\n") for line in f if line.strip()]
 
    header = parse_asc_header(lines)
    data_lines = []
    for l in lines[2:]:
        if not l[:2].isdigit():
            continue
        if int(l[:2]) == 0:
            break  # Bay=00是trailer，不是货物数据
        data_lines.append(l)
 
    if header["record_count"] is not None and len(data_lines) != header["record_count"]:
        raise ValueError(
            f"{asc_path}: 行数校验失败，RECORD={header['record_count']}，实际读到{len(data_lines)}行"
        )
 
    df = parse_asc_lines(data_lines)
 
    df["bay_idx"], df["row_idx"], df["tier_idx"] = zip(
        *df.apply(lambda x: phy_to_idx(x.bay_p, x.row_p, x.tier_p), axis=1)
    )
    df["POL"] = df["pol_raw"].map(STSE_PORT_MAP).fillna(-1).astype(int)
    df["POD"] = df["pod_raw"].map(STSE_PORT_MAP).fillna(-1).astype(int)
 
    iso_parsed = df["iso_raw"].apply(parse_iso_code)
    df["length"] = [t[0] for t in iso_parsed]
    df["height"] = [t[1] for t in iso_parsed]
    df["type"] = [t[2] for t in iso_parsed]
 
    df["is_IMDG"] = df["imdg_raw"].str.len() > 0
 
    return df[["bay_idx", "row_idx", "tier_idx", "POL", "POD", "length", "height", "type", "weight", "status", "is_IMDG"]], header["pol"]
 
 
def batch_parse_asc(raw_dir, bayplan_dir):
    """遍历raw_dir下所有.ASC文件，按header里的POL(通过STSE_PORT_MAP编号)存到bayplan_dir，
    命名规则: {编号}_{POL三字码}_DEP.csv。"""
    os.makedirs(bayplan_dir, exist_ok=True)
    written = {}
    for fname in os.listdir(raw_dir):
        if not fname.upper().endswith(".ASC"):
            continue
        try:
            df, pol_code = parse_asc_file(os.path.join(raw_dir, fname))
        except ValueError as e:
            print(f"警告: {fname} 解析失败，跳过 ({e})")
            continue
        if pol_code not in STSE_PORT_MAP:
            print(f"警告: {fname} header POL='{pol_code}' 不在STSE_PORT_MAP里，跳过")
            continue
        out_name = f"{STSE_PORT_MAP[pol_code]}_{pol_code}_DEP.csv"
        out_path = os.path.join(bayplan_dir, out_name)
        if pol_code in written:
            print(f"警告: '{out_name}' 被 {fname} 覆盖（之前来自 {written[pol_code]}）")
        written[pol_code] = fname
        df.to_csv(out_path, index=False)