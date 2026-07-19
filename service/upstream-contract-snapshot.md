# 上游契约快照

> 现状记录文档，不含改造建议、不含schema映射方案。数据来自一次真实跑通：
> `main.py` 的 `ensure_geometry()` + `ensure_cbf()` 数据准备逻辑 + `Vessel.load_vessel()`
> 加载真实STSE数据（`data/STSE/geometry/full_slot_table.csv` + `data/STSE/cbf/cbf.json`），
> `random.seed(8245)`，`solve(vessel, is_debug=False, snapshots=snapshots, best=best)`，
> 与 `debug/tail_real_run.py` 同款调用方式。船体7个big bay，全船1003个slot。
>
> 目的：给后续写 `translator.py` 的人提供唯一真相源，不用重新翻源码猜数据结构。

---

## 1. `Vessel.proj_cell_to_vessel()` / `evaluate_xxx` / `solve()` snapshot 的真实结构

### 1.1 `solve()` 产出的 `snapshots`

`solve(vessel, is_debug=False, snapshots=snapshots, best=best)` 中 `snapshots` 是调用方传入的**空dict**，`solve()` 内部原地写入（`CSP_solver.py:243` `snapshots[vessel.current_pol] = vessel.snapshot()`，港口回溯失败时 `CSP_solver.py:257` `del snapshots[vessel.current_pol]`）。

真实跑完（7港航次）后：

```python
type(snapshots)        # <class 'dict'>
sorted(snapshots.keys())  # [0, 1, 2, 3, 4, 5, 6]  —— 每个key是一个POL（int），不是list
```

**是多个快照，每港一个**（departure时刻的快照），不是单个快照。key就是港口编号（int，对应 `STSE_PORT_MAP` 的value）。

每个 `snapshots[pol]` 是 `Vessel.snapshot()`（`VesselClass.py:356`）的返回值，真实结构：

```python
snapshots[0].keys()   # dict_keys(['cell', 'cbf', 'current_pol'])

type(snapshots[0]["cell"])         # <class 'numpy.ndarray'>
snapshots[0]["cell"].shape         # (7, 2, 2)  —— (bay, lr, hd)，n_bay=7为STSE船型固定值

type(snapshots[0]["cbf"])          # <class 'dict'>  —— 该港departure时刻的cbf切面（当时剩余需求）
snapshots[0]["current_pol"]        # 0  (int)
```

`cell` 数组每个元素是一个 `dict`，**字段数量不固定**——已赋值的cell和未赋值的cell字段数不同：

```python
# 未赋值cell（_EMPTY_RECORD模板，只有4个key）：
cell[0, 0, 1]
# {'POD': -1, 'POL': -1, 'GP_count': 0, 'RF_count': 0}

# 已赋值cell（assign()写入，多出4个下划线前缀的记账字段）：
cell[0, 0, 0]
# {'POD': 4, 'POL': 0, 'GP_count': 3, 'RF_count': 0,
#  '_gp_from_gp': 0, '_gp_from_hc': 3, '_rf_from_rf': 0, '_rf_from_hr': 0}

# 带RF的cell：
cell[2, 0, 0]
# {'POD': 6, 'POL': 0, 'GP_count': 5, 'RF_count': 1,
#  '_gp_from_gp': 1, '_gp_from_hc': 4, '_rf_from_rf': 0, '_rf_from_hr': 1}
```

字段含义：`POD`/`POL` 为-1表示未赋值；`GP_count`/`RF_count` 是该cell实际装的箱量（≤各自capacity）；`_gp_from_gp`/`_gp_from_hc`/`_rf_from_rf`/`_rf_from_hr` 是 `assign()`（`VesselClass.py:266`）记账用的字段，只服务 `unassign()` 回滚，不代表真实HC/HR身份（真实HC/HR身份由 `proj_cell_to_vessel` 的HC贴标逻辑另算，见下）。

### 1.2 `Vessel.proj_cell_to_vessel()` 的真实返回

签名：`proj_cell_to_vessel(self, cell_state=None, original_cbf=None) -> pd.DataFrame`（`VesselClass.py:417`）。`cell_state` 不传则用 `self.cell`，传则接受 `snapshot()` 格式的dict（内部取其 `"cell"`）。`original_cbf` 必传，用于HC/HR贴标预算池（求解前的原始cbf，`solve()` 之前 `copy.deepcopy(vessel.cbf)`）。

真实调用：`result_vessel.proj_cell_to_vessel(cell_state=snapshots[0], original_cbf=original_cbf)`

返回一个 `pandas.DataFrame`，**行数=全船slot数（1003），与 `full_slot_table.csv` 行数一致，逐20ft-slot记录**（不是逐cell记录——一个40ft cell会展开写入到摊满该cell需求量的若干个物理slot行上，摊不满的slot保持POD=-1）：

| 列名 | dtype | 含义/来源 |
|---|---|---|
| `bay_idx` | int64 | 透传自 `full_slot_table.csv` |
| `row_idx` | int64 | 透传自 `full_slot_table.csv` |
| `tier_idx` | int64 | 透传自 `full_slot_table.csv` |
| `lr` | int64 | 透传自 `full_slot_table.csv`（0=left,1=right） |
| `hd` | int64 | 透传自 `full_slot_table.csv`（0=hold,1=deck） |
| `can_40ft` | bool | 透传自 `full_slot_table.csv` |
| `can_20ft` | bool | 透传自 `full_slot_table.csv` |
| `can_reefer` | bool | 透传自 `full_slot_table.csv` |
| `POL` | int64 | -1=未装货；否则为装货港编号 |
| `POD` | int64 | -1=未分配；否则为目的港编号 |
| `GP_count` | int64 | 0或1（这个slot是否装了一个GP箱） |
| `RF_count` | int64 | 0或1（这个slot是否装了一个RF箱） |
| `is_hc` | bool | 是否被贴上高箱(HC/HR)标签，proj_cell_to_vessel内部二次分配算出，不是assign()阶段的字段 |
| `is_20ft` | bool | **当前恒为False**，docstring原话："预留给post-solve补丁模块...当前proj_cell_to_vessel完全没有拆分逻辑，这一列恒为False，纯占位" |

真实样例（未占用行）：

```
   bay_idx  row_idx  tier_idx  lr  hd  can_40ft  can_20ft  can_reefer  POL  POD  GP_count  RF_count  is_hc  is_20ft
0        0        5         1   1   0     False      True       False   -1   -1         0         0  False    False
1        0        5         2   1   0     False      True       False   -1   -1         0         0  False    False
```

真实样例（占用行，占用行数=72/1003）：

```
    bay_idx  row_idx  tier_idx  lr  hd  can_40ft  can_20ft  can_reefer  POL  POD  GP_count  RF_count  is_hc  is_20ft
33        2        5         0   1   0      True     False       False    0    4         1         0   True    False
34        2        5         1   1   0      True     False       False    0    4         1         0   True    False
41        2        6         2   1   0      True     False       False    0    4         1         0   True    False
```

其它统计值（POL=0这一港快照）：`is_hc` value_counts = `{False: 935, True: 68}`；`is_20ft` value_counts = `{False: 1003}`（全False，无一例外）；`GP_count` value_counts = `{0: 935, 1: 68}`；`RF_count` value_counts = `{0: 999, 1: 4}`；出现的POD值样例 `[4, 5, 6]`（均为`numpy.int64`类型，不是原生`int`）。

`export_bayplan()` 对每个 `pol in sorted(snapshots.keys())` 调用一次本函数（`VesselClass.py:867`），所以全航次会产出7份这样的DataFrame（每港一份）。

### 1.3 `evaluate_xxx` 系列真实返回结构

均为**纯裸数值，不带任何阈值判断（无OK/WARNING/CRITICAL字样）**。

**`evaluate_crane_time(vessel, snapshots, k=2, crane_rate=1.0, port_names=PORT_NAMES, if_debug=False)`**
返回 `list[dict]`，长度=港口数（7）。每个dict字段与真实值（POL=0这条）：

```python
{
  "pol": 0,                 # int
  "label": "SHP",           # str，来自port_names映射
  "discharge_tally": array([0, 0, 0, 0, 0, 0, 0]),   # numpy.ndarray，按bay统计
  "loading_tally":   array([10, 0, 21, 0, 15, 0, 0]),# numpy.ndarray
  "bay_total":       array([10, 0, 21, 0, 15, 0, 0]),# numpy.ndarray
  "split": 2,                # int，切分点bay index
  "work1": 31.0,   "wait1": 0.0,  "time1": 31.0,     # float
  "work2": 15.0,   "wait2": 16.0, "time2": 15.0,     # float
  "makespan": 31.0,          # float
  "utilization": 0.7419354838709677,  # float，可能为None（n_bay<2且total=0时）
  "time_port": 31.0,         # float，等于makespan
}
```
注意：`discharge_tally`/`loading_tally`/`bay_total` 是 `numpy.ndarray`（不是list），直接塞进JSON会报错，需要 `.tolist()`。

**`evaluate_crane_intensity(vessel, snapshots, target_ci=2, port_names=PORT_NAMES, if_debug=False)`**
返回 `list[dict]`，长度=港口数。字段：

```python
{
  "pol": 0, "label": "SHP",
  "discharge_tally": array([0,0,0,0,0,0,0]),
  "loading_tally":   array([10,0,21,0,15,0,0]),
  "bay_total":       array([10,0,21,0,15,0,0]),
  "ci": 2.1904761904761907,   # float，可能为None（本港无吊车动作时）
}
```

**`evaluate_pod_discharge_spread(vessel, snapshots, port_names=PORT_NAMES, if_debug=False)`**
返回 `dict`（不是list），key为POD（int），value为该POD的分散度指标：

```python
spread_res.keys()   # dict_keys([3, 4, 5, 6, ...])  —— int类型的POD
spread_res[3]        # {'variance': 98.48979591836735, 'range': 25, 'ci': 2.04}
# variance: float, range: int, ci: float或None
```

**`evaluate_ci_theoretical_ceiling(vessel)`**
返回单个 `numpy.float64` 标量（不是dict/list），真实值 `2.9506172839506175`。与具体某次求解无关，只取决于船体几何。

**`evaluate_pod_leverage(cbf)`**（未在真实数据上重新dump，签名与返回结构见 `utils/evaluate.py:343`）：返回 `dict`，key为POD，value为 `{"total": int, "contrib": list[(pol, qty)], "leverages": {pol: float}}`；调用方必须传入求解前的原始cbf，不能传求解后被扣减过的 `vessel.cbf`。

---

## 2. `full_slot_table.csv` 字段记录

真实文件：`data/STSE/geometry/full_slot_table.csv`，**1003行，8列**。

```
columns: bay_idx, row_idx, tier_idx, lr, hd, can_40ft, can_20ft, can_reefer
dtypes:  全部int64，除 can_40ft/can_20ft/can_reefer 是 bool
```

| 字段 | 含义（从调用处反推） |
|---|---|
| `bay_idx` | 0-base bay索引，来自 `build_vessel_geometry()` 对原始 `Bay` 列改名（`utils/vessel_io.py:80`）。不等同于 `VesselClass` 里的 `big_bay`（0..6大箱编号）——`bay_idx` 是物理bay槽位编号（含放不了40ft大箱的bay_idx=0），`big_bay` 是 `STSE_BAY_PAIRS` 里每对(b0,b1)的代表index，仅覆盖大箱可用的7对 |
| `row_idx` | 0-base row索引，来自原始 `Row` 列改名。与物理row标签的对应关系见第3节 `STSE_ROW_LABELS` |
| `tier_idx` | 0-base tier索引，来自原始 `Tier` 列改名。与物理tier标签的对应关系见第3节 `STSE_TIER_LABELS` |
| `lr` | `(row_idx >= STSE_HATCH_SPLIT).astype(int)`，`STSE_HATCH_SPLIT=5`，即 row_idx<5 为 left(0)，>=5 为 right(1) |
| `hd` | `(tier_idx >= STSE_DECK_TIER).astype(int)`，`STSE_DECK_TIER=4`，即 tier_idx<4 为 hold(0)，>=4 为 deck(1) |
| `can_40ft` | 是否能装40ft大箱（`find_can_40ft()`：只标在 `STSE_BAY_PAIRS` 每对的b0一侧，且(row,tier)在b0/b1两侧交集内） |
| `can_20ft` | 是否只能装20ft箱（`find_can_20ft()`：与`can_40ft`互斥且覆盖全部有效槽位——bay_idx=0全部True；配对bay里不在两侧交集内的位置各自算True） |
| `can_reefer` | 是否带reefer插座（`find_can_reefer()`：读 `reefer_slots.csv`，并做了跨b0/b1镜像：只要b0或b1任一侧是reefer，两侧都标记为True） |

真实样例（前5行）：

```
   bay_idx  row_idx  tier_idx  lr  hd  can_40ft  can_20ft  can_reefer
0        0        5         1   1   0     False      True       False
1        0        5         2   1   0     False      True       False
2        0        5         3   1   0     False      True       False
3        0        5         4   1   1     False      True       False
4        0        5         5   1   1     False      True       False
```

`can_20ft`/`can_40ft` 真实取值分布：

```
can_20ft value_counts: {False: 956, True: 47}
can_40ft value_counts: {False: 525, True: 478}
```

`can_20ft`/`can_40ft` **两者都为False**的行（对应 `bay-plan-api-design.md` 里定义的 `unavailableReason: "STRUCTURAL"`）：

```
count = 478, ratio = 478/1003 ≈ 47.66%
```

样例（含一个 `can_reefer=True` 但 `can_40ft`/`can_20ft` 均False的行，即reefer插座存在于一个当前两种箱长都装不了的槽位上）：

```
    bay_idx  row_idx  tier_idx  lr  hd  can_40ft  can_20ft  can_reefer
79        3        5         0   1   0     False     False       False
80        3        5         1   1   0     False     False       False
81        3        5         2   1   0     False     False       False
82        3        5         3   1   0     False     False       False
83        3        5         4   1   1     False     False        True
```

---

## 3. 现有映射关系现状

### 3.1 `STSE_PORT_MAP`（`utils/vessel_io.py:40`）

真实完整内容（只有8个港口，全量，因为本来就不长）：

```python
STSE_PORT_MAP = {"SHP": 0, "TXG": 1, "DLC": 2, "YKK": 3, "NGO": 4, "TYO": 5, "YOK": 6, "LYG": 7}
```

`main.py:46` 反转它得到 `PORT_NAMES = {v: k for k, v in STSE_PORT_MAP.items()}`：

```python
PORT_NAMES = {0: 'SHP', 1: 'TXG', 2: 'DLC', 3: 'YKK', 4: 'NGO', 5: 'TYO', 6: 'YOK', 7: 'LYG'}
```

真实航次（`data/STSE/raw/` 下的.cbf文件）只用到编号0-6（LYG=7未出现在真实cbf.json的key里，`_PORT_MERGE_MAP` 把LYG的货量并到YKK，见第4节）。

`STSE_PORT_COLORS`（`utils/vessel_io.py:42`）是港口三字码到hex颜色的映射，供 `export_bayplan`/`plot_bayplan` 内部画图用，与本次schema设计无关（颜色决策已明确要放在前端），此处仅记录其存在，不列全表。

### 3.2 bay 物理编号 <-> 代码 bay_idx 的对应规则

`utils/vessel_io.py` 顶部注释与 `phy_to_idx`/`idx_to_phy_bay` 函数（`utils/vessel_io.py:60-74`）：

```
phy Bay01 -> idx0（只放20ft，不参与大箱pair）
phy Bay03 -> idx2, Bay05 -> idx3, ... Bay29 -> idx15
40ft大箱占用相邻一对(idx偶,idx偶+1)，即STSE_BAY_PAIRS里的7对，
重新编号为big_bay 0..6（用pair里的第一个bay_idx，即b0，作为代表）
```

正向转换代码（`phy_to_idx`，用于ASC解析）：
```python
def phy_to_idx(bay_p, row_p, tier_p):
    bay_p = int(bay_p)
    b_idx = (bay_p - 1) // 2
    if bay_p > 1:
        b_idx += 1
    ...
```

逆向转换代码（`idx_to_phy_bay`，只做bay_idx->物理Bay码，仅用于viz的物理坐标显示模式）：
```python
def idx_to_phy_bay(bay_idx: int) -> str:
    bay_p = 1 if bay_idx == 0 else 2 * bay_idx - 1
    return str(bay_p).zfill(2)
```

`STSE_BAY_PAIRS`（`utils/vessel_io.py:29`）真实完整内容：

```python
STSE_BAY_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
# _BIG_BAY_OF_B0 = {2:0, 4:1, 6:2, 8:3, 10:4, 12:5, 14:6}
N_BAY = 7
```

即：`big_bay` (0..6，`VesselClass.py` 的cell数组第一维、`proj_cell_to_vessel`/评估函数里所有"按bay"统计用的index) 与物理bay_idx的关系是"取pair的b0作为代表"，b0/b1两个bay_idx共享同一个big_bay。

### 3.3 row/tier 的 idx 与物理值的对应规则

`STSE_ROW_LABELS`/`STSE_TIER_LABELS`（`utils/vessel_io.py:34-35`），列表下标即 `row_idx`/`tier_idx`，列表元素是物理标签字符串（两位数字zfill）：

```python
STSE_ROW_LABELS  = ["10", "08", "06", "04", "02", "00", "01", "03", "05", "07", "09"]
# row_idx=0 -> 物理Row"10"，row_idx=5 -> 物理Row"00"，row_idx=10 -> 物理Row"09"

STSE_TIER_LABELS = ["02", "04", "06", "08", "82", "84", "86", "88", "90", "92"]
# tier_idx=0 -> 物理Tier"02"，tier_idx=4 -> 物理Tier"82"（hold/deck分界处），tier_idx=9 -> 物理Tier"92"
```

`phy_to_idx()` 用 `STSE_ROW_LABELS.index(...)` / `STSE_TIER_LABELS.index(...)` 做正向查表；代码库里**没有**现成的 `idx_to_phy_row`/`idx_to_phy_tier` 函数（只有bay方向有 `idx_to_phy_bay`）——`utils/viz.py` 的物理坐标显示模式（`use_phy_labels=True`）如何取row/tier物理标签需另行确认，本文档只记录"没有现成反向函数"这一事实，不做处理。

`STSE_HATCH_SPLIT = 5`（row_idx<5 -> lr=0/left, >=5 -> lr=1/right），`STSE_DECK_TIER = 4`（tier_idx<4 -> hd=0/hold, >=4 -> hd=1/deck），两者都在 `build_vessel_geometry()`（`utils/vessel_io.py:77-85`）里直接派生成 `lr`/`hd` 两列写入 `full_slot_table.csv`。

---

## 4. `parse_cbf_file`/`batch_parse_cbf` 现状

### 4.1 单行解析失败（`parse_cbf_file`，`utils/vessel_io.py:157`）

```python
_CBF_DATA_LINE_RE = re.compile(r"^46\s+([A-Z]{2}[A-Z]{3})\s+([A-Z0-9]{4})\s+(\d+)")

def parse_cbf_file(cbf_path) -> pd.DataFrame:
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
            ...
```

行为：以"46"开头但正则不匹配的行，**打印一条 `print()` 警告，直接跳过该行，不收集、不抛异常**。不以"46"开头的行静默跳过（不算异常，是文件格式约定：只有"46"开头的行是货量数据行）。

### 4.2 未知港口码

在 `parse_cbf_file` 内部（POD码）：
```python
rows = [
    {"POD": STSE_PORT_MAP.get(pod_code, -1), "length": length, "type": box_type, "count": count}
    for (pod_code, length, box_type), count in counts.items()
]
```
未知POD码**不警告、不跳过**，直接记为 `POD=-1`，照常进DataFrame。是否有下游代码专门处理 `POD=-1` 的行未在本次范围内确认。

在 `batch_parse_cbf`/`batch_parse_cbf_with_20`（POL码，从文件名推断）：
```python
pol_code = re.split(r"[ _]+", os.path.splitext(fname)[0])[-1].upper()
if pol_code not in STSE_PORT_MAP:
    print(f"警告: {fname} 文件名末段POL='{pol_code}' 不在STSE_PORT_MAP里，跳过")
    continue
```
未知POL码：打印警告，**整个文件被跳过**（不解析该文件的任何货量数据）。

同一POL被多个文件覆盖时也只打印警告，不报错：
```python
if pol_num in written:
    print(f"警告: POL={pol_num}({pol_code}) 被 {fname} 覆盖（之前来自 {written[pol_num]}）")
written[pol_num] = fname
```

### 4.3 关键字段缺失/无法识别的箱型长度

在 `cbf_df_to_dict`/`cbf_df_to_dict_with_20`（`utils/vessel_io.py:186`/`206`，把DataFrame聚合成最终cbf.json用的dict）：
```python
if row.length == "UNKNOWN":
    print(f"警告: POD={pod} type={row.type} length无法识别，count={row['count']}已跳过")
    continue
```
`length` 无法识别（来自 `parse_iso_code` 返回 `None`，聚合时被填成字符串 `"UNKNOWN"`）的行：打印警告，跳过，不计入总数。

`parse_iso_code`（`utils/vessel_io.py:283`）本身遇到无法识别的ISO代码：
```python
if not m:
    print(f"警告: 无法识别的ISO代码 '{iso_raw}'")
    return None, None, None
```
返回三个None，交给上层 `cbf_df_to_dict` 按上面的"length==UNKNOWN"分支跳过。

未登记箱型（不在 `_ISO_TYPE_HEIGHT` 里）：
```python
height = _ISO_TYPE_HEIGHT.get(box_type)
if height is None:
    print(f"警告: 未登记的箱型 '{box_type}' (来自ISO '{iso_raw}')")
```
只警告height取不到，**不影响box_type本身照常返回并被后续流程使用**（`_bucket_type`/`cbf_df_to_dict`里未识别类型会归入GP）。

### 4.4 现状总结（不做改造，只陈述事实）

当前所有容错路径都是**print到stdout + 跳过该条数据**，没有任何地方收集一份可返回给调用方的 `warnings: list`。`parse_cbf_file`/`batch_parse_cbf`/`batch_parse_cbf_with_20` 函数签名都不接受、也不返回warnings相关参数。

---

## 5. `export_bayplan()` 现状

签名：`export_bayplan(self, snapshots: dict, out_dir: str, original_cbf: dict, port_names: dict = None, if_csv: bool = False, if_plot_phy: bool = False) -> list`（`VesselClass.py:831`）。

真实执行流程（`VesselClass.py:842-881`）：

1. 打印 `"[export_bayplan]"`。
2. 遍历所有snapshots收集出现过的POD集合，算出 `port_colors: {POD: hex颜色}`（`STSE_PORT_COLORS`按港口三字码查，查不到则用 `_default_port_colors` 自动生成）——这一步是纯为画图服务的颜色分配，不产出结构化数据。
3. `for pol in sorted(snapshots.keys())`：
   - `df = self.proj_cell_to_vessel(cell_state=snapshots[pol], original_cbf=original_cbf)` —— 见第1.2节，这是**这个循环体内唯一的结构化中间产物**。
   - 若 `if_csv=True`：`df.to_csv(csv_path, index=False)` 落盘一份 `{pol}_{code}_DEP_bayplan.csv`。
   - 无条件调用 `plot_bayplan(df, ...)`（`utils/viz.py:204`）落盘PNG。`plot_bayplan` 内部直接消费同一个 `df`（见docstring `"slots: Vessel.proj_cell_to_vessel()的输出"`），只做matplotlib渲染，**没有再产出任何JSON-friendly的中间层**——`_render_bayplan`（`utils/viz.py:90`）直接在这个DataFrame上按行画图，返回值只是文件路径字符串。
4. 返回 `paths: list[str]`（csv和png路径交替）。

**结论（陈述事实，非建议）**：`export_bayplan()` 本身除了内部调用的 `proj_cell_to_vessel()` 之外，不产出任何额外的结构化中间数据；`proj_cell_to_vessel()` 返回的DataFrame（第1.2节）就是从cell级解到CSV/PNG之间唯一存在的结构化产物，`plot_bayplan`/`_render_bayplan` 是这个DataFrame的纯消费端，不反向产出新结构。

---

## 附：真实运行环境记录

- 数据来源：`data/STSE/geometry/full_slot_table.csv`（1003行）+ `data/STSE/cbf/cbf.json`（真实STSE 2545E/2546W航次货量）
- `Vessel.load_vessel(geometry_dir, cbf_json_path)`，`current_pol` 未传，取cbf里最小POL（=0）为起点
- `random.seed(8245)`，`solve(vessel, is_debug=False, snapshots=snapshots, best=best)` 一次成功（`success=True`），7港全部完成
- `original_cbf = copy.deepcopy(vessel.cbf)`（求解前深拷贝，供 `proj_cell_to_vessel`/`evaluate_pod_leverage` 使用）
- `PORT_NAMES`（=`main.py`里反转的`STSE_PORT_MAP`）：`{0: 'SHP', 1: 'TXG', 2: 'DLC', 3: 'YKK', 4: 'NGO', 5: 'TYO', 6: 'YOK', 7: 'LYG'}`
