# 集装箱船舶配载系统 —— 后端服务化 & 输出接口设计

> v2.0 精简版，供 Claude Code 开发参考

## 0. 总原则

- **JSON 只承载算法的最终输出结果，不承载算法本身。** 后端对外的一切数据都是"结果事实"（放置结果、评估结果），不包含任何中间过程、评分细节、规则参数。
- **渲染 100% 在前端完成。** 后端不做颜色、图标、布局等视觉决策，只给前端渲染所需的结构化事实。
- **"事实"与"评估"是两类结果，但都由算法产出、都通过 JSON 传给前端。** 不是"事实在后端算、评估在前端算"，而是两者都是算法的最终输出，只是在 schema 里分层放置，避免混在箱位对象里难以维护。

---

## 1. JSON Schema（TypeScript interface）

```typescript
// ============================================================
// 顶层响应：一次配载方案的完整结果
// ============================================================
interface BayPlanResult {
  planId: string;
  voyageId: string;
  /** 基于哪个 planId 重算而来，初次计算为 null */
  parentPlanId: string | null;
  generatedAt: string;

  vessel: VesselInfo;
  voyageLegs: VoyageLeg[];
  legend: LegendEntry[];
  bays: Bay[];

  /** 评估结果，由算法产出，结构见 §1.5 */
  metrics: MetricsBlock;

  appliedConstraints: UserConstraint[];
}

// ============================================================
// 船舶 & 航线
// ============================================================
interface VesselInfo {
  vesselId: string;
  vesselName: string;
  bayOrder: string[]; // bay 物理顺序，船艏→船艉
}

interface VoyageLeg {
  sequence: number;   // 航线顺序
  portCode: string;   // 如 "CNSHA"
  portName: string;
}

// ============================================================
// 图例：只给分组身份和顺序，颜色由前端决定
// ============================================================
interface LegendEntry {
  groupKey: string;      // 通常等于 portCode，也可以是 "EMPTY" "RESTRICTED" 等特殊分组
  label: string;
  displayOrder: number;
  kind: "POD" | "ATTRIBUTE" | "STATUS";
}

// ============================================================
// Bay
// ============================================================
interface Bay {
  bayId: string;
  bayNumber: string;
  gridBounds: {
    minRow: number;
    maxRow: number;
    minTier: number;
    maxTier: number;
  };
  deckBoundaryTier: number; // tier < 该值为舱内，反之为甲板上
  slots: Slot[];
}

// ============================================================
// Slot：一个物理箱位
// ============================================================
interface Slot {
  slotId: string; // 由 bay+row+tier 派生，物理稳定不变
  row: number;
  tier: number;

  status: "OCCUPIED" | "EMPTY" | "UNAVAILABLE" | "RESERVED";
  unavailableReason?: "USER_CONSTRAINT" | "STRUCTURAL" | "LASHING_EQUIPMENT" | "OTHER";

  container: ContainerPlacement | null;
}

// ============================================================
// ContainerPlacement：一个箱子在某 slot 上的放置事实
// （不含任何箱号信息，箱子不作为独立可追踪实体维护）
// ============================================================
interface ContainerPlacement {
  pod: string; // 对应 legend.groupKey
  pol: string;
  sizeType: "20GP" | "40GP" | "40HC" | "45HC" | string;
  attributeFlags: AttributeFlag[]; // H / 危险品 / 冷藏 / 超重 / 超限 等
  weightKg?: number;

  /** 动画预留字段（当前不实现，仅占位）
   *  方案A：算法可选给出不透明的放置顺序号，前端可用可不用；
   *  不携带任何算法评分/规则依据，仅表示呈现顺序。 */
  placementSequence?: number;
  placementBatch?: number;
}

type AttributeFlag =
  | "HIGH_CUBE"
  | "HAZARDOUS"
  | "REEFER"
  | "OVERWEIGHT"
  | "OUT_OF_GAUGE"
  | string;

// ============================================================
// 指标层：与箱位事实分离存放，同样是算法输出的"结果事实"
// ============================================================
interface MetricsBlock {
  global: MetricEntry[];
  byBay: Record<string, MetricEntry[]>; // key 为 bayId
}

interface MetricEntry {
  key: string;      // 如 "STABILITY_CI"
  label: string;
  value: number;
  unit?: string;
  status?: "OK" | "WARNING" | "CRITICAL";
  thresholds?: { warning?: number; critical?: number };
}

// ============================================================
// 用户约束（未来功能占位）
// ============================================================
interface UserConstraint {
  constraintId: string;
  type: "SLOT_UNAVAILABLE" | "COLUMN_UNAVAILABLE" | "OTHER";
  target: Record<string, unknown>;
  submittedAt: string;
}
```

---

## 2. API 接口

### 资源模型
```
Voyage —— 航次（POL/POD序列 + 各港CBF）
  └─ Plan —— 一次配载方案结果（BayPlanResult）
       └─ Constraint —— 基于某 Plan 提交的约束，触发新 Plan
```

### 端点

```
POST /api/v1/voyages
body: { "voyageLegs": [{ "sequence": 1, "portCode": "CNSHA" }, ...], "vesselId": "VSL-001" }
→ 201 { "voyageId": "VOY-20260716-001" }
```

```
POST /api/v1/voyages/{voyageId}/cbf   (multipart/form-data: portCode, file)
→ 202 { "cbfId": "CBF-CNSHA-001", "status": "ACCEPTED" }
```

```
POST /api/v1/voyages/{voyageId}/plans
→ 202 { "planId": "PLAN-0001", "status": "QUEUED" }
```

```
GET /api/v1/voyages/{voyageId}/plans/{planId}
→ 200 { "status": "COMPLETED", "result": <BayPlanResult> }
→ 200 { "status": "PROCESSING" }
→ 200 { "status": "FAILED", "errorCode": "PLAN_INFEASIBLE" }
```

```
POST /api/v1/voyages/{voyageId}/plans/{planId}/constraints
body: { "type": "COLUMN_UNAVAILABLE", "target": { "bayId": "BAY-06", "row": 2 } }
→ 202 { "newPlanId": "PLAN-0002", "parentPlanId": "PLAN-0001", "status": "QUEUED" }
```

```
GET /api/v1/voyages/{voyageId}/plans/diff?from=PLAN-0001&to=PLAN-0002
→ 200 {
    "changedSlots": [
      { "slotId": "BAY-06-ROW02-TIER82", "before": <ContainerPlacement|null>, "after": <ContainerPlacement|null> }
    ]
  }
```
> diff 以 `slotId` 为比对单位（位置稳定），不依赖箱子身份追踪。

### 统一错误格式
```json
{ "errorCode": "CBF_FORMAT_INVALID", "message": "...", "requestId": "req-abc123" }
```
`errorCode` 为预定义业务枚举；`message` 不含底层异常/堆栈/模块路径。

---

## 3. 服务分层

```
前端（浏览器）
  - SVG/Canvas 渲染、配色、图例、（未来）动画
  - 只消费 BayPlanResult / diff 结果
        │ HTTPS/JSON
API 网关层
  - 鉴权、限流、请求校验、统一错误格式化
  - 不暴露任何调试/内部端点
        │ 内部调用（不对公网暴露）
业务服务层
  - Voyage/Plan/Constraint 生命周期管理
  - CBF 解析校验
  - 【反腐层】将算法原始输出翻译为白名单 schema（§1）
  - plan 间 diff 计算（基于 slotId）
  - CSV/PNG 内部存档（不对外暴露）
        │ 进程内调用 / 内网消息队列（不联网暴露）
算法核心层（黑箱）
  - 纯计算：输入船期+CBF，输出原始配载结果 + 评估指标
  - 不提供任何网络接口，不含调试端点
  - 中间过程/评分/规则参数不得流出此层
```

**硬性边界**：算法核心层与业务服务层之间不用 HTTP，走进程内调用或内网队列；API 网关层错误处理走统一中间件，杜绝个别端点泄露原始异常。

---

## 4. 遗留问题

1. CBF 文件具体格式待补充解析规则。
2. 异步计算通知机制（轮询 vs WebSocket/SSE）待前端交互设计确定后再定。

## 5. claude 回复
策略方案：现有排箱算法后端服务化
1. 现状盘点
输入侧

船型/几何：utils/vessel_io.py 的 build_vessel_geometry + find_can_40ft/20ft/reefer，从 data/STSE/geometry/all_slots.csv（idx 版槽位表）+ reefer_slots.csv 构建出 full_slot_table.csv，落盘缓存（main.py:ensure_geometry）。这是纯 pandas 批处理，STSE 船型强绑定（STSE_BAY_PAIRS、STSE_ROW_LABELS 等硬编码常量）。
CBF（货量预报）：batch_parse_cbf 解析 data/STSE/raw/*.cbf 原始文件（固定列正则），产出 cbf.json（{POL:{POD:{"GP":n,"HC":n,"RF":n,"HR":n}}}），同样走"若已存在则跳过"的落盘缓存（main.py:ensure_cbf）。
两者最终通过 Vessel.load_vessel(geometry_dir, cbf_json_path) 汇合成一个 Vessel 实例——这是唯一真正的"输入接口"，其余都是文件路径拼接。
核心计算逻辑

VesselClass.py::Vessel：静态几何 + 动态搜索状态的数据层，assign/unassign/discharge/undischarge/snapshot/restore 是回溯搜索的状态机原语。
CSP_solver.py::solve()：唯一的递归搜索入口，cal_candidates/mrv_select/_pod_try_order 是启发式打分细节。这两个文件是"算法核心层"本体，边界很清晰——它们不产生任何文件 I/O、不依赖 STSE 具体常量（N_BAY/坐标只是通过 Vessel.__init__ 传入的 full_slot_table 派生），可以直接原样当黑箱调用。
输出侧

solve() 本身不产生对外结果，只在 vessel.cell（内存态）和 snapshots: {POL: snapshot_dict} 里留下解。
Vessel.proj_cell_to_vessel() 把 cell 级解投影回 slot 级 DataFrame（列：bay_idx,row_idx,tier_idx,lr,hd,can_40ft,can_20ft,can_reefer,POL,POD,GP_count,RF_count,is_hc），这是唯一现成的、可对外暴露候选的中间数据结构。
Vessel.export_bayplan() 编排：对每港调 proj_cell_to_vessel → 落盘 CSV（可选）+ 调 utils/viz.py::plot_bayplan 出 PNG。PNG 渲染（配色、图例、田字格布局）目前 100% 在后端 matplotlib 完成，这与设计文档"渲染 100% 前端做"的原则是主要冲突点，改造时要绕开而不是复用这部分。
utils/evaluate.py：一组只读评估函数（CI、吊车耗时、POD 分散度等），输入是 vessel + snapshots，输出是 Python dict/list，这是 MetricsBlock 的天然数据源。
utils/tail.py：尾箱二次安置管线（build_unified_tail_list → scan_host_candidates → match_tails_to_hosts），产出安置台账，目前只在 main.py 里打印摘要，没有结构化落盘。
结论：算法核心层 = VesselClass.Vessel + CSP_solver.solve；utils/evaluate.py + utils/tail.py 的后处理部分性质上也偏"核心计算"（产出评估事实），应该跟 solver 一起留在黑箱内，只有 utils/viz.py 的渲染部分要被替换/隔离。

2. 映射关系
以 proj_cell_to_vessel() 产出的 slot 级 DataFrame（下称"slot_df"）为准，逐字段看：

slot_df / Vessel 现有字段	目标 schema	现成/需新增/需排除
bay_idx	Bay.bayNumber（配合 idx_to_phy_bay() 转真实 Bay 码）、Slot.slotId 派生	现成，但需要一层 idx→bayId 命名转换
row_idx, tier_idx	Slot.row, Slot.tier	现成，若要给前端"真实物理刻度"需配合 STSE_ROW_LABELS/STSE_TIER_LABELS 转换（idx→物理两位数码），否则前端拿到的是内部 0-base 索引，不直观也不稳定
can_40ft/can_20ft/can_reefer	Slot.status（UNAVAILABLE，或空位判断依据）	can_20ft=True 的槽位现在渲染成浅灰"未决策"——这是求解器暂不处理 20ft 的已知局限，需要业务服务层决定它该报 UNAVAILABLE 还是干脆不出现在 slots[] 里（不能直接照搬现在的"浅灰"视觉语义，那是渲染决策不是事实）
POD	ContainerPlacement.pod（配合 PORT_NAMES/STSE_PORT_MAP 转三字码）	现成，需数字↔三字码转换（反腐层职责）
POL	ContainerPlacement.pol	现成，同上需转码
GP_count/RF_count	派生 ContainerPlacement.sizeType（40GP/40HC 等）、attributeFlags 里的 REEFER	需新增判断逻辑：GP_count>0 且 is_hc→40HC，否则 40GP；RF_count>0→追加 REEFER flag。注意 GP_count/RF_count 语义是"槽位是否被占用(0/1)"不是箱型本身，这层翻译不能省
is_hc	AttributeFlag: HIGH_CUBE	现成 bool，直接映射
（无）weight	ContainerPlacement.weightKg	需新增：当前 cell 级解没有重量概念（_ISO_TYPE_HEIGHT/ASC 解析里有 weight 字段但只用于历史 ASC 数据回放，不是 solver 输出的一部分），schema 里标了 ? 可选，短期直接省略即可
（无）placementSequence/Batch	同名字段	明确"占位不实现"，直接不产出
capacity_total/capacity_rf/capacity_hc（Vessel 静态几何）	Bay.gridBounds、Slot.status=UNAVAILABLE	需要新增一次性转换：capacity_total==0 的 cell → 反映到具体 slot 的 UNAVAILABLE（unavailableReason: STRUCTURAL）
evaluate_crane_time/evaluate_crane_intensity 等返回的 dict	MetricsBlock.global / byBay	现成数值，需要包一层 key/label/unit/status/thresholds 的映射表（比如 time_port→"CRANE_TIME_PORT"），这层映射本身就是"术语白名单化"，顺带把内部打分细节（Cost_adj、CI 内部中间量等）挡在外面
_ci_current_pol_score/_ci_future_pod_score/priority() 排序键	不导出	这些是 MRV/启发式打分的中间过程，属于"算法核心层黑箱"内部，绝对不能流出，文档 §0 明确禁止
vessel.cbf（剩余尾货）、_tail_source2_log/_tail_source3_log	不直接导出；unified_tail_list/placements/unplaced（utils/tail.py）经过白名单化后可能对应未来的"未装箱"状态或者 MetricsBlock 里一个 tail 相关条目	尾箱诊断日志是调试用，不应出现在对外 JSON；但"最终有多少箱子没装上"这个事实本身如果产品需要展示，要走 MetricsBlock 而不是直接吐日志结构
排除清单（明确不出现在对外 JSON）：_gp_from_gp/_gp_from_hc/_rf_from_rf/_rf_from_hr（assign 记账内部字段）、current_port_bay_load、bay_capacity_share、port_budget、_ci_* 打分函数的任何中间返回值、tail_threshold、ci_pol_enabled/ci_pod_enabled 这类消融实验开关。

3. 改动范围最小化：反腐层放哪
结论：新增独立模块包裹，不改动 Vessel/solve/utils/evaluate.py 的任何返回值签名。

理由：

Vessel.proj_cell_to_vessel() 和 utils/evaluate.py 的各 evaluate_xxx 已经是"结构化只读输出"（DataFrame / dict），不是打印语句，具备被包装的条件，不需要重构它们本身。
export_bayplan() 内部耦合了 CSV 落盘 + PNG 渲染两个副作用，不要在这个函数里加 JSON 输出分支——它应该保持"内部存档专用"（文档 §3 服务分层里也明确 CSV/PNG 是业务服务层内部存档，不对外暴露），JSON 转换要在更上层发生。
建议新增目录：


service/
  __init__.py
  translator.py   # 反腐层本体：slot_df + evaluate结果 + Vessel静态几何 -> BayPlanResult dict
  codes.py        # POD/POL数字<->三字码, bay_idx<->bayId, idx<->物理Row/Tier 的双向映射封装
                  # （现在这些映射零散分布在 vessel_io.STSE_PORT_MAP / idx_to_phy_bay 等处，
                  #   codes.py 只是收拢引用，不重新实现）
  errorcodes.py   # 业务错误枚举 + 统一异常类型
translator.py 的函数签名类似：


def to_bay_plan_result(vessel: Vessel, snapshot: dict, original_cbf: dict,
                        eval_results: dict, plan_id: str, voyage_id: str,
                        port_names: dict) -> dict:  # 直接是可以json.dumps的dict
输入就是现有函数已经在产出的东西（proj_cell_to_vessel 的 df + evaluate_* 的返回值），不需要 solver 侧改一行代码。

4. 服务框架落地
技术栈现状：纯 Python + pandas/numpy/matplotlib，pyproject.toml 是 setuptools 打包，没有任何 web 框架依赖，main.py 是同步脚本式入口。

框架选型（2选1，不展开）：

FastAPI：类型化 request/response（可以直接用 TypedDict/pydantic 对着 §1 的 TypeScript interface 写一份 1:1 的 pydantic model，天然贴合"JSON schema 强约束"的设计原则），异步友好，适合后续接轮询/SSE。
Flask + Celery（如果团队更熟悉同步框架 / 已有 Celery 基建做异步任务）：solve() 是 CPU 密集型递归搜索，跑一次可能是秒级到分钟级，Plan 计算天然需要异步任务队列（文档 API 里 POST /plans 直接返回 202 QUEUED，跟 Celery/RQ 的任务模型完全对应）。
无论选哪个，算法核心层跑在独立进程/worker 里（Celery worker 或 FastAPI 的 BackgroundTasks/外部任务队列），不要在 HTTP 请求线程里同步跑 solve()——这是文档 §3 "算法核心层不提供网络接口"的字面要求，也是现实的性能要求（回溯搜索可能长跑）。

目录结构建议（贴合文档三层）：


api/                    # API网关层：路由、鉴权、限流、统一错误中间件
  routes/voyages.py
  routes/plans.py
  routes/constraints.py
  middleware/error_handler.py

service/                # 业务服务层
  translator.py          # 反腐层（见上）
  codes.py
  errorcodes.py
  voyage_service.py      # Voyage/Plan/Constraint生命周期、CBF解析校验编排
  plan_diff.py           # 基于slotId的plan间diff
  storage.py             # CSV/PNG内部存档 + plan结果持久化（DB或文件，待定）
  tasks.py               # 异步任务定义（Celery task或等价物），调用算法核心层

# 算法核心层：原样不动
VesselClass.py
CSP_solver.py
utils/vessel_io.py       # CBF/geometry解析，业务服务层复用而非重写
utils/evaluate.py
utils/tail.py
utils/viz.py             # 仅CSV/PNG内部存档路径继续用，不作为对外JSON数据源
api/ 和 service/ 之间进程内直接调用即可（同一 Python 进程/同一 Celery worker），不需要额外协议层；"内部调用不对公网暴露"这条边界主要体现在路由层不直接 import VesselClass/CSP_solver，一律经过 service/ 转发。

5. 分阶段实施顺序
反腐层单测先行：拿 debug/tail_real_run.py 或 CSP_solver.py __main__ 里现成的 4-bay 测试场景跑出一份真实 (vessel, snapshots)，喂给 translator.to_bay_plan_result()，对照 §1 schema 手工核对每个字段（尤其是 sizeType/attributeFlags 的派生逻辑、bay_idx→bayId 转换）。
风险点：现有测试场景港口都是数字编号（0/1/2/3），没有真实三字码，codes.py 的双向映射在小场景下可能覆盖不全，需要专门为 STSE 真实 STSE_PORT_MAP 场景再跑一遍。
最简单的一个 GET 接口跑通 schema：先做 GET /plans/{planId}，读一份预先跑好、落盘缓存的 plan 结果（不接 Celery/异步），验证"HTTP → service → translator → JSON"整条链路能吐出合法的 BayPlanResult。
风险点：slotId 需要"物理稳定不变"（diff 依赖它），目前 bay_idx-row_idx-tier_idx 组合已经具备这个性质，但要在这一步就定死生成规则，后续不能改，否则 diff 端点会全部失效。
补 POST /plans（异步任务）：接入任务队列，solve() 挪到 worker 里跑，先只支持"排队→跑完→GET 拿结果"，不做进度推送。
风险点：solve() 目前是纯内存递归，长跑场景下需要考虑超时/取消机制（现有代码没有这个概念，_solve_call_count 只是打印日志，没有硬性上限），异步化时要么加一个深度/时间上限，要么接受"跑不完就一直占着 worker"的风险。
CBF 上传解析接口：包装 batch_parse_cbf/parse_cbf_file，这里改动量最小，因为解析函数已经是纯函数、吃文件返回 dict。
风险点：现有解析对"文件名末段当 POL 三字码""行格式固定 regex"这类假设没有容错，直接暴露成上传接口前必须补业务级校验（CBF_FORMAT_INVALID 错误码），否则脏文件会让内部异常直接穿透。
Constraint → 新 Plan（重算）+ diff 接口：这是最后一步，因为它依赖 1-4 全部就绪，且需要在 Vessel/solve() 层面确认"约束"（比如某 slot 强制不可用）具体怎么注入求解器——这是当前代码完全没有的能力，需要新增 Vessel 构造参数或 is_valid 掩码叠加逻辑，属于对算法核心层的真实改动，不再是纯包装。
6. 待确认问题清单
CBF 上传格式：文档 §4 遗留问题已提到"具体解析规则待补充"——现有 parse_cbf_file 是针对 CASP 导出的 .cbf 定长/正则格式，前端上传的文件是否就是同一种格式？如果前端允许自由格式（Excel/CSV），需要另一套解析器。
异步通知机制：轮询 vs SSE/WebSocket（文档遗留问题2），直接决定第 3 步任务队列要不要附带进度回调。
用户约束（Constraint）具体语义：COLUMN_UNAVAILABLE/SLOT_UNAVAILABLE 提交后，是"在原 Plan 基础上打个补丁重算"还是"完全重新跑一次 solve()"？如果是前者，需要 Vessel 支持"从某个 snapshot 状态出发、锁定部分 slot 不可用后继续/重跑"，这在当前 snapshot/restore 机制上是可行的，但需要 solver 侧确认。
异常处理规范：现有代码里几乎全部是 print(f"警告: ...") + 继续执行（比如 parse_cbf_file 遇到无法解析的行、batch_parse_cbf 遇到未知 POL 码），没有异常类型体系。业务服务层要把这些"警告后跳过"的分支改成收集校验结果（哪些行被跳过、为什么），再决定是 400 CBF_FORMAT_INVALID 还是允许部分跳过继续处理——这个策略需要产品/算法团队一起定，不是纯技术决定。
current_pol/POD 数字编号 vs 三字码的对外契约：现有 STSE_PORT_MAP 硬编码在 vessel_io.py，是否要在 POST /voyages 时由前端传入完整的港口列表来动态生成，还是继续复用这份硬编码表？如果未来要支持非 STSE 船型/航线，这里现在的强耦合需要解耦。
20ft 槽位在 schema 里怎么表达：当前视觉上专门标"浅灰=20ft未决策"，是产品事实的一部分还是纯视觉遗留？需要产品确认 Slot.status 该给 EMPTY（假装可用）还是新增语义（比如 RESERVED/UNAVAILABLE 之外的第三态），这个決定会连带影响 slot 总数与前端网格渲染逻辑。
Plan 持久化载体：export_bayplan 目前落盘 CSV/PNG 到本地目录，BayPlanResult 是否需要落库（供 diff、历史查询），还是每次都基于 snapshot 文件重新走 translator？这决定 service/storage.py 是文件存储还是接数据库。