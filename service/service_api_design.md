# 集装箱船舶配载系统 —— 后端服务化 & 输出接口设计

> v3.0，供 Claude Code 开发参考

## 0. 总原则

- **JSON 只承载算法的最终输出结果，不承载算法本身。** 中间过程、评分细节、规则参数、搜索轨迹一律不进入对外 JSON，即使字段名看起来中立。
- **渲染 100% 在前端完成。** 后端不做颜色、图标、布局等视觉决策。
- **"事实"与"评估"都是算法产出的结果**，一起通过 JSON 传给前端，只是在 schema 里分层放置，避免混在箱位对象里难以维护。
- **不维护箱子唯一身份（无 containerId）**，箱子不作为可跨次追踪的独立实体。
- **Constraint / 重算 / diff 本版不做**，仅在数据结构上不预留会产生歧义的字段，功能落地时再单独设计。

---

## 1. JSON Schema（TypeScript interface）

```typescript
// ============================================================
// 顶层响应：一次配载方案的完整结果
// ============================================================
interface BayPlanResult {
  planId: string;
  voyageId: string;
  generatedAt: string;

  vessel: VesselInfo;
  voyageLegs: VoyageLeg[];
  legend: LegendEntry[];
  bays: Bay[];
  metrics: MetricsBlock;
}

// ============================================================
// 船舶 & 航线
// ============================================================
interface VesselInfo {
  vesselId: string;
  vesselName: string;
  bayOrder: string[]; // bay 物理顺序，船艏→船艉
}

// 输入：POST /voyages 时传 portCodes: string[]（有序数组，无需单独传 sequence）
// 输出：voyageLegs 的顺序 = 输入数组下标，由后端生成，不接受客户端显式指定 sequence
interface VoyageLeg {
  sequence: number;   // 由后端根据输入数组下标生成
  portCode: string;   // 如 "CNSHA"，需在 STSE_PORT_MAP 中可查
  portName: string;
}

// ============================================================
// 图例：只给分组身份和顺序，颜色由前端决定
// ============================================================
interface LegendEntry {
  groupKey: string;      // 通常等于 portCode，或 "EMPTY"/"RESTRICTED" 等特殊分组
  label: string;
  displayOrder: number;  // 与 voyageLegs.sequence 保持一致
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
  /** 数据源：full_slot_table.csv。can_20ft/can_40ft 均为 false 的槽位记为 STRUCTURAL 不可用 */
  unavailableReason?: "STRUCTURAL" | "USER_CONSTRAINT" | "OTHER"; // USER_CONSTRAINT 为未来预留，本版不产出

  /** 槽位物理承载能力，来自 full_slot_table.csv 的 can_20ft/can_40ft，与 status 独立 */
  capability?: { can20ft: boolean; can40ft: boolean };

  container: ContainerPlacement | null;
}

// ============================================================
// ContainerPlacement：一个箱子在某 slot 上的放置事实
// 不含箱号信息，箱子不作为独立可追踪实体维护
// ============================================================
interface ContainerPlacement {
  pod: string; // 对应 legend.groupKey
  pol: string;
  sizeType: "20GP" | "40GP" | "40HC" | "45HC" | string;
  attributeFlags: AttributeFlag[]; // H / 危险品 / 冷藏 / 超重 / 超限 等
  weightKg?: number;
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
// 本版只做单条 metric 的透传展示，不做阈值判断/计算，前端仅渲染为表格
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
  status?: "OK" | "WARNING" | "CRITICAL";     // 本版不产出，字段保留但为空
  thresholds?: { warning?: number; critical?: number }; // 本版不产出
}
```

**已明确不做的字段/功能（不在本版 schema 中出现）：**
- `containerId`：不维护箱子身份
- `placementSequence` / `placementBatch`：存在泄露 solver 搜索轨迹（变量排序启发式、回溯路径）的风险，且当前无实际功能依赖，暂不加入。动画功能启动时若需要顺序信息，只能由 translator 基于**最终结果事实**（POD分组、bay物理顺序等）在业务层生成一套独立的展示顺序规则，不得读取 solver 的 snapshot/搜索状态。
- `UserConstraint`、`parentPlanId`、diff 相关结构：Constraint 功能本版不做，相关端点和字段一并不建，避免消费端先于生产端存在造成语义空转。

---

## 2. API 接口

### 资源模型
```
Voyage —— 航次（POL/POD序列 + 各港CBF）
  └─ Plan —— 一次配载方案结果（BayPlanResult），落盘后不可变
```

### 端点

```
POST /api/v1/voyages
body: { "portCodes": ["CNSHA","SGSIN","NLRTM"], "vesselId": "VSL-001" }
→ 201 { "voyageId": "VOY-20260716-001" }
→ 422 { "errorCode": "VOYAGE_PORT_NOT_SUPPORTED", ... }  // portCode 不在 STSE_PORT_MAP 中
```

```
POST /api/v1/voyages/{voyageId}/cbf   (multipart/form-data: 多个 .cbf 文件)
→ 202 { "cbfId": "CBF-BATCH-001", "status": "ACCEPTED", "warnings": ["port ABC 未知，已跳过第12行", ...] }
→ 422 { "errorCode": "CBF_FORMAT_INVALID", ... }  // 解析结果为空/关键字段缺失
```
> `warnings` 来自 `parse_cbf_file`/`batch_parse_cbf` 新增的可选 `warnings: list` 收集参数（TODO，现有 print 逻辑不变，只是多开一个输出通道）。service 层拿到 warnings 后自行判断：有 warnings 但结果非空 → 200/202 + 透传 warnings；结果为空或关键字段缺失 → `CBF_FORMAT_INVALID`。

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

> 本版不建 Constraint 端点、不建 diff 端点。

### 统一错误格式
```json
{ "errorCode": "CBF_FORMAT_INVALID", "message": "...", "requestId": "req-abc123" }
```
`errorCode` 为预定义业务枚举；`message` 不含底层异常/堆栈/模块路径。

---

## 3. 服务分层

```
前端（浏览器）
  - SVG/Canvas 渲染、配色、图例
  - 只消费 BayPlanResult
        │ HTTPS/JSON
API 网关层
  - 鉴权、限流、请求校验、统一错误格式化
        │ 内部调用（不对公网暴露）
业务服务层
  - Voyage/Plan 生命周期管理
  - CBF 解析校验（复用现有 parse_cbf_file/batch_parse_cbf，接 warnings）
  - 【反腐层 translator.py】：snapshot + eval_results + Vessel静态几何 -> BayPlanResult
  - CSV/PNG 内部存档（不对外暴露）
  - Plan 结果落盘（storage/plans/{planId}/result.json），生成后不可变
        │ 进程内调用 / 任务队列
算法核心层（黑箱，不改动返回值签名）
  - Vessel.proj_cell_to_vessel() / evaluate_xxx：结构化只读输出，直接作为 translator 输入
  - export_bayplan()：保持"仅内部存档"职责，不加 JSON 输出分支
  - 不提供任何网络接口
```

目录建议：
```
service/
  translator.py    # 反腐层本体
  codes.py          # POD/POL数字<->三字码、bay_idx<->bayId、idx<->物理Row/Tier 映射封装（收拢现有引用，不重新实现）
  errorcodes.py     # 业务错误枚举
  storage.py        # 文件存储：result.json + CSV/PNG 内部存档
```

**边界要求**：算法核心层与业务服务层之间不用 HTTP，进程内调用或任务队列；API 网关层统一错误中间件，杜绝个别端点泄露原始异常。

---

## 4. 存储设计

- 文件存储，不接数据库。
- `storage/plans/{planId}/result.json`：对外 JSON，计算完成时生成一次，**之后不可变**（即使 translator 逻辑后续更新，历史 plan 的 JSON 不回溯变化）。
- `storage/plans/{planId}/archive/`：CSV/PNG，内部审计用，不挂对外路由，前端可通过下载弹窗获取。
- `GET /plans/{planId}` 直接读已落盘的 `result.json`，不重复调用 translator。

---

## 5. 遗留问题

1. 20ft 箱逻辑：solver 当前不支持小箱决策，待 solver 补齐后再评估是否需要在 schema 中体现相关状态。
2. CBF 解析容错：`warnings` 收集参数为 TODO，需在 `parse_cbf_file`/`batch_parse_cbf` 中新增，不改变现有 print 行为。
3. STSE_PORT_MAP 硬编码：当前继续复用；未来支持任意航线时，评估是否由前端在 `POST /voyages` 时传入完整港口定义动态生成。