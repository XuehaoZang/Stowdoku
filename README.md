## TODO
1. 高箱悬空问题解决，现在需要看遍历row的顺序，现在有点散乱；此外有些地方仓内可以放4tier补充一个H，可以先判断一下高箱比例，决定4个的slot是放2高还是3高
2. 尾箱处理有bug，分析完之后看看能配进去多少箱子；现在运行会有assertion error
3. 20ft的cbf成功生成了，完成了只拆分40ft，保存数据合理；下一步是给有些单数小箱打补丁装到only20去；第三步再考虑利用bay 01，加这个搜索前的分配；此外cbf是不是统一成涵盖20ft的？
4. 接受HR全额进尾箱，等proj补上再自动收敛；现在新proj的第一步"摊RF"里，is_hc字段没用到
5. 为了优化尾箱逻辑，可以考虑把尾箱也分配到cell里，然后把这个只装了4个箱子的cell的candidate相应改成比装进来的tail的pod要小，容量相应减小4.这样一来，箱子可以成功进来，不影响下一阶段搜索，就是可能稍微看上去乱一点
（乱一点可以考如何安排尾箱的位置调整）

## 递归回溯的搜索结构
```text
根节点（init）
├── 装箱决策1（port0）
│   ├── 装箱决策2
│   │   ├── ...装完port0所有箱子
│   │   │   └── [discharge port1] ← 特殊节点，记录快照
│   │   │       ├── 装箱决策N（port1）
│   │   │       │   └── 发现dead slot → 回溯
│   │   │       │       ├── 回溯几步 → 还在port1装载层
│   │   │       │       └── port1装载全部穷举失败 → 回溯越过discharge节点
│   │   │       │           → 回到port0装载层继续试其他分支
```

## 配载数据集 (Stowage Dataset) 说明

1. init (初始状态)

格式: 每个半舱视为一个格子，田字格的三维数组 (Bay $\times$ Tier $\times$ Row)，目前是4 $\times$ 2 $\times$ 2。-1是空位，可自由配载，其他值表示已占用。

2. cbf (货量需求预测)

格式: { "出发港": { "目的港": 数量 } }
示例: "0": { "1": 10, "2": 6 } 表示在港口 0，需装上 10个 0->1 的箱子、6个 0->2 的箱子。

3. 核心任务

将 cbf 中的箱子合法排入空间。
规则底线: 满足 init 的硬约束，且绝对禁止翻箱（上层箱子的目的港，绝对不能比下层箱子的目的港更远）。

### Dataset 设计
1. 完全装满16个位置，有0->1, 0->2, 1->3，必须需要学习到2要提前竖着放而不能平放，否则在港口1装1->3的时候会导致翻箱
测试结果
```text
init:
bay0 | bay1 | bay2 | bay3
 1  X |  X  X |  X  X |  X  X
 X  X |  X  X |  X  2 |  X  X

[Departure] from POL=0 出发状态:
bay0 | bay1 | bay2 | bay3
 1  1 |  1  1 |  1  2 |  2  2
 1  1 |  1  1 |  1  2 |  2  2

[Arrive] at POL=1, 卸了10个箱子，推进到POL=1
[Departure] from POL=1 出发状态:
bay0 | bay1 | bay2 | bay3
 3  3 |  3  3 |  3  2 |  2  2
 3  3 |  3  3 |  3  2 |  2  2

[Arrive] at POL=2, 卸了6个箱子，推进到POL=2
final:
bay0 | bay1 | bay2 | bay3
 3  3 |  3  3 |  3  X |  X  X
 3  3 |  3  3 |  3  X |  X  X
```

2. 基于1，引入了4个空位。

3. 基于1，引入了4个空位，在1港加入了1->4。

3. 基于1，引入了2个空位，在0港加入了0->4.

## Install & Run

```bash
git clone <repo-url>
cd Stowdoku
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python CSP_solver.py
```

`pip install -e .` 会一并安装 `pyproject.toml` 中声明的依赖（pandas / numpy / matplotlib）。`data/` 和 `arxiv/` 不在 git 版本控制中，需要单独拷贝到项目根目录下。