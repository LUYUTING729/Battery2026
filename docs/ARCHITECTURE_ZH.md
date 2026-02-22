# MDVRP-RL BCP 项目架构详解

本文档解释项目的函数功能、调用逻辑与计算方法，帮助快速理解并二次开发。

## 1. 整体调用链

入口有两类：
- 单实例：`bpc.search.solver.solve_instance`
- 批实验：`bpc.search.solver.run_experiment`

单实例主链：
1. `load_instance` 读取 SQLite + CSV。
2. `generate_initial_columns` 生成初始列池。
3. 建根节点并进入 BCP 树循环。
4. 节点内调用 `_solve_node`：
   - `MasterProblem.solve` 解 RMP；
   - `separate_cuts` 分离 violated cuts 并回灌 RMP；
   - `extract_duals` 提取对偶；
   - `price_columns` 定价生成负化简成本列；
   - `add_columns` 增列。
5. 若节点整数则尝试更新 incumbent；否则 `pick_branch_arc` + `split_node` 分支。
6. 直到队列空、超时或达 gap 阈值。
7. 落盘：`solution.json/routes.csv/trace.csv/metrics.csv`。

## 2. 数据层与约束语义

### 2.1 输入数据
- `customers.csv`: 客户需求
- `depots.csv`: 中心容量
- `vehicles.csv`: 车辆初始位置
- `cost_matrix.csv`: 成本弧权
- `dist_matrix.csv`: 距离/能耗弧权

### 2.2 数据校验
`load_instance` 会校验：
- 必需文件存在、列名完整、数值可解析；
- 客户需求 `q_i <= U`；
- 中心容量非负；
- 只保留合法弧，且禁止 `depot -> depot`。

## 3. RMP（集合划分主问题）

在 `MasterProblem` 中实现：
- 变量：`lambda_r`（列变量）
- 约束：
  - 覆盖：每个客户恰好 1 次；
  - 车辆：每车至多 1 条路线；
  - 中心：中心出车上限；
  - 分支行：禁弧/强制弧；
  - cuts 行：RCC/Clique/SRI。

### 3.1 增量增列
`add_columns` 会把新列对所有约束的系数一次性填入。
- 覆盖系数来自 `a_i`
- 分支系数来自 `arc_flags`
- cut 系数由 `CutDefinition.coefficient` 计算

### 3.2 对偶输出
`extract_duals` 返回：
- `cover_pi`, `vehicle_alpha`, `depot_beta`
- 分支弧对偶 `branch_arc_dual`
- 由 clique cut 聚合得到的客户惩罚项

## 4. Pricing（RCSPP + ng-route）

在 `pricing/ng_pricing.py`：

### 4.1 标签状态
`Label = (node, rc, load, dist, memory, visited)`
- `rc`: 累计 reduced cost
- `memory`: ng 访问记忆
- `visited`: 用于重构路线与覆盖向量

### 4.2 扩展规则
扩展 `cur -> nxt` 时：
1. 弧可行（未被禁用，图中存在）
2. 容量与续航可行：
   - `load <= U`
   - `dist + completion_lb <= Q`
3. reduced cost 递推：
   - `+ arc_cost`
   - `- cover_pi[nxt]`（首次访问客户）
   - `- branch_arc_dual[(u,v)]`
   - `- clique_bonus[nxt]`

### 4.3 剪枝
- 2-cycle elimination
- dominance（同节点下 rc/load/dist 非劣 + memory 子集）

### 4.4 负列判定
能回中心时计算完整 `final_rc`。只有 `final_rc < 0` 的路线加入新列。

## 5. Cuts 分离

`cuts/separators.py` 实现启发式分离：
- RCC：`lhs < ceil(q(S)/U)` 视为违反
- Clique：冲突团上 `lhs > 1` 视为违反
- SRI：根节点启发式模板

分离顺序：RCC -> Clique -> SRI（根节点）。

## 6. 分支与搜索

### 6.1 分支变量
由 `aggregated_arc_flow` 计算 `x_uv = Σ λ_r b_uvr`。
选择最接近 0.5 的分数弧。

### 6.2 子节点约束
- 左支：禁弧 `x_uv <= 0`
- 右支：强制弧 `x_uv >= 1`

### 6.3 节点选择
- `best_bound`：按下界最小优先
- `depth_first`：按深度优先

## 7. 稳定化（预留）

`DualStabilizer` 支持 box/penalty/hybrid 三种方式，当前主流程尚未强耦合调用，可在后续将 `extract_duals` 结果送入 `stabilize` 后再定价。

## 8. 输出与复现

每次求解会输出：
- `solution.json`
- `routes.csv`
- `trace.csv`（迭代轨迹）
- `metrics.csv`

复现实验建议固定：
- `random_seed`
- 同一配置文件
- 相同 batch/instance 数据版本
