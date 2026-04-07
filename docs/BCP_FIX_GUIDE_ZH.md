# MDVRP Branch-and-Price 项目问题修复指南（含数理推导）

本文针对当前实现中已识别的关键问题，给出：
- 数学层面的原因与正确性影响；
- 工程落地的修复路径；
- 建议的代码改造点与验证方案。

适用代码版本：当前仓库 `src/bpc/*`。

---

## 1. 问题一：定价 reduced cost 与主问题不一致（最关键）

### 1.1 现状
当前 RMP 中包含：
- 覆盖约束；
- 车辆约束；
- 车场约束；
- 分支弧约束；
- 以及 `RCC / SRI / Clique` cut。

但定价中 reduced cost 仅使用了：
- 覆盖对偶；
- 车辆、车场对偶；
- 分支弧对偶；
- clique customer bonus。

即 `RCC/SRI` 的对偶没有进入定价。

### 1.2 数理推导
将 RMP 写成标准形式（最小化）：

\[
\min_{\lambda \ge 0} \sum_{r\in\Omega} c_r\lambda_r
\]

约束可统一写作
\[
A\lambda = 1,\quad B\lambda \le b,\quad G\lambda \ge g.
\]

对应对偶变量分别为
- 覆盖等式：\(\pi\)（自由）；
- \(\le\) 约束：\(\alpha\le 0\)；
- \(\ge\) 约束：\(\gamma\ge 0\)。

任意列 \(r\) 的 reduced cost 必须是
\[
\bar c_r = c_r - \pi^\top a_r - \alpha^\top b_r - \gamma^\top g_r.
\]

只要 RMP 中存在有效约束（例如 RCC/SRI）而其对偶未进入 \(\bar c_r\)，则“\(\bar c_r\ge 0\) 对所有列成立”不再等价于当前受限主问题最优。于是列生成停止条件在数学上失效。

### 1.3 影响
- 可能“假收敛”：报告 `new_columns=0`，但实际仍有违反完整对偶定价条件的列。
- 报告的 root bound 可能偏弱甚至不可信（对应当前 cut-augmented RMP）。
- 不能严格宣称 BCP 的最优性证明链条完整。

### 1.4 修复建议
1. 在 `DualValues` 中新增通用 cut 对偶容器，例如：
   - `cut_duals: Dict[str, float]`（cut_id -> Pi）；
   - 或更结构化：`rcc_duals`, `sri_duals`, `clique_duals`。
2. 在 `MasterProblem.extract_duals()` 中导出全部 cut dual。
3. 在定价中统一 reduced cost 计算：
   \[
   \bar c_r = c_r - \sum_i \pi_i a_{ir} - \alpha_{v(r)} - \beta_{d(r)} - \sum_{(u,v)}\mu_{uv}b_{uvr} - \sum_{k\in\mathcal{K}}\eta_k\,\phi_k(r)
   \]
   其中 \(\phi_k(r)\) 为 cut \(k\) 对列 \(r\) 的系数。
4. 对无法鲁棒进入 DP 的 cut（典型是复杂 SRI）采用策略分离：
   - 方案 A：根节点只加“pricing-compatible” cuts；
   - 方案 B：切换为分层定价（启发式 DP + 必要时 MIP pricing）；
   - 方案 C：cut-and-price 的外层循环中控制非鲁棒 cut 的启用层级。

### 1.5 代码落点
- `src/bpc/core/types.py`：扩展 `DualValues`。
- `src/bpc/rmp/master_problem.py`：`extract_duals` 导出所有 cut dual。
- `src/bpc/pricing/ng_pricing.py`：统一 reduced cost 递推与收尾公式。

---

## 2. 问题二：分支节点缺少可行化（Phase-I）导致潜在误剪枝

### 2.1 现状
当前节点若 `rmp.solve()` 不可行，则直接返回 `inf` 并跳过。该判定基于“当前列池”，不是“该分支节点真实可行域”。

### 2.2 数理解释
列生成框架中，节点主问题真实可行性是：
\[
\exists\lambda\ge0\ \text{s.t.}\ A\lambda=1,\ B\lambda\le b,\ G\lambda\ge g
\]
其中 \(\lambda\) 定义在完整列集 \(\Omega\)。

而你当前求的是受限列集 \(\Omega'\subset\Omega\) 下的可行性。若
\[
\mathcal{F}(\Omega')=\emptyset,\quad \mathcal{F}(\Omega)\neq\emptyset,
\]
则“RMP infeasible”仅说明列集不够，不应剪枝。

### 2.3 影响
- 可能把本应继续生成列的节点直接删除，破坏完整性。
- 尤其在 `forced_arc` 分支时更容易发生。

### 2.4 修复建议
1. 增加 Phase-I（人工变量）模型：
   - 覆盖/强制分支等关键约束加入松弛变量 \(s\ge0\)；
   - 先最小化 \(M\sum s\)（或二阶段：先 \(\sum s\)，后原目标）；
   - 若最优 \(\sum s=0\)，说明节点在扩充列后可达可行。
2. 当 RMP 不可行时，不立刻剪枝，先进入“可行化定价”模式：
   - 优先生成满足 forced arc 的列；
   - 必要时用 MIP pricing 构造至少一条可行列。
3. 增加节点状态 `infeasible_by_columns` 与日志字段，区分“真不可行”与“列不足不可行”。

### 2.5 代码落点
- `src/bpc/rmp/master_problem.py`：新增 `solve_phase1()`。
- `src/bpc/search/solver.py`：`_solve_node` 中分流 infeasible 处理逻辑。
- `src/bpc/pricing/ng_pricing.py`：支持“满足 forced arcs 的优先定价”。

---

## 3. 问题三：整数停滞提前退出不是严格收敛条件

### 3.1 现状
当“解整数 + 目标不变 + 活跃列签名不变”连续达到阈值就提前 break。

### 3.2 数学风险
列生成的严格停止通常要求：
- 完整定价问题无负 reduced-cost 列（对当前 RMP 对偶）。

“基解稳定”是经验信号，不是必要充分条件。若提前退出，则存在漏列风险。

### 3.3 修复建议
- 将该逻辑降级为“早停候选”，必须附加一次严格验证：
  1. 执行一次强化定价（提高搜索深度/关闭启发式截断）；
  2. 或执行 MIP pricing 校验无负列；
  3. 仅当验证通过才允许停止。
- 提供配置开关：`enable_integral_stall_early_stop`（默认关闭）。

### 3.4 代码落点
- `src/bpc/search/solver.py`：替换 `stall_count` 的直接 break。

---

## 4. 问题四：稳定化模块未接入主循环

### 4.1 现状
配置里 `enable_stabilization=true`，但求解循环直接使用原始对偶进行定价。

### 4.2 理论背景
对偶振荡会导致：
- 反复生成相似列；
- 迭代尾部效率差；
- 根节点列数膨胀。

稳定化可理解为在对偶空间做投影/正则：
- Box: \(\tilde\pi = \Pi_{[c-w,c+w]}(\pi)\)
- Penalty: \(\tilde\pi=(1-\omega)\pi+\omega c\)

### 4.3 修复建议
1. 在每个节点维护 stabilizer（或对偶中心）。
2. `extract_duals()` 后先稳定化，再用于定价。
3. 对 root 与非 root 使用不同参数（root 更强）。
4. 增加诊断指标：
   - dual drift；
   - 重复列比例；
   - 每次定价负列数量。

### 4.4 代码落点
- `src/bpc/search/solver.py`：引入 `DualStabilizer` 调用链。
- `src/bpc/stabilization/dual_stabilizer.py`：扩展支持结构化 dual（不止扁平 dict）。

---

## 5. 问题五：实验几乎全部 `nodes_processed=1`

### 5.1 现象解释
`nodes_processed=1` 并不必然错误，但在一整批实例都如此时，应优先排查：
- 根节点 LP 是否总能整数（模型结构/列池偏强）；
- 或“分支条件未被触发”（例如只要 LP 分数但 pick_arc 返回空）；
- 或早停逻辑与截断导致未进入分支。

### 5.2 修复建议
1. 在 trace 增加分支诊断字段：
   - `is_integral`, `fractional_arc_count`, `chosen_branch_arc`。
2. 增加 root 强校验：若 `is_integral=True` 但仍有显著负列候选，标记 warning。
3. 在实验报告中增加“分支树深度统计/分支触发率”。

### 5.3 代码落点
- `src/bpc/search/solver.py`：trace 扩字段。
- `src/bpc/cli/experiment_demand.py`：汇总分支统计。

---

## 6. 问题六：cut 分离贡献弱（尤其 SRI 模板过粗）

### 6.1 现状
你当前 trace 中常见 `cuts_added=0`，说明分离器在现有实例/列空间下命中率低。

### 6.2 原因
- RCC 枚举仅到子集大小 3，覆盖有限；
- SRI 使用固定 0.5 权重模板，针对性弱；
- clique 默认配置常关闭。

### 6.3 修复建议
1. 将 cut 模块分为两类：
   - pricing-compatible（优先）；
   - pricing-disruptive（谨慎，必要时仅 root）。
2. RCC 候选改为“基于分数解支持集”的自适应生成，而非全局小枚举。
3. SRI 改为基于当前分数覆盖行的启发式 lifting，而非固定模板。
4. 给每轮分离增加 KPI：`violated_count`, `max_violation`, `effective_cuts_added`。

### 6.4 代码落点
- `src/bpc/cuts/separators.py`：候选生成策略重写。
- `src/bpc/search/solver.py`：记录 cut efficacy 指标。

---

## 7. 问题七：`solve_seed` 未真正驱动求解随机性

### 7.1 现状
实验记录了 `solve_seed`，但求解主流程几乎未使用随机数。

### 7.2 影响
- 难以解释“不同 seed 运行”差异来源；
- 实验可重复性字段与算法行为不一致。

### 7.3 修复建议
- 明确两种策略二选一：
  1. **确定性模式**：移除 `solve_seed`；
  2. **随机化模式**：在以下环节显式用 RNG：
     - 同分支候选打破平局；
     - pricing 扩展顺序扰动；
     - cut 候选采样。
- 将 `seed` 写入 `solution.json` 与 trace 头信息。

---

## 8. 关键修复的“最小可行路线图（建议按顺序）”

1. **第一阶段（正确性优先）**
- 修复 reduced cost 与所有 active constraints 一致性；
- 增加 Phase-I 防误剪枝；
- 关闭整数停滞早停（或加严格校验）。

2. **第二阶段（收敛效率）**
- 接入 dual stabilization；
- 改造 cut 分离策略和统计；
- 加强定价验证（启发式 + 精确校验）。

3. **第三阶段（实验可信度）**
- 完善 trace/metrics 字段；
- 分支触发率、cut有效率、定价成功率三类诊断看板；
- 统一 deterministic/reproducible 实验协议。

---

## 9. 建议新增的验证实验（必须）

### 9.1 单元测试
- `test_reduced_cost_consistency_with_cuts`：随机抽列，比较“显式对偶公式”与定价内部计算。
- `test_phase1_no_false_prune`：构造受限列池不可行但全列可行的节点。
- `test_stall_stop_guard`：验证早停前严格定价校验会阻止误停。

### 9.2 回归实验
- 同一实例对比：
  1. 当前版本；
  2. 修复后（仅正确性改动）；
  3. 修复后+稳定化。
- 比较指标：
  - root LP 值；
  - 最终最优值；
  - nodes processed；
  - 总列数；
  - runtime。

### 9.3 正确性审计
每个节点保存并可复算：
- 当前对偶；
- 新增列的 reduced cost 组成项；
- 停止时最优性证据（“无负列”校验日志）。

---

## 10. 推荐的数据结构改造（便于长期维护）

将 reduced cost 统一改成“约束插件求和”模式：

\[
\bar c_r = c_r - \sum_{\kappa\in\mathcal{C}} y_\kappa\,\phi_\kappa(r)
\]

其中：
- \(\mathcal{C}\) 是当前激活约束集合（含 cuts、branching、基础约束）；
- 每类约束都实现 `coefficient(route_or_label)`；
- 定价和 RMP 共用同一系数接口，避免日后再出现不一致。

这能从架构上杜绝“RMP 加了约束但定价忘记加对偶项”的问题。

---

## 11. 结论

从理论完整性上，你现在最需要优先修的是两件事：
1. **定价与主问题严格一致（包含 cut 对偶）**；
2. **节点不可行判定从“受限列池不可行”升级为“真不可行”**。

只要这两点落地，项目就会从“工程上能跑”跃迁到“可做严谨最优性声明”的 BCP 实现。后续再叠加稳定化和更强分离，性能会更可控。
