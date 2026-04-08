# MDVRP 输出方案（按文件字段细化）

基于 [BPC_MONITORING_METHODOLOGY.md](/Users/lu/Desktop/MDVRP/docs/BPC_MONITORING_METHODOLOGY.md) 的字段字典，结合当前代码实现（`src/bpc/search/solver.py`），给出一版可落地的输出方案。

## 1. 设计原则

1. 运行开始即创建目录和核心文件（含表头/占位 JSON）。
2. 迭代级、节点级数据采用 append-only。
3. 单例摘要文件采用覆盖更新（始终保留最新状态）。
4. 字段分为通用算法字段与业务字段，优先避免混写。

## 2. 目录结构（建议）

```text
run_xxx/
  run_config.json
  run_result.json
  solution.json
  routes.csv
  metrics.csv
  trace.csv
  bp_snapshots.csv
  label_setting/
    labels.csv
    pricing_stats.csv
  rmp/
    columns.csv
    selected_columns.csv
    coverage_duals.csv
    solution.json
  branch/
    node_*_candidates.json
    node_*_strong_eval.json
    node_*_decision.json
  analysis/
    incumbent_updates.csv
    node_summary.csv
    root_bound_events.csv
    root_gap_series.csv
    cut_violations.csv
    infeasible_stats.csv
    node_depths.csv
    node_time_breakdown.csv
    column_lifetimes.csv
    run_summary.json
```

## 3. 核心求解层（当前优先落地）

### 3.1 `run_config.json`

- 写入方式：启动时一次性写入（overwrite）
- 触发时机：run 初始化
- 字段：
  - `instance_id`
  - `config`（完整 SolverConfig）
  - `start_time`
  - `code_version`（可选，git sha）

### 3.2 `solution.json`

- 写入方式：启动时占位 + 结束时覆盖
- 触发时机：run 初始化 / run 结束
- 字段：
  - `status`
  - `obj_primal`
  - `obj_dual`
  - `gap`
  - `stats`（对象）
  - `routes`（数组，每个元素含 `vehicle_id,depot_id,customer_seq,cost,load,dist`）

### 3.3 `routes.csv`

- 写入方式：启动时建表头，结束时覆盖全量最终解
- 触发时机：run 初始化 / run 结束
- 字段：
  - `vehicle_id`
  - `depot_id`
  - `customer_seq`
  - `cost`
  - `load`
  - `dist`

### 3.4 `metrics.csv`

- 写入方式：启动时建表头，结束时覆盖
- 触发时机：run 初始化 / run 结束
- 字段：
  - `metric`
  - `value`

### 3.5 `trace.csv`

- 写入方式：append-only（实时）
- 触发时机：每次 CG 迭代 / 异常节点返回
- 字段（当前实现）：
  - `node_id`
  - `depth`
  - `cg_iter`
  - `lp_obj`
  - `cuts_added`
  - `new_columns`
  - `active_columns`
  - `iter_time_sec`
  - `cuts_added_rcc`
  - `cuts_added_clique`
  - `cuts_added_sri`
  - `is_integral`
  - `art_violation_post_solve`

### 3.6 `bp_snapshots.csv`

- 写入方式：append-only（实时）
- 触发时机：每次 CG 迭代
- 字段（当前实现）：
  - `instance_id`
  - `node_id`
  - `depth`
  - `cg_iter`
  - `lp_obj_before_pricing`
  - `cuts_added_this_iter`
  - `cuts_added_rcc`
  - `cuts_added_clique`
  - `cuts_added_sri`
  - `new_columns_added`
  - `active_columns`
  - `is_integral`
  - `art_violation`
  - `iter_runtime_sec`

### 3.7 `analysis/node_summary.csv`

- 写入方式：append-only（实时）
- 触发时机：节点处理完成或剪枝时
- 字段（当前实现）：
  - `instance_id`
  - `node_id`
  - `depth`
  - `is_root`
  - `node_runtime_sec`
  - `node_lb`
  - `node_ub_if_integer`
  - `node_gap`
  - `node_is_integer`
  - `cg_iters`
  - `columns_start`
  - `columns_added_total`
  - `columns_end`
  - `cuts_added_total`
  - `cuts_added_rcc`
  - `cuts_added_clique`
  - `cuts_added_sri`
  - `pricing_calls`
  - `pricing_negative_found_calls`
  - `rmp_solve_calls`
  - `cut_sep_calls`
  - `rmp_time_sec`
  - `cut_sep_time_sec`
  - `pricing_time_sec`
  - `art_violation_final`
  - `prune_reason`
  - `fail_stage`

### 3.8 `branch/branch_decisions.csv`

- 写入方式：append-only（实时）
- 触发时机：每次分支决策后
- 字段（当前实现）：
  - `instance_id`
  - `parent_node_id`
  - `depth`
  - `frac_arc_count`
  - `selected_arc_u`
  - `selected_arc_v`
  - `selected_arc_flow`
  - `selected_arc_dist_to_half`
  - `left_child_id`
  - `right_child_id`

### 3.9 `analysis/incumbent_updates.csv`

- 写入方式：append-only（实时）
- 触发时机：incumbent 更新时
- 字段（当前实现）：
  - `instance_id`
  - `node_id`
  - `depth`
  - `update_reason`
  - `old_ub`
  - `new_ub`
  - `improvement`
  - `global_time_sec`
  - `route_count`

### 3.10 `run_result.json`

- 写入方式：覆盖更新（实时快照）
- 触发时机：初始化后、每个节点后、run 结束
- 字段（当前实现）：
  - `instance_id`
  - `status`
  - `obj_primal`
  - `obj_dual`
  - `gap`
  - `runtime_sec`
  - `nodes_processed`
  - `global_columns`
  - `global_columns_final`
  - `best_lb`
  - `root_lb`
  - `nodes_pruned_by_bound`
  - `nodes_pruned_infeasible`
  - `max_open_nodes`
  - `max_depth`
  - `cg_iters_total`
  - `rmp_solve_calls_total`
  - `cuts_separation_calls_total`
  - `pricing_calls_total`
  - `time_rmp_sec`
  - `time_cut_sep_sec`
  - `time_pricing_sec`
  - `time_branch_sec`
  - `time_other_sec`
  - `time_to_first_feasible_sec`
  - `time_to_best_incumbent_sec`
  - `solve_seed`
  - `fail_after_cuts_nodes`
  - `fail_before_cuts_nodes`
  - `fail_infeasible_by_columns_nodes`

## 4. 扩展层（按方法论文档建议新增）

### 4.1 Pricing 层

- `label_setting/labels.csv`（debug+）
  - `iteration,rank,path,served,cost,red_cost,time,energy,latest_departure`
- `label_setting/pricing_stats.csv`（standard+）
  - `iteration,num_negative_rc_columns,min_reduced_cost,strong_negative_count,weak_negative_count,time_elapsed`

### 4.2 RMP 层

- `rmp/columns.csv`（standard+，append-only）
  - `id,cost,len_path,served_size,served,duration,energy,dep_window,meta,creation_iteration,creation_node_id,pricing_mode`
- `rmp/selected_columns.csv`（standard+）
  - `id,x_value,cost,served_size,served,meta`
- `rmp/coverage_duals.csv`（standard+）
  - `customer,dual_pi`
- `rmp/solution.json`（standard+）
  - 顶层：`objective,picked,duals`
  - `picked[]`：`id,x,cost,served,meta`

### 4.3 Branch 节点 JSON 层（debug+）

- `branch/node_*_candidates.json`
  - 顶层：`node_id,depth,lower_bound,candidates`
  - `candidates[]`：`var_id,x_value,score,down_lb,up_lb,eval_time,reason`
- `branch/node_*_strong_eval.json`
  - 同上，强调 `down_lb/up_lb/score/reason`
- `branch/node_*_decision.json`
  - 顶层：`node_id,depth,lower_bound,decision,candidates`
  - `decision`：`chosen,reason,left_fixes,right_fixes`
  - `left_fixes/right_fixes[]`：`var_id,lb,ub`

### 4.4 Analysis 层（standard+）

- `analysis/root_bound_events.csv`
  - `time,event_type,bound_before,bound_after,delta_bound,node_id`
- `analysis/root_gap_series.csv`
  - `time,iteration,gap`
- `analysis/cut_violations.csv`
  - `cut_id,time,max_violation_before,max_violation_after`
- `analysis/infeasible_stats.csv`
  - `iteration,time,num_subproblem_calls,num_infeasible_subproblems,ratio`
- `analysis/node_depths.csv`
  - `node_id,depth,status`
- `analysis/node_time_breakdown.csv`
  - `node_id,time_rmp,time_pricing,time_cut_separation,time_subproblem,total_time`
- `analysis/column_lifetimes.csv`
  - `column_id,creation_iteration,times_selected_in_rmp,lifetime_iterations`
- `analysis/run_summary.json`
  - `instance_name,variant_name,status,root_gap_end,infeasible_ratio,columns_generated`

## 5. 监控等级与文件开关

1. `minimal`：`solution.json,routes.csv,metrics.csv,monitoring/run_summary.csv,trace.csv`
2. `standard`：`minimal + monitoring 全套 + analysis/* + pricing_stats + selected_columns + coverage_duals`
3. `debug`：`standard + labels.csv + branch/node_*_*.json + model.lp`
4. `forensic`：`debug + dual drift + RC consistency + 内存/树宽监控`

## 6. 实施顺序（建议）

1. 固化核心 10 个文件（第 3 节）字段与写入策略。
2. 增加 `run_config.json` 与 `monitor_level` 配置。
3. 接入 Pricing/RMP 扩展文件（第 4.1、4.2）。
4. 接入 Branch JSON 和 Analysis 层（第 4.3、4.4）。
