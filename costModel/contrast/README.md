# Triton-Ascend Official Cost Model Baseline

该目录提供项目基线：直接调用 Triton-Ascend 安装中的官方 `costmodel_bench`，不编译 Kernel、不执行测试、不做硬件采样。

## 数据来源

脚本读取：

- `costModel/datasets/<operator>/data/<role>/metadata.json`
- TTIR：`costModel/datasets/<operator>/data/<role>/ir/ttir/*.ttir.mlir`
- 实测延迟标签：metadata 或 `profiling/profile_summary.json`

候选只在同一 operator、相同输入/输出 shape 和 dtype 的 workload 内进行排名比较。

## 运行

从 `/workspace` 执行：

```bash
python Agent/costModel/contrast/run_official_costmodel.py --dry-run
python Agent/costModel/contrast/run_official_costmodel.py
```

常用筛选参数：

```bash
python Agent/costModel/contrast/run_official_costmodel.py --operator _rms_norm_kernel --limit 20
python Agent/costModel/contrast/run_official_costmodel.py --roles p0,p50,p100
python Agent/costModel/contrast/run_official_costmodel.py --no-auto-pid
```

参数绑定配置会生成或读取：

```text
Agent/costModel/contrast/official_costmodel_bindings.json
```

优先级是：`default` < `operators` < `candidates` < metadata 中的绑定字段。TTIR 中出现 `tt.get_program_id` 时，脚本默认补充静态 `pid_x/y/z=0`；出现 `tt.get_num_programs` 且缺少绑定时，会在候选清单中标记。

## 当前预测语义

官方文档中的返回值是静态 Cost Model 估计，不等价于完整 grid 的端到端硬件延迟。因此本基线将主要评价指标设为：

- group 内 Spearman
- group 内 Kendall tau-b
- pairwise ranking accuracy
- Top-1 accuracy / 选中实际最优候选比例
- Top-5 hit rate
- selection regret

不把 MAE、MAPE、RMSE 作为主指标，避免把静态估计和完整 kernel 实测时间错误地当成同一量纲。

## 输出

每次运行会生成一个时间戳目录：

- `candidate_inventory.csv`：候选、TTIR、绑定、实测标签和 workload 签名
- `predictions.csv`：每个候选的官方 Cost Model 预测
- `skipped_candidates.csv`：跳过原因
- `summary.json`：汇总指标和运行信息

## 已验证环境

当前环境已验证：

- Triton-Ascend 3.2.0
- `triton.backends.ascend.runtime.costmodel_runtime.costmodel_bench` 可导入
- 单候选调用可成功返回预测值

## 简洁预测入口

需要快速查看官方 Cost Model 预测和已有实测延迟时，使用：

```bash
python Agent/costModel/contrast/predict_costmodel.py
python Agent/costModel/contrast/predict_costmodel.py --operator _rms_norm_kernel --role p0 --role p50 --role p100
python Agent/costModel/contrast/predict_costmodel.py --limit 20
python Agent/costModel/contrast/predict_costmodel.py --limit 20 --json
```

表格输出每行包含：

```text
operator  role  candidate  predicted_us  real_us  real_source  status
```

其中 `predicted_us` 来自 Triton-Ascend 官方 `costmodel_bench`；`real_us` 和 `real_source` 来自当前数据集 metadata 或 profile_summary 中的实测完整 Kernel 标签（如果数据存在）。当某个候选没有有效实测标签时，脚本会显示 `-`。
