# Triton-Ascend Cost Model 预测入口

只保留一个入口：根据算子目录读取已有 TTIR，调用 Triton-Ascend 官方 `costmodel_bench`，并在控制台打印预测值。

## 使用

从 `/workspace/Agent` 运行：

```bash
python costModel/contrast/predict_costmodel.py --operator _rms_norm_kernel
```

算子名就是 `costModel/datasets` 下的目录名，例如：

```bash
python costModel/contrast/predict_costmodel.py --operator _act_quant_kernel
python costModel/contrast/predict_costmodel.py --operator solve_tril_16x16_kernel
```

## 参数绑定

可选绑定配置：

```text
costModel/contrast/official_costmodel_bindings.json
```

脚本会读取 metadata 和绑定配置，并默认将 TTIR 中出现的 `pid_x/y/z` 设为 `0`。

## 输出

```text
role  candidate  predicted_us  real_us
```

- `predicted_us`：Triton-Ascend 官方 Cost Model 静态预测
- `real_us`：datasets 中已有的 msprof 实测延迟
