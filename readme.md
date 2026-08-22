<p align="center">
  <img src="assets/school_logo.png" alt="西北大学校徽" height="200"/>
  <img src="assets/school_name.png" alt="西北大学" height="200"/>
</p>

---

# 项目简介

基于进化算法的Triton自动优化系统，利用大语言模型进行代码重构，通过进化算法进行多代演化，自动迭代出最优的算子。

- 📘 [技术说明文档](https://pan.baidu.com/s/1SmCDd2pi6TdLaqVv-Xb6Bw?pwd=1234)
- 🎬 [幻灯片，介绍视频](https://pan.baidu.com/s/1SmCDd2pi6TdLaqVv-Xb6Bw?pwd=1234)

提取码：`1234`

---
## 🏆 核心特性

- **🧠 初始化导流**：从种子代码出发，结合rag检索结果，完成初始化不同方向生成策略发掘，提升父代质量
- **📚 RAG增强**: 基于知识库的优化策略检索，为初始化导流提供知识参考
- **🚀 并行请求**: llm并行请求，减少模型请求时间，提高优化效率
- **🎯 双卡测评**: 利用机器中的两张卡进行npu测评，提高优化效率
- **🛡️ AST门禁**: 基于抽象语法树的代码安全检查，防止非法代码进入测评，提高测试效率

## 📁 项目结构

```
/workspace/user_data/Agent/
├── RAG/                            # RAG知识库模块
│   ├── data/                       # 知识文件（优化策略文档）
│   ├── knowledge/                  # 向量索引目录
│   │   ├── index.faiss             # FAISS向量索引
│   │   └── index.pkl               # 索引元数据
│   ├── .hf_cache/                  # HuggingFace模型缓存
│   ├── kb_cli.py                   # 知识库构建命令行工具
│   └── knowledge_node.py           # 知识节点定义与检索逻辑
├── baseline/                       # 基线数据目录
│   └── baseline.json               # 官方基线耗时数据
├── datasets/                       # 输入数据集（待优化算子）
│   └── kernel_name/                # 算子目录
│       ├── kernel_name.py          # 主代码（必需）
│       ├── kernel_name_1.py        # 变体代码 1（可选）
│       ├── kernel_name_2.py~10.py  # 变体代码 2~10（可选）
│       ├── variants/               # 变体代码子目录（可选）
│       └── test_kernel_name_1.py   # 测试用例（必需）
├── output/                         # 输出目录（优化结果）
│   └── kernel_name/                # 优化后的算子目录
│       ├── kernel_name_best.py     # 最优代码
│       ├── kernel_name_v1.py       # Top 1 代码
│       ├── kernel_name_v2.py       # Top 2 代码
│       ├── kernel_name_v3.py       # Top 3 代码
│       ├── kernel_name_v4.py       # Top 4 代码
│       ├── kernel_name_v5.py       # Top 5 代码
│       └── kernel_name_stats.json  # 统计信息（耗时、加速比等）
├── saved_codes/                    # 生成代码存档（按代数保存）
├── set_env/                        # 环境变量配置目录
├── .gitignore                      # Git忽略文件
├── config.py                       # 配置模块（进化参数、LLM配置）
├── evolutionary_algorithm.py       # 进化算法核心（种群管理、世代迭代）
├── executor.py                     # Triton执行器（NPU评估、双卡并行）
├── genetic_operators.py            # 遗传算子（变异、交叉、策略分析、代码清洗）
├── llm_interface.py                # LLM接口（模型调用、Token统计、重试机制）
├── main.py                         # 命令行入口
├── optimizer_agent.py              # 优化器Agent主类（协调各组件）
└── readme.md                       # 项目说明文档
```

## 🚀 快速开始

### 1.申请机器

- Triton-Ascend : 3.2.1
- CANN          : 9.0.0 
- Torch-npu     : 2.7.1
- Python        : 3.11

### 2.安装依赖

```bash
pip install torch triton langchain langchain-openai httpx
```

### 3.appikey配置

编辑 `set_env/set_api_test.py`，配置可用的DeepSeek apikey,apiurl

### 4.向量模型导入

系统使用BGE中文向量模型进行算子特征检索，配置如下：

```python
embedding_model = "BAAI/bge-small-zh-v1.5"  # 向量模型
cache_folder = "./RAG/.hf_cache"            # 模型本地缓存目录路径
```

**向量模型下载**：

首次运行时，系统会自动从 HuggingFace 镜像下载模型到 `./RAG/.hf_cache` 目录。

```bash
# 设置HuggingFace镜像源（国内加速镜像站）
export HF_ENDPOINT=https://hf-mirror.com

将知识库文档放入`RAG/data/`目录，然后运行构建脚本：

# 放入知识文档（支持 .txt, .md, .pdf, .docx 等格式）
# 文档格式要求：每条优化规则以 ⚛️ 或 ACTION_ 开头

# 构建 FAISS 向量索引
cd RAG
python kb_cli.py --build

# 测试检索功能
python kb_cli.py --search
```

### 5.运行优化

```bash
# 优化单个算子
python main.py --input-dir ./datasets --output-dir ./output --kernel _act_quant_kernel

# 优化多个算子
python main.py --input-dir ./datasets --output-dir ./output --kernel _act_quant_kernel _rms_norm_kernel

# 优化目录下所有算子
python main.py --input-dir ./datasets --output-dir ./output

# 自定义进化参数
python main.py --input-dir ./datasets --output-dir ./output \
               --population-size 10 --max-generations 5

# 启用调试模式
python main.py --input-dir ./datasets --output-dir ./output --debug
```

## 🔧 配置说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `population_size` | int | 6 | 种群大小 |
| `max_generations` | int | 2 | 最大进化代数 |
| `crossover_rate` | float | 0.8 | 交叉概率 |
| `mutation_rate` | float | 0.5 | 变异概率 |
| `elite_ratio` | float | 0.34 | 精英保留比例 |
| `llm_temperature` | float | 0.2 | 生成温度 |
| `max_llm_tokens` | int | 16888 | 最大 Token 数 |
| `model_switch_prob` | float | 0.2 | 变异时模型切换概率 |
| `baseline_json` | str | ./baseline/baseline.json | 基线数据路径 |

## 🎯 优化流程

```
  Step 1: 加载种子代码                                        
    └─ 读取 baseline_code 和多个 seed_codes                   
                                                             
  Step 2: RAG 策略发掘                         
    └─ 对种子进行npu测评 → 根据种子源码和profiling生成算子瓶颈 → 检索知识库 → 生成优化策略蓝图             
                                                             
  Step 3: 生成初始种群                      
    └─ 根据策略蓝图生成多个候选算子                   
                                                             
  Step 4: NPU压测                             
    └─ 双卡并行测试 → 计算加速比和Fitness               
                                                             
  Step 5: 进化迭代                         
    ├─ 选择: 锦标赛选择                                       
    ├─ 交叉: 基于speedup选择主干父代，嫁接指令/超参           
    ├─ 变异: 根据speedup和世代锁定结构，仅做微调
    └─ Step 4                
                                                             
  Step 6: 输出结果                                            
    └─ 保存Top 5代码 + 统计信息             
```

## 🧬 遗传算子

### 变异类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `param_tuning` | 超参与 Launch 调优 | 已达到 1.8x |
| `arithmetic_and_mask` | 算术降级与指令精简 | 已达到 1.8x |
| `structure_rewrite` | 网格维度与分块结构重构 | 初始阶段或加速比 < 1.8x |

### 交叉策略

- **主干父代选择**: 基于相对Baseline的 `speedup` 选择
- **嫁接规则**: 仅允许移植供体父代的局部指令优化和超参配置
- **骨架保护**: 100% 保留主干父代的 Grid 映射、指针偏移算式和片上布局

## 📊 评估指标

- **Fitness**: 相对种子代码的加速比（`seed_time / current_time`）
- **Speedup**: 相对官方 Baseline 的加速比（`baseline_time / current_time`）
- **Success Rate**: 代码通过功能测试和性能测试的比例

## 📝 对官方文件的修改

对官方主文件修改的说明（相应修改部分已做注释说明）：

1. **`executor.py`**
   - **增加了里面的一点参数，以实现双卡测评**

2. **`set_env/`**
   - **配置了自己的apiKey和apiUrl**

3. **`genetic_operators.py`、`evolutionary_algorithm.py`**
   - **重点修改文件，详见具体文件**

4. **`llm_interface.py`**
   - **引入线程锁机制，彻底避免多线程调用llm的计数错乱，以实现模型的并发调用。**
   - **精准提取API返回的Token数量，并增加单次调用耗时（s）监控，便于调试。**

5. **`optimizer_agent.py`**
   - **增加了一句代码,在优化流程结束时自动调用 `self.llm.print_call_summary()`，展示LLM耗时与Token消耗统计。**

6. **`main.py`**
   - **预设HuggingFace缓存路径，确保无网络环境下加载RAG模型。**
   - **`--kernel` 参数升级支持指定多个算子进行优化（如 `--kernel kernel1 kernel2`）。**

7. **`datasets/`**
   - **修改了里面的算子的变体代码**

## 📄 许可证

MIT License
