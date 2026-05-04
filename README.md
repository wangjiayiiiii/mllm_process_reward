# Project A: MLLM Complex Reasoning Evaluation Pipeline

本项目对应代码考核中的 **项目 A：复杂推理与过程评估（MLLM & Process Reward）**。

目标是从零实现一个面向多模态大模型的评测流水线，覆盖：

- 真实多模态数据集下载与统一格式转换
- Qwen2.5-VL-3B-Instruct 推理
- 手写答案解析、归一化、匹配和 Accuracy 计算
- Self-consistency / best-of-N 推理
- Verifier / reranker 探索
- Oracle@N 与 bad case analysis

本项目没有使用 `VLMEvalKit`、`OpenCompass`、`lm-evaluation-harness` 等现成评测框架。

## 1. 项目结构

```text
.
├── configs/
│   ├── model.yaml
│   ├── eval.yaml
│   ├── eval_mathvision_testmini.yaml
│   ├── eval_vstar_bench.yaml
│   ├── verifier.yaml
│   └── verifier_mathvision_n4.yaml
├── data/
│   ├── samples/
│   ├── processed/
│   └── raw/
├── scripts/
│   ├── prepare_real_datasets.py
│   ├── run_inference.py
│   ├── run_eval.py
│   ├── run_self_consistency.py
│   ├── analyze_candidates.py
│   ├── rule_verifier_rerank.py
│   ├── build_verifier_data.py
│   ├── train_verifier.py
│   ├── score_with_verifier.py
│   ├── mllm_verifier_rerank.py
│   └── summarize_bad_cases.py
├── src/aibox_project_a/
│   ├── data/
│   ├── prompts/
│   ├── inference/
│   ├── eval/
│   ├── verifier/
│   └── utils/
├── outputs/
│   ├── predictions/
│   ├── eval_reports/
│   └── verifier/
├── requirements.txt
├── 运行说明.txt
└── 实验记录.txt
```

## 2. 环境安装

建议使用 Linux GPU 服务器，Python 3.10+。

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch`
- `transformers`
- `vllm`
- `openai`
- `datasets`
- `qwen-vl-utils`
- `scikit-learn`
- `accelerate`

## 3. 模型服务

正式推理使用 vLLM OpenAI-compatible API。

先启动 Qwen2.5-VL-3B-Instruct：

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --limit-mm-per-prompt '{"image":1,"video":0}'
```

然后确认 `configs/model.yaml` 中配置如下：

```yaml
model:
  name: Qwen/Qwen2.5-VL-3B-Instruct
  backend: vllm
  api_base: http://localhost:8000/v1
  api_key: EMPTY
```

如果只想测试代码流程，可以把 `backend` 改成 `mock`，不需要 GPU。

## 4. 数据准备

本项目使用两个数据集：

- `MathVision-testmini`
- `V*Bench`

运行：

```bash
python scripts/prepare_real_datasets.py \
  --dataset mathvision_testmini \
  --output data/processed/mathvision_testmini.jsonl

python scripts/prepare_real_datasets.py \
  --dataset vstar_bench \
  --output data/processed/vstar_bench.jsonl
```

统一后的 JSONL schema 主要包含：

```json
{
  "id": "sample_id",
  "question": "question text",
  "choices": ["choice A", "choice B"],
  "answer": "ground truth answer",
  "image_path": "path/to/image.jpg",
  "dataset": "mathvision",
  "metadata": {}
}
```

## 5. 一键运行核心流程

前提：vLLM 服务已经启动，且 `configs/model.yaml` 的 backend 为 `vllm`。

下面命令会依次完成数据准备、greedy baseline、self-consistency、Oracle@4 分析和 bad case analysis：

```bash
set -e

python scripts/prepare_real_datasets.py \
  --dataset mathvision_testmini \
  --output data/processed/mathvision_testmini.jsonl

python scripts/prepare_real_datasets.py \
  --dataset vstar_bench \
  --output data/processed/vstar_bench.jsonl

python scripts/run_inference.py \
  --input data/processed/mathvision_testmini.jsonl \
  --output outputs/predictions/mathvision_testmini_predictions.jsonl \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/mathvision_testmini_predictions.jsonl \
  --report outputs/eval_reports/mathvision_testmini_report.json \
  --bad-cases outputs/eval_reports/mathvision_testmini_bad_cases.jsonl

python scripts/run_inference.py \
  --input data/processed/vstar_bench.jsonl \
  --output outputs/predictions/vstar_bench_predictions.jsonl \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/vstar_bench_predictions.jsonl \
  --report outputs/eval_reports/vstar_bench_report.json \
  --bad-cases outputs/eval_reports/vstar_bench_bad_cases.jsonl

python scripts/run_self_consistency.py \
  --input data/processed/mathvision_testmini.jsonl \
  --output outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl \
  --candidates-output outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --n 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl \
  --report outputs/eval_reports/mathvision_self_consistency_n4_report.json \
  --bad-cases outputs/eval_reports/mathvision_self_consistency_n4_bad_cases.jsonl

python scripts/analyze_candidates.py \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --report outputs/eval_reports/mathvision_self_consistency_n4_candidates_report.json \
  --details-output outputs/eval_reports/mathvision_self_consistency_n4_candidates_details.jsonl

python scripts/summarize_bad_cases.py \
  --predictions outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --report outputs/eval_reports/mathvision_bad_case_summary.json \
  --samples-output outputs/eval_reports/mathvision_bad_case_samples.jsonl \
  --max-samples-per-category 8
```

更完整的分步骤命令见 `运行说明.txt`。

## 6. Greedy Baseline

MathVision-testmini：

```bash
python scripts/run_inference.py \
  --input data/processed/mathvision_testmini.jsonl \
  --output outputs/predictions/mathvision_testmini_predictions.jsonl \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/mathvision_testmini_predictions.jsonl \
  --report outputs/eval_reports/mathvision_testmini_report.json \
  --bad-cases outputs/eval_reports/mathvision_testmini_bad_cases.jsonl
```

V*Bench：

```bash
python scripts/run_inference.py \
  --input data/processed/vstar_bench.jsonl \
  --output outputs/predictions/vstar_bench_predictions.jsonl \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/vstar_bench_predictions.jsonl \
  --report outputs/eval_reports/vstar_bench_report.json \
  --bad-cases outputs/eval_reports/vstar_bench_bad_cases.jsonl
```

## 7. Self-consistency N=4

MathVision-testmini：

```bash
python scripts/run_self_consistency.py \
  --input data/processed/mathvision_testmini.jsonl \
  --output outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl \
  --candidates-output outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --n 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl \
  --report outputs/eval_reports/mathvision_self_consistency_n4_report.json \
  --bad-cases outputs/eval_reports/mathvision_self_consistency_n4_bad_cases.jsonl
```

V*Bench：

```bash
python scripts/run_self_consistency.py \
  --input data/processed/vstar_bench.jsonl \
  --output outputs/predictions/vstar_self_consistency_n4_predictions.jsonl \
  --candidates-output outputs/predictions/vstar_self_consistency_n4_candidates.jsonl \
  --n 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --batch-size 1

python scripts/run_eval.py \
  --predictions outputs/predictions/vstar_self_consistency_n4_predictions.jsonl \
  --report outputs/eval_reports/vstar_self_consistency_n4_report.json \
  --bad-cases outputs/eval_reports/vstar_self_consistency_n4_bad_cases.jsonl
```

## 8. Oracle@4 分析

```bash
python scripts/analyze_candidates.py \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --report outputs/eval_reports/mathvision_self_consistency_n4_candidates_report.json \
  --details-output outputs/eval_reports/mathvision_self_consistency_n4_candidates_details.jsonl

python scripts/analyze_candidates.py \
  --candidates outputs/predictions/vstar_self_consistency_n4_candidates.jsonl \
  --report outputs/eval_reports/vstar_self_consistency_n4_candidates_report.json \
  --details-output outputs/eval_reports/vstar_self_consistency_n4_candidates_details.jsonl
```

## 9. Verifier / Reranker 探索

### 9.1 Rule verifier

```bash
python scripts/rule_verifier_rerank.py \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --output outputs/predictions/mathvision_rule_verifier_n4_predictions.jsonl \
  --scores-output outputs/eval_reports/mathvision_rule_verifier_n4_scores.jsonl

python scripts/run_eval.py \
  --predictions outputs/predictions/mathvision_rule_verifier_n4_predictions.jsonl \
  --report outputs/eval_reports/mathvision_rule_verifier_n4_report.json \
  --bad-cases outputs/eval_reports/mathvision_rule_verifier_n4_bad_cases.jsonl
```

### 9.2 Learned DistilBERT verifier

```bash
python scripts/build_verifier_data.py \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --train-output outputs/verifier/mathvision_n4_train.jsonl \
  --valid-output outputs/verifier/mathvision_n4_valid.jsonl \
  --valid-ratio 0.2

python scripts/train_verifier.py \
  --config configs/verifier_mathvision_n4.yaml

python scripts/score_with_verifier.py \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --verifier-dir outputs/verifier/mathvision_n4_checkpoints \
  --output outputs/predictions/mathvision_trained_verifier_n4_predictions.jsonl \
  --report outputs/eval_reports/mathvision_trained_verifier_n4_report.json \
  --scores-output outputs/eval_reports/mathvision_trained_verifier_n4_scores.jsonl \
  --report-subset-ids-from outputs/verifier/mathvision_n4_valid.jsonl \
  --batch-size 16
```

### 9.3 MLLM-as-Verifier

```bash
python scripts/mllm_verifier_rerank.py \
  --model-config configs/model.yaml \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --output outputs/predictions/mathvision_mllm_verifier_n4_predictions.jsonl \
  --report outputs/eval_reports/mathvision_mllm_verifier_n4_report.json \
  --judge-output outputs/eval_reports/mathvision_mllm_verifier_n4_judge.jsonl
```

## 10. Bad Case Analysis

```bash
python scripts/summarize_bad_cases.py \
  --predictions outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl \
  --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl \
  --report outputs/eval_reports/mathvision_bad_case_summary.json \
  --samples-output outputs/eval_reports/mathvision_bad_case_samples.jsonl \
  --max-samples-per-category 8

python scripts/summarize_bad_cases.py \
  --predictions outputs/predictions/vstar_bench_predictions.jsonl \
  --candidates outputs/predictions/vstar_self_consistency_n4_candidates.jsonl \
  --report outputs/eval_reports/vstar_bad_case_summary.json \
  --samples-output outputs/eval_reports/vstar_bad_case_samples.jsonl \
  --max-samples-per-category 8
```

## 11. 实验结果

| Dataset | Method | Accuracy | Parse Success Rate | Note |
| --- | --- | ---: | ---: | --- |
| MathVision-testmini | Greedy baseline | 12.50% | 100% | 原始 prompt |
| MathVision-testmini | Prompt v2 | 12.17% | 100% | 强化视觉检查与计算验证，未提升 |
| MathVision-testmini | Self-consistency N=4 majority | 21.38% | 100% | MathVision 主结果 |
| MathVision-testmini | Rule verifier rerank | 21.05% | 100% | 未超过 majority |
| MathVision-testmini | DistilBERT verifier valid subset | 21.67% | 100% | 文本 verifier，看不到图 |
| MathVision-testmini | MLLM-as-Verifier | 17.43% | 100% | Qwen 3B judge 未超过 majority |
| MathVision-testmini | Oracle@4 | 39.80% | - | reranker 理论上限 |
| V*Bench | Greedy baseline | 74.87% | 100% | V*Bench 主结果 |
| V*Bench | Self-consistency N=4 majority | 71.73% | 100% | 低于 greedy baseline |
| V*Bench | Oracle@4 | 87.43% | - | 候选集合上限较高 |

核心发现：

- MathVision-testmini 上 self-consistency 从 12.50% 提升到 21.38%，说明 test-time scaling 对复杂视觉数学推理有效。
- V*Bench 上 greedy baseline 已经较高，self-consistency majority 反而下降，说明采样候选可能引入噪声。
- 两个数据集的 Oracle@4 都明显高于实际选择结果，说明多候选生成有潜力，瓶颈在 verifier/reranker。
- 当前 rule verifier、DistilBERT learned verifier、Qwen2.5-VL-3B judge 都没有稳定超过 majority voting，说明轻量 verifier 选择能力不足。

## 12. Execution Logs 要求

建议最终提交时保留以下文本 log 或终端截图，用于证明代码真实可执行。

### 必需日志

1. 环境与依赖安装日志

```text
pip install -r requirements.txt
```

证明 `requirements.txt` 可复现环境。

2. vLLM 服务启动日志

```text
vllm serve Qwen/Qwen2.5-VL-3B-Instruct ...
```

需要看到模型加载成功、OpenAI-compatible server 正常监听端口。

3. 数据准备日志

```text
python scripts/prepare_real_datasets.py --dataset mathvision_testmini ...
python scripts/prepare_real_datasets.py --dataset vstar_bench ...
```

需要看到数据成功写入 `data/processed/*.jsonl`，图片保存路径正常。

4. Greedy baseline 推理与评测日志

```text
python scripts/run_inference.py --input data/processed/mathvision_testmini.jsonl ...
python scripts/run_eval.py --predictions outputs/predictions/mathvision_testmini_predictions.jsonl ...
python scripts/run_inference.py --input data/processed/vstar_bench.jsonl ...
python scripts/run_eval.py --predictions outputs/predictions/vstar_bench_predictions.jsonl ...
```

日志中应包含进度条、保存路径、最终 `accuracy` 和 `parse_success_rate`。

5. Self-consistency N=4 日志

```text
python scripts/run_self_consistency.py --input data/processed/mathvision_testmini.jsonl ...
python scripts/run_eval.py --predictions outputs/predictions/mathvision_self_consistency_n4_predictions.jsonl ...
```

推荐同时保留 V*Bench 的 self-consistency 日志。

6. Oracle@4 分析日志

```text
python scripts/analyze_candidates.py --candidates outputs/predictions/mathvision_self_consistency_n4_candidates.jsonl ...
python scripts/analyze_candidates.py --candidates outputs/predictions/vstar_self_consistency_n4_candidates.jsonl ...
```

需要看到 `oracle_at_n`、`majority_accuracy`、`oracle_majority_gap`。

7. Verifier / reranker 日志

至少保留以下三类之一，推荐三类都保留：

```text
python scripts/rule_verifier_rerank.py ...
python scripts/train_verifier.py --config configs/verifier_mathvision_n4.yaml
python scripts/mllm_verifier_rerank.py ...
```

这些日志用于证明完成了进阶任务 Verifier/RM Design 的探索。

8. Bad case analysis 日志

```text
python scripts/summarize_bad_cases.py ...
```

需要看到 `candidate_generation_failure` 与 `reranker_or_voting_failure` 的统计。

### 建议保存的日志文件名

```text
logs/00_install.txt
logs/01_vllm_server.txt
logs/02_prepare_data.txt
logs/03_mathvision_baseline.txt
logs/04_vstar_baseline.txt
logs/05_mathvision_self_consistency.txt
logs/06_vstar_self_consistency.txt
logs/07_oracle_analysis.txt
logs/08_rule_verifier.txt
logs/09_trained_verifier.txt
logs/10_mllm_verifier.txt
logs/11_bad_case_analysis.txt
```

如果用终端截图，也建议截图中包含完整命令、进度条结束状态和最终输出字典。

## 13. 关键实现说明

- `src/aibox_project_a/eval/parser.py`：解析 `Final Answer:`、`\boxed{}` 等最终答案。
- `src/aibox_project_a/eval/normalizer.py`：归一化数字、LaTeX、选项文本。
- `src/aibox_project_a/eval/matcher.py`：实现规则匹配与容错比较。
- `scripts/run_eval.py`：手写 Accuracy 评测入口。
- `scripts/run_self_consistency.py`：N=4 多候选生成与多数投票。
- `scripts/analyze_candidates.py`：Oracle@N 与候选多样性分析。
- `scripts/*verifier*.py`：rule verifier、learned verifier、MLLM-as-verifier。

## 14. 结论

最终推荐主结果：

- MathVision-testmini：Self-consistency N=4 majority，Accuracy 21.38%。
- V*Bench：Greedy baseline，Accuracy 74.87%。

Verifier/RM Design 的核心发现：

- Oracle@4 显示候选集合中存在较多正确答案。
- 当前 3B 模型和轻量 verifier 还不能稳定选出正确候选。
- 后续更有潜力的方向是训练可看图的 multimodal verifier，或使用更强 MLLM 作为 judge。
