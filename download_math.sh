#!/usr/bin/env bash
# =============================================================================
# 下载并处理 MATH (lighteval) 数据集到当前目录 ./math/
#   - 输出: ./math/train.parquet, ./math/test.parquet
#   - 与 state-ratio-experiment/scripts/preprocess_math.py 输出格式完全一致
#   - 不依赖 verl, 仅需 datasets / pandas / pyarrow
#
# 用法:
#   bash download_math.sh                  # 输出到 ./math
#   OUT_DIR=/nfs/datasets/math bash download_math.sh
#
# 可选环境变量:
#   HF_ENDPOINT   HuggingFace 镜像 (国内/内网建议设置 https://hf-mirror.com)
#   HF_HOME       HF 缓存目录, 默认 ./.hf_cache
#   OUT_DIR       输出目录, 默认 ./math
# =============================================================================
set -euo pipefail

OUT_DIR="${OUT_DIR:-$(pwd)/math}"
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
# 如已在外部设置过 HF_ENDPOINT 则保留, 否则默认使用 hf-mirror (国内可达)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUT_DIR" "$HF_HOME"

echo ">>> OUT_DIR     = $OUT_DIR"
echo ">>> HF_HOME     = $HF_HOME"
echo ">>> HF_ENDPOINT = $HF_ENDPOINT"

# 依赖检查
python3 - <<'PY'
import sys
import importlib.util  # py3.12+ 必须显式导入
missing = [m for m in ("datasets", "pandas", "pyarrow") if importlib.util.find_spec(m) is None]
if missing:
    print(f"[ERR] 缺少依赖: {missing}\n请执行: pip install datasets pandas pyarrow", file=sys.stderr)
    sys.exit(1)
PY

python3 - "$OUT_DIR" <<'PY'
import os
import re
import sys

import datasets

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


# -----------------------------------------------------------------------------
# 与 verl.utils.reward_score.math_reward 中等价的两个工具函数
# (从 Hendrycks 的 MATH 评测代码移植, 不依赖 verl)
# -----------------------------------------------------------------------------
def last_boxed_only_string(string: str):
    """返回字符串里最后一个 \\boxed{...} 或 \\fbox{...} 子串, 找不到返回 None."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str):
    """去掉 \\boxed{...} 外壳, 返回里面的内容."""
    if s is None:
        return ""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left and s[-1] == "}", f"Bad boxed string: {s!r}"
    return s[len(left) : -1]


def extract_solution(solution_str: str) -> str:
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return ""
    try:
        return remove_boxed(boxed)
    except AssertionError:
        return ""


print(f">>> Loading {DATA_SOURCE} from huggingface (endpoint={os.environ.get('HF_ENDPOINT')}) ...", flush=True)
dataset = datasets.load_dataset(DATA_SOURCE)

train_ds = dataset["train"]
test_ds = dataset["test"]


def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("problem") + " " + INSTRUCTION
        answer = example.pop("solution")
        solution = extract_solution(answer)
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx},
        }
    return process_fn


train_ds = train_ds.map(function=make_map_fn("train"), with_indices=True)
test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)

train_path = os.path.join(out_dir, "train.parquet")
test_path = os.path.join(out_dir, "test.parquet")
train_ds.to_parquet(train_path)
test_ds.to_parquet(test_path)

print(f">>> Saved: {train_path}  ({len(train_ds)} examples)")
print(f">>> Saved: {test_path}  ({len(test_ds)} examples)")
PY

echo ""
echo ">>> Done. 你可以在训练脚本中这样使用:"
echo "    export DATASET_ROOT=$(dirname "$OUT_DIR")"
echo "    export MATH_DIR=$OUT_DIR"
