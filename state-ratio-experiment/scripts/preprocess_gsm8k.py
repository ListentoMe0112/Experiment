"""
Preprocess the GSM8K dataset to parquet format.
Adapted from verl/examples/data_preprocess/gsm8k.py
"""

import argparse
import os
import re

import datasets


def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/gsm8k",
        help="The save directory for the preprocessed dataset.",
    )
    args = parser.parse_args()

    data_source = "openai/gsm8k"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        'Let\'s think step by step and output the final answer after "####".'
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"GSM8K saved to {local_save_dir}/")
    print(f"  train: {len(train_dataset)} examples")
    print(f"  test:  {len(test_dataset)} examples")
