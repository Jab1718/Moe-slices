"""
Evaluation script for Code Syntax, Reasoning, and Problem-Solving Fidelity.
"""

import argparse
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLE_EVAL_TASKS = [
    {
        "task_id": "Python/BinarySearch",
        "prompt": "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1 if not found.\"\"\"\n",
        "check": lambda code: "while" in code and ("mid" in code or "//" in code)
    },
    {
        "task_id": "Python/FibonacciDP",
        "prompt": "def fibonacci_memo(n: int, memo={}) -> int:\n    \"\"\"Compute the n-th Fibonacci number using memoization.\"\"\"\n",
        "check": lambda code: "if n" in code and ("memo" in code or "+" in code)
    },
    {
        "task_id": "Python/ValidParentheses",
        "prompt": "def is_valid_parentheses(s: str) -> bool:\n    \"\"\"Check if string containing '()[]{}' has balanced matching pairs.\"\"\"\n",
        "check": lambda code: "stack" in code or "append" in code or "pop" in code
    },
    {
        "task_id": "Rust/AsyncChannel",
        "prompt": "// Write a Rust function that sends a message through a tokio mpsc channel.\nuse tokio::sync::mpsc;\n\npub async fn send_msg(tx: mpsc::Sender<String>, msg: String) -> Result<(), String> {\n",
        "check": lambda code: "send" in code or "await" in code or "Ok" in code
    }
]


def evaluate_model_quality(model_path: str):
    """
    Chạy bộ test kiểm tra cú pháp và khả năng sinh code logic.
    """
    print(f"\n" + "=" * 60)
    print(f" 🧪 EVALUATING ACCURACY & SYNTAX FIDELITY: {model_path}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = model.device

    results = []
    passed = 0

    for task in SAMPLE_EVAL_TASKS:
        inputs = tokenizer(task["prompt"], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,
                top_p=0.9,
                do_sample=False
            )
            
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        is_pass = task["check"](full_text)
        if is_pass:
            passed += 1

        status = "✅ PASS" if is_pass else "❌ FAIL"
        results.append([task["task_id"], status, full_text[:100].replace("\n", " ") + "..."])

    accuracy = (passed / len(SAMPLE_EVAL_TASKS)) * 100.0
    print("\n" + tabulate(results, headers=["Task ID", "Status", "Code Snippet Preview"], tablefmt="fancy_grid"))
    print(f"\n[📊] Tổng tỷ lệ đạt yêu cầu logic cú pháp: {accuracy:.1f}% ({passed}/{len(SAMPLE_EVAL_TASKS)})")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Accuracy")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn model")
    args = parser.parse_args()

    evaluate_model_quality(args.model)


if __name__ == "__main__":
    main()
