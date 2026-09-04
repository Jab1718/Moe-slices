"""
Hugging Face Hub Uploader for Sliced Qwen3.8-Flash-Coder (85GB BF16).
Hỗ trợ upload toàn bộ checkpoint Safetensors, Config, Tokenizer và Model Card.
"""

import os
import argparse
import sys
from huggingface_hub import HfApi, create_repo, login


def upload_to_hub(model_dir: str, repo_id: str, token: str = None, private: bool = False):
    print("=" * 80)
    print(" 🚀 BẮT ĐẦU QUY TRÌNH XUẤT BẢN LÊN HUGGING FACE HUB")
    print(f"    • Thư mục Checkpoint : {model_dir}")
    print(f"    • Target Repository   : {repo_id}")
    print(f"    • Private Repo        : {private}")
    print("=" * 80)

    # 1. Xác thực Token
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("[*] Đang đăng nhập Hugging Face bằng token cung cấp...")
        login(token=hf_token)
    else:
        print("[*] Sử dụng phiên đăng nhập Hugging Face có sẵn trong hệ thống...")

    api = HfApi(token=hf_token)

    # 2. Tạo Repository trên Hugging Face nếu chưa tồn tại
    try:
        print(f"[*] Kiểm tra / Tạo Repository: {repo_id} ...")
        create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
        print(f"[✓] Repository đã sẵn sàng: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[!] Thông báo tạo repo: {e}")

    # 3. Tải toàn bộ thư mục lên Hub với cơ chế multi-part chunked upload
    print(f"\n[*] Đang upload toàn bộ checkpoint (~85GB BF16) lên https://huggingface.co/{repo_id} ...")
    print("    (Quá trình này sử dụng streaming chunk upload với thanh tiến trình trực tiếp)")

    try:
        future = api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Qwen3.8-Flash-Coder-85GB-BF16 (Sliced MoE Subnet 160 Experts, 1-cycle DoRA Calibration)",
            ignore_patterns=["*.tmp", "*.pyc", "__pycache__"]
        )
        print("\n" + "=" * 80)
        print(f" 🎉 XUẤT BẢN THÀNH CÔNG RỰC RỠ LÊN HUGGING FACE HUB!")
        print(f" 🔗 Link truy cập mô hình: https://huggingface.co/{repo_id}")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Lỗi khi upload: {e}")
        print("💡 Gợi ý: Hãy kiểm tra token của bạn có quyền WRITE (Token Role: Write) tại: https://huggingface.co/settings/tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Sliced MoE Model to Hugging Face Hub")
    parser.add_argument("--model-dir", type=str, default="./qwen3.8_flash_coder_85gb_bf16", help="Đường dẫn thư mục model")
    parser.add_argument("--repo-id", type=str, default="Jab1718/qwen3.8-flash-coder-85gb-bf16", help="Tên repo trên HF")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write Permission)")
    parser.add_argument("--private", action="store_true", help="Đặt repo ở chế độ Private")
    args = parser.parse_args()

    upload_to_hub(args.model_dir, args.repo_id, args.token, args.private)
