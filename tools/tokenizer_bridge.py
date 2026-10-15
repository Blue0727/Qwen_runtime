import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Convert text and token IDs using the local tokenizer."
    )
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    # 根据脚本位置定位模型，不依赖当前工作目录。
    root = Path(__file__).resolve().parents[1]
    tokenizer_path = (
        root / "models" / "Qwen2.5-0.5B" / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if args.mode == "encode":
        # 兼容 UTF-8 BOM，保留文本中的空格和换行。
        text = args.input_path.read_text(encoding="utf-8-sig")

        # Base 模型直接续写原文，不额外添加特殊 token。
        ids = tokenizer.encode(
            text, add_special_tokens=False
        ).ids

        args.output_path.write_text(
            json.dumps(ids) + "\n",
            encoding="utf-8",
        )
        print("Encoded tokens:", len(ids))

    else:
        ids = json.loads(
            args.input_path.read_text(encoding="utf-8-sig")
        )

        # 整段解码，避免逐 token 解码破坏多字节字符。
        text = tokenizer.decode(ids, skip_special_tokens=True)
        args.output_path.write_text(text, encoding="utf-8")
        print("Decoded tokens:", len(ids))


if __name__ == "__main__":
    main()