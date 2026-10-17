import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]


class GenerateRequest(BaseModel):
    """网页提交的生成参数。"""

    text: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1)


def read_message(process: subprocess.Popen) -> dict:
    """读取一行 JSON；进程退出或协议损坏时报告错误。"""

    line = process.stdout.readline()

    if not line:
        raise RuntimeError("C++ 推理进程已断开，请查看服务终端日志")

    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError("C++ 返回了无效的 JSON") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时创建共享资源，关闭时回收 C++ 进程。"""

    tokenizer = Tokenizer.from_file(
        str(ROOT / "models" / "Qwen2.5-0.5B" / "tokenizer.json")
    )

    # 使用参数列表启动，路径中的空格和中文由 subprocess 处理。
    process = subprocess.Popen(
        [
            str(ROOT / "build" / "Release" / "qwen_chat.exe"),
            "16",
            "q4",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    try:
        # C++ 加载完成并返回 ready 后，服务才开始接收请求。
        ready = read_message(process)

        if ready.get("type") != "ready":
            raise RuntimeError("C++ 推理进程没有返回 ready")

        app.state.tokenizer = tokenizer
        app.state.process = process
        app.state.capacity = ready["context_capacity"]
        app.state.lock = Lock()

        yield

    finally:
        # 关闭输入管道，让 C++ 的 getline 循环读到 EOF 后退出。
        try:
            process.stdin.close()
        except OSError:
            pass

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        process.stdout.close()


app = FastAPI(
    title="Qwen Runtime",
    lifespan=lifespan,
)
@app.get("/")

def index():
    """返回本地聊天页面。"""
    return FileResponse(ROOT / "tools" / "index.html")


@app.post("/generate")
def generate(body: GenerateRequest, request: Request):
    """完成一次：文字编码 → C++ 推理 → 新增文本解码。"""

    state = request.app.state

    # 一次请求占用完整的收发过程，避免其他请求读走当前结果。
    with state.lock:
        input_ids = state.tokenizer.encode(
            body.text,
            add_special_tokens=False,
        ).ids

        if not input_ids:
            raise HTTPException(
                status_code=400,
                detail="输入没有产生有效 token",
            )

        available = state.capacity - len(input_ids)

        if body.max_new_tokens > available:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"输入占用 {len(input_ids)} 个 token，"
                    f"当前最多还能生成 {max(0, available)} 个 token，"
                    "请缩短输入或降低生成上限"
                ),
            )

        payload = {
            "input_ids": input_ids,
            "max_new_tokens": body.max_new_tokens,
        }

        process = state.process

        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()

            result = read_message(process)

        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            ) from exc

        if result.get("type") == "error":
            raise HTTPException(
                status_code=502,
                detail=result["message"],
            )

        if result.get("type") != "result":
            raise HTTPException(
                status_code=502,
                detail="C++ 返回了未知的响应类型",
            )

        output_ids = result["output_ids"]

        # 整段解码，避免逐 token 解码拆开多字节字符。
        text = state.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )

        return {
            "text": text,
            "prompt_tokens": len(input_ids),
            "generated_tokens": len(output_ids),
            "prefill_ms": result["prefill_ms"],
            "decode_ms": result["decode_ms"],
            "decode_forward_calls": result["decode_forward_calls"],
        }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        workers=1,
        reload=False,
    )