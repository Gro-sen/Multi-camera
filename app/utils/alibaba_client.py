from typing import Optional
import time
import random

try:
    from openai import OpenAI
    import openai as _openai_module
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    _openai_module = None
    HAS_OPENAI = False

try:
    import httpx
except Exception:
    httpx = None


class AlibabaOpenAIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        if not HAS_OPENAI:
            raise ImportError("openai库未安装。请运行: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url

    def generate(
        self,
        model: str,
        prompt: str,
        images=None,
        options: Optional[dict] = None,
        format: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> str:
        options = options or {}
        temperature = options.get("temperature", 0.1)
        top_p = options.get("top_p", 1.0)

        if images:
            image_b64 = images[0]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{"role": "user", "content": prompt}]

        # 带重试的调用，处理临时网络/连接错误
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout
                )
                return completion.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                # 特定错误类型识别（如果可用）
                is_conn_err = False
                if httpx is not None and isinstance(e, getattr(httpx, "ConnectError", ())):
                    is_conn_err = True
                if _openai_module is not None and isinstance(e, getattr(_openai_module, "APIConnectionError", ())):
                    is_conn_err = True

                # 对连接类错误尝试重试；对其它明显不可重试错误可立即抛出
                if attempt >= max_retries or not is_conn_err:
                    raise

                backoff = min(2 ** attempt, 10) + random.uniform(0, 0.5)
                time.sleep(backoff)

        # 达到这里说明重试失败，抛出最后一次异常
        if last_exc:
            raise last_exc
        return ""

    def close(self):
        # OpenAI client may not require close; try best-effort
        try:
            if hasattr(self.client, "close"):
                self.client.close()
            elif hasattr(self.client, "session"):
                try:
                    self.client.session.close()
                except Exception:
                    pass
        except Exception:
            pass