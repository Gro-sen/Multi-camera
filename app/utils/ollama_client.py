import os
import time
import random
from typing import Optional

import requests

DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
DEFAULT_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def generate(
        self,
        model: str,
        prompt: str,
        images=None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        timeout = timeout if timeout is not None else DEFAULT_OLLAMA_TIMEOUT
        max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES

        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        if images:
            payload["images"] = images

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "") or ""
            except requests.RequestException as e:
                last_exc = e
                # 如果最后一次仍然失败，则向上抛出
                if attempt >= max_retries:
                    raise
                # 指数退避 + 小抖动
                backoff = min(2 ** attempt, 10) + random.uniform(0, 0.5)
                time.sleep(backoff)
        # 理论上不会到这里，但为稳妥返回或抛出最后异常
        if last_exc:
            raise last_exc
        return ""

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass