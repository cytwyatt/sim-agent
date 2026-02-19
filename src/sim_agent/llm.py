from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class OpenAIClient:
    api_key: str | None
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 60

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self.base_url}/chat/completions", data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError):
            return None
        except Exception:
            return None

        try:
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None
