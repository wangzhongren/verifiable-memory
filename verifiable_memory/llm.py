"""OpenAI 兼容 LLM 适配器（纯标准库 urllib，零第三方依赖）。

配置全部来自环境变量，密钥只从环境读取、绝不打印：
- VM_BASE_URL   形如 https://api.deepseek.com/v1（别名 OPENAI_BASE_URL）
- VM_API_KEY    密钥（别名 OPENAI_API_KEY）
- VM_MODEL      模型名（别名 OPENAI_MODEL）

温度固定 0：解析必须是可复现的功能，不是采样创作。失败只向上抛
LLMError，宿主（parser）决定是否用后备文法兜底。
"""

import json
import os
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 60


class LLMError(Exception):
    """网络、认证或协议层失败。"""


class LLM:
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or os.environ.get('VM_BASE_URL')
                         or os.environ.get('OPENAI_BASE_URL') or '').rstrip('/')
        self.api_key = api_key or os.environ.get('VM_API_KEY') or os.environ.get('OPENAI_API_KEY') or ''
        self.model = model or os.environ.get('VM_MODEL') or os.environ.get('OPENAI_MODEL') or ''
        if not self.base_url or not self.api_key or not self.model:
            raise LLMError('LLM 未配置：需要 VM_BASE_URL / VM_API_KEY / VM_MODEL'
                           '（或 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL）')

    @property
    def label(self):
        """给证据链看的标识：只含 base_url 与模型名，绝不含密钥。"""
        return f'{self.base_url}@{self.model}'

    def chat(self, system, user, temperature=0.0):
        """一次 chat completion，返回助手的文本回复。"""
        payload = {'model': self.model, 'temperature': temperature,
                   'messages': [{'role': 'system', 'content': system},
                                {'role': 'user', 'content': user}]}
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.api_key}'})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:300]
            raise LLMError(f'HTTP {exc.code}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f'请求失败: {exc}') from exc
        try:
            return body['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f'响应缺 choices/message: {body!r:.300}') from exc
