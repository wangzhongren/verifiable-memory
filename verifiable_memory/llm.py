"""OpenAI / Anthropic 兼容调用（纯标准库，读取本地配置，不回写认证信息）。

默认读取 ~/.config/verifiable-memory/config.json，本地配置优先于环境配置。
没有本地配置时，VM_API_STYLE 可显式选择 openai / anthropic；否则使用 VM_*/OPENAI_*
配置，无该组配置时选择 ANTHROPIC_*。两组配置不交叉借用密钥。
Anthropic 的 AUTH_TOKEN 使用 Bearer，API_KEY 使用 x-api-key。
温度 0 仅固定采样设置，不承诺模型输出确定性；错误不静默兜底。
"""

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 60
DEFAULT_CONFIG_PATH = Path.home() / '.config' / 'verifiable-memory' / 'config.json'


class LLMError(Exception):
    """网络、认证或协议层失败。"""


class LLM:
    @staticmethod
    def has_local_config(config_path=None):
        return Path(config_path or DEFAULT_CONFIG_PATH).is_file()

    def __init__(self, base_url=None, api_key=None, model=None, api_style=None, config_path=None):
        path = Path(config_path or DEFAULT_CONFIG_PATH)
        config = {}
        if path.exists() or config_path is not None:
            try:
                config = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise LLMError(f'无法读取本地模型配置：{path}（需要有效 JSON 文件）') from None
            allowed = {'api_style', 'base_url', 'model', 'api_key', 'auth_token'}
            if (not isinstance(config, dict) or set(config) - allowed
                    or not all(isinstance(v, str) and v.strip() for v in config.values())):
                raise LLMError('本地模型配置只允许 api_style、base_url、model、api_key、auth_token 字符串字段')
        if config:
            self.api_style = api_style or config.get('api_style', 'openai')
            self.base_url = (base_url or config.get('base_url', '')).rstrip('/')
            self.model = model or config.get('model', '')
            if not api_key and config.get('api_key') and config.get('auth_token'):
                raise LLMError('本地模型配置的 api_key 和 auth_token 只能选择一种')
            token_auth = not api_key and bool(config.get('auth_token'))
            self.api_key = api_key or config.get('auth_token') or config.get('api_key', '')
            self.auth_header = ('Authorization' if token_auth or self.api_style == 'openai'
                                else 'x-api-key')
            self._validate_settings()
            return
        configured_style = api_style or os.environ.get('VM_API_STYLE')
        if configured_style is None:
            generic = any((base_url, api_key, model)) or any(os.environ.get(k) for k in (
                'VM_BASE_URL', 'VM_API_KEY', 'VM_MODEL',
                'OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL'))
            anthropic = any(os.environ.get(k) for k in (
                'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN',
                'ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL'))
            configured_style = 'anthropic' if anthropic and not generic else 'openai'
        if configured_style not in ('openai', 'anthropic'):
            raise LLMError('VM_API_STYLE 必须是 openai 或 anthropic')
        self.api_style = configured_style
        family = 'ANTHROPIC' if self.api_style == 'anthropic' else 'OPENAI'
        self.base_url = (base_url or os.environ.get('VM_BASE_URL')
                         or os.environ.get(f'{family}_BASE_URL') or '').rstrip('/')
        self.model = model or os.environ.get('VM_MODEL') or os.environ.get(f'{family}_MODEL') or ''
        self.api_key = api_key or os.environ.get('VM_API_KEY') or ''
        self.auth_header = 'x-api-key' if self.api_style == 'anthropic' else 'Authorization'
        if not self.api_key:
            if self.api_style == 'anthropic' and os.environ.get('ANTHROPIC_AUTH_TOKEN'):
                self.api_key = os.environ['ANTHROPIC_AUTH_TOKEN']
                self.auth_header = 'Authorization'
            else:
                self.api_key = os.environ.get(f'{family}_API_KEY') or ''
        self._validate_settings()

    def _validate_settings(self):
        if self.api_style not in ('openai', 'anthropic'):
            raise LLMError('api_style 必须是 openai 或 anthropic')
        if not self.base_url or not self.api_key or not self.model:
            raise LLMError('模型配置不完整：本地配置需要 base_url、model 和 '
                           'auth_token（或 api_key）；也兼容 VM_* / OPENAI_* / ANTHROPIC_* '
                           '环境配置。同一配置来源必须完整，不跨组借用密钥。')

    @property
    def label(self):
        """不包含认证信息的协议与模型标识。"""
        return f'{self.api_style}:{self.model}'

    def _safe_error(self, message):
        return str(message).replace(self.api_key, '[REDACTED]')[:300]

    def chat(self, system, user, temperature=0.0):
        """调用所选协议，只接受非空文本响应。"""
        headers = {'Content-Type': 'application/json', self.auth_header:
                   ('Bearer ' + self.api_key if self.auth_header == 'Authorization' else self.api_key)}
        if self.api_style == 'anthropic':
            suffix = '/messages' if self.base_url.endswith('/v1') else '/v1/messages'
            url = self.base_url + suffix
            headers['anthropic-version'] = '2023-06-01'
            payload = {'model': self.model, 'max_tokens': 1024, 'temperature': temperature,
                       'system': system, 'messages': [{'role': 'user', 'content': user}]}
        else:
            url = self.base_url + '/chat/completions'
            payload = {'model': self.model, 'temperature': temperature,
                       'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user}]}
        request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = self._safe_error(exc.read().decode('utf-8', errors='replace'))
            raise LLMError(f'HTTP {exc.code}: {detail}') from None
        except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(f'请求失败: {self._safe_error(exc)}') from None
        try:
            if self.api_style == 'anthropic':
                text = '\n'.join(part['text'] for part in body['content']
                                 if part.get('type') == 'text')
            else:
                text = body['choices'][0]['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('empty or non-text response')
            return text
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise LLMError(f'{self.api_style} 响应缺少有效文本内容') from None
