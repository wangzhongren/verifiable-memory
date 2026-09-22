"""模型接口与 CLI 回归：本地 HTTP 假服务，无外部网络或真实密钥。"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifiable_memory import data, parser
from verifiable_memory import llm as llm_module
from verifiable_memory.llm import LLM, LLMError
from verifiable_memory.session import Session


class LLMChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # 测试绝不读取开发者真实模型配置；CLI 使用独立空文件隔离。
        config_patch = patch.object(llm_module, 'DEFAULT_CONFIG_PATH', self.root / 'default-config.json')
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.env_config = self.root / 'env-only.json'
        self.env_config.write_text('{}', encoding='utf-8')
        self.requests = []
        self.responses = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append({'path': self.path, 'headers': dict(self.headers), 'body': body})
                status, response = owner.responses.pop(0) if owner.responses else (500, {'error': 'fixture exhausted'})
                encoded = json.dumps(response).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f'http://127.0.0.1:{self.server.server_port}/proxy'
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('VM_', 'OPENAI_', 'ANTHROPIC_'))}
        self.env.update({'ANTHROPIC_BASE_URL': self.base, 'ANTHROPIC_AUTH_TOKEN': 'fixture-secret',
                         'ANTHROPIC_MODEL': 'fixture-model[1m]'})

    def reply(self, op, style='anthropic'):
        text = json.dumps(op, ensure_ascii=False)
        body = ({'content': [{'type': 'text', 'text': text}]} if style == 'anthropic'
                else {'choices': [{'message': {'content': text}}]})
        self.responses.append((200, body))

    def cli(self, *args, expected=0, force_llm=True):
        command = [sys.executable, str(ROOT / 'cli.py'), '--session', str(self.root / 'memory.db'),
                   '--config', str(self.env_config)]
        if force_llm:
            command.append('--llm')
        proc = subprocess.run([*command, *map(str, args)], cwd=self.root, env=self.env,
                              capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)
        return proc

    def local_config(self, **overrides):
        config = {'api_style': 'anthropic', 'base_url': self.base, 'model': 'local-model',
                  'auth_token': 'local-secret', **overrides}
        path = self.root / 'default-config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        return path

    def test_default_local_config_needs_no_environment(self):
        self.local_config()
        self.reply(data.query_record('x'))
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(LLM.has_local_config())
            llm = LLM()
            self.assertEqual(llm.model, 'local-model')
            llm.chat('s', 'u')
        self.assertEqual(self.requests[0]['headers']['Authorization'], 'Bearer local-secret')

    def test_local_config_precedes_environment_without_mixing(self):
        path = self.local_config()
        with patch.dict(os.environ, self.env, clear=True):
            self.assertEqual(LLM().api_key, 'local-secret')
            config = json.loads(path.read_text())
            del config['auth_token']
            path.write_text(json.dumps(config))
            with self.assertRaises(LLMError):
                LLM()  # 本地文件不完整时不得借用环境里的另一把密钥

    def test_bad_local_config_fails_without_echoing_secrets(self):
        path = self.root / 'invalid.json'
        for content in ('{"auth_token":"private-value",', '["private-value"]',
                        '{"auth_token": null}', '{"typo": "private-value"}'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(LLMError) as caught:
                    LLM(config_path=path)
                self.assertNotIn('private-value', str(caught.exception))
        with self.assertRaises(LLMError):
            LLM(config_path=self.root / 'missing.json')
        path = self.local_config(api_key='other-key')
        with self.assertRaises(LLMError):
            LLM(config_path=path)

    def test_local_config_auto_parses_natural_language_only(self):
        path = self.local_config()
        self.env = {k: v for k, v in self.env.items()
                    if not k.startswith(('VM_', 'OPENAI_', 'ANTHROPIC_'))}
        self.reply(data.teach_fact('项目主干', 'main'))
        proc = self.cli('--config', path, '--json', 'teach', '请记下项目主干是main。', force_llm=False)
        self.assertEqual(json.loads(proc.stdout)['source'], 'llm')
        self.assertEqual(len(self.requests), 1)
        proc = self.cli('--config', path, '--json', 'ask', '查询 项目主干', force_llm=False)
        self.assertEqual(json.loads(proc.stdout)['source'], 'fallback')
        self.assertEqual(len(self.requests), 1)
        self.cli('--config', path, '--no-llm', 'ask', '项目主干是什么？', expected=1, force_llm=False)
        self.assertEqual(len(self.requests), 1)

    def test_anthropic_auth_token_and_proxy_path(self):
        self.reply(data.teach_fact('项目主干', 'main'))
        with patch.dict(os.environ, self.env, clear=True):
            llm = LLM()
        got = llm.chat('system', 'user')
        self.assertEqual(json.loads(got)['content'], 'main')
        request = self.requests[0]
        self.assertEqual(request['path'], '/proxy/v1/messages')
        self.assertEqual(request['headers']['Authorization'], 'Bearer fixture-secret')
        self.assertEqual(request['headers']['Anthropic-Version'], '2023-06-01')
        self.assertNotIn('X-Api-Key', request['headers'])
        self.assertEqual(request['body']['system'], 'system')
        self.assertEqual(request['body']['model'], 'fixture-model[1m]')
        self.assertGreater(request['body']['max_tokens'], 0)
        self.assertNotIn('fixture-secret', llm.label)

    def test_anthropic_api_key_and_existing_v1_suffix(self):
        env = {k: v for k, v in self.env.items() if k != 'ANTHROPIC_AUTH_TOKEN'}
        env.update(ANTHROPIC_API_KEY='api-key', ANTHROPIC_BASE_URL=self.base + '/v1/')
        self.responses.append((200, {'content': [{'type': 'thinking', 'thinking': 'ignored'},
                                                {'type': 'text', 'text': 'one'},
                                                {'type': 'text', 'text': 'two'}]}))
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(LLM().chat('s', 'u'), 'one\ntwo')
        self.assertEqual(self.requests[0]['path'], '/proxy/v1/messages')
        self.assertEqual(self.requests[0]['headers']['X-Api-Key'], 'api-key')
        self.assertNotIn('Authorization', self.requests[0]['headers'])

    def test_openai_compatibility_and_configuration_precedence(self):
        env = dict(self.env, OPENAI_BASE_URL=self.base + '/v1', OPENAI_API_KEY='openai-key',
                   OPENAI_MODEL='openai-model')
        self.reply(data.query_record('x'), 'openai')
        with patch.dict(os.environ, env, clear=True):
            llm = LLM()
            self.assertEqual(llm.api_style, 'openai')
            self.assertEqual(json.loads(llm.chat('s', 'u')), data.query_record('x'))
        request = self.requests[0]
        self.assertEqual(request['path'], '/proxy/v1/chat/completions')
        self.assertEqual(request['headers']['Authorization'], 'Bearer openai-key')
        self.assertEqual(request['body']['messages'][0], {'role': 'system', 'content': 's'})
        with patch.dict(os.environ, dict(env, VM_API_STYLE='anthropic'), clear=True):
            self.assertEqual(LLM().api_key, 'fixture-secret')

    def test_partial_configuration_does_not_borrow_other_family_key(self):
        with patch.dict(os.environ, dict(self.env, OPENAI_BASE_URL=self.base), clear=True):
            with self.assertRaises(LLMError):
                LLM()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMError):
                LLM()
            with self.assertRaises(LLMError):
                LLM(api_style='invalid')

    def test_generic_and_explicit_constructor_overrides(self):
        env = dict(self.env, VM_API_STYLE='anthropic', VM_BASE_URL=self.base + '/v1',
                   VM_API_KEY='generic-key', VM_MODEL='generic-model')
        with patch.dict(os.environ, env, clear=True):
            llm = LLM()
            self.assertEqual((llm.api_key, llm.model, llm.auth_header),
                             ('generic-key', 'generic-model', 'x-api-key'))
            llm = LLM(self.base, 'explicit-key', 'explicit-model', api_style='openai')
            self.assertEqual((llm.api_style, llm.api_key, llm.model),
                             ('openai', 'explicit-key', 'explicit-model'))

    def test_http_error_masks_key_and_cli_does_not_traceback(self):
        self.responses.append((401, {'error': 'denied fixture-secret'}))
        proc = self.cli('ask', '查询 x', expected=1)
        self.assertIn('HTTP 401', proc.stderr)
        self.assertIn('[REDACTED]', proc.stderr)
        self.assertNotIn('fixture-secret', proc.stdout + proc.stderr)
        self.assertNotIn('Traceback', proc.stderr)
        session = Session.load(self.root / 'memory.db')
        self.addCleanup(session.close)
        self.assertEqual(session.n_ops, 0)

    def test_network_error_masks_key(self):
        with patch.dict(os.environ, self.env, clear=True):
            llm = LLM()
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('fixture-secret')):
            with self.assertRaises(LLMError) as caught:
                llm.chat('s', 'u')
        self.assertNotIn('fixture-secret', str(caught.exception))

    def test_malformed_responses_raise_llmerror(self):
        for style, body in [('openai', {'choices': [{'message': {'content': None}}]}),
                            ('openai', {'choices': []}), ('anthropic', {'content': []}),
                            ('anthropic', {'content': [{'type': 'text', 'text': None}]}),
                            ('anthropic', {'content': [None]}), ('anthropic', None)]:
            with self.subTest(style=style, body=body), patch.dict(os.environ, {}, clear=True):
                self.responses.append((200, body))
                with self.assertRaises(LLMError):
                    LLM(self.base, 'key', 'model', api_style=style).chat('s', 'u')

    def test_native_anthropic_cli_fact_correction_and_replay(self):
        cases = [('teach', '记下项目主干为main', data.teach_fact('项目主干', 'main')),
                 ('teach', '证据格式为session@v1', data.teach_fact('证据格式', 'session@v1')),
                 ('ask', '项目主干是什么？', data.query_record('项目主干')),
                 ('correct', '证据格式改为session@v2', data.correct_fact('证据格式', 'session@v2')),
                 ('ask', '证据格式是什么？', data.query_record('证据格式'))]
        for category, text, op in cases:
            self.reply(op)
            result = json.loads(self.cli('--json', category, text).stdout)
            self.assertEqual(result['status'], 'ok')
            if op['op'] == 'query_record':
                self.assertEqual(result['result']['answer'], 'main' if op['name'] == '项目主干' else 'session@v2')
        evidence, replayed, verified = [self.root / name for name in ('session.json', 'replay.json', 'verify.json')]
        self.cli('export', '--out', evidence)
        for script, args in [('replay.py', ['--session', evidence, '--out', replayed]),
                             ('verify.py', ['--session', evidence, '--replayed', replayed, '--out', verified])]:
            proc = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                  capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(verified.read_text())['n_failures'], 0)

    def test_paraphrases_cli_handles_both_two_and_three_field_dicts(self):
        for _, expected in data.PARAPHRASES:
            self.reply(expected)
        target = self.root / 'results' / 'paraphrases.json'
        result = self.cli('paraphrases', '--out', target)
        self.assertIn('10/10', result.stdout)
        report = json.loads(target.read_text())
        self.assertEqual(report['protocol'], 'verifiable_memory/paraphrases@v2')
        self.assertEqual(report['matches'], 10)
        self.assertEqual(report['rows'][0]['known_slots'], [])
        for request, row, (_, expected) in zip(self.requests, report['rows'], data.PARAPHRASES):
            self.assertEqual(row['got'], expected)
            self.assertIn(json.dumps(row['known_slots'], ensure_ascii=False),
                          request['body']['messages'][0]['content'])
        before = target.read_bytes()
        self.cli('paraphrases', '--out', target, expected=1)
        self.assertEqual(len(self.requests), 10)  # 拒绝覆盖在任何网络调用之前
        self.assertEqual(target.read_bytes(), before)

    def test_paraphrases_keep_strict_scoring_and_record_mismatch(self):
        for i, (_, expected) in enumerate(data.PARAPHRASES):
            self.reply({**expected, 'content': '方向是右'} if i == 7 else expected)
        target = self.root / 'mismatch.json'
        result = self.cli('paraphrases', '--out', target, expected=1)
        self.assertIn('9/10', result.stdout)
        report = json.loads(target.read_text())
        self.assertFalse(report['rows'][7]['match'])
        self.assertEqual(report['rows'][7]['got']['content'], '方向是右')
        self.assertEqual(report['rows'][7]['expected']['content'], '方向是朝右')

    def test_benchmark_contexts_satisfy_operation_preconditions(self):
        self.assertEqual(len(data.PARAPHRASES), len(data.PARAPHRASE_KNOWN_SLOTS))
        for i, ((_, op), known) in enumerate(zip(data.PARAPHRASES, data.PARAPHRASE_KNOWN_SLOTS)):
            session = Session.create(self.root / f'context-{i}.json')
            for slot in known:
                seed = (data.teach_fact(slot['name'], 'fixture') if slot['kind'] == 'fact'
                        else data.teach_rule(slot['name'], ['反转']))
                self.assertEqual(session.apply(seed, category='teach', source='test', utterance='seed')['status'], 'ok')
            category = 'ask' if op['op'] in ('query_record', 'apply_rule') else op['op'].split('_')[0]
            self.assertEqual(session.apply(op, category=category, source='test', utterance='test')['status'], 'ok')

    def test_malformed_model_operation_retries_without_host_repair(self):
        class Stub:
            def __init__(self, replies):
                self.replies = iter(replies)
                self.calls = 0

            def chat(self, system, user):
                self.calls += 1
                return next(self.replies)

        llm = Stub(['{"op": []}', '{"op": "teach_fact", "name": "版本", "content": "v2.0"}'])
        op, source = parser.parse('记下版本为v2.0', [], llm)
        self.assertEqual((op['content'], source, llm.calls), ('v2.0', 'llm', 2))
        llm = Stub(['{"op": []}', '{"op": []}'])
        with self.assertRaises(parser.ParserError):
            parser.parse('教事实 版本：v2.0', [], llm)
        self.assertEqual(llm.calls, 2)  # 不再静默转用后备文法


if __name__ == '__main__':
    unittest.main(verbosity=2)
