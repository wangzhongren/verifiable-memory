"""训练、自动分叉、弃权、旧模型失效与完整审计的行为测试。"""

import hashlib
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifiable_memory import policy
from verifiable_memory.audit import entry_digest
from verifiable_memory.session import Session
from verifiable_memory.storage import export_evidence
import replay
import verify


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'graph.db'
        self.model = self.root / 'branch.policy.json'
        self.training = ROOT / 'examples/branch_policy_training.jsonl'
        session = Session.create(self.db)
        self.addCleanup(session.close)
        operations = [
            {'op': 'teach_entity', 'name': 'A', 'vector': [0, 0]},
            {'op': 'teach_entity', 'name': 'B', 'vector': [1, 0]},
            {'op': 'teach_entity', 'name': 'D', 'vector': [0, 1]},
            {'op': 'teach_entity', 'name': 'C', 'vector': [1, 1]},
            {'op': 'teach_vector_action', 'name': '右移', 'delta': [1, 0]},
            {'op': 'teach_vector_action', 'name': '上移', 'delta': [0, 1]},
            {'op': 'link_entities', 'name': 'AB', 'source': 'A', 'action': '右移', 'target': 'B'},
            {'op': 'link_entities', 'name': 'AD', 'source': 'A', 'action': '上移', 'target': 'D'},
            {'op': 'link_entities', 'name': 'BC', 'source': 'B', 'action': '上移', 'target': 'C'},
        ]
        for op in operations:
            entry = session.apply(op, category='vector', source='test',
                                  utterance=json.dumps(op, ensure_ascii=False))
            self.assertEqual(entry['status'], 'ok', entry)
        session.close()

    def cli(self, *args, code=0, as_json=False):
        command = [sys.executable, str(ROOT / 'cli.py'), '--session', str(self.db)]
        if as_json:
            command.append('--json')
        proc = subprocess.run([*command, *map(str, args)], cwd=self.root,
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, code, proc.stdout + proc.stderr)
        return json.loads(proc.stdout) if as_json else proc

    def train(self):
        output = self.cli('policy-train', '--data', self.training, '--out', self.model)
        self.assertIn('13 条标注', output.stdout)
        return json.loads(self.model.read_text())

    def test_trained_policy_selects_multi_step_branch_and_audits_it(self):
        model = self.train()
        self.assertEqual(model['n_examples'], 13)
        self.assertEqual(stat.S_IMODE(self.model.stat().st_mode), 0o600)
        event = self.cli('policy-route', '--model', self.model, '--source', 'A',
                         '--query', '向右然后向上', as_json=True)
        self.assertEqual(event['status'], 'ok')
        self.assertEqual(event['result']['answer'], 'C')
        self.assertEqual(event['op']['path'], ['AB', 'BC'])
        self.assertEqual(event['result']['kind'], 'policy_route')
        self.assertFalse(event['result']['abstained'])
        self.assertEqual(event['result']['policy_sha256'], hashlib.sha256(
            self.model.read_bytes()).hexdigest())
        self.assertEqual(len(event['result']['trace']), 2)
        evidence, replayed, verified = [self.root / name for name in
                                        ('session.json', 'replayed.json', 'verified.json')]
        self.cli('export', '--out', evidence)
        for script, args in [('replay.py', ['--session', evidence, '--out', replayed]),
                             ('verify.py', ['--session', evidence, '--replayed', replayed,
                                           '--out', verified])]:
            proc = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(verified.read_text())['n_failures'], 0)

    def test_other_input_selects_other_branch(self):
        self.train()
        event = self.cli('policy-route', '--model', self.model, '--source', 'A',
                         '--query', '向上', as_json=True)
        self.assertEqual((event['op']['path'], event['result']['answer']), (['AD'], 'D'))

    def test_unseen_input_abstains_and_logs_decision(self):
        self.train()
        event = self.cli('policy-route', '--model', self.model, '--source', 'A',
                         '--query', '无关请求', code=2, as_json=True)
        self.assertTrue(event['result']['abstained'])
        self.assertIsNone(event['result']['answer'])
        self.assertEqual(event['result']['trace'], [])
        self.assertEqual(event['op']['path'], [])
        self.assertEqual(event['op']['decisions'][0]['choice'], None)
        rendered = self.cli('policy-route', '--model', self.model, '--source', 'A',
                            '--query', '无关请求', code=2)
        self.assertIn('答案：未判定', rendered.stdout)
        evidence, replayed, verified = [self.root / name for name in
                                        ('abstained-session.json', 'abstained-replayed.json',
                                         'abstained-verified.json')]
        self.cli('export', '--out', evidence)
        for script, args in [('replay.py', ['--session', evidence, '--out', replayed]),
                             ('verify.py', ['--session', evidence, '--replayed', replayed,
                                           '--out', verified])]:
            proc = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(verified.read_text())['n_failures'], 0)

    def test_holdout_and_hop_limit_abstain(self):
        self.train()
        held_out = self.root / 'held-out.jsonl'
        held_out.write_text(json.dumps({'source': 'A', 'query': '请先右移再上移',
                                        'edge': 'AB'}, ensure_ascii=False) + '\n')
        evaluated = self.cli('policy-eval', '--model', self.model,
                             '--data', held_out)
        report = json.loads(evaluated.stdout)
        self.assertEqual(report['n_examples'], 1)
        self.assertEqual(report['correct'], 0)
        self.assertEqual(report['abstentions'], 1)
        event = self.cli('policy-route', '--model', self.model, '--source', 'A',
                         '--query', '请先右移再上移', code=2, as_json=True)
        self.assertTrue(event['result']['abstained'])
        event = self.cli('policy-route', '--model', self.model, '--source', 'A',
                         '--query', '向右然后向上', '--max-hops', 1,
                         code=2, as_json=True)
        self.assertEqual(event['op']['path'], ['AB'])
        self.assertEqual(event['result']['reason'], 'max_hops')
        self.assertIsNone(event['result']['answer'])

    def test_training_is_reproducible_and_rejects_bad_labels(self):
        rows = [json.loads(line) for line in self.training.read_text().splitlines()]
        session = Session.load(self.db)
        try:
            self.assertEqual(policy.train(rows, session.store.slots),
                             policy.train(rows, session.store.slots))
            rows[0]['edge'] = 'AD'  # 合法但语义错误的标注仍可训练，非真值核验。
            self.assertNotEqual(policy.train(rows, session.store.slots)['weights'],
                                policy.train([json.loads(line) for line in
                                              self.training.read_text().splitlines()],
                                             session.store.slots)['weights'])
            rows[0]['edge'] = '不存在的边'
            with self.assertRaises(policy.PolicyError):
                policy.train(rows, session.store.slots)
            rows = [row for row in [json.loads(line) for line in
                    self.training.read_text().splitlines()] if row['edge'] != 'AD']
            with self.assertRaisesRegex(policy.PolicyError, '缺少这些有向边的正例'):
                policy.train(rows, session.store.slots)
            rows = [row for row in [json.loads(line) for line in
                    self.training.read_text().splitlines()]
                    if not (row['source'] == 'B' and row['edge'] is None)]
            with self.assertRaisesRegex(policy.PolicyError, '缺少这些源实体的弃权样本'):
                policy.train(rows, session.store.slots)
        finally:
            session.close()

    def test_graph_change_marks_policy_stale_without_writing_new_event(self):
        self.train()
        session = Session.load(self.db)
        entry = session.apply({'op': 'correct_entity', 'name': 'B', 'vector': [2, 0]},
                              category='vector', source='test', utterance='correct B')
        self.assertEqual(entry['status'], 'ok')
        before = session.n_ops
        session.close()
        result = self.cli('policy-route', '--model', self.model, '--source', 'A',
                          '--query', '向右然后向上', code=1)
        self.assertIn('模型已过期', result.stderr)
        again = Session.load(self.db)
        self.assertEqual(again.n_ops, before)
        again.close()

    def test_rejects_overwrite_and_untrained_graph(self):
        self.train()
        before = self.model.read_bytes()
        self.cli('policy-train', '--data', self.training, '--out', self.model, code=1)
        self.assertEqual(self.model.read_bytes(), before)
        bad = self.root / 'bad.jsonl'
        bad.write_text('{"source":"A","query":"错误","edge":"缺失"}\n')
        self.cli('policy-train', '--data', bad, '--out', self.root / 'bad.policy.json', code=1)

    def test_forged_policy_path_is_rejected_inside_store(self):
        session = Session.load(self.db)
        try:
            op = {'op': 'route_entities', 'name': 'C', 'source': 'A', 'query': '向右',
                  'path': ['AD', 'BC'], 'decisions': [
                      {'source': 'A', 'choice': 'AD', 'candidates': []},
                      {'source': 'D', 'choice': 'BC', 'candidates': []}],
                  'policy_sha256': '0' * 64, 'abstained': False,
                  'reason': 'leaf', 'max_hops': 8}
            event = session.apply(op, category='policy', source='test', utterance='forged')
            self.assertEqual(event['status'], 'error')
            self.assertIn('无效或反向边', event['error'])
        finally:
            session.close()

    def test_independent_verifier_detects_rehashed_route_trace(self):
        self.train()
        self.cli('policy-route', '--model', self.model, '--source', 'A',
                 '--query', '向右然后向上', as_json=True)
        session_path = self.root / 'forged-session.json'
        payload = export_evidence(self.db, session_path)
        event = payload['entries'][-1]
        event['result']['trace'][0]['after'] = ['9', '9']
        event['entry_hash'] = entry_digest(event)
        payload['log_head'] = event['entry_hash']
        session_path.write_text(json.dumps(payload, ensure_ascii=False))
        checks, answers, terminal, failures = replay.replay(payload['entries'], payload['capacity'])
        self.assertGreater(failures, 0)
        replayed = self.root / 'forged-replayed.json'
        replayed.write_text(json.dumps({
            'source_session_sha256': hashlib.sha256(session_path.read_bytes()).hexdigest(),
            'n_entries': len(payload['entries']), 'n_failures': 0,
            'terminal_state_hash': terminal, 'answers': answers, 'checks': checks,
        }))
        _, problems = verify.verify(session_path, replayed)
        self.assertTrue(any('向量结果独立重算不一致' in problem for problem in problems), problems)


if __name__ == '__main__':
    unittest.main(verbosity=2)
