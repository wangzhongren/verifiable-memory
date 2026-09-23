"""实体向量、有向动作、多步推导的跨进程与独立核验测试。"""

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay
import verify
from verifiable_memory import vectors
from verifiable_memory.audit import entry_digest
from verifiable_memory.session import Session
from verifiable_memory.storage import export_evidence


class VectorChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'entities.db'
        self.session = Session.create(self.db)
        self.addCleanup(self.session.close)

    def apply(self, op, ok=True, session=None):
        session = session or self.session
        result = session.apply(op, category='vector', source='test',
                               utterance=json.dumps(op, ensure_ascii=False))
        self.assertEqual(result['status'], 'ok' if ok else 'error', result)
        return result

    def entity(self, name, vector):
        return self.apply({'op': 'teach_entity', 'name': name, 'vector': vector})

    def action(self, name, delta):
        return self.apply({'op': 'teach_vector_action', 'name': name, 'delta': delta})

    def link(self, name, source, action, target, ok=True):
        return self.apply({'op': 'link_entities', 'name': name, 'source': source,
                           'action': action, 'target': target}, ok)

    def read(self, name, source=None, max_hops=None, ok=True):
        op = {'op': 'derive_entities' if source else 'query_record', 'name': name}
        if source:
            op['source'] = source
        if max_hops is not None:
            op['max_hops'] = max_hops
        return self.apply(op, ok)

    def chain(self):
        self.entity('A', [0, 0])
        self.entity('B', [1, 0])
        self.entity('C', [1, 2])
        self.action('向右', [1, 0])
        self.action('向上', [0, 2])
        self.link('AB', 'A', '向右', 'B')
        self.link('BC', 'B', '向上', 'C')

    def verify_export(self, path=None):
        path = path or self.db
        source = self.root / 'session.json'
        payload = export_evidence(path, source)
        checks, answers, terminal, failures = replay.replay(payload['entries'], payload['capacity'])
        self.assertEqual(failures, 0, checks)
        replayed = self.root / 'replayed.json'
        replayed.write_text(json.dumps({
            'source_session_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'n_entries': len(payload['entries']), 'n_failures': failures,
            'terminal_state_hash': terminal, 'answers': answers, 'checks': checks,
        }), encoding='utf-8')
        result, problems = verify.verify(source, replayed)
        self.assertFalse(problems, result)
        return payload

    def test_quantization_and_entity_query(self):
        self.entity('小数', [0.1, 0.1234564, -1.25])
        rec = self.session.store.slots['小数']
        self.assertEqual(rec['vector_units'], [100000, 123456, -1250000])
        self.assertEqual(rec['vector_scale'], 1000000)
        self.assertEqual(self.read('小数')['result']['answer'], ['0.1', '0.123456', '-1.25'])
        self.verify_export()

    def test_embedding_sized_vector_and_reusable_action(self):
        base = [round(i / 1000, 6) for i in range(768)]
        shifted = [round(value + 0.25, 6) for value in base]
        self.entity('原始', base)
        self.entity('平移后', shifted)
        self.action('平移', [0.25] * 768)
        self.link('原始到平移', '原始', '平移', '平移后')
        result = self.read('平移后', '原始')
        self.assertEqual(len(result['result']['vector']), 768)
        self.assertEqual(result['result']['trace'][0]['edge'], '原始到平移')
        self.verify_export()

    def test_rejects_invalid_vectors_without_side_effects(self):
        self.entity('A', [1, 2])
        for name, value in [('empty', []), ('bool', [True]), ('nan', [math.nan]),
                            ('inf', [math.inf]), ('text', ['1']),
                            ('large', [1000001]), ('too_many', [0] * 4097)]:
            with self.subTest(name=name):
                before = self.session.terminal_state_hash()
                event = self.apply({'op': 'teach_entity', 'name': name, 'vector': value}, False)
                self.assertEqual(event['state_hash_before'], event['state_hash_after'])
                self.assertEqual(self.session.terminal_state_hash(), before)
        self.verify_export()

    def test_directional_composition_and_independent_proof(self):
        self.chain()
        candidate = self.apply({'op': 'apply_vector_action', 'name': '向右', 'source': 'A'})
        self.assertEqual(candidate['result']['matches'], ['B'])
        derived = self.read('C', 'A')
        self.assertEqual(derived['result']['answer'], 'C')
        self.assertEqual([step['edge'] for step in derived['result']['trace']], ['AB', 'BC'])
        self.assertEqual(derived['result']['vector'], ['1', '2'])
        self.assertEqual(derived['result']['trace'][0]['delta'], ['1', '0'])
        first = derived['result']['trace'][0]
        self.assertEqual(first['edge_written_by'], self.session.store.slots['AB']['written_by'])
        self.assertEqual(first['source_hash'], self.session.store.slot_hashes()['A'])
        self.assertEqual(first['action_hash'], self.session.store.slot_hashes()['向右'])
        self.assertEqual(first['target_hash'], self.session.store.slot_hashes()['B'])
        self.assertTrue(self.read('AB')['result']['active'])
        self.verify_export()

    def test_reverse_and_mismatched_link_are_rejected(self):
        self.chain()
        self.link('BA', 'B', '向右', 'A', False)
        self.link('AC', 'A', '向右', 'C', False)
        self.action('三维', [1, 2, 3])
        self.link('ABC', 'A', '三维', 'B', False)
        self.link('缺实体', 'A', '向右', '不存在', False)
        self.read('A', 'C', ok=False)
        self.assertEqual(set(self.session.store.slots).intersection({'BA', 'AC', 'ABC'}), set())
        self.verify_export()

    def test_apply_action_can_derive_vector_without_link(self):
        self.entity('A', [0, 1])
        self.entity('B', [2, 1])
        self.entity('B别名', [2, 1])
        self.action('右移', [2, 0])
        result = self.apply({'op': 'apply_vector_action', 'name': '右移', 'source': 'A'})
        self.assertEqual(result['result']['output_vector'], ['2', '1'])
        self.assertEqual(result['result']['matches'], ['B', 'B别名'])
        self.read('B', 'A', ok=False)  # 数值候选不是已确认的有向边
        self.verify_export()

    def test_corrections_invalidate_old_edges_until_relinked(self):
        self.chain()
        self.apply({'op': 'correct_entity', 'name': 'B', 'vector': [2, 0]})
        self.assertFalse(self.read('AB')['result']['active'])
        self.assertFalse(self.read('BC')['result']['active'])
        self.read('C', 'A', ok=False)
        self.apply({'op': 'correct_vector_action', 'name': '向右', 'delta': [2, 0]})
        self.apply({'op': 'correct_entity', 'name': 'C', 'vector': [2, 2]})
        self.link('AB2', 'A', '向右', 'B')
        self.link('BC2', 'B', '向上', 'C')
        derived = self.read('C', 'A')
        self.assertEqual([step['edge'] for step in derived['result']['trace']], ['AB2', 'BC2'])
        self.assertEqual(derived['result']['trace'][0]['action_revision'], 2)
        self.verify_export()

    def test_hop_limit_and_lexicographic_tie_break(self):
        self.chain()
        self.link('AA', 'A', '向右', 'B')
        self.read('C', 'A', max_hops=1, ok=False)
        result = self.read('C', 'A', max_hops=2)
        self.assertEqual([step['edge'] for step in result['result']['trace']], ['AA', 'BC'])
        self.apply({'op': 'derive_entities', 'name': 'C', 'source': 'A', 'max_hops': 0}, False)
        self.verify_export()

    def test_sqlite_reload_and_checkpoint(self):
        self.chain()
        for _ in range(505):
            self.read('A')
        self.assertEqual(self.session.n_ops, 512)
        restored = Session.load(self.db)
        self.addCleanup(restored.close)
        event = self.apply({'op': 'derive_entities', 'source': 'A', 'name': 'C'}, session=restored)
        self.assertEqual(len(event['result']['trace']), 2)
        self.verify_export()

    def test_json_sqlite_evidence_equivalence(self):
        json_session = Session.create(self.root / 'entities.json')
        self.addCleanup(json_session.close)
        self.chain()
        for event in self.session.entries:
            replayed = self.apply(event['op'], session=json_session)
            self.assertEqual(replayed['status'], event['status'])
        self.apply({'op': 'derive_entities', 'source': 'A', 'name': 'C'})
        self.apply({'op': 'derive_entities', 'source': 'A', 'name': 'C'}, session=json_session)
        exported = self.root / 'equivalent.json'
        export_evidence(self.db, exported)
        self.assertEqual(exported.read_bytes(), json_session.path.read_bytes())

    def test_cli_across_processes(self):
        def call(op, code=0):
            proc = subprocess.run([sys.executable, str(ROOT / 'cli.py'), '--session',
                                   str(self.db), '--json', 'vector', json.dumps(op, ensure_ascii=False)],
                                  cwd=self.root, capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, code, proc.stdout + proc.stderr)
            return json.loads(proc.stdout)

        call({'op': 'teach_entity', 'name': 'A', 'vector': [0, 0]})
        call({'op': 'teach_entity', 'name': 'B', 'vector': [1, 2]})
        call({'op': 'teach_vector_action', 'name': '移动', 'delta': [1, 2]})
        call({'op': 'link_entities', 'name': 'AB', 'source': 'A', 'action': '移动', 'target': 'B'})
        derived = call({'op': 'derive_entities', 'source': 'A', 'name': 'B'})
        self.assertEqual(derived['result']['trace'][0]['after'], ['1', '2'])
        rendered = subprocess.run([sys.executable, str(ROOT / 'cli.py'), '--session',
                                   str(self.db), 'vector', json.dumps(
                                       {'op': 'derive_entities', 'source': 'A', 'name': 'B'})],
                                  cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn('A --移动[AB]--> B', rendered.stdout)
        self.assertIn('有效', subprocess.run(
            [sys.executable, str(ROOT / 'cli.py'), '--session', str(self.db), 'status'],
            cwd=self.root, capture_output=True, text=True, timeout=15).stdout)
        call({'op': 'link_entities', 'name': 'BA', 'source': 'B', 'action': '移动', 'target': 'A'}, 1)
        self.verify_export()

    def test_independent_verifier_catches_forged_path_even_with_rehashed_log(self):
        self.chain()
        self.read('C', 'A')
        payload = self.verify_export()
        entry = payload['entries'][-1]
        entry['result']['trace'][1]['after'] = ['9', '9']
        entry['entry_hash'] = entry_digest(entry)
        payload['log_head'] = entry['entry_hash']
        session_path = self.root / 'forged.json'
        session_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        checks, answers, head, failures = replay.replay(payload['entries'], payload['capacity'])
        self.assertGreater(failures, 0)
        replayed_path = self.root / 'forged-replayed.json'
        replayed_path.write_text(json.dumps({
            'source_session_sha256': hashlib.sha256(session_path.read_bytes()).hexdigest(),
            'n_entries': len(payload['entries']), 'n_failures': 0,
            'terminal_state_hash': head, 'answers': answers, 'checks': checks}))
        _, problems = verify.verify(session_path, replayed_path)
        self.assertTrue(any('向量结果独立重算不一致' in p for p in problems), problems)


if __name__ == '__main__':
    unittest.main(verbosity=2)
