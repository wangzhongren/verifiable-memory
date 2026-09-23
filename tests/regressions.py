"""检查点、完整日志和失败事务的回归测试（纯标准库）。"""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay
import verify
from verifiable_memory import storage, store
from verifiable_memory.audit import AuditError, check_evidence
from verifiable_memory.session import Session, SessionError


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def create(self, name='memory.db'):
        session = Session.create(self.root / name)
        self.addCleanup(session.close)
        return session

    def load(self, path):
        session = Session.load(path)
        self.addCleanup(session.close)
        return session

    def apply(self, session, kind, name='x', content='a', category=None):
        op = {'op': kind, 'name': name}
        if kind.endswith('_fact'):
            op['content'] = content
        category = category or ('ask' if kind == 'query_record' else kind.split('_')[0])
        return session.apply(op, category=category, source='test',
                             utterance=f'{kind} {name} {content}')

    def seed_checkpoint(self, session):
        self.apply(session, 'teach_fact')
        for _ in range(511):
            self.apply(session, 'query_record')
        self.assertEqual(session.n_ops, 512)

    def evidence(self, session):
        path = self.root / 'evidence.json'
        return path, storage.export_evidence(session.path, path)

    def verify_payload(self, path, raw):
        # 即便为被改文件提供匹配的重放来源摘要，独立核验也必须检查事件链。
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        checks, answers, head, failures = replay.replay(raw['entries'], raw['capacity'])
        rep = self.root / 'replayed.json'
        rep.write_text(json.dumps({
            'source_session_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'n_entries': len(raw['entries']), 'n_failures': failures,
            'terminal_state_hash': head, 'answers': answers, 'checks': checks,
        }), encoding='utf-8')
        return failures, verify.verify(path, rep)[1]

    def test_checkpoint_reload_read_write_and_error(self):
        session = self.create()
        self.seed_checkpoint(session)
        loaded = self.load(session.path)
        self.assertEqual(loaded.entries, [])
        self.assertEqual(self.apply(loaded, 'query_record')['op_id'], 'op-513')
        self.assertEqual(self.apply(loaded, 'correct_fact', content='b')['op_id'], 'op-514')
        error = self.apply(loaded, 'query_record', name='missing')
        self.assertEqual((error['op_id'], error['status']), ('op-515', 'error'))
        restored = self.load(session.path)
        self.assertEqual(restored.n_ops, 515)
        self.assertEqual(len(restored.entries), 3)
        self.assertEqual(restored.store.slots['x']['content'], 'b')
        path, raw = self.evidence(restored)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_continuous_writes_beyond_two_checkpoints(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        for i in range(1, 1026):
            self.apply(session, 'correct_fact', content=str(i))
            if i in (511, 512, 1023, 1024, 1025):
                restored = self.load(session.path)
                self.assertEqual(restored.n_ops, i + 1)
                self.assertEqual(restored.store.slots, session.store.slots)
                self.assertEqual(restored.log_head, session.log_head)

    def test_subprocesses_continue_after_checkpoint(self):
        session = self.create()
        self.seed_checkpoint(session)
        for command, utterance in [('ask', '查询 x'),
                                   ('correct', '更正事实 x：新值'), ('ask', '查询 x')]:
            result = subprocess.run(
                [sys.executable, str(ROOT / 'cli.py'), '--session', str(session.path),
                 command, utterance], cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('新值', result.stdout)
        self.assertEqual(self.load(session.path).n_ops, 515)

    def test_concurrent_reads_writes_errors_after_checkpoint(self):
        session = self.create()
        self.seed_checkpoint(session)
        jobs = []
        for i in range(5):
            for command, utterance, expected in [
                ('ask', '查询 x', 0), ('correct', f'更正事实 x：值{i}', 0),
                ('ask', '查询 missing', 1),
            ]:
                process = subprocess.Popen(
                    [sys.executable, str(ROOT / 'cli.py'), '--session', str(session.path),
                     command, utterance], cwd=self.root,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                jobs.append((process, expected))
        results = [(p.communicate(), p.returncode, expected) for p, expected in jobs]
        for output, code, expected in results:
            self.assertEqual(code, expected, ''.join(output))
        restored = self.load(session.path)
        self.assertEqual(restored.n_ops, 527)
        path, raw = self.evidence(restored)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_stale_session_retries_after_read_only_event(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        stale = self.load(session.path)
        before = stale.terminal_state_hash()
        self.apply(session, 'query_record')
        self.assertEqual(before, session.terminal_state_hash())
        result = self.apply(stale, 'correct_fact', content='b')
        self.assertEqual(result['op_id'], 'op-003')
        self.assertEqual(stale.n_ops, 3)

    def test_failed_commit_restores_memory_and_can_retry(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        before = (session.terminal_state_hash(), session.n_ops, session.log_head)
        # 在 ops 已插入、state 即将写入时注入真正的 SQLite 事务失败。
        session.storage.conn.execute(
            "CREATE TRIGGER reject_state BEFORE INSERT ON state BEGIN "
            "SELECT RAISE(ABORT, 'injected failure'); END")
        with self.assertRaises(SessionError):
            session.apply({'op': 'correct_fact', 'name': 'x', 'content': 'b'},
                          category='correct', source='test', utterance='correct x',
                          max_attempts=1)
        self.assertEqual((session.terminal_state_hash(), session.n_ops, session.log_head), before)
        self.assertEqual(self.load(session.path).n_ops, 1)
        session.storage.conn.execute('DROP TRIGGER reject_state')
        self.assertEqual(self.apply(session, 'correct_fact')['op_id'], 'op-002')

    def test_every_query_field_is_hash_protected(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        self.apply(session, 'query_record')
        path, original = self.evidence(session)
        self.assertEqual(self.verify_payload(path, original), (0, []))
        replacements = {'source': 'forged', 'utterance': 'forged request',
                        'category': 'correct', 'op_id': 'op-099',
                        'op': {'op': 'query_record', 'name': 'other'},
                        'status': 'error', 'state_hash_before': '0' * 64,
                        'state_hash_after': '0' * 64, 'prev_entry_hash': '0' * 64,
                        'entry_hash': '0' * 64, 'result': {'answer': 'forged'}}
        self.assertEqual(set(replacements), set(original['entries'][1]))
        for key, value in replacements.items():
            with self.subTest(field=key):
                raw = copy.deepcopy(original)
                raw['entries'][1][key] = value
                with self.assertRaises(AuditError):
                    check_evidence(raw)
                failures, _ = self.verify_payload(path, raw)
                self.assertGreater(failures, 0)
                # 构造形状有效的陈旧重放报告，确保 verify 的事件链检查独立生效。
                rep = json.loads((self.root / 'replayed.json').read_text())
                rep['n_failures'] = 0
                (self.root / 'replayed.json').write_text(json.dumps(rep))
                _, problems = verify.verify(path, self.root / 'replayed.json')
                self.assertTrue(any('日志' in problem for problem in problems), problems)

    def test_error_and_certificate_tampering(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        self.apply(session, 'query_record', name='missing')
        path, original = self.evidence(session)
        for index, key, value in [(0, 'proof', {}), (1, 'error', 'forged error')]:
            with self.subTest(field=key):
                raw = copy.deepcopy(original)
                raw['entries'][index][key] = value
                failures, problems = self.verify_payload(path, raw)
                self.assertGreater(failures, 0)
                self.assertTrue(problems)

    def test_missing_hash_deleted_reordered_and_truncated_events(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        self.apply(session, 'query_record')
        self.apply(session, 'query_record')
        path, original = self.evidence(session)
        variants = []
        for key in ('entry_hash', 'prev_entry_hash'):
            raw = copy.deepcopy(original)
            del raw['entries'][1][key]
            variants.append(raw)
        raw = copy.deepcopy(original)
        raw['entries'][1], raw['entries'][2] = raw['entries'][2], raw['entries'][1]
        variants.append(raw)
        for index in (0, 1, 2):
            raw = copy.deepcopy(original)
            del raw['entries'][index]
            variants.append(raw)
        for i, raw in enumerate(variants):
            with self.subTest(variant=i):
                with self.assertRaises(AuditError):
                    check_evidence(raw)
                self.assertTrue(self.verify_payload(path, raw)[1])

    def test_sqlite_metadata_tampering_before_and_after_checkpoint(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        for checkpointed in (False, True):
            if checkpointed:
                for _ in range(511):
                    self.apply(session, 'query_record')
            self.apply(session, 'query_record')
            seq = session.n_ops
            for column in ('source', 'utterance', 'category'):
                with self.subTest(checkpoint=checkpointed, column=column):
                    original = session.storage.conn.execute(
                        f'SELECT {column} FROM ops WHERE seq = ?', (seq,)).fetchone()[0]
                    session.storage.conn.execute(
                        f'UPDATE ops SET {column} = ? WHERE seq = ?', ('forged', seq))
                    with self.assertRaises(SessionError):
                        Session.load(session.path)
                    path, raw = self.evidence(session)
                    self.assertTrue(self.verify_payload(path, raw)[1])
                    session.storage.conn.execute(
                        f'UPDATE ops SET {column} = ? WHERE seq = ?', (original, seq))

    def test_checkpoint_and_current_cache_independently_checked(self):
        session = self.create()
        self.seed_checkpoint(session)
        self.apply(session, 'correct_fact', content='b')
        for table in ('state', 'checkpoint_state'):
            with self.subTest(table=table):
                original = session.storage.conn.execute(
                    f'SELECT record FROM {table} WHERE name = ?', ('x',)).fetchone()[0]
                record = json.loads(original)
                record['content'] = 'forged'
                session.storage.conn.execute(f'UPDATE {table} SET record = ?',
                                             (json.dumps(record),))
                with self.assertRaises(SessionError):
                    Session.load(session.path)
                session.storage.conn.execute(f'UPDATE {table} SET record = ?', (original,))

    def test_category_rejection_is_replayable(self):
        session = self.create()
        entry = self.apply(session, 'teach_fact', category='ask')
        self.assertEqual(entry['status'], 'error')
        path, raw = self.evidence(session)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_error_messages_do_not_depend_on_sqlite_row_order(self):
        session = self.create()
        self.apply(session, 'teach_fact', name='a')
        self.apply(session, 'teach_fact', name='b')
        self.apply(session, 'correct_fact', name='a', content='new')
        for _ in range(509):
            self.apply(session, 'query_record', name='a')
        restored = self.load(session.path)
        self.apply(restored, 'query_record', name='missing')
        self.apply(restored, 'correct_fact', name='missing')
        path, raw = self.evidence(restored)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_legacy_formats_fail_closed_without_rewriting(self):
        session = self.create('legacy.json')
        self.apply(session, 'teach_fact')
        raw = json.loads(session.path.read_text())
        raw['format'] = 'verifiable_memory_01/session@v1'
        session.path.write_text(json.dumps(raw))
        before = session.path.read_bytes()
        with self.assertRaises(SessionError):
            Session.load(session.path)
        self.assertEqual(session.path.read_bytes(), before)
        db = self.create('legacy.db')
        self.apply(db, 'teach_fact')
        db.storage.conn.execute("UPDATE meta SET value = 'verifiable_memory_01/sqlite@v1' "
                                "WHERE key = 'format'")
        with self.assertRaises(SessionError):
            Session.load(db.path)
        self.assertEqual(db.storage.conn.execute('SELECT COUNT(*) FROM ops').fetchone()[0], 1)

    def test_empty_evidence_and_failed_first_operation(self):
        session = self.create()
        path, raw = self.evidence(session)
        self.assertEqual(self.verify_payload(path, raw), (0, []))
        self.apply(session, 'query_record', name='missing')
        path, raw = self.evidence(session)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_stale_replay_report_rejected(self):
        session = self.create()
        self.apply(session, 'teach_fact')
        path, raw = self.evidence(session)
        self.assertEqual(self.verify_payload(path, raw), (0, []))
        self.apply(session, 'query_record')
        self.evidence(session)
        _, problems = verify.verify(path, self.root / 'replayed.json')
        self.assertIn('replayed.json 不属于当前证据文件', problems)

    def test_fact_content_500_boundary_and_history(self):
        session = self.create()
        for length in (200, 201, 499, 500):
            with self.subTest(length=length):
                event = self.apply(session, 'teach_fact', name=f'事实{length}', content='记' * length)
                self.assertEqual(event['status'], 'ok')
        # 混合中文、ASCII 和 emoji 按 Python len() 的 Unicode 码点计数。
        mixed = '记A🙂。' * 125
        self.assertEqual(len(mixed), 500)
        corrected = self.apply(session, 'correct_fact', name='事实500', content=mixed)
        self.assertEqual(corrected['result']['revision'], 2)
        before = session.terminal_state_hash()
        for kind, name in [('teach_fact', '超长'), ('correct_fact', '事实500')]:
            event = self.apply(session, kind, name=name, content='记' * 501)
            self.assertEqual(event['status'], 'error')
            self.assertIn('1–500', event['error'])
            self.assertEqual(event['state_hash_before'], event['state_hash_after'])
            self.assertEqual(session.terminal_state_hash(), before)
        restored = self.load(session.path)
        self.assertEqual(restored.store.slots['事实500']['content'], mixed)
        path, raw = self.evidence(restored)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_legacy_200_limit_errors_still_replay(self):
        session = self.create()
        self.apply(session, 'teach_fact', content='旧内容')
        with patch.object(store, 'CONTENT_MAX', 200):
            for kind, name in [('teach_fact', '新记录'), ('correct_fact', 'x')]:
                event = self.apply(session, kind, name=name, content='旧' * 201)
                self.assertEqual(event['error'], 'content 必须是 1–200 字的字符串')
        restored = self.load(session.path)
        self.assertEqual(restored.store.slots['x']['content'], '旧内容')
        event = self.apply(restored, 'correct_fact', content='新' * 500)
        self.assertEqual(event['status'], 'ok')
        path, raw = self.evidence(restored)
        self.assertEqual(self.verify_payload(path, raw), (0, []))

    def test_500_character_facts_through_real_cli_and_import(self):
        db = self.root / 'long-facts.db'

        def run(*args, code=0, as_json=True):
            command = [sys.executable, str(ROOT / 'cli.py'), '--session', str(db), '--no-llm']
            if as_json:
                command.append('--json')
            proc = subprocess.run([*command, *map(str, args)], cwd=self.root,
                                  capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, code, proc.stdout + proc.stderr)
            return json.loads(proc.stdout) if as_json else proc

        original = '甲' * 500
        updated = '乙' * 500
        self.assertEqual(run('teach', f'教事实 长事实：{original}')['status'], 'ok')
        self.assertEqual(run('ask', '查询 长事实')['result']['answer'], original)
        self.assertEqual(run('correct', f'更正事实 长事实：{updated}')['status'], 'ok')
        run('correct', f'更正事实 长事实：{updated}多', code=1, as_json=False)
        self.assertEqual(run('ask', '查询 长事实')['result']['answer'], updated)
        knowledge = self.root / 'long.jsonl'
        knowledge.write_text('\n'.join(json.dumps({'kind': 'fact', 'name': name, 'content': content})
                            for name, content in [('导入500', original), ('导入501', original + '多')]))
        result = run('import', knowledge, code=1, as_json=False)
        self.assertIn('新增 1', result.stdout)
        self.assertIn('失败 1', result.stdout)
        self.assertEqual(run('ask', '查询 导入500')['result']['answer'], original)
        restored = self.load(db)
        path, raw = self.evidence(restored)
        self.assertEqual(self.verify_payload(path, raw), (0, []))


if __name__ == '__main__':
    unittest.main(verbosity=2)
