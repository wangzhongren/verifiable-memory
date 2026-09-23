"""模型自测→教材核对→错题记忆→候选再训练的端到端测试。"""

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

from verifiable_memory import policy, store
from verifiable_memory.session import Session


class SelfStudyChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'graph.db'
        self.old = self.root / 'old.policy.json'
        self.new = self.root / 'new.policy.json'
        self.new_data = self.root / 'new.policy-data.jsonl'
        self.training = ROOT / 'examples/branch_policy_training.jsonl'
        s = Session.create(self.db)
        items = [
            {'op':'teach_entity','name':'A','vector':[0,0]},
            {'op':'teach_entity','name':'B','vector':[1,0]},
            {'op':'teach_entity','name':'D','vector':[0,1]},
            {'op':'teach_entity','name':'C','vector':[1,1]},
            {'op':'teach_vector_action','name':'右移','delta':[1,0]},
            {'op':'teach_vector_action','name':'上移','delta':[0,1]},
            {'op':'link_entities','name':'AB','source':'A','action':'右移','target':'B'},
            {'op':'link_entities','name':'AD','source':'A','action':'上移','target':'D'},
            {'op':'link_entities','name':'BC','source':'B','action':'上移','target':'C'},
        ]
        for item in items:
            self.assertEqual(s.apply(item,category='vector',source='test',
                             utterance=json.dumps(item,ensure_ascii=False))['status'],'ok')
        s.close()
        self.cli('policy-train','--data',self.training,'--out',self.old)

    def cli(self,*args,code=0,json_output=False):
        command=[sys.executable,str(ROOT/'cli.py'),'--session',str(self.db)]
        if json_output:
            command.append('--json')
        proc=subprocess.run([*command,*map(str,args)],cwd=self.root,
                            capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,code,proc.stdout+proc.stderr)
        return json.loads(proc.stdout) if json_output else proc

    def book(self,rows,name='book.jsonl'):
        path=self.root/name
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        return path

    def study(self,book,eval_path=None,code=0,out=None,data_out=None):
        args=['policy-self-study','--model',self.old,'--data',self.training,
              '--book',book,'--out',out or self.new,
              '--data-out',data_out or self.new_data]
        if eval_path:
            args += ['--eval',eval_path]
        return self.cli(*args,code=code,json_output=True)

    def test_exact_correction_wins_and_full_log_verifies(self):
        book=self.book([
            {'source':'A','query':'我要右行','edge':'AB'},
            {'source':'B','query':'从B向右走','edge':None},
        ])
        prior=self.cli('policy-route','--model',self.old,'--source','B',
                       '--query','从B向右走',json_output=True)
        self.assertEqual(prior['op']['path'],['BC'])
        report=self.study(book)
        self.assertEqual((report['checked'],report['model_misses'],
                          report['exact_corrections_written']),(2,2,2))
        self.assertEqual(len(report['correction_op_ids']),2)
        model=json.loads(self.new.read_text())
        self.assertEqual(model['promotion'],'candidate')
        self.assertEqual(stat.S_IMODE(self.new.stat().st_mode),0o600)
        self.assertEqual(stat.S_IMODE(self.new_data.stat().st_mode),0o600)
        self.assertEqual(model['parent_model_sha256'],hashlib.sha256(self.old.read_bytes()).hexdigest())
        corrected=self.cli('policy-route','--model',self.old,'--source','A',
                           '--query','我要右行',json_output=True)
        self.assertEqual((corrected['op']['path'],corrected['result']['answer']),(['AB'],'B'))
        self.assertEqual(corrected['result']['reason'],'override_stop')
        spaced=self.cli('policy-route','--model',self.old,'--source','A',
                        '--query',' 我要右行 ',json_output=True)
        self.assertEqual(spaced['op']['path'],['AB'])
        refused=self.cli('policy-route','--model',self.old,'--source','B',
                         '--query','从B向右走',code=2,json_output=True)
        self.assertEqual(refused['result']['reason'],'confirmed_abstain')
        still_old=self.cli('policy-route','--model',self.old,'--source','A',
                           '--query','向上',json_output=True)
        self.assertEqual(still_old['op']['path'],['AD'])
        override=self.cli('vector',json.dumps({'op':'query_record',
            'name':store.override_name('A','我要右行')},ensure_ascii=False),json_output=True)
        self.assertEqual((override['result']['origin'],override['result']['active']),('book',True))
        self.assertIn('book:',override['result']['judge'])
        feedback=self.root/'from-book.policy-feedback.jsonl'
        self.cli('policy-export-feedback','--out',feedback)
        exported=[json.loads(line) for line in feedback.read_text().splitlines()]
        self.assertEqual(len(exported),2)
        self.assertEqual(stat.S_IMODE(feedback.stat().st_mode),0o600)
        evidence,replayed,verified=[self.root/name for name in
            ('session.json','replayed.json','verified.json')]
        self.cli('export','--out',evidence)
        for script,args in [('replay.py',['--session',evidence,'--out',replayed]),
                            ('verify.py',['--session',evidence,'--replayed',replayed,
                                          '--out',verified])]:
            proc=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],
                                cwd=self.root,capture_output=True,text=True,timeout=30)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.assertEqual(json.loads(verified.read_text())['n_failures'],0)

    def test_protection_rejects_model_but_keeps_book_corrected_answer(self):
        book=self.book([
            {'source':'A','query':'我要右行','edge':'AB'},
            {'source':'A','query':'请从A往上走一步','edge':'AD'},
            {'source':'B','query':'从B向右走','edge':None},
            {'source':'B','query':'B返回A','edge':None},
        ])
        protected=self.book([{'source':'B','query':'从B向右走','edge':'BC'}],
                            'protected.jsonl')
        report=self.study(book,protected,code=2)
        self.assertEqual(report['model_misses'],4)
        self.assertEqual(report['exact_corrections_written'],4)
        self.assertIn('独立评估退步',report['model_update_rejected'])
        self.assertFalse(self.new.exists() or self.new_data.exists())
        corrected=self.cli('policy-route','--model',self.old,'--source','B',
                           '--query','从B向右走',code=2,json_output=True)
        self.assertTrue(corrected['result']['abstained'])

    def test_existing_answer_is_not_rewritten_or_retrained(self):
        book=self.book([{'source':'A','query':'向上','edge':'AD'}])
        before=Session.load(self.db);count=before.n_ops;before.close()
        report=self.study(book)
        after=Session.load(self.db)
        self.assertEqual(after.n_ops,count)
        after.close()
        self.assertEqual(report['model_misses'],0)
        self.assertIsNone(report['new_model'])
        self.assertFalse(self.new.exists() or self.new_data.exists())

    def test_book_conflict_is_rejected_before_any_write(self):
        book=self.book([
            {'source':'A','query':'我要右行','edge':'AB'},
            {'source':'A','query':'我要右行','edge':'AD'},
        ])
        before=Session.load(self.db);count=before.n_ops;before.close()
        result=self.cli('policy-self-study','--model',self.old,'--data',self.training,
                        '--book',book,'--out',self.new,'--data-out',self.new_data,code=1)
        self.assertIn('标签冲突',result.stderr)
        after=Session.load(self.db)
        self.assertEqual(after.n_ops,count)
        after.close()

    def test_revised_book_overrides_prior_book_version(self):
        first=self.book([{'source':'A','query':'我要右行','edge':'AB'}])
        self.study(first)
        changed=self.book([{'source':'A','query':'我要右行','edge':'AD'}],
                          'revised-book.jsonl')
        second=self.study(changed,code=0,out=self.root/'other.policy.json',
                          data_out=self.root/'other.policy-data.jsonl')
        self.assertEqual(second['exact_corrections_written'],1)
        routed=self.cli('policy-route','--model',self.old,'--source','A',
                        '--query','我要右行',json_output=True)
        self.assertEqual((routed['op']['path'],routed['result']['answer']),(['AD'],'D'))
        record=self.cli('vector',json.dumps({'op':'query_record',
                    'name':store.override_name('A','我要右行')},ensure_ascii=False),json_output=True)
        self.assertEqual(record['result']['revision'],2)

    def test_book_correction_expires_with_graph_revision(self):
        self.study(self.book([{'source':'A','query':'我要右行','edge':'AB'}]))
        session=Session.load(self.db)
        changed=session.apply({'op':'correct_entity','name':'B','vector':[2,0]},
                              category='vector',source='test',utterance='change B')
        self.assertEqual(changed['status'],'ok')
        session.close()
        record=self.cli('vector',json.dumps({'op':'query_record',
                    'name':store.override_name('A','我要右行')},ensure_ascii=False),json_output=True)
        self.assertFalse(record['result']['active'])
        routed=self.cli('policy-route','--model',self.old,'--source','A',
                        '--query','我要右行',code=1)
        self.assertIn('模型已过期',routed.stderr)


if __name__=='__main__':
    unittest.main(verbosity=2)
