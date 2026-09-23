"""反馈增量训练、版本链、保留旧样本和独立评估回归门槛。"""

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
from verifiable_memory.session import Session

FEEDBACK_ONE = [
    {'source':'A','query':'我要右行','edge':'AB'},
    {'source':'A','query':'请从A往上走一步','edge':'AD'},
    {'source':'B','query':'从B向右走','edge':None},
    {'source':'B','query':'B返回A','edge':None},
]
FEEDBACK_TWO = [
    {'source':'A','query':'请从A往右走一步','edge':'AB'},
    {'source':'B','query':'B向上移动到C','edge':'BC'},
]


class ContinualChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'graph.db'
        self.base_data = ROOT / 'examples/branch_policy_training.jsonl'
        self.model1 = self.root / 'one.policy.json'
        self.data2 = self.root / 'two.policy-data.jsonl'
        self.model2 = self.root / 'two.policy.json'
        self.data3 = self.root / 'three.policy-data.jsonl'
        self.model3 = self.root / 'three.policy.json'
        session = Session.create(self.db)
        ops = [
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
        for op in ops:
            self.assertEqual(session.apply(op,category='vector',source='test',
                            utterance=json.dumps(op,ensure_ascii=False))['status'],'ok')
        session.close()
        self.feedback1 = self.root / 'first.policy-feedback.jsonl'
        self.feedback2 = self.root / 'second.policy-feedback.jsonl'
        for path, rows in ((self.feedback1, FEEDBACK_ONE), (self.feedback2, FEEDBACK_TWO)):
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))

    def cli(self, *args, code=0, as_json=False):
        command=[sys.executable,str(ROOT/'cli.py'),'--session',str(self.db)]
        if as_json:
            command.append('--json')
        proc=subprocess.run([*command,*map(str,args)],cwd=self.root,
                            capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,code,proc.stdout+proc.stderr)
        return json.loads(proc.stdout) if as_json else proc

    def train(self):
        self.cli('policy-train','--data',self.base_data,'--out',self.model1)
        return json.loads(self.model1.read_text())

    def learn(self,model,data,feedback,out,data_out,eval_file=None,code=0):
        args=['policy-learn','--model',model,'--data',data,'--feedback',feedback,
              '--out',out,'--data-out',data_out]
        if eval_file:
            args.extend(['--eval',eval_file])
        return self.cli(*args,code=code)

    def test_two_feedback_rounds_warm_start_and_version_chain(self):
        first=self.train()
        first_bytes=self.model1.read_bytes()
        original_bytes=self.base_data.read_bytes()
        report2=json.loads(self.learn(self.model1,self.base_data,self.feedback1,
                                      self.model2,self.data2).stdout)
        second=json.loads(self.model2.read_text())
        self.assertEqual((report2['generation'],report2['feedback']['before'],
                          report2['feedback']['after']),(2,0,4))
        self.assertEqual((report2['retained']['before'],report2['retained']['after']),(13,13))
        self.assertEqual(second['parent_model_sha256'],hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(second['generation'],2)
        self.assertEqual(second['promotion'],'candidate')
        self.assertEqual(first_bytes,self.model1.read_bytes())
        self.assertEqual(original_bytes,self.base_data.read_bytes())
        self.assertEqual(stat.S_IMODE(self.data2.stat().st_mode),0o600)
        self.assertEqual(stat.S_IMODE(self.model2.stat().st_mode),0o600)
        data2=[json.loads(line) for line in self.data2.read_text().splitlines()]
        self.assertEqual(second['training_sha256'],policy.training_digest(data2))

        report3=json.loads(self.learn(self.model2,self.data2,self.feedback2,
                                      self.model3,self.data3).stdout)
        third=json.loads(self.model3.read_text())
        self.assertEqual(report3['generation'],3)
        self.assertEqual(third['promotion'],'candidate')
        self.assertEqual(report3['feedback']['before'],0)
        self.assertEqual(report3['feedback']['after'],2)
        self.assertEqual(third['parent_model_sha256'],hashlib.sha256(self.model2.read_bytes()).hexdigest())
        self.assertEqual(third['training_sha256'],policy.training_digest(
            [json.loads(line) for line in self.data3.read_text().splitlines()]))
        session=Session.load(self.db)
        try:
            self.assertEqual(policy.evaluate(third,
                [json.loads(line) for line in self.base_data.read_text().splitlines()],
                session.store.slots)['correct'],13)
        finally:
            session.close()

    def test_feedback_changes_choice_without_rewriting_old_model(self):
        self.train()
        old_model_bytes=self.model1.read_bytes()
        before=self.cli('policy-route','--model',self.model1,'--source','B',
                        '--query','从B向右走',as_json=True)
        self.assertEqual(before['op']['path'],['BC'])
        self.learn(self.model1,self.base_data,self.feedback1,self.model2,self.data2)
        rejected=self.cli('policy-route','--model',self.model2,'--source','B',
                          '--query','从B向右走',code=1)
        self.assertIn('候选版',rejected.stderr)
        after=self.cli('policy-route','--model',self.model2,'--source','B',
                       '--query','从B向右走','--allow-candidate',code=2,as_json=True)
        self.assertEqual(after['op']['path'],[])
        self.assertTrue(after['result']['abstained'])
        self.assertEqual(self.model1.read_bytes(),old_model_bytes)

    def test_rejects_mismatched_history_without_outputs(self):
        self.train()
        tampered=self.root/'tampered.policy-data.jsonl'
        tampered.write_text(self.base_data.read_text().replace('向右然后向上','向左然后向上'))
        result=self.learn(self.model1,tampered,self.feedback1,self.model2,self.data2,code=1)
        self.assertIn('训练摘要不符',result.stderr)
        self.assertFalse(self.model2.exists() or self.data2.exists())

    def test_protected_eval_blocks_regression(self):
        self.train()
        protected=self.root/'protected.jsonl'
        # 与新反馈冲突的独立标签：旧模型判 BC，新模型拟弃权。
        protected.write_text(json.dumps({'source':'B','query':'从B向右走','edge':'BC'},
                                        ensure_ascii=False)+'\n')
        failed=self.learn(self.model1,self.base_data,self.feedback1,
                          self.model2,self.data2,protected,code=1)
        self.assertIn('独立评估退步',failed.stderr)
        self.assertFalse(self.model2.exists() or self.data2.exists())

    def test_stable_external_eval_promotes_only_that_candidate(self):
        self.train()
        limited=self.root/'limited-eval.jsonl'
        limited.write_text(json.dumps({'source':'A','query':'把A往右挪','edge':'AB'},
                                      ensure_ascii=False)+'\n')
        report=json.loads(self.learn(self.model1,self.base_data,self.feedback1,
                                     self.model2,self.data2,limited).stdout)
        self.assertEqual(report['independent_eval'],{'before':1,'after':1,'n_examples':1})
        self.assertEqual(json.loads(self.model2.read_text())['promotion'],'evaluated')
        result=self.cli('policy-route','--model',self.model2,'--source','B',
                        '--query','从B向右走',code=2,as_json=True)
        self.assertTrue(result['result']['abstained'])

    def test_feedback_replaces_conflicting_label_by_source_and_query(self):
        self.train()
        old_data_bytes=self.base_data.read_bytes()
        rows=[json.loads(line) for line in self.base_data.read_text().splitlines()]
        new=[{'source':'A','query':'向上','edge':'AB'}]
        merged,replaced=policy.merge_feedback(rows,new)
        self.assertEqual((len(merged),replaced),(len(rows),1))
        self.assertEqual(next(row for row in merged if row['source']=='A'
                              and row['query']=='向上')['edge'],'AB')
        self.assertEqual(self.base_data.read_bytes(),old_data_bytes)

    def test_graph_revision_still_requires_new_training_graph(self):
        self.train()
        session=Session.load(self.db)
        session.apply({'op':'correct_entity','name':'B','vector':[2,0]},
                      category='vector',source='test',utterance='change B')
        session.close()
        result=self.learn(self.model1,self.base_data,self.feedback1,
                          self.model2,self.data2,code=1)
        self.assertIn('模型已过期',result.stderr)
        self.assertFalse(self.model2.exists() or self.data2.exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
