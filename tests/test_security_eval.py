import asyncio
import json
import unittest
import uuid
import shutil
from pathlib import Path

from gate import check_gate
from src.security_eval import dataset, runtime_probe, summarize, evaluate


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / '.cache' / ('eval-test-' + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_all_five_categories_and_twenty_ordinary_questions(self):
        attacks, normal = dataset()
        self.assertEqual(len(attacks), 10)
        self.assertEqual(len(normal), 20)
        self.assertEqual({a['kind'] for a in attacks}, {'direct','indirect','memory','cross_user','tool_abuse'})

    def test_actual_runtime_probes_before_after(self):
        attacks, _ = dataset()
        for item in (a for a in attacks if a['mode'] == 'runtime'):
            before = asyncio.run(runtime_probe(item, 'v4', self.directory))
            after = asyncio.run(runtime_probe(item, 'v5', self.directory / str(item['id'])))
            succeeds = lambda data: any(m in json.dumps(data) for m in item['fail_if'])
            self.assertTrue(succeeds(before))
            self.assertFalse(succeeds(after))

    def test_incomplete_model_calls_cannot_be_reported_as_zero_asr(self):
        asyncio.run(evaluate(self.directory, 'contracts'))
        data = json.loads((self.directory / 'results_v5.json').read_text(encoding='utf-8'))
        self.assertEqual(data['metrics']['attacks_completed'], 4)
        self.assertIsNone(data['metrics']['asr'])
        self.assertIsNone(data['metrics']['false_refusals'])
        self.assertFalse(check_gate(data))
        self.assertEqual(data['metrics']['leak'], 0)

    def test_gate_recomputes_metrics_and_requires_live(self):
        attacks, golden = dataset()
        rows = [dict(a, completed=True, attack_succeeded=False, boundary_ok=True) for a in attacks]
        normal = [dict(a, completed=True, false_refusal=False, correct=True) for a in golden]
        data = {'attacks':rows,'golden':normal,'manifest':{'backend':'live'},'metrics':summarize(rows,normal)}
        self.assertTrue(check_gate(data))
        for row in rows[:3]:
            row['attack_succeeded'] = True
        self.assertFalse(check_gate(data))
        data['metrics'] = summarize(rows,normal)
        self.assertFalse(check_gate(data))
        self.assertTrue(check_gate(data, max_asr=.3))
        data['manifest']['backend'] = 'contracts'
        self.assertFalse(check_gate(data, max_asr=.3))

    def test_gate_denies_empty_and_nan(self):
        self.assertFalse(check_gate({}))
        self.assertFalse(check_gate({'manifest':{'backend':'live'},'metrics':{'complete':True,'asr':float('nan'),'leak':0}}))


if __name__ == '__main__':
    unittest.main()
