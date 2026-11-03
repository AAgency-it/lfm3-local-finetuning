"""The generic (synthetic-only) dataset: deterministic, labels exact, bench separate from training."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'src'))
import build_generic as B  # noqa: E402
from lfm3_local import extract  # noqa: E402


class ExtractRowTests(unittest.TestCase):
    def test_same_seed_same_row(self):
        self.assertEqual(B.extract_row(7), B.extract_row(7))

    def test_row_uses_the_skill_prompt_and_a_parsable_answer(self):
        row = B.extract_row(7)
        self.assertEqual(row['task'], 'extract')
        system, user, answer = [m['content'] for m in row['messages']]
        self.assertEqual(system, extract.SYSTEM)
        self.assertTrue(user.startswith('Frase: '))
        self.assertEqual(set(json.loads(answer)), {'data', 'importo_euro'})


class LogRowTests(unittest.TestCase):
    def test_one_row_per_line_with_its_label(self):
        rows = B.log_rows(3)
        _, lines = B.gen_logs.make_document(3)
        self.assertEqual(len(rows), len(lines))
        self.assertEqual([r['messages'][-1]['content'] for r in rows], [label for _, label in lines])

    def test_prompt_marks_the_target_line(self):
        rows = B.log_rows(3)
        _, lines = B.gen_logs.make_document(3)
        self.assertIn('RIGA: ' + lines[0][0], rows[0]['messages'][1]['content'])
        self.assertEqual(rows[0]['messages'][0]['content'], B.LOG_SYSTEM)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.d = B.build(n_extract=60, n_log_docs=9, n_bench_extract=30, n_bench_log_docs=3)

    def test_bench_never_inside_training(self):
        bench = {r['messages'][1]['content'] for r in self.d['bench']}
        for name in ('train', 'dev'):
            self.assertFalse(bench & {r['messages'][1]['content'] for r in self.d[name]}, name)

    def test_public_extraction_bench_is_in_the_bench_and_not_in_training(self):
        public = [json.loads(x)['testo'] for x in open(ROOT / 'bench' / 'extract_bench.jsonl', encoding='utf-8') if x.strip()]
        bench = {r['messages'][1]['content'] for r in self.d['bench']}
        train = {r['messages'][1]['content'] for r in self.d['train'] + self.d['dev']}
        for t in public:
            self.assertIn('Frase: ' + t, bench)
            self.assertNotIn('Frase: ' + t, train)

    def test_both_tasks_everywhere(self):
        for name in ('train', 'dev', 'bench'):
            self.assertEqual({r['task'] for r in self.d[name]}, {'extract', 'logs'}, name)

    def test_deterministic(self):
        again = B.build(n_extract=60, n_log_docs=9, n_bench_extract=30, n_bench_log_docs=3)
        self.assertEqual(self.d, again)


class BaselineTests(unittest.TestCase):
    def test_baselines_are_between_0_and_1(self):
        d = B.build(n_extract=20, n_log_docs=2, n_bench_extract=20, n_bench_log_docs=2)
        base = B.baselines(d['bench'])
        self.assertEqual(set(base), {'extract_regex', 'logs_majority'})
        for v in base.values():
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 1)


if __name__ == '__main__':
    unittest.main()
