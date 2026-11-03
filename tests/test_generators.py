"""The synthetic generators are deterministic per seed and produce well-formed labels."""
import re
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data_synthetic'))
import gen_extract  # noqa: E402
import gen_logs  # noqa: E402


class ExtractGeneratorTests(unittest.TestCase):
    def test_same_seed_same_sentence(self):
        self.assertEqual(gen_extract.make_sentence(42), gen_extract.make_sentence(42))

    def test_answers_are_well_formed(self):
        for seed in range(500):
            text, ans = gen_extract.make_sentence(seed)
            self.assertTrue(text.strip())
            if ans['data'] is not None:
                date.fromisoformat(ans['data'])
            if ans['importo_euro'] is not None:
                self.assertIsInstance(ans['importo_euro'], (int, float))
                self.assertGreater(ans['importo_euro'], 0)

    def test_written_year_appears_in_text_when_date_is_set(self):
        for seed in range(300):
            text, ans = gen_extract.make_sentence(seed)
            if ans['data']:
                y = ans['data'][:4]
                self.assertTrue(y in text or y[2:] in text, (seed, text, ans))


class LogGeneratorTests(unittest.TestCase):
    def test_same_seed_same_document(self):
        self.assertEqual(gen_logs.make_document(7), gen_logs.make_document(7))

    def test_every_family_has_both_labels_somewhere_and_only_valid_labels(self):
        for fam in gen_logs.FAMIGLIE:
            labels = set()
            for seed in range(60):
                _, lines = gen_logs.make_document(seed, [fam])
                self.assertTrue(lines)
                labels |= {lab for _, lab in lines}
            self.assertEqual(labels, {gen_logs.R, gen_logs.N}, fam.__name__)

    def test_no_drive_letter_paths_in_generated_text(self):
        drive = re.compile(r'\b[A-Za-z]:[\\/]')
        for seed in range(300):
            _, lines = gen_logs.make_document(seed)
            self.assertFalse([l for l, _ in lines if drive.search(l)])


if __name__ == '__main__':
    unittest.main()
