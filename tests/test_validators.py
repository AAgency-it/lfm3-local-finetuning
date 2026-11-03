"""Unit tests with no model and no server: validators, regex extraction, sentence splitting, the client's failure path."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from lfm3_local import client, extract, generic, key_points, rewrite  # noqa: E402


class ExtractTests(unittest.TestCase):
    def test_regex_unambiguous(self):
        r = extract.by_regex('Il bonifico da 480 € è stato fatto il 3/12/2026.')
        self.assertEqual(r, {'data': '2026-12-03', 'importo_euro': 480.0})

    def test_regex_gives_up_on_ambiguity(self):
        self.assertIsNone(extract.by_regex('Il listino passa da 120 a 135 euro dal 1 gennaio 2027.'))
        self.assertIsNone(extract.by_regex('Consegna il 9 dicembre 2026; costo di duemila euro.'))

    def test_percentage_is_not_an_amount(self):
        self.assertEqual(extract.amounts_in_text('Il fatturato è cresciuto del 12% nel 2026.'), [])

    def test_validator_rejects_invented_date(self):
        ok, why = extract.validate('Nessuna novità sul pagamento.', {'data': '2025-12-13', 'importo_euro': None})
        self.assertFalse(ok)
        self.assertIn('date not written in the sentence', why)

    def test_validator_rejects_null_when_present(self):
        ok, _ = extract.validate('Saldo di 450 euro.', {'data': None, 'importo_euro': None})
        self.assertFalse(ok)

    def test_validator_accepts_right_answer(self):
        ok, _ = extract.validate('Saldo di 450 euro il 5/11/2026.', {'data': '2026-11-05', 'importo_euro': 450})
        self.assertTrue(ok)


class RewriteTests(unittest.TestCase):
    def test_numbers_must_match(self):
        ok, why = rewrite.validate('Ti scrivo per ricordarti che il saldo è di 450 euro.', 'Il saldo è di 540 euro.')
        self.assertFalse(ok)
        self.assertIn('numbers differ from the original', why)

    def test_shorter_and_finished(self):
        ok, _ = rewrite.validate('Ti scrivo per ricordarti che il saldo è di 450 euro.', 'Saldo: 450 euro.')
        self.assertTrue(ok)


class GenericTests(unittest.TestCase):
    def test_piece_check(self):
        self.assertEqual(generic.check_piece('def f(x):\n    return x', ['return'], [], 'def f'), [])
        self.assertIn('contains ```', generic.check_piece('```py\nx\n```', [], [], None))
        self.assertIn('() unbalanced', generic.check_piece('f((x)', [], [], None))

    def test_classify_needs_two_labels(self):
        self.assertEqual(generic.classify('x', 'y', ['solo'])['outcome'], 'escalate')

    def test_deny_pattern(self):
        r = generic.classify('Is this a legal matter?', 'text', ['yes', 'no'], deny=r'legal')
        self.assertEqual(r['outcome'], 'escalate')


class KeyPointsTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(key_points.split_sentences('Uno. Due!\n\nTre?'), ['Uno.', 'Due!', 'Tre?'])

    def test_prompt_edges(self):
        s = ['a', 'b']
        self.assertEqual(key_points.sentence_prompt(s, 0), 'PRIMA: \nTARGET: a\nDOPO: b')


class ClientTests(unittest.TestCase):
    def test_too_long(self):
        self.assertEqual(client.ask('s', 'x' * 5000)['reason'], 'too_long')

    def test_server_off_never_raises(self):
        import os
        os.environ['LFM_SERVER'] = 'http://127.0.0.1:9'
        try:
            self.assertFalse(client.healthy())
            self.assertEqual(client.ask('s', 'u')['reason'], 'server')
        finally:
            del os.environ['LFM_SERVER']


if __name__ == '__main__':
    unittest.main()
