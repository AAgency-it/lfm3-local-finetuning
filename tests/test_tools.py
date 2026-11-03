"""Tests for the publishing tools: anonymised results, the cleanliness scan (tree and git history), the numbers check."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import check_clean  # noqa: E402
import check_numbers  # noqa: E402
import check_template_overlap  # noqa: E402
import publish_results  # noqa: E402

LABELS = {'acme': 'set A', 'globex': 'set B'}


class AnonymiseTests(unittest.TestCase):
    def test_keys_become_labels(self):
        out = publish_results.anonymise({'claim': {'acme': 0.963, 'Globex': 0.988}}, LABELS)
        self.assertEqual(out, {'claim': {'set A': 0.963, 'set B': 0.988}})

    def test_strings_inside_lists_are_relabelled(self):
        out = publish_results.anonymise({'errors': [['ACME->globex', 3]]}, LABELS)
        self.assertEqual(out, {'errors': [['set A->set B', 3]]})

    def test_short_code_replaced_only_as_whole_word(self):
        out = publish_results.anonymise({'per_classe': {'ACM': 1.0}, 'note': 'acmecorp->ACM'},
                                        {'ACM': 'cat 7', 'acmecorp': 'set B'})
        self.assertEqual(out, {'per_classe': {'cat 7': 1.0}, 'note': 'set B->cat 7'})

    def test_per_case_detail_is_dropped(self):
        out = publish_results.anonymise({'n': 5, 'detail': [1, 2], 'dettaglio': [3], 'previsioni': ['x']}, LABELS)
        self.assertEqual(out, {'n': 5})

    def test_a_count_under_a_detail_name_is_kept(self):
        out = publish_results.anonymise({'casi': 50, 'detail': [1]}, LABELS)
        self.assertEqual(out, {'casi': 50})

    def test_internal_names_become_public_ones(self):
        out = publish_results.anonymise({'zone_AB': {'f1': 0.4}, 'brand': {'baseline_in_ledger': 0.8}, 'abilita': 'key_zone'}, {},
                                        rename=[('baseline_in_ledger', 'baseline_hand_written_label'), ('zone', 'notes')])
        self.assertEqual(out, {'notes_AB': {'f1': 0.4}, 'category': {'baseline_hand_written_label': 0.8}, 'abilita': 'key_notes'})

    def test_leftover_forbidden_word_fails(self):
        with self.assertRaises(ValueError):
            publish_results.anonymise({'note': 'initech said so'}, LABELS, forbidden=['initech'])


class CleanScanTests(unittest.TestCase):
    def test_forbidden_word_found_in_json_text(self):
        hits = check_clean.scan_text('{"set": "Acme Corp", "f1": 0.4}', ['acme'])
        self.assertEqual([h[1] for h in hits], ['forbidden word'])

    def test_clean_text_has_no_hits(self):
        self.assertEqual(check_clean.scan_text('<svg><text>set A 0.96</text></svg>', ['acme']), [])

    def test_package_at_version_is_not_an_email(self):
        self.assertEqual(check_clean.scan_text('npm WARN deprecated glob@3.0.4 and request@3.4.2', []), [])

    def test_email_still_found(self):
        at = '@'  # built at run time so this file stays clean for the scanner
        self.assertTrue(check_clean.scan_text(f'write to someone{at}company.it', []))

    def test_windows_path_found(self):
        drive = 'Q' + ':'  # built at run time so this file stays clean for the scanner
        self.assertTrue(check_clean.scan_text(f'model at {drive}\\models\\x.gguf', []))

    def test_history_finds_word_removed_later(self):
        with tempfile.TemporaryDirectory() as d:
            run = lambda *a: subprocess.run(['git', '-C', d, *a], check=True, capture_output=True)  # noqa: E731
            run('init', '-q')
            run('config', 'user.name', 'T')
            run('config', 'user.email', 'noreply@example.com')
            Path(d, 'a.txt').write_text('made for acme\n', encoding='utf-8')
            run('add', '.')
            run('commit', '-qm', 'one')
            Path(d, 'a.txt').write_text('made for set A\n', encoding='utf-8')
            run('commit', '-qam', 'two')
            hits = check_clean.scan_history(Path(d), ['acme'])
        self.assertTrue(any(h[1] == 'forbidden word' for h in hits))


class NumbersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'published_results').mkdir()
        (self.root / 'published_results' / 'run.json').write_text(json.dumps({'chiave': {'f1': 0.9634}}), encoding='utf-8')
        (self.root / 'README.md').write_text('F1 **0.963** after fine-tuning', encoding='utf-8')
        (self.root / 'README.it.md').write_text('F1 **0,963** dopo il fine-tuning', encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def claim(self, **kw):
        c = {'key': 'kp_f1', 'file': 'run.json', 'path': 'chiave.f1', 'value': 0.963, 'digits': 3}
        c.update(kw)
        return c

    def test_matching_claim_passes(self):
        self.assertEqual(check_numbers.verify([self.claim()], self.root), [])

    def test_wrong_value_fails(self):
        probs = check_numbers.verify([self.claim(value=0.97)], self.root)
        self.assertTrue(any('value' in p for p in probs))

    def test_number_missing_from_readme_fails(self):
        (self.root / 'README.it.md').write_text('niente numeri', encoding='utf-8')
        probs = check_numbers.verify([self.claim()], self.root)
        self.assertTrue(any('README.it.md' in p for p in probs))

    def test_path_as_list_reaches_keys_with_dots(self):
        self.assertEqual(check_numbers.lookup({'per_threshold': {'0.5': {'f1': 0.444}}}, ['per_threshold', '0.5', 'f1']), 0.444)


if __name__ == '__main__':
    unittest.main()


class TemplateOverlapTests(unittest.TestCase):
    def test_placeholders_match_any_value(self):
        self.assertTrue(check_template_overlap.follows('Saldo di {A} da versare entro il {D}.',
                                                       'Saldo di 450 euro da versare entro il 5 dicembre 2026.'))

    def test_other_wording_does_not_match(self):
        self.assertFalse(check_template_overlap.follows('Saldo di {A} da versare entro il {D}.',
                                                        'Il saldo di 450 euro va versato entro il 5 dicembre 2026.'))

    def test_published_bench_has_19_sentences_built_on_a_template(self):
        self.assertEqual(len(check_template_overlap.overlapping()), 19)
