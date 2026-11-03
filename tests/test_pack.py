"""pack_for_cloud.split_lines: every part stays under the byte limit and no line is lost or reordered."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
import pack_for_cloud  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_parts_stay_under_limit_and_keep_every_line(self):
        lines = [f'{{"n": {i}, "t": "{"x" * (i % 7)}"}}\n' for i in range(50)]
        parts = pack_for_cloud.split_lines(lines, max_bytes=100)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(sum(len(l.encode('utf-8')) for l in p) <= 100 for p in parts))
        self.assertEqual([l for p in parts for l in p], lines)

    def test_multibyte_characters_count_as_bytes(self):
        lines = ['è' * 30 + '\n'] * 3  # 61 bytes each, 31 characters
        self.assertEqual(len(pack_for_cloud.split_lines(lines, max_bytes=100)), 3)


if __name__ == '__main__':
    unittest.main()
