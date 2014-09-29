from unittest import TestCase

class TestPhraseTable(TestCase):
    def test_phrase_table(self):
        import numpy as np

        import m2m
        import dummy_data

        lines, io = dummy_data.generate_phrase_table()

        pt = m2m.PhraseTable(io)
        for i, (f_words, e_words, scores) in enumerate(pt.iter()):
            line = lines[i]
            sections = line.split(" ||| ")
            f_ph = sections[0].split(" ")
            e_ph = sections[1].split(" ")
            ss = np.array(map(float, sections[2].split(" ")), dtype=np.float)

            self.assertEqual(f_words, f_ph)
            self.assertEqual(e_words, e_ph)

            for a, b in zip(scores, ss):
                self.assertAlmostEqual(a, b)

        io.close()
