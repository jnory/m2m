from unittest import TestCase


class TestMosesIniReader(TestCase):
    def test_moses_ini_reader(self):
        import m2m
        import dummy_data

        lines, io = dummy_data.generate_moses_ini()
        reader = m2m.MosesIniReader(io)

        self.assertEqual(reader.distortion_limit, 6)
        self.assertEqual(reader.stack, 100)

        self.assertEqual(reader.config['abc'], ['def', 'ghi'])
        self.assertEqual(reader.feature_conf['A'], dict())
        self.assertEqual(reader.feature_conf['B'], dict(name='X'))
        self.assertEqual(reader.feature_conf['C'], dict(name='Y', path='Z'))
        self.assertEqual(reader.feature_conf['D'], dict(path='W'))

        expected = [0.2]
        for a, b in zip(reader.weight_conf['X'], expected):
            self.assertAlmostEqual(a, b)

        expected = [0.5, 0.6]
        for a, b in zip(reader.weight_conf['Y'], expected):
            self.assertAlmostEqual(a, b)

        expected = [0.2, 0.3, 0.5]
        for a, b in zip(reader.weight_conf['A0'], expected):
            self.assertAlmostEqual(a, b)

        expected = [0.4, 0.7, 0.2]
        for a, b in zip(reader.weight_conf['D0'], expected):
            self.assertAlmostEqual(a, b)
