#! -*- coding: utf-8 -*-

import numpy as np
import os
from utils.smart_fp import smart_open


class PhraseTable(object):
    def __init__(self, path_or_file):
        if isinstance(path_or_file, str) \
                or isinstance(path_or_file, unicode):
            self._path = path_or_file
            self._fp = None
        else:
            self._path = None
            self._fp = path_or_file

    @staticmethod
    def _parse_line(line):
        parts = line.split(' ||| ')
        f_words = parts[0]
        e_words = parts[1]
        scores = parts[2]

        return f_words, e_words, scores

    @staticmethod
    def _parse_line_premitive(line):
        f_words, e_words, scores = PhraseTable._parse_line(line)
        f_words = f_words.split(' ')
        e_words = e_words.split(' ')
        scores = np.fromstring(scores, dtype=np.float, sep=' ')

        return f_words, e_words, scores

    def _get_file_object(self):
        if self._path is not None:
            fp = smart_open(self._path)
        else:
            fp = self._fp
            fp.seek(0)
        return fp

    def _close_if_needed(self, fp):
        if self._fp == fp:
            fp.close()

    def iter(self):
        fp = self._get_file_object()
        for line in fp:
            f_words, e_words, scores = self._parse_line(line)
            yield (f_words, e_words, scores)
        self._close_if_needed(fp)


class MosesIniReader(object):
    def __init__(self, path):
        self.config = {}
        self.feature_conf = {}
        self.weight_conf = {}

        fp = smart_open(path)
        self._parse(fp)
        self._parse_feature()
        self.parse_weight()
        self.distortion_limit = int(self.config['distortion-limit'][0])

    def _parse(self, fp):
        name = None
        for line in fp:
            line = line.strip()

            if line[0] == '#':
                continue
            if line[0] == '[':
                name = line[1:-1]
            elif line != '':
                self.config.setdefault(name, []).append(line)

    def _parse_feature(self):
        feature_list = self.config['feature']
        for line in feature_list:
            f_conf = line.split(' ')
            f_name = f_conf[0]
            f_conf_dict = {}
            for conf in f_conf[1:]:
                conf_name, conf_info = conf.split('=')
                f_conf_dict[conf_name] = conf_info
            self.feature_conf[f_name] = f_conf_dict

    def _parse_weight(self):
        weight_list = self.config['weight']
        for line in weight_list:
            name, weights = line.split('=')
            weights = np.fromstring(weights, dtype=np.float, sep=' ')
            self.weight_conf[name] = weights


class LM(object):
    def __init__(self, path, weight):
        self.path = path
        self.weight = weight


class TM(object):
    def __init__(self, path, weights):
        self.path = path
        self.weights = weights


class Converter(object):
    multiply_weight = dict(
        WordPenalty=-1,
        KenLM=2.30258509299404568401,
        Distortion=-1,
        PhraseDictionaryMemory=1,
    )

    def __init__(self, path):
        self.reader = MosesIniReader(path)

    def __call__(self, output_directory):
        mtplz_ini = os.path.join(output_directory, 'mtplz.ini')
        self._write_ini(mtplz_ini)

        phrase_table = os.path.join(output_directory, 'phrase_table')
        self._write_pt(phrase_table)

        mtplz_wrap_sh = os.path.join(output_directory, 'mtplz_wrap.sh')

    def _write_ini(self, mtplz_ini):
        fp_ini = open(mtplz_ini, 'w')
        fp_ini.write('{0} {1}\n'.format('target_word_insertion', self.word_penalty))
        fp_ini.write('{0} {1}\n'.format('lm', self.lm.weight))
        fp_ini.write('{0} {1}\n'.format('distortion', self.distortion_weight))
        fp_ini.write('{0} {1}\n'.format('phrase_table', ' '.join(map(str, self.phrase_table.weights.tolist()))))
        fp_ini.close()

    def _write_pt(self, path_pt):
        pt = PhraseTable(self.phrase_table.path)
        fp_pt = open(path_pt, 'w')
        for f_words, e_words, scores in pt.iter():
            scores = np.log(scores)
            fp_pt.write('{0} ||| {1} ||| {2}\n'.format(
                ' '.join(f_words),
                ' '.join(e_words),
                ' '.join(map(str, scores.tolist()))))
        fp_pt.close()

    def _get_name(self, feature_class_name):
        return self.reader.feature_conf[feature_class_name]['name']

    def _get_path(self, feature_class_name):
        return self.reader.feature_conf[feature_class_name]['path']

    def _get_weight(self, feature_class_name):
        feature_name = self._get_name(feature_class_name)
        return self.reader.weight_conf[feature_name] * self.multiply_weight[feature_class_name]

    @property
    def distortion_weight(self):
        return self._get_weight('Distortion')[0]

    @property
    def lm(self):
        lm_path = self._get_path('KenLM')
        lm_weight = self._get_weight('KenLM')
        return LM(lm_path, lm_weight[0])

    @property
    def word_penalty(self):
        return self._get_weight('WordPenalty')

    @property
    def phrase_table(self):
        tm_path = self._get_path('PhraseDictionaryMemory')
        tm_weight = self._get_weight('PhraseDictionaryMemory')
        return TM(tm_path, tm_weight)


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='moses.ini converter')
    parser.add_argument('--ini', help='path to a moses.ini.')
    parser.add_argument('--output', help='path to output directory.')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()


if __name__ == '__main__':
    main()

