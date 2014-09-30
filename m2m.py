#! -*- coding: utf-8 -*-

import numpy as np
import os
from utils.smart_fp import smart_open
import gzip


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
    def _parse_line_primitive(line):
        """
        :rtype : list, list, np.ndarray
        """
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
            f_words, e_words, scores = self._parse_line_primitive(line)
            yield (f_words, e_words, scores)
        self._close_if_needed(fp)


class MosesIniReader(object):
    DEFAULT_STACK_SIZE = 200
    DEFAULT_DISTORTION_LIMIT = -1

    def __init__(self, path):
        self.config = {}
        self.feature_conf = {}
        self.weight_conf = {}

        if isinstance(path, str):
            fp = smart_open(path)
        else:
            fp = path
            fp.seek(0)
        self._parse(fp)
        self._parse_feature()
        self._parse_weight()
        self.distortion_limit = int(
            self.config.get('distortion-limit', [self.DEFAULT_DISTORTION_LIMIT])[0])
        self.stack = int(self.config.get('stack', [self.DEFAULT_STACK_SIZE])[0])

    def _parse(self, fp):
        name = None
        for line in fp:
            line = line.strip()

            if len(line) == 0 or line[0] == '#':
                continue
            if line[0] == '[':
                name = line[1:-1]
            else:
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
        KENLM=2.30258509299404568401,
        Distortion=-1,
        PhraseDictionaryMemory=1,
    )

    def __init__(self, path, decoder_path):
        self.reader = MosesIniReader(path)
        self.decoder_path = decoder_path
        self._lm = None
        self._pt = None

    def __call__(self, output_directory):
        if not os.path.exists(output_directory):
            os.mkdir(output_directory, 0755)

        mtplz_ini = os.path.join(output_directory, 'mtplz.ini')
        self._write_ini(mtplz_ini)

        phrase_table = os.path.join(output_directory, 'phrase_table.gz')
        self._write_pt(phrase_table)

        mtplz_wrap_sh = os.path.join(output_directory, 'mtplz_wrap.sh')
        self._write_wrapper(mtplz_wrap_sh, mtplz_ini, phrase_table)

    def _write_ini(self, mtplz_ini):
        fp_ini = open(mtplz_ini, 'w')
        fp_ini.write('{0} {1}\n'.format('target_word_insertion', self.word_penalty))
        fp_ini.write('{0} {1}\n'.format('lm', self.lm.weight))
        fp_ini.write('{0} {1}\n'.format('distortion', self.distortion_weight))
        fp_ini.write('{0} {1}\n'.format('phrase_table', ' '.join(map(str, self.phrase_table.weights.tolist()))))
        fp_ini.close()

    def _write_pt(self, path_pt):
        pt = PhraseTable(self.phrase_table.path)
        fp_pt = gzip.open(path_pt, 'w')
        for f_words, e_words, scores in pt.iter():
            scores = np.log(scores)
            fp_pt.write('{0} ||| {1} ||| {2}\n'.format(
                ' '.join(f_words),
                ' '.join(e_words),
                ' '.join(map(str, scores.tolist()))))
        fp_pt.close()

    def _write_wrapper(self, mtplz_wrap_sh, mtplz_ini, phrase_table):
        fp_sh = open(mtplz_wrap_sh, 'w')
        script = [
            "#!/bin/bash",
            "LM={0}".format(self.lm.path),
            "PT={0}".format(phrase_table),
            "WEIGHTS={0}".format(mtplz_ini),
            "BEAM={0}".format(self.reader.stack),
            "LIMIT={0}".format(self.reader.distortion_limit),
            "BIN={0}".format(self.decoder_path),
            "LC_ALL=C $BIN --lm $LM --phrase $PT --weights_file $WEIGHTS --beam $BEAM --reordering $LIMIT"
        ]
        script = '\n'.join(script)
        fp_sh.write(script)
        fp_sh.close()
        os.chmod(mtplz_wrap_sh, 0755)

    def _get_name(self, feature_class_name):
        return self.reader.feature_conf[feature_class_name].get('name', '{0}0'.format(feature_class_name))

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
        if self._lm is None:
            lm_path = self._get_path('KENLM')
            lm_weight = self._get_weight('KENLM')
            self._lm = LM(lm_path, lm_weight[0])
            return self._lm
        else:
            return self._lm

    @property
    def word_penalty(self):
        return self._get_weight('WordPenalty')[0]

    @property
    def phrase_table(self):
        if self._pt is None:
            tm_path = self._get_path('PhraseDictionaryMemory')
            tm_weight = self._get_weight('PhraseDictionaryMemory')
            self._pt = TM(tm_path, tm_weight)
            return self._pt
        else:
            return self._pt


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='moses.ini converter')
    parser.add_argument('--ini', help='path to a moses.ini.')
    parser.add_argument('--decoder', help='path to a decoder.')
    parser.add_argument('--output', help='path to output directory.')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    converter = Converter(args.ini, args.decoder)
    converter(args.output)


if __name__ == '__main__':
    main()

