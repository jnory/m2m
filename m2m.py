#! -*- coding: utf-8 -*-

import gzip
import sys
import numpy as np


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
            if self._path[-3:] == '.gz':
                fp = gzip.open(self._path)
            elif self._path == '-':
                fp = sys.stdin
            else:
                fp = open(self._path)
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
        self._path = path

    def _read_block(self, fp):
        pass



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

