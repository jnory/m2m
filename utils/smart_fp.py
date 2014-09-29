import sys
import gzip

__all__ = ['smart_open']


def smart_open(path):
    if path[-3:] == '.gz':
        fp = gzip.open(path)
    elif path == '-':
        fp = sys.stdin
    else:
        fp = open(path)
    return fp
