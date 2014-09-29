import tempfile
import heapq

__all__ = ["sort_on_disk"]


def sort_and_write(block, temporary_dir):
    fp = tempfile.TemporaryFile(mode="w+", dir=temporary_dir)

    block.sort()
    for line in block:
        fp.write(line)
        fp.write("\n")
    fp.seek(0)
    return fp


def merge_and_write(parts, temporary_dir):
    fp = tempfile.TemporaryFile(mode="w+", dir=temporary_dir)

    for line in heapq.merge(*parts):
        line = line.rstrip("\n")
        fp.write(line)
        fp.write("\n")

    for part in parts:
        part.close()

    fp.seek(0)
    return fp


def sort_on_disk(fp, temporary_dir, block_size=5000):
    """
    sort and unique a file on disk.
    :param fp: a file io object. must be seek(0).
    :param temporary_dir: a directory to use by sort.
    :return: a file io object.
    """

    block = []
    parts = []
    for line in fp:
        line = line.rstrip("\n")
        block.append(line)
        if len(block) >= block_size:
            tmp_fp = sort_and_write(block, temporary_dir)
            parts.append(tmp_fp)
            block = []
    if len(block) > 0:
        tmp_fp = sort_and_write(block, temporary_dir)
        parts.append(tmp_fp)

    return merge_and_write(parts, temporary_dir)

