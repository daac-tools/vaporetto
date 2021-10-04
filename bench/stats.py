#!/usr/bin/env python3

import collections
import math
import re
import sys


RE_DICT = [
    ('kytea', re.compile(r'Elapsed-kytea: ([0-9\.]+) \[sec\]')),
    ('vaporetto', re.compile(r'Elapsed: ([0-9\.]+) \[sec\]')),
    ('mecab', re.compile(r'Elapsed-mecab: ([0-9\.]+) \[sec\]')),
    ('kuromoji', re.compile(r'Elapsed-kuromoji: ([0-9\.]+) \[sec\]')),
    ('lindera', re.compile(r'Elapsed-lindera: ([0-9\.]+) \[sec\]')),
    ('sudachi', re.compile(r'Elapsed-sudachi: ([0-9\.]+) \[sec\]')),
    ('sudachi.rs', re.compile(r'Elapsed-sudachi.rs: ([0-9\.]+) \[sec\]')),
]

N_CHARS = 16318893


def mean_std(times: list[float]) -> (float, float):
    speeds = [N_CHARS / time for time in times]
    mean = sum(speeds) / len(speeds)
    dist = sum((speed - mean) ** 2 for speed in speeds) / len(speeds)
    return mean, math.sqrt(dist)


def _main():
    times = collections.defaultdict(list)
    for line in sys.stdin:
        for name, r in RE_DICT:
            m = r.match(line)
            if m is not None:
                times[name].append(float(m.group(1)))
                break

    for name, _ in RE_DICT:
        mean, std = mean_std(times[name])
        print(f'{name} {mean} {std}')


if __name__ == '__main__':
    _main()
