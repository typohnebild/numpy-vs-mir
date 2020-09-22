#!/bin/env python

import numpy as np
import pandas as pd
import optparse
import matplotlib.pyplot as plt
import os.path

import matplotlib.colors as mcolor
from itertools import cycle


DEFAULT_FILE = '../Python/results/outfile_cip1e3_1609_intel_1_numba'
DEFAULT_OUT = '../graphs'
NAMES = [
    'size',
    'dimension',
    'time',
    'cycles',
    'error',
    'scalar_single',
    'scalar_double',
    '128b_packed_double',
    '128b_packed_single',
    '256b_packed_double',
    '256b_packed_single',
    'empty',
]


def read_file(path):
    infos = ''
    with open(path, 'r') as file_des:
        line = file_des.readline()  # read frist line
        if line.startswith('##'):
            line = file_des.readline()
            while not line.startswith('##'):
                infos += line
                line = file_des.readline()
        df = pd.read_csv(
            file_des,
            sep=':',
            comment='#',
            thousands=',')
        del df['empty']
        df['FLOP'] = (df['scalar_single'] +
                      df['scalar_double'] +
                      df['128b_packed_double'] +
                      df['128b_packed_single'] +
                      df['256b_packed_double'] +
                      df['256b_packed_single'])
        df['FLOPS'] = df['FLOP'] / df['time']
        return infos, df


def flops(df, label):
    df.groupby('size').median().FLOPS.plot(label=label)
    plt.ylabel('FLOP/s')


def flop(df, label):
    df.groupby('size').median().FLOP.plot(label=label)
    plt.ylabel('Flop')


def time(df, label):
    df.groupby('size').median().time.plot(label=label)
    plt.ylabel('time in s')


def subplots(frames, base_path, column):
    color = cycle(mcolor.TABLEAU_COLORS.keys())
    plt.clf()
    fig, axes = plt.subplots(len(frames), 1, sharex=True)
    for i, axe in enumerate(axes):
        name, df = frames[i]
        g = df.groupby('size').median()[column]
        g.plot(label=name, ax=axe, marker='o', color=next(color))
        axe.grid(color='b', linestyle='-', linewidth=0.2, alpha=0.5)
        axe.set(ylabel=column)
        axe.set_title(name)

    fig.tight_layout()
    fig.savefig(f'{base_path}_{column}_subplots.png')


def extract_name(path):
    parts = os.path.basename(path).split('_')
    if len(parts) == 4:
        return f'D with Mir ({parts[-1]})'
    return ' '.join(parts[-3:])


def extract_date_host(path):
    return ''.join(os.path.basename(path).split('_')[1:3])


def plot(frames, func, base_path, title):
    plt.clf()
    plt.title(title)
    for name, df in frames:
        func(df, name)
    plt.legend()
    plt.minorticks_on()
    plt.grid(color='b', linestyle='-', linewidth=0.2, alpha=0.5)
    plt.savefig(f'{base_path}_{func.__name__}.png')


def main():
    parser = optparse.OptionParser(usage="usage: %prog [options] files...")
    parser.add_option('-o',
                      action='store',
                      dest='outpath',
                      default=DEFAULT_OUT,
                      help='path to save pictures')
    parser.add_option('-s',
                      action='store_true',
                      dest='subs',
                      default=False,
                      help='also print a subplot for every single file')

    options, args = parser.parse_args()
    if not args:
        parser.print_usage()
        exit(1)

    frames = []
    name = extract_date_host(args[0])
    base_name = f'{options.outpath}/{name}'

    for arg in args:
        _, df = read_file(arg)
        frames.append((extract_name(arg), df))

    plot(frames, flops, base_name, 'Floating Point Operations / second')
    # plot(frames, flop, base_name, 'Floating Point Operations')
    plot(frames, time, base_name, 'Time in Seconds')

    if options.subs:
        subplots(frames, base_name, 'FLOPS')
        subplots(frames, base_name, 'time')


if __name__ == '__main__':
    main()
