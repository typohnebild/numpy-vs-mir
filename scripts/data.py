#!/bin/env python

import numpy as np
import pandas as pd
import optparse
import matplotlib.pyplot as plt
import os.path

import matplotlib.colors as mcolor
from itertools import cycle

plt.rcParams['figure.figsize'] = (16, 9)

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


def plot_cache_lines(fig):
    l1 = 32e3
    l2 = 256e3
    l3 = 12288e3

    def cache2size(x):
        return np.sqrt(x / 8)

    fig.axvline(cache2size(l1), color='black', ls='--')
    fig.axvline(cache2size(l2), color='black', ls='--')
    fig.axvline(cache2size(l3), color='black', ls='--')


def plot_membandwidth(fig):
    # calculation from there https://www.cs.virginia.edu/stream/ref.html
    # take scale value and divide by 16 since it produces 1 flops per 24 byte
    # writen
    # mem_band = 22490.3 * 1e6  # convert from MB/s to B/s
    # take triad value and divide by 16 since it produces 1 flops per 24 byte
    # writen
    mem_band = 25104.3 * 4 / 3 * 1e6  # convert Triad * 4/3 (write-allocate) from MB/s to B/s
    fig.axhline(mem_band / 12, color='black', ls=':')


def subplots(frames, base_path, column):
    color = cycle(mcolor.TABLEAU_COLORS.keys())
    plt.clf()

    fig, axes = plt.subplots((len(frames) + 1) // 2,
                             2,
                             figsize=(12, 15))

    overflow = len(frames) < len(axes.flat)
    if overflow:
        axes.flat[3].axis('off')

    for i, frame in enumerate(frames):
        name, df = frame
        axe = overflow and axes.flat[i] if i < 3 else axes.flat[i + 1]
        g = df.groupby('size').median()[column]
        g.plot(label=name, ax=axe, marker='o', color=next(color))
        axe.grid(
            color='b',
            linestyle='-',
            linewidth=0.3,
            alpha=0.5,
            which='major')
        axe.grid(
            color='b',
            linestyle='-',
            linewidth=0.1,
            alpha=0.5,
            which='minor')

        axe.set(ylabel=column)
        axe.minorticks_on()
        axe.set_title(name)
        plot_cache_lines(axe)

    fig.tight_layout()
    fig.savefig(f'{base_path}_{column}_subplots.png', bbox_inches='tight')


def plot(frames, func, base_path, title):
    plt.clf()
    if func.__name__ == 'flops' and len(frames) == 11:
        plot_membandwidth(plt)
    plt.title(title)
    for name, df in frames:
        func(df, name)
    plt.legend()
    plt.minorticks_on()
    plt.grid(color='b', linestyle='-', linewidth=0.3, alpha=0.5, which='major')
    plt.grid(color='b', linestyle='-', linewidth=0.1, alpha=0.5, which='minor')
    plot_cache_lines(plt)

    plt.savefig(f'{base_path}_{func.__name__}.png', bbox_inches='tight')


def extract_name(path):
    parts = os.path.basename(path).split('_')
    if len(parts) == 4:
        return f'D with Mir ({parts[-1]})'
    if len(parts) == 5:
        return f'D with Mir ({parts[-2]})'
    return ' '.join(parts[-4:-1])


def extract_date_host(path):
    return ''.join(os.path.basename(path).split('_')[1:3])


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

    parser.add_option('-g',
                      action='store_true',
                      dest='groups',
                      default=False,
                      help='also print a plots for every group')

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

    plot(frames, flops, base_name, 'Floating Point Operations per second')

    plot(frames, time, base_name, 'Time in Seconds')

    if options.groups:
        only_D = [x for x in frames if x[0].startswith("D")]
        only_numba = [x for x in frames if x[0].split(" ")[-1] == "numba"]
        only_nonumba = [x for x in frames if x[0].split(" ")[-1] == "nonumba"]
        plot(only_D, flops, f'{base_name}D',
             'Floating Point Operations per second D with Mir')
        plot(only_D, time, f'{base_name}D', 'Time in Seconds D with Mir')
        plot(only_numba, flops, f'{base_name}numba',
             'Floating Point Operations per second with numba')
        plot(
            only_numba,
            time,
            f'{base_name}numba',
            'Time in Seconds with numba')
        plot(
            only_nonumba,
            flops,
            f'{base_name}nonumba',
            'Floating Point Operations per second without numba')
        plot(
            only_nonumba,
            time,
            f'{base_name}nonumba',
            'Time in Seconds without numba')

    if options.subs:
        subplots(frames, base_name, 'FLOPS')
        subplots(frames, base_name, 'time')


if __name__ == '__main__':
    main()
