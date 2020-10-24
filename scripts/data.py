#!/bin/env python
"""
A not to well writen script to produces the figures.

It uses pandas to read the outfiles
arguments: outfiles
options:
    -o : sets the path were the figures are stored
    -s : produces the plots for each outfile
    -g : produces the plots for every group (D, numba, nonumba)
    --nl : disables the printing of the lines for caches/bandwidth
"""

import optparse
import os.path
from copy import copy
from itertools import cycle

import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'both'
plt.style.use('ggplot')

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

parser.add_option(
    '--nl',
    action='store_false',
    dest='lines',
    default=True,
    help='do NOT plot cache size and memory bandwidth in plots')

options, args = parser.parse_args()

COLORS = cycle([*mcolor.TABLEAU_COLORS.keys(), 'darkcyan'])
COLOR_TABLE = {}


def name_2_color(name):
    if name not in COLOR_TABLE:
        COLOR_TABLE[name] = next(COLORS)

    return COLOR_TABLE[name]


def read_file(path):
    """ reads the outfiles and extracts the data and infos """
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


def name_2_linestyle(name):
    if '1' in name:
        return (0, (1, 1))  # densly dotted
    if '8' in name:
        return (0, (5, 10))  # loosley dashed
    return '-'


def flops(df, label=None):
    df.FLOPS.plot(
        label=label,
        marker='x',
        color=name_2_color(label),
        ls=name_2_linestyle(label))
    plt.ylabel('FLOP/s')


def flop(df, label):
    df.FLOP.plot(
        label=label,
        color=name_2_color(label),
        ls=name_2_linestyle(label))
    plt.ylabel('Flop')


def time(df, label):
    df.time.plot(
        label=label,
        color=name_2_color(label),
        ls=name_2_linestyle(label))
    plt.ylabel('time in s')


def cycles(df, label):
    df.cycles.plot(label=label, marker='x', color=name_2_color(label))
    plt.ylabel('Number of used MG-Cycels')


def plot_cache_lines(fig):
    # if not options.lines:
    #     return

    l1 = 32e3
    l2 = 256e3
    l3 = 12288e3

    def cache2size(x):
        return np.sqrt(x / 8)

    l1s = cache2size(l1)
    l2s = cache2size(l2)
    l3s = cache2size(l3)

    l1_line = fig.axvline(
        l1s,
        color='black',
        ls=':',
        label='L1 cache (32K) ',
        alpha=0.7)

    fig.axvline(
        l2s,
        color='black',
        ls='--',
        label='L2 cache (256K)',
        alpha=0.7)

    fig.axvline(
        l3s,
        color='black',
        ls='-.',
        label='L3 cache (12288K)',
        alpha=0.7)


def plot_membandwidth(fig):
    """
    Plot Memory Bandwidth line.

    Calculation from there
    https://moodle.rrze.uni-erlangen.de/pluginfile.php/16786/mod_resource/content/1/09_06_04-2020-PTfS.pdf
    take triad value and divide by 16 since it produces 2 flops per 32 byte
    writen
    convert Triad * 4/3 (write-allocate) from MB/s to B/s
    """
    if not options.lines:
        return
    mem_band = (25104.3 * (4 / 3) * 1e6) / 16
    fig.axhline(
        mem_band,
        color='black',
        ls=':',
        alpha=0.7)
    fig.annotate(
        'Memory Bandwidth',
        xy=(10, mem_band),
        xycoords='data',
        xytext=(10, mem_band + 0.15 * mem_band),
        textcoords='data',
        arrowprops=dict(facecolor='black', shrink=0.15),
        horizontalalignment='right',
        verticalalignment='center'
    )


def subplots(frames, base_path, column):
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
        g = df[column]
        g.plot(
            label=name,
            ax=axe,
            marker='x',
            color=name_2_color(name),
            ls=name_2_linestyle(name))

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
    plt.minorticks_on()
    plot_cache_lines(plt)
    plt.legend(ncol=2)

    plt.savefig(f'{base_path}_{func.__name__}.png', bbox_inches='tight')


def extract_name(path):
    parts = os.path.basename(path).split('_')
    if len(parts) == 4:
        return f'D with Mir ({parts[-1]})'
    if len(parts) == 5:
        return f'D with Mir ({parts[-2]})'
    return ' '.join(parts[-4:-1])


def extract_picture_name(path):
    splits = os.path.basename(path).split('_')
    return splits[-1]


def main():

    if not args:
        parser.print_usage()
        exit(1)

    frames = []
    name = extract_picture_name(args[0])
    base_name = f'{options.outpath}/{name}'

    for arg in args:
        _, df = read_file(arg)
        frames.append((extract_name(arg), df.groupby('size').median()))

    plot(frames, flops, base_name, 'Floating Point Operations per second')

    plot(frames, time, base_name, 'Time in Seconds')

    # plot(frames, cycles, base_name, 'MG-Cycles till convergence')

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
