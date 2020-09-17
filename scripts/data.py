#!/bin/env python

import numpy as np
import pandas as pd
import optparse
import matplotlib.pyplot as plt
import os.path

DEFAULT_FILE = '../Python/results/outfile_cip1e3_1609_intel_1_numba'
DEFAULT_OUT = '../graphs'
NAMES = [
    'N',
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
            header=None,
            thousands=',',
            names=NAMES)
        del df['empty']
        df['FLOPS'] = (df['scalar_single'] +
                       df['scalar_double'] +
                       df['128b_packed_double'] +
                       df['128b_packed_single'] +
                       df['256b_packed_double'] +
                       df['256b_packed_single'])
        df['FLOPSS'] = df['FLOPS'] / df['time']
        return infos, df


def flopss(df, label):
    df.groupby('N').mean().FLOPSS.plot(label=label)
    plt.ylabel('Flops/sec')


def flops(df, label):
    df.groupby('N').mean().FLOPS.plot(label=label)
    plt.ylabel('Flops')


def time(df, label):
    df.groupby('N').mean().time.plot(label=label)
    plt.ylabel('Time in sec')


def extract_name(path):
    parts = os.path.basename(path).split('_')
    if len(parts) == 3:
        return 'D with Mir'
    return ' '.join(parts[-3:])


def extract_date_host(path):
    return ''.join(os.path.basename(path).split('_')[1:3])


def plot(frames, func, base_path):
    plt.clf()
    for name, df in frames:
        func(df, name)
    plt.legend()
    plt.minorticks_on()
    plt.grid(color='b', linestyle='-', linewidth=0.2, alpha=0.5)
    plt.savefig(f'{base_path}_{func.__name__}.png')


def main():
    parser = optparse.OptionParser()
    parser.add_option('-o',
                      action='store',
                      dest='outpath',
                      default=DEFAULT_OUT,
                      help='path to save pictures')

    options, args = parser.parse_args()

    frames = []
    name = extract_date_host(args[0])
    base_name = f'{options.outpath}/{name}'

    for arg in args:
        _, df = read_file(arg)
        frames.append((extract_name(arg), df))

    plot(frames, flopss, base_name)
    plot(frames, flops, base_name)
    plot(frames, time, base_name)


if __name__ == '__main__':
    main()
