import matplotlib.pyplot as plt
import numpy as np


def read_file(filepath):
    ret = [[], [], []]
    with open(filepath, 'r') as target:
        lines = target.readlines()
        infos = ""
        if lines[0].startswith('##'):
            line = lines.pop(0)
            line = lines.pop(0)
            while not line.startswith('##'):
                infos += line
                line = lines.pop(0)

        for line in lines:
            if line.count(':') == 1:
                N, time = line.split(':')
                ret[0].append(float(N))
                ret[1].append(float(time))
                ret[2].append(float(time))
            else:
                N, *flop, time, _ = line.split(':')
                ret[0].append(float(N))
                ret[1].append(sum([float(x.replace(',', '')) for x in flop]))
                ret[2].append(float(time))
    # this assures that we have that sorted by N
    return list(zip(*sorted(zip(*ret), key=lambda x: x[0]))), infos


def draw_flops(input_data):
    flops = np.array(input_data[1]) / np.array(input_data[2])
    plt.plot(input_data[0], flops)
    plt.ylabel("Flops")
    plt.xlabel("Problem size")


def draw_time(input_data):
    plt.plot(input_data[0], input_data[1])
    plt.ylabel("Time in s")
    plt.xlabel("Problem size")
