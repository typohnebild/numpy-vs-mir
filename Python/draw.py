import matplotlib.pyplot as plt
import numpy as np


def read_file(filepath):
    ret = [[], [], []]
    with open(filepath, 'r') as target:
        for line in target:
            N, *flop, time, _ = line.split(':')
            ret[0].append(float(N))
            ret[1].append(sum([float(x.replace(',', '')) for x in flop]))
            ret[2].append(float(time))
    return ret


def draw(input_data):
    flops = np.array(input_data[1]) / np.array(input_data[2])
    plt.plot(input_data[0], flops)
    plt.ylabel("Flops")
    plt.xlabel("Problem size")
