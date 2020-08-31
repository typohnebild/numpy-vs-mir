import matplotlib.pyplot as plt
import numpy as np
import os


def plot_multiple_flops_sec(str_startswith):
    for file in os.listdir("results"):
        if file.startswith(str_startswith):
            draw_flops_sec(*read_file(f"results/{file}"))
    plt.legend()
    plt.minorticks_on()
    plt.grid(color='b', linestyle='-', linewidth=0.2, alpha=0.5)


def plot_multiple_flops(str_startswith):
    for file in os.listdir("results"):
        if file.startswith(str_startswith):
            draw_flops(*read_file(f"results/{file}"))
    plt.legend()
    plt.minorticks_on()
    plt.grid(color='b', linestyle='-', linewidth=0.2, alpha=0.5)


def plot_multiple_times(str_startswith):
    for file in os.listdir("results"):
        if file.startswith(str_startswith):
            draw_time(*read_file(f"results/{file}"))
    plt.legend()
    plt.minorticks_on()
    plt.grid(color='b', linestyle='-', linewidth=0.2, alpha=0.5)


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
    return list(zip(*sorted(zip(*ret), key=lambda x: x[0]))), os.path.basename(filepath)


def draw_flops_sec(input_data, str_legend):
    print(str_legend)
    sizes, flops_sec, _, _ = avg_reduce(input_data)
    plt.plot(sizes, flops_sec, label=str_legend)
    plt.ylabel("Flops/sec")
    plt.xlabel("Problem size")


def draw_flops(input_data, str_legend):
    print(str_legend)
    sizes, _, _, flops = avg_reduce(input_data)
    plt.plot(sizes, flops, label=str_legend)
    plt.ylabel("Flops")
    plt.xlabel("Problem size")


def draw_time(input_data, str_legend):
    print(str_legend)
    sizes, _, sec, _ = avg_reduce(input_data)
    plt.plot(sizes, sec, label=str_legend)
    plt.ylabel("Time in s")
    plt.xlabel("Problem size")


def avg_reduce(input_data):
    dict = {}
    for n, flops, time  in zip(*input_data):
        if not n in dict:
            dict[n]=[]
        dict[n].append((flops, time))
    sizes, flops_sec, sec, flops = [],[],[], []
    for key, value in dict.items():
        sizes.append(key)
        flops_sec.append(sum([flop/time for flop, time in value])/len(value))
        sec.append(sum([time for _, time in value])/len(value))
        flops.append(sum([flops for flops, _ in value])/len(value))
    return sizes, flops_sec, sec, flops
