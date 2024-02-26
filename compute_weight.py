import numpy as np
import matplotlib.pyplot as plt
from pic_name import pic_name
import os


num_sample = 201
height = 96
width = 96
name = pic_name

location_c3 = []
location_h1 = []
location_h2 = []
location_h3 = []
location_v1 = []
location_v2 = []
location_v3 = []
location_d1 = []
location_d2 = []
location_d3 = []

if not os.path.exists("../weight/{}".format(name)):
    os.mkdir("../weight/{}".format(name))

file = open('../distribution/{}/c3.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_c3.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/h1.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_h1.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/h2.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_h2.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/h3.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_h3.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/v1.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_v1.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/v2.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_v2.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/v3.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_v3.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/d1.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_d1.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/d2.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_d2.append((int(curLine[0]), int(curLine[1])))
file = open('../distribution/{}/d3.txt'.format(name), encoding='utf8')
for line in file.readlines():
    curLine = line.strip().split(' ')
    location_d3.append((int(curLine[0]), int(curLine[1])))

for element in range(128):
    weight = np.zeros((height, width))
    for loc in location_c3:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_c3 = open('../distribution/{}/c3/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        c3 = []
        for line in file_c3.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            c3.append(curLine)
        c3 = np.array(c3)
        weight[i][j], _ = np.polyfit(x, c3, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_h1:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_h1 = open('../distribution/{}/h1/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        h1 = []
        for line in file_h1.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            h1.append(curLine)
        h1 = np.array(h1)
        j += height // 2
        weight[i][j], _ = np.polyfit(x, h1, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_h2:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_h2 = open('../distribution/{}/h2/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        h2 = []
        for line in file_h2.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            h2.append(curLine)
        h2 = np.array(h2)
        j += height // 4
        weight[i][j], _ = np.polyfit(x, h2, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_h3:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_h3 = open('../distribution/{}/h3/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        h3 = []
        for line in file_h3.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            h3.append(curLine)
        h3 = np.array(h3)
        j += height // 8
        weight[i][j], _ = np.polyfit(x, h3, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_v1:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_v1 = open('../distribution/{}/v1/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        v1 = []
        for line in file_v1.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            v1.append(curLine)
        v1 = np.array(v1)
        i += height // 2
        weight[i][j], _ = np.polyfit(x, v1, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_v2:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_v2 = open('../distribution/{}/v2/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        v2 = []
        for line in file_v2.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            v2.append(curLine)
        v2 = np.array(v2)
        i += height // 4
        weight[i][j], _ = np.polyfit(x, v2, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_v3:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_v3 = open('../distribution/{}/v3/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        v3 = []
        for line in file_v3.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            v3.append(curLine)
        v3 = np.array(v3)
        i += height // 8
        weight[i][j], _ = np.polyfit(x, v3, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_d1:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_d1 = open('../distribution/{}/d1/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        d1 = []
        for line in file_d1.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            d1.append(curLine)
        d1 = np.array(d1)
        i += height // 2
        j += height // 2
        weight[i][j], _ = np.polyfit(x, d1, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_d2:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_d2 = open('../distribution/{}/d2/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        d2 = []
        for line in file_d2.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            d2.append(curLine)
        d2 = np.array(d2)
        i += height // 4
        j += height // 4
        weight[i][j], _ = np.polyfit(x, d2, 1)
        weight[i][j] = abs(weight[i][j])
    for loc in location_d3:
        i = loc[0]
        j = loc[1]
        # load data
        x = np.linspace(1, num_sample, num_sample)
        file_d3 = open('../distribution/{}/d3/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), encoding='utf8')
        d3 = []
        for line in file_d3.readlines():
            curLine = line.strip().split(' ')
            for itr in range(len(curLine)):
                curLine[itr] = float(curLine[itr][:-5]) / float(pow(10, int(curLine[itr][-1])))
            d3.append(curLine)
        d3 = np.array(d3)
        i += height // 8
        j += height // 8
        weight[i][j], _ = np.polyfit(x, d3, 1)
        weight[i][j] = abs(weight[i][j])
    np.savetxt('../weight/{}/weight{}.txt'.format(name, element), weight)