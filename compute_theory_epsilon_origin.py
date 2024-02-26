import numpy as np
import matplotlib.pyplot as plt
from numpy.random import geometric as geo
import cv2
from pywt import dwt2, idwt2
import pywt
import face_rec as fc
import argparse
import random
from scipy.optimize import Bounds, minimize
import time
from pic_name import pic_name
import os


name = pic_name
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}".format(name))
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/origin".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/origin".format(name))
if not os.path.exists("./data/{}".format(name)):
    os.mkdir("./data/{}".format(name))
if not os.path.exists("./data/{}/origin".format(name)):
    os.mkdir("./data/{}/origin".format(name))

origin_size = 96
sample = 2000
x4 = []
y4 = []
y4b = []

# haar wavelet
for kk in range(145, 146):
    if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/origin/{}".format(name, kk / 10)):
        os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/origin/{}".format(name, kk / 10))
    b = kk / 10
    p = 0.02
    all_pixels = []
    for i in range(origin_size):
        for j in range(origin_size):
            all_pixels.append([i, j])
    utility_pix = 0
    utility_b = 0
    PSNR = 0
    accuracy = 0
    theory_eps = 0
    count = 0
    for iii in range(10000):
        # compute theory epsilon
        sensitivity = 0.01
        file_weight = open('../weight/{}/weight0.txt'.format(name), encoding='utf8')
        weight = []
        for line in file_weight.readlines():
            cur = line.strip().split(' ')
            for itr in range(len(cur)):
                cur[itr] = float(cur[itr][:-5]) / float(pow(10, int(cur[itr][-1])))
            weight.append(cur)
        weight = np.array(weight)
        k = min(geo(p, 1)[0], 100)
        pixels = random.sample(all_pixels, k)
        fenmu = 0
        for pp in range(len(pixels)):
            fenmu += pow(pow(weight[pixels[pp][0]][pixels[pp][1]] * 60000 * b, 2) * (
                        pow(1 - p, pp + 1) - pow(1 - p, pow(origin_size, 2) + 1)), 0.5)
        if fenmu != 0:
            theory_eps += sensitivity / fenmu * 128
            count += 1
    print(count)
    print(theory_eps / count)