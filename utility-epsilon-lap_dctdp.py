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
import argparse
from pic_name import pic_name
import os
from skimage.metrics import structural_similarity as compare_ssim


def compute_psnr(I, K):
    MSE = 0
    ori_size = 96
    for iiii in range(ori_size):
        for jjjj in range(ori_size):
            MSE += 1 / pow(ori_size, 2) * pow(I[iiii][jjjj] - K[iiii][jjjj], 2)
    return 10 * np.log10(pow(255, 2) / MSE)


def laplace(noise_scale, size):
    return list(np.random.laplace(0, scale=noise_scale, size=size))


name = pic_name
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}".format(name))
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/dctdp".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/dctdp".format(name))
if not os.path.exists("./data/{}".format(name)):
    os.mkdir("./data/{}".format(name))
if not os.path.exists("./data/{}/dctdp".format(name)):
    os.mkdir("./data/{}/dctdp".format(name))

b_begin = 293
x7 = []
y7 = []
y7b = []
for iii in range(b_begin, b_begin + 1):
    if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/dctdp/{}".format(name, iii / 10)):
        os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/dctdp/{}".format(name, iii / 10))
    sample = 2000
    p = 0.02
    origin_size = 96
    all_pixels = []
    for i in range(origin_size):
        for j in range(origin_size):
            all_pixels.append([i, j])
    utility_pix = 0
    utility_b = 0
    img = cv2.imread('../img/{}_gray.png'.format(name), 0)
    img = np.float32(img)
    psnr = 0
    ssim = 0
    accuracy = 0
    for itr in range(pow(origin_size, 2)):
        utility_b += 2 * pow(iii / 10, 2) * (pow(1 - p, itr + 1) - pow(1 - p, pow(origin_size, 2) + 1))
    for itr in range(sample):
        k = min(geo(p, 1)[0], 100)
        pixels = random.sample(all_pixels, k)
        img_dct = cv2.dct(img)
        img_dct = np.array(img_dct)
        for pix in pixels:
            noise = laplace(iii / 10, 1)[0]
            img_dct[pix[0]][pix[1]] += noise
        img_idct = np.array(cv2.idct(img_dct), np.float32)
        for ii in range(origin_size):
            for jj in range(origin_size):
                utility_pix += pow(img_idct[ii][jj] - img[ii][jj], 2)
        psnr += compute_psnr(img_idct, img)
        cv2.imwrite('../img/add_noise_reverse_haar_wavelet/{}/dctdp/{}/{}_noise_{}.png'.format(name, iii / 10, name, itr), img_idct)
        imageA = cv2.imread('../img/{}_gray.png'.format(name))
        imageB = cv2.imread('../img/add_noise_reverse_haar_wavelet/{}/dctdp/{}/{}_noise_{}.png'.format(name, iii / 10, name, itr))
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        ssim += score
    ssim /= sample
    psnr /= sample
    utility_pix /= sample
    # compute p(b)
    face_rec = fc.face_recognition()  # 创建对象
    face_rec.inputPerson(name='{}_origin'.format(name),
                         img_path=f'\\..\\img\\{name}_gray.png')  # name中写第一个人名字，img_name为图片名字，注意要放在faces文件夹中
    vector = face_rec.create128DVectorSpace()  # 提取128维向量，是dlib.vector类的对象
    person_data1 = fc.savePersonData(face_rec, vector)  # 将提取出的数据保存到data文件夹，为便于操作返回numpy数组，内容还是一样的
    count_another = 0
    count_none = 0
    for k in range(sample):
        # 导入第二张图片，并提取特征向量
        face_rec.inputPerson(name='{}_noise'.format(name),
                             img_path=f'\\..\\img\\add_noise_reverse_haar_wavelet\\{name}\\dctdp\\{iii / 10}\\{name}_noise_{k}.png')
        if face_rec.create128DVectorSpace() is not None:
            vector = face_rec.create128DVectorSpace()  # 提取128维向量，是dlib.vector类的对象
            person_data2 = fc.savePersonData(face_rec, vector)
            # 计算欧式距离，判断是否是同一个人
            if not fc.comparePersonData(person_data1, person_data2):
                count_another += 1
        else:
            count_none += 1
    pb = (count_another + count_none) / sample
    accuracy = 1 - pb
    # compute epsilon
    # - epsilon = ln(pb)
    if (count_another + count_none) > 0:
        real_epsilon = -np.log(pb)
        # prepare data to draw picture
        x7.append([real_epsilon, iii / 10, psnr, ssim])
        y7.append(utility_b)
        y7b.append(utility_pix)
np.savetxt('./data/{}/dctdp/x7_{}.txt'.format(name, b_begin), x7)
np.savetxt('./data/{}/dctdp/y7_{}.txt'.format(name, b_begin), y7)
np.savetxt('./data/{}/dctdp/y7b_{}.txt'.format(name, b_begin), y7b)