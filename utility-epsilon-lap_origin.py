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


def origin(b):
    p = 0.02
    all_pixels = []
    for i in range(origin_size):
        for j in range(origin_size):
            all_pixels.append([i, j])
    utility_pix = 0
    utility_b = 0
    psnr = 0
    ssim = 0
    for itr in range(pow(origin_size, 2)):
        utility_b += 2 * pow(b, 2) * (pow(1 - p, itr + 1) - pow(1 - p, pow(origin_size, 2) + 1))
    for itr in range(sample):
        # 几何分布选取像素点
        k = min(geo(p, 1)[0], 100)
        pixels = random.sample(all_pixels, k)
        pixels_c3 = []
        pixels_h3 = []
        pixels_v3 = []
        pixels_d3 = []
        pixels_h2 = []
        pixels_v2 = []
        pixels_d2 = []
        pixels_h1 = []
        pixels_v1 = []
        pixels_d1 = []
        for i in range(len(pixels)):
            if pixels[i][0] < 12 and pixels[i][1] < 12:
                pixels_c3.append(pixels[i])
            elif pixels[i][0] < 12 and 12 <= pixels[i][1] < 24:
                pixels_h3.append(pixels[i])
            elif 12 <= pixels[i][0] < 24 and pixels[i][1] < 12:
                pixels_v3.append(pixels[i])
            elif 12 <= pixels[i][0] < 24 and 12 <= pixels[i][1] < 24:
                pixels_d3.append(pixels[i])
            elif pixels[i][0] < 24 and 24 <= pixels[i][1] < 48:
                pixels_h2.append(pixels[i])
            elif 24 <= pixels[i][0] < 48 and pixels[i][1] < 24:
                pixels_v2.append(pixels[i])
            elif 24 <= pixels[i][0] < 48 and 24 <= pixels[i][1] < 48:
                pixels_d2.append(pixels[i])
            elif pixels[i][0] < 48 and 48 <= pixels[i][1] < 96:
                pixels_h1.append(pixels[i])
            elif 48 <= pixels[i][0] < 96 and pixels[i][1] < 48:
                pixels_v1.append(pixels[i])
            elif 48 <= pixels[i][0] < 96 and 48 <= pixels[i][1] < 96:
                pixels_d1.append(pixels[i])
        img = cv2.imread('../img/{}_gray.png'.format(name), 0).astype(np.float64)
        origin_c1, (origin_h1, origin_v1, origin_d1) = dwt2(img, 'haar')
        origin_c2, (origin_h2, origin_v2, origin_d2) = dwt2(origin_c1, 'haar')
        origin_c3, (origin_h3, origin_v3, origin_d3) = dwt2(origin_c2, 'haar')
        for pix in pixels_c3:
            noise = laplace(b, 1)[0]
            origin_c3[pix[0]][pix[1]] += noise
        for pix in pixels_h3:
            noise = laplace(b, 1)[0]
            origin_h3[pix[0]][pix[1] - 12] += noise
        for pix in pixels_v3:
            noise = laplace(b, 1)[0]
            origin_v3[pix[0] - 12][pix[1]] += noise
        for pix in pixels_d3:
            noise = laplace(b, 1)[0]
            origin_d3[pix[0] - 12][pix[1] - 12] += noise
        for pix in pixels_h2:
            noise = laplace(b, 1)[0]
            origin_h2[pix[0]][pix[1] - 24] += noise
        for pix in pixels_v2:
            noise = laplace(b, 1)[0]
            origin_v2[pix[0] - 24][pix[1]] += noise
        for pix in pixels_d2:
            noise = laplace(b, 1)[0]
            origin_d2[pix[0] - 24][pix[1] - 24] += noise
        for pix in pixels_h1:
            noise = laplace(b, 1)[0]
            origin_h1[pix[0]][pix[1] - 48] += noise
        for pix in pixels_v1:
            noise = laplace(b, 1)[0]
            origin_v1[pix[0] - 48][pix[1]] += noise
        for pix in pixels_d1:
            noise = laplace(b, 1)[0]
            origin_d1[pix[0] - 48][pix[1] - 48] += noise
        origin_c2_ = pywt.idwt2((origin_c3, (origin_h3, origin_v3, origin_d3)), 'haar')
        origin_c1_ = pywt.idwt2((origin_c2_, (origin_h2, origin_v2, origin_d2)), 'haar')
        img1 = pywt.idwt2((origin_c1_, (origin_h1, origin_v1, origin_d1)), 'haar')
        for ii in range(origin_size):
            for jj in range(origin_size):
                utility_pix += pow(img1[ii][jj] - img[ii][jj], 2)
        psnr += compute_psnr(img1, img)
        cv2.imwrite('../img/add_noise_reverse_haar_wavelet/{}/origin/{}/{}_noise_{}.png'.format(name, b, name, itr),
                    img1)
        imageA = cv2.imread('../img/{}_gray.png'.format(name))
        imageB = cv2.imread('../img/add_noise_reverse_haar_wavelet/{}/origin/{}/{}_noise_{}.png'.format(name, b, name, itr))
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
                             img_path=f'\\..\\img\\add_noise_reverse_haar_wavelet\\{name}\\origin\\{b}\\{name}_noise_{k}.png')
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
        x4.append([real_epsilon, b, psnr, ssim])
        y4.append(utility_b)
        y4b.append(utility_pix)


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


b_begin = 95
# haar wavelet
for kk in range(b_begin, b_begin + 1):
    if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/origin/{}".format(name, kk / 10)):
        os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/origin/{}".format(name, kk / 10))
    origin(kk / 10)

# save data
np.savetxt('./data/{}/origin/x4_{}.txt'.format(name, b_begin), x4)
np.savetxt('./data/{}/origin/y4_{}.txt'.format(name, b_begin), y4)
np.savetxt('./data/{}/origin/y4b_{}.txt'.format(name, b_begin), y4b)