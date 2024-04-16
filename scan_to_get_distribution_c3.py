import cv2
import numpy as np
import face_rec as fc
from matplotlib import pyplot as plt
from pywt import dwt2, idwt2
import pywt
import argparse
from pic_name import pic_name
import os


origin_size = 96
name = pic_name

if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}".format(name))
    os.mkdir("../distribution/{}".format(name))
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/c3".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/c3".format(name))
    os.mkdir("../distribution/{}/c3".format(name))

for i in range(origin_size // 8):
    for j in range(origin_size // 8):
        # 读取灰度图
        img = cv2.imread('../img/{}_gray.png'.format(name), 0).astype(np.float64)
        # 对img进行haar小波变换：
        origin_c1, (origin_h1, origin_v1, origin_d1) = dwt2(img, 'haar')
        origin_c2, (origin_h2, origin_v2, origin_d2) = dwt2(origin_c1, 'haar')
        origin_c3, (origin_h3, origin_v3, origin_d3) = dwt2(origin_c2, 'haar')
        # 筛选掉像素值为0的像素点
        if origin_c3[i][j] > 0:
            face_rec = fc.face_recognition()   # 创建对象

            sample = [[0 for _ in range(201)] for _ in range(128)]
            mean = []

            for k in range(201):
                origin_c3_, (origin_h3_, origin_v3_, origin_d3_) = dwt2(origin_c2, 'haar')
                noise_c3_ = 0.1 * k
                origin_c3_[i][j] += noise_c3_

                # reverse haar wavelet
                origin_c2_ = pywt.idwt2((origin_c3_, (origin_h3_, origin_v3_, origin_d3_)), 'haar')
                origin_c1_ = pywt.idwt2((origin_c2_, (origin_h2, origin_v2, origin_d2)), 'haar')
                img1 = pywt.idwt2((origin_c1_, (origin_h1, origin_v1, origin_d1)), 'haar')
                cv2.imwrite('../img/add_noise_reverse_haar_wavelet/{}/c3/{}_reverse_{}_{}.png'.format(name, name, i, j), img1)

                face_rec.inputPerson(name='musk_face_reverse',
                                     img_path=f'\\..\\img\\add_noise_reverse_haar_wavelet\\{name}\\c3\\{name}_reverse_{i}_{j}.png')
                vector = face_rec.create128DVectorSpace()
                list_vector = list(vector)
                for itr in range(128):
                    sample[itr][k] = list_vector[itr]

            mean = []
            for k in range(128):
                mean.append(np.mean(sample[k]))
            for k in range(128):
                for m in range(201):
                    sample[k][m] -= mean[k]

            for element in range(128):
                np.savetxt('../distribution/{}/c3/sample_minus_mean_{}_{}_{}.txt'.format(name, element, i, j), sample[element])
                plt.cla()
                plt.plot(sample[element], 'bo-', markersize=2)
                plt.xlabel("number")
                plt.ylabel("value")
                plt.savefig("../distribution/{}/c3/{}_{}_{}_{}.png".format(name, element, i, j, mean[element]))