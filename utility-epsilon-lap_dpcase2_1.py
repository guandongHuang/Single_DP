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


def linear_pixels():
    pixels = []
    file = open('../distribution/{}/c3.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]), int(cur[1])])
    file = open('../distribution/{}/h1.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]), int(cur[1]) + 48])
    file = open('../distribution/{}/h2.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]), int(cur[1]) + 24])
    file = open('../distribution/{}/h3.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]), int(cur[1]) + 12])
    file = open('../distribution/{}/v1.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]) + 48, int(cur[1])])
    file = open('../distribution/{}/v2.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]) + 24, int(cur[1])])
    file = open('../distribution/{}/v3.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]) + 12, int(cur[1])])
    file = open('../distribution/{}/d1.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]) + 48, int(cur[1]) + 48])
    file = open('../distribution/{}/d2.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]) + 24, int(cur[1]) + 24])
    file = open('../distribution/{}/d3.txt'.format(name), encoding='utf8')
    for line in file.readlines():
        cur = line.strip().split(' ')
        pixels.append([int(cur[0]) + 12, int(cur[1]) + 12])
    return pixels


def laplace(noise_scale, size):
    return list(np.random.laplace(0, scale=noise_scale, size=size))


# 目标函数：
def func(xx, params):
    bk = []
    for i in range(len(xx)):
        bk.append(params[i])
    p = params[2 * len(xx)]
    mp = params[2 * len(xx) + 1]
    sum_obj = 0
    for i in range(len(xx)):
        sum_obj += pow(bk[i] + xx[i], 2) * (pow(1 - p, i + 1) - pow(1 - p, mp + 1))
    return sum_obj


# 约束条件，包括等式约束和不等式约束
def con(xx, params):
    bk = []
    pixels = []
    for i in range(len(xx)):
        bk.append(params[i])
    for i in range(len(xx), 2 * len(xx)):
        pixels.append(params[i])
    p = params[2 * len(xx)]
    mp = params[2 * len(xx) + 1]
    sensitivity = params[2 * len(xx) + 2]
    c = params[2 * len(xx) + 3]
    result = 0
    for kk in range(len(xx)):
        for i in range(128):
            up = 0
            down = 0
            file_weight_i = open('../weight/{}/weight{}.txt'.format(name, i), encoding='utf8')
            weight_i = []
            for line in file_weight_i.readlines():
                cur = line.strip().split(' ')
                for itr in range(len(cur)):
                    cur[itr] = float(cur[itr][:-5]) / float(pow(10, int(cur[itr][-1])))
                weight_i.append(cur)
            weight_i = np.array(weight_i)
            # (x,y)为像素点的坐标
            for kkk in range(len(xx)):
                x = pixels[kkk][0]
                y = pixels[kkk][1]
                down += pow(c * weight_i[x][y], 2) * pow(bk[kkk] + xx[kkk], 2) * (
                            pow(1 - p, kkk + 1) - pow(1 - p, mp + 1))
            down = pow(down, 1.5)
            x = pixels[kk][0]
            y = pixels[kk][1]
            up += sensitivity * pow(c * weight_i[x][y], 2) * (bk[kk] + xx[kk]) * (
                        pow(1 - p, kk + 1) - pow(1 - p, mp + 1)) * xx[kk]
            result += up / down
    return result


# 定义目标函数和约束条件的梯度
def gradient(xx, params):
    bk = []
    pixels = []
    for i in range(len(xx)):
        bk.append(params[i])
    for i in range(len(xx), 2 * len(xx)):
        pixels.append(params[i])
    p = params[2 * len(xx)]
    mp = params[2 * len(xx) + 1]
    l_gra = []
    for kk in range(len(pixels)):
        l_gra.append(2 * (bk[kk] + xx[kk]) * (pow(1 - p, kk + 1) - pow(1 - p, mp + 1)))
    return l_gra


def constraint_jac(xx, params):
    bk = []
    pixels = []
    for i in range(len(xx)):
        bk.append(params[i])
    for i in range(len(xx), 2 * len(xx)):
        pixels.append(params[i])
    p = params[2 * len(xx)]
    mp = params[2 * len(xx) + 1]
    sensitivity = params[2 * len(xx) + 2]
    c = params[2 * len(xx) + 3]
    l_cons = []
    res = 0
    for kk in range(len(pixels)):
        for i in range(128):
            file_weight_i = open('../weight/{}/weight{}.txt'.format(name, i), encoding='utf8')
            weight_i = []
            for line in file_weight_i.readlines():
                cur = line.strip().split(' ')
                for itr in range(len(cur)):
                    cur[itr] = float(cur[itr][:-5]) / float(pow(10, int(cur[itr][-1])))
                weight_i.append(cur)
            weight_i = np.array(weight_i)
            temp = 0
            for kkk in range(len(pixels)):
                x = pixels[kkk][0]
                y = pixels[kkk][1]
                temp += pow(c * weight_i[x][y], 2) * pow(bk[kkk] + xx[kkk], 2) * (pow(1 - p, kkk + 1) - pow(1 - p, mp + 1))
            x = pixels[kk][0]
            y = pixels[kk][1]
            up = sensitivity * pow(c * weight_i[x][y], 2) * (pow(1 - p, kk + 1) - pow(1 - p, mp + 1)) * (bk[kk] + 2 * xx[kk]) * temp - 3 * pow(sensitivity * pow(c * weight_i[x][y], 2) * (bk[kk] + 2 * xx[kk]) * (pow(1 - p, kk + 1) - pow(1 - p, mp + 1)), 2) * xx[kk]
            down = 0
            for kkk in range(len(pixels)):
                x = pixels[kkk][0]
                y = pixels[kkk][1]
                down += pow(c * weight_i[x][y], 2) * pow(bk[kkk] + xx[kkk], 2) * (pow(1 - p, kkk + 1) - pow(1 - p, mp + 1))
            down = pow(down, 2.5)
            res += up / down
        l_cons.append(res)
    return l_cons


def dp_case2_1(theory_epsilon, c, count, origin_pixels):
    # compute b for every pixel
    numerator = 0
    p = 0.02
    mp = 96 * 96
    sensitivity = 0.01
    if count == 0:
        pixels = linear_pixels()
        for pixel in pixels:
            for i in range(128):
                file_weight_i = open('../weight/{}/weight{}.txt'.format(name, i), encoding='utf8')
                weight_i = []
                for line in file_weight_i.readlines():
                    cur = line.strip().split(' ')
                    for itr in range(len(cur)):
                        cur[itr] = float(cur[itr][:-5]) / float(pow(10, int(cur[itr][-1])))
                    weight_i.append(cur)
                weight_i = np.array(weight_i)
                # (x,y)为像素点的坐标
                x = pixel[0]
                y = pixel[1]
                if weight_i[x][y] > 0:
                    numerator += sensitivity / (c * weight_i[x][y])
            denominator = pow((1 - p - pow(1 - p, mp + 1)) / p - mp * pow(1 - p, mp + 1), 0.5)
            pixel.append(numerator / denominator / theory_epsilon)
    else:
        pixels = origin_pixels
        origin_pixels = []
    file_weight = open('../weight/{}/weight0.txt'.format(name), encoding='utf8')
    weight = []
    for line in file_weight.readlines():
        cur = line.strip().split(' ')
        for itr in range(len(cur)):
            cur[itr] = float(cur[itr][:-5]) / float(pow(10, int(cur[itr][-1])))
        weight.append(cur)
    weight = np.array(weight)
    for pixel in pixels:
        pixel.append(weight[pixel[0]][pixel[1]])
    pixels = sorted(pixels, key=lambda pi: -pi[3])
    pixels = pixels[:50]
    for itr_time in range(1):
        delta_k = []
        #formula 3(compute delta_k)
        start_time = time.time()
        bk = []
        for pix in pixels:
            bk.append(pix[2])
        print(bk)
        x0 = np.array(bk)
        params = []
        params.extend(bk)
        params.extend(pixels)
        params.append(p)
        params.append(mp)
        params.append(sensitivity)
        params.append(c)
        bounds = Bounds([-min(bk) + 1e-4] * len(bk), [200] * len(bk))
        cons = {
            'type': 'eq',
            'fun': lambda xx: con(xx, params),
            'jac': lambda xx: constraint_jac(xx, params)
        }
        res = minimize(func, x0, args=(params), method='SLSQP', jac=gradient, constraints=[cons], bounds=bounds)
        # print(res)
        for idx in range(len(pixels)):
            delta_k.append(res.x[idx])
        for idx in range(len(bk)):
            bk[idx] += delta_k[idx]
        for idx in range(len(pixels)):
            pixels[idx][2] = bk[idx]
        end_time = time.time()
        print('time:', end_time - start_time)
    # add noise
    # pixel:[x,y,b,weight]
    utility_b = 0
    utility_pix = 0
    # origin_pixels
    origin_pixels.extend(pixels)
    psnr = 0
    ssim = 0
    accuracy = 0
    for itr in range(sample):
        k = min(geo(p, 1)[0], 100)
        pixels = origin_pixels[:k]
        for itritr in range(len(pixels)):
            utility_b += 2 * pow(pixels[itritr][2], 2)
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
            # print(pix[3])
            noise = laplace(pix[2], 1)[0]
            origin_c3[pix[0]][pix[1]] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_h3:
            noise = laplace(pix[2], 1)[0]
            origin_h3[pix[0]][pix[1] - 12] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_v3:
            noise = laplace(pix[2], 1)[0]
            origin_v3[pix[0] - 12][pix[1]] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_d3:
            noise = laplace(pix[2], 1)[0]
            origin_d3[pix[0] - 12][pix[1] - 12] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_h2:
            noise = laplace(pix[2], 1)[0]
            origin_h2[pix[0]][pix[1] - 24] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_v2:
            noise = laplace(pix[2], 1)[0]
            origin_v2[pix[0] - 24][pix[1]] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_d2:
            noise = laplace(pix[2], 1)[0]
            origin_d2[pix[0] - 24][pix[1] - 24] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_h1:
            noise = laplace(pix[2], 1)[0]
            origin_h1[pix[0]][pix[1] - 48] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_v1:
            noise = laplace(pix[2], 1)[0]
            origin_v1[pix[0] - 48][pix[1]] += noise
            # utility_b += pow(pix[2], 2)
        for pix in pixels_d1:
            noise = laplace(pix[2], 1)[0]
            origin_d1[pix[0] - 48][pix[1] - 48] += noise
            # utility_b += pow(pix[2], 2)
        origin_c2_ = pywt.idwt2((origin_c3, (origin_h3, origin_v3, origin_d3)), 'haar')
        origin_c1_ = pywt.idwt2((origin_c2_, (origin_h2, origin_v2, origin_d2)), 'haar')
        img1 = pywt.idwt2((origin_c1_, (origin_h1, origin_v1, origin_d1)), 'haar')
        for ii in range(origin_size):
            for jj in range(origin_size):
                utility_pix += pow(img[ii][jj] - img1[ii][jj], 2)
        psnr += compute_psnr(img1, img)
        cv2.imwrite('../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1/{}/{}/{}_noise_{}.png'.format(name, int(c), theory_epsilon, name, itr), img1)
        imageA = cv2.imread('../img/{}_gray.png'.format(name))
        imageB = cv2.imread('../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1/{}/{}/{}_noise_{}.png'.format(name, int(c), theory_epsilon, name, itr))
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        ssim += score
    ssim /= sample
    psnr /= sample
    utility_pix /= sample
    utility_b /= sample
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
                             img_path=f'\\..\\img\\add_noise_reverse_haar_wavelet\\{name}\\dp_case2_1\\{int(c)}\\{theory_epsilon}\\{name}_noise_{k}.png')
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
        x2_1.append([real_epsilon, theory_epsilon, psnr, ssim])
        y2_1.append(utility_b)
        y2b_1.append(utility_pix)


parser = argparse.ArgumentParser()
# jolie,6*10^6
# sorkin,6*10^7
parser.add_argument('--c', type=float, default=60000000)
args = parser.parse_args()

name = pic_name
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}".format(name))
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1".format(name)):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1".format(name))
if not os.path.exists("./data/{}".format(name)):
    os.mkdir("./data/{}".format(name))
if not os.path.exists("./data/{}/dp_case2_1".format(name)):
    os.mkdir("./data/{}/dp_case2_1".format(name))
if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1/{}".format(name, int(args.c))):
    os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1/{}".format(name, int(args.c)))

origin_size = 96
sample = 2000
x2_1 = []
y2_1 = []
y2b_1 = []
y1_parameter = 1478677.27581093
origin_pixels = []

restart = True
count = 0
b_begin = 340
while restart:
    for eps in range(b_begin, b_begin + 1):
        print("count = ", count)
        if not os.path.exists("../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1/{}/{}".format(name, int(args.c), eps / 1000)):
            os.mkdir("../img/add_noise_reverse_haar_wavelet/{}/dp_case2_1/{}/{}".format(name, int(args.c), eps / 1000))
        dp_case2_1(eps / 1000, args.c, count, origin_pixels)
        print(x2_1)
        print(x2_1[count])
        print(y1_parameter / x2_1[count][0])
        print(y2_1)
        print(y2_1[count])
        if y2_1[count] > y1_parameter / x2_1[count][0]:
            count += 1
            print('---------------------')
            break
        else:
            restart = False
            break

# # save data
np.savetxt('./data/{}/dp_case2_1/x2_1_{}.txt'.format(name, b_begin), x2_1)
np.savetxt('./data/{}/dp_case2_1/y2_1_{}.txt'.format(name, b_begin), y2_1)
np.savetxt('./data/{}/dp_case2_1/y2b_1_{}.txt'.format(name, b_begin), y2b_1)