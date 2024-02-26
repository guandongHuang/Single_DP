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


name = pic_name

img = cv2.imread('../img/{}_gray.png'.format(name), 0).astype(np.float64)
origin_c1, (origin_h1, origin_v1, origin_d1) = dwt2(img, 'haar')
origin_c2, (origin_h2, origin_v2, origin_d2) = dwt2(origin_c1, 'haar')
origin_c3, (origin_h3, origin_v3, origin_d3) = dwt2(origin_c2, 'haar')

cv2.imwrite('../img/{}_c1.png'.format(name), origin_c1)
cv2.imwrite('../img/{}_h1.png'.format(name), origin_h1)
cv2.imwrite('../img/{}_v1.png'.format(name), origin_v1)
cv2.imwrite('../img/{}_d1.png'.format(name), origin_d1)
# hwt1 = cv2.imread('../img/{}.png'.format(name), 0).astype(np.float64)
# for i in range(0, 48):
#     for j in range(0, 48):
#         hwt1[i][j] = origin_c1[i][j]
#         hwt1[i][j + 48] = origin_h1[i][j]
#         hwt1[i + 48][j] = origin_v1[i][j]
#         hwt1[i + 48][j + 48] = origin_d1[i][j]
# cv2.imwrite('../img/{}1.png'.format(name), hwt1)

cv2.imwrite('../img/{}_c2.png'.format(name), origin_c2)
cv2.imwrite('../img/{}_h2.png'.format(name), origin_h2)
cv2.imwrite('../img/{}_v2.png'.format(name), origin_v2)
cv2.imwrite('../img/{}_d2.png'.format(name), origin_d2)
# hwt2 = cv2.imread('../img/{}_c1.png'.format(name), 0).astype(np.float64)
# for i in range(0, 24):
#     for j in range(0, 24):
#         hwt2[i][j] = origin_c2[i][j]
#         hwt2[i][j + 24] = origin_h2[i][j]
#         hwt2[i + 24][j] = origin_v2[i][j]
#         hwt2[i + 24][j + 24] = origin_d2[i][j]
# cv2.imwrite('../img/{}2.png'.format(name), hwt2)

cv2.imwrite('../img/{}_c3.png'.format(name), origin_c3)
cv2.imwrite('../img/{}_h3.png'.format(name), origin_h3)
cv2.imwrite('../img/{}_v3.png'.format(name), origin_v3)
cv2.imwrite('../img/{}_d3.png'.format(name), origin_d3)
# hwt3 = cv2.imread('../img/{}_c2.png'.format(name), 0).astype(np.float64)
# for i in range(0, 12):
#     for j in range(0, 12):
#         hwt3[i][j] = origin_c3[i][j]
#         hwt3[i][j + 12] = origin_h3[i][j]
#         hwt3[i + 12][j] = origin_v3[i][j]
#         hwt3[i + 12][j + 12] = origin_d3[i][j]
# cv2.imwrite('../img/{}3.png'.format(name), hwt3)