import cv2
from pic_name import pic_name


name = pic_name
img = cv2.imread("../img/{}.png".format(name))
cropped = img[0:96, 0:96]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("../img/{}_cropped.png".format(name), cropped)