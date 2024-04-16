import cv2
from PIL import Image
from pic_name import pic_name


name = pic_name
color_img = cv2.imread("../img/{}_crop_face/{}.png".format(name, name))
gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
gray = Image.fromarray(gray_img)
gray.save("../img/{}_gray.png".format(name))