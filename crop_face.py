import numpy as np
import cv2
import os
from pic_name import pic_name


def crop_face(input_folder_path, output_folder_path):
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    images = os.listdir(input_folder_path)
    for image in images:
        image_path = os.path.join(input_folder_path, image)
        img = cv2.imread(image_path)
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30))

        # 无法识别面部的图片
        if len(faces) == 0:
            print(f"No face found in {image_path}")
            return

        for (x, y, w, h) in faces:
            cropped_img = img[y:y + h, x:x + w]
            # 调整图像大小为512x512
            resized = cv2.resize(cropped_img, (96, 96), interpolation=cv2.INTER_AREA)
            # 将图像保存到输出目录
            output_path = os.path.join(output_folder_path, image)
            cv2.imwrite(output_path, resized)


if __name__ == "__main__":
    input_folder = "../img/{}/".format(pic_name)
    output_folder = "../img/{}_crop_face/".format(pic_name)
    # 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    crop_face(input_folder, output_folder)
    print('Done!')