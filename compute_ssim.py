from skimage.metrics import structural_similarity as compare_ssim
import cv2
from pic_name import pic_name


name = pic_name
sample = 2000
method = 'scan'
para = '10.3'

imageA = cv2.imread('../img/{}_gray.png'.format(name))
ssim = 0
for i in range(sample):
    imageB = cv2.imread('../img/add_noise_reverse_haar_wavelet/{}/{}/{}/{}_noise_{}.png'.format(name, method, para, name, i))
    # imageB = cv2.imread('../img/add_noise_reverse_haar_wavelet/{}/{}/60000000/{}/{}_noise_{}.png'.format(name, method, para, name, i))

    # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    ssim += score

    # # 6. You can print only the score if you want
    # print("SSIM: {}".format(score))
print(ssim / sample)