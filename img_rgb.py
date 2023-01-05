
import cv2 as cv
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import sys
import time



img_a = data.cat()
plt.imshow(img_a)


canal_r = img_a[:,:,0]
canal_g = img_a[:,:,1]
canal_b = img_a[:,:,2]

f, ax = plt.subplots(2, 2, figsize=(10,6))
#f, ax = plt.subplots(2, 2)

ax[0,0].imshow(img_a)
ax[0,0].set_title('Image RGB')
ax[0,0].axis('off')

ax[0,1].imshow(canal_r, cmap='gray')
ax[0,1].set_title('Canal Rojo')
ax[0,1].axis('off')

ax[1,0].imshow(canal_g, cmap='gray')
ax[1,0].set_title('Canal Verde')
ax[1,0].axis('off')

ax[1,1].imshow(canal_b, cmap='gray')
ax[1,1].set_title('Canal Azul')
ax[1,1].axis('off')


def rgb_to_gray(img_rgb):
    img_gray = 0.2989 * img_rgb[:,:,0] + 0.5870 * img_rgb[:,:,1] + 0.1140 * img_rgb[:,:,2]
    return img_gray


# Mi funcion
img_gray = rgb_to_gray(img_a)

# funcion de opencv
img_gray2 = cv.cvtColor(img_a, cv.COLOR_RGB2GRAY)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title('My funcion Gray')
ax[0].axis('off')

ax[1].imshow(img_gray2, cmap='gray')
ax[1].set_title('Funcion OpenCV')
ax[1].axis('off')

def my_filter(img, kernel):
    rows, cols = img.shape
    k_rows, k_cols = kernel.shape
    dst_img = img.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            tmp_value = 0
            for i in range(k_rows):
                for j in range(k_cols):
                    tmp_value += kernel[i, j] * img[x+i-1, y+j-1]
            dst_img[x, y] = tmp_value

    return dst_img

def my_filter2(img, kernel):
    rows, cols = img.shape
    k_rows, k_cols = kernel.shape
    dst_img = img.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            dst_img[x, y] = (img[x-1:x+2, y-1:y+2] * kernel[:,:]).sum()

    return dst_img

image = cv.imread("data/salt-snoopy.png", 0)
if image is None:
    sys.exit("Couldn't read image...")
plt.imshow(image, cmap='gray')
print(image.shape)


kernel = np.ones((3,3))/9
# my filtro
start = time.time()
dst_img = my_filter2(image, kernel)
end = time.time()
print(end-start)
# filtro de opencv
start = time.time()
#dst_img2 = my_filter(image, kernel)
dst_img2 = cv.filter2D(image, cv.CV_8U, kernel)
end = time.time()
print(end-start)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(dst_img, cmap='gray')
ax[0].set_title('My filter')
ax[0].axis('off')

ax[1].imshow(dst_img2, cmap='gray')
ax[1].set_title('Filter OpenCV')
ax[1].axis('off')


kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
#print(kernel)
# my filtro
start = time.time()
dst_img = my_filter2(image, kernel)
end = time.time()
print(end-start)
# filtro de opencv
start = time.time()
dst_img2 = cv.GaussianBlur(image, (3,3), 1.0)
end = time.time()
print(end-start)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(dst_img, cmap='gray')
ax[0].set_title('My filter gausiano')
ax[0].axis('off')

ax[1].imshow(dst_img2, cmap='gray')
ax[1].set_title('Gausianblur OpenCV')
ax[1].axis('off')

def my_medianfilter(img, ksize):
    rows, cols = img.shape
    krows, kcols = ksize
    dst_img = img.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            arr_tmp = (img[x-1:x+2, y-1:y+2]).flatten()
            dst_img[x, y] = np.sort(arr_tmp)[len(arr_tmp)//2]

    return dst_img

# my filtro
start = time.time()
dst_img = my_medianfilter(image, (5,5))
end = time.time()
print(end-start)
# filtro de opencv
start = time.time()
dst_img2 = cv.medianBlur(image, 3)
end = time.time()
print(end-start)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(dst_img, cmap='gray')
ax[0].set_title('My filter median')
ax[0].axis('off')

ax[1].imshow(dst_img2, cmap='gray')
ax[1].set_title('MedianBlur OpenCV')
ax[1].axis('off')

def my_minfilter(img, ksize=3):
    rows, cols = img.shape
    dst_img = img.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            arr_tmp = (img[x-1:x+2, y-1:y+2]).flatten()
            dst_img[x, y] = arr_tmp.min()

    return dst_img

def my_maxfilter(img, ksize=3):
    rows, cols = img.shape
    dst_img = img.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            arr_tmp = (img[x-1:x+2, y-1:y+2]).flatten()
            dst_img[x, y] = arr_tmp.max()

    return dst_img

coffe = data.coins()
#coffe_g = cv.cvtColor(coffe, cv.COLOR_RGB2GRAY)
plt.imshow(coffe, cmap='gray')

# my filtro
start = time.time()
dst_img = my_minfilter(coffe)
end = time.time()
print(end-start)
# filtro de opencv
start = time.time()
dst_img2 = my_maxfilter(coffe)
end = time.time()
print(end-start)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(dst_img, cmap='gray')
ax[0].set_title('My filter minimum')
ax[0].axis('off')

ax[1].imshow(dst_img2, cmap='gray')
ax[1].set_title('My filter maximum')
ax[1].axis('off')

def my_threshold(img, vthreshold):
    rows, cols = img.shape
    dst_img = img.copy()
    for x in range(rows):
        for y in range(cols):
            if img[x,y] > vthreshold:
                dst_img[x, y] = 255
            else:
                dst_img[x, y] = 0

    return dst_img

threshold = 130
# my filtro
start = time.time()
dst_img = my_threshold(coffe, threshold)
end = time.time()
print(end-start)
# filtro de opencv
start = time.time()
ret, dst_img2 = cv.threshold(coffe, threshold, 255, cv.THRESH_BINARY)
end = time.time()
print(end-start)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(dst_img, cmap='gray')
ax[0].set_title('My thresholding')
ax[0].axis('off')

ax[1].imshow(dst_img2, cmap='gray')
ax[1].set_title('OpenCV thresholding')
ax[1].axis('off')
