
from skimage import data, filters
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = data.camera()
plt.imshow(image, cmap='gray')
print(image.ndim)

def rgb_to_gray(img):
    if img.ndim > 2:
        img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        img_gray = img
    return img_gray

img_gray = rgb_to_gray(image)
plt.imshow(img_gray, cmap='gray')

print(img_gray.dtype)
print(img_gray.max(), img_gray.min())

img_gray = img_gray[:,:]/255

print(img_gray.dtype)
print(img_gray.max(), img_gray.min())

plt.imshow(img_gray, cmap='gray')

def my_filter(img, kernel):
    rows, cols = img.shape
    dst_img = img.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            dst_img[x, y] = (img[x-1:x+2, y-1:y+2] * kernel[:,:]).sum()

    return dst_img


gx_prewitt = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype='float64')
gy_prewitt = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype='float64')
print(gx_prewitt)
print(gy_prewitt)

gx_p_img = my_filter(img_gray, gx_prewitt)
gy_p_img = my_filter(img_gray, gy_prewitt)
print(gx_p_img.max(), gx_p_img.min())
print(gy_p_img.max(), gy_p_img.min())

gx_sk = filters.prewitt_v(img_gray)
gy_sk = filters.prewitt_h(img_gray)
print(gx_sk.max(), gx_sk.min())
print(gy_sk.max(), gy_sk.min())

f, ax = plt.subplots(2, 2, figsize=(10,10))

ax[0,0].imshow(gx_p_img, cmap='gray')
ax[0,0].set_title("Gx Prewitt")
ax[0,0].axis('off')

ax[0,1].imshow(gy_p_img, cmap='gray')
ax[0,1].set_title("Gy Prewitt")
ax[0,1].axis('off')

ax[1,0].imshow(gx_sk, cmap='gray')
ax[1,0].set_title("Gx Prewitt Skimage")
ax[1,0].axis('off')

ax[1,1].imshow(gy_sk, cmap='gray')
ax[1,1].set_title("Gy Prewitt Skimage")
ax[1,1].axis('off')

prewitt_img = np.sqrt(gx_p_img[:,:]**2 + gy_p_img[:,:]**2)

edges_p = filters.prewitt(img_gray)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(edges_p, cmap='gray')
ax[0].set_title("Prewitt Skimage")
ax[0].axis('off')

ax[1].imshow(prewitt_img, cmap='gray')
ax[1].set_title("My prewitt edge")
ax[1].axis('off')

gx_sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gy_sobel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
print(gx_sobel)
print(gy_sobel)

gx_s_img = my_filter(img_gray, gx_sobel)
gy_s_img = my_filter(img_gray, gy_sobel)
print(gx_s_img.max(), gx_s_img.min())
print(gy_s_img.max(), gy_s_img.min())

gx_s_sk = filters.sobel_v(img_gray)
gy_s_sk = filters.sobel_h(img_gray)
print(gx_s_sk.max(), gx_s_sk.min())
print(gy_s_sk.max(), gy_s_sk.min())

f, ax = plt.subplots(2, 2, figsize=(10,10))

ax[0,0].imshow(gx_s_img, cmap='gray')
ax[0,0].set_title("Gx Sobel")
ax[0,0].axis('off')

ax[0,1].imshow(gy_s_img, cmap='gray')
ax[0,1].set_title("Gy Sobel")
ax[0,1].axis('off')

ax[1,0].imshow(gx_s_sk, cmap='gray')
ax[1,0].set_title("Gx Sobel Skimage")
ax[1,0].axis('off')

ax[1,1].imshow(gy_s_sk, cmap='gray')
ax[1,1].set_title("Gy Sobel Skimage")
ax[1,1].axis('off')

sobel_img = np.sqrt(gx_s_img[:,:]**2 + gy_s_img[:,:]**2)

skimage_sobel = filters.sobel(img_gray)

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(skimage_sobel, cmap='gray')
ax[0].set_title("Sobel Skimage")
ax[0].axis('off')

ax[1].imshow(sobel_img, cmap='gray')
ax[1].set_title("Sobel edge")
ax[1].axis('off')


k_laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) # se puede usar tambien [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
laplace_img = my_filter(img_gray, k_laplace)
maxValue = laplace_img.max()
minValue = laplace_img.min()
print(maxValue, minValue)

laplace_img = (laplace_img[:,:] - minValue)/(maxValue-minValue)
print(laplace_img.max(), laplace_img.min())
print(laplace_img.dtype)

img_laplace = cv.Laplacian(image, cv.CV_8UC1, ksize=3)
print(img_laplace.max(), img_laplace.min())

f, ax = plt.subplots(1, 2, figsize=(10,10))

ax[0].imshow(laplace_img, cmap='gray')
ax[0].set_title("My Laplace operator")
ax[0].axis('off')

ax[1].imshow(img_laplace, cmap='gray')
ax[1].set_title("Laplace OpenCV")
ax[1].axis('off')
