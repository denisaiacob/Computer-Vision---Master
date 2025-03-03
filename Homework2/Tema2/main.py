from math import gamma

import cv2
import numpy as np


# new_image = np.zeros_like(image, dtype=np.uint8)


# Lab 2 - RGB to Grayscale Conversions
# 1. Simple averaging Gray = (R+G+B ) / 3 (better use R/3+G/3+B/3) .
def simple_averaging(image):
    row, col, _ = image.shape
    for i in range(row):
        for j in range(col):
            # image[i, j] = sum(image[i, j] / 3)
            b, g, r = image[i, j]
            gray = r // 3 + g // 3 + b // 3
            image[i, j] = (gray, gray, gray)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


# 2. Weighted average Gray= 0.3R+0.59G+0.11B or
# Gray = 0.2126R+0.7152G+0.0722B or
# Gray = 0.299R+0.587G+ 0.114B
def weighted_averaging(image):
    row, col, _ = image.shape
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            gray = (0.3 * r + 0.59 * g + 0.11 * b)
            image[i, j] = (gray, gray, gray)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


# 3. Desaturation (Gray = (min(R,G,B)+max(R,G,B)) / 2)
def desaturation(image):
    row, col, _ = image.shape
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j].astype(float)
            gray = (min(r, g, b) + max(r, g, b)) // 2
            image[i, j] = (gray, gray, gray)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


# 4. Decomposition Maximum Gray = max(R,G,B)
# Minimum Gray = min(R,G,B)
def decomposition_max(image):
    row, col, _ = image.shape
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            gray = max(r, g, b)
            image[i, j] = (gray, gray, gray)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


def decomposition_min(image):
    row, col, _ = image.shape
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            gray = min(r, g, b)
            image[i, j] = (gray, gray, gray)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


# 5. Single color channel Gray=R or Gray=G or Gray=B
def single_color_chanel(image):
    row, col, _ = image.shape
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            image[i, j] = (b, b, b)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


# 6. Custom number of grey shades(2 and 256 is accepted; 2 results in a black-and-white image, while 256 gives you an image identical)
def custom_shades(image):
    row, col, _ = image.shape
    number_shades = 60
    conversion_factor = 255 / (number_shades - 1)

    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j].astype(float)  # Convert to float to avoid overflow
            average_value = (r + g + b) / 3.0
            gray_index = int(average_value / conversion_factor + 0.5)
            gray = int(gray_index * conversion_factor)
            image[i, j] = (gray, gray, gray)

    cv2.imshow('Grayscale Image', image)
    cv2.waitKey()


# 6.2
def custom_shades2(image):
    row, col, _ = image.shape

    # weighted average
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            gray = (0.3 * r + 0.59 * g + 0.11 * b)
            image[i, j] = (gray, gray, gray)
    image = image[:, :, 0]

    p = np.random.randint(2, 255)
    # p = 60
    print(p)
    a = np.random.randint(1, 255, size=(p,))
    # a=[]
    # for i in range(p):
    #     a.append((i+1)*(255//p))

    output_image = np.zeros((row, col), dtype=np.uint8)

    for i in range(p):
        if i==0:
            lower_bound=0
        else:
            lower_bound = a[i]

        if i==p-1:
            upper_bound=255
        else:
            upper_bound = a[i+1]

        mask = (image >= lower_bound) & (image <= upper_bound)
        if np.any(mask):
            avg = int(image[mask].mean())
        else:
            avg = 0

        output_image[mask] = avg

    cv2.imshow('Grayscale Image', output_image)
    cv2.waitKey()


# 7. Custom number of grey shades with error-diffusion dithering
# Implement both the Floyd-Steinberg and the Stucki Dithering algorithms.
def floyd_steinberg_dithering(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    row, col = gray_image.shape
    output_image = np.zeros((row, col), dtype=np.uint8)

    for y in range(row):
        for x in range(col):
            old_pixel = gray_image[y, x]
            if old_pixel > 127:
                new_pixel = 255
            else:
                new_pixel = 0
            output_image[y, x] = new_pixel

            error = old_pixel - new_pixel
            if x + 1 < col:
                gray_image[y, x + 1] += error * (7 / 16)
            if x - 1 >= 0 and y + 1 < row:
                gray_image[y + 1, x - 1] += error * (3 / 16)
            if y + 1 < row:
                gray_image[y + 1, x] += error * (5 / 16)
            if x + 1 < col and y + 1 < row:
                gray_image[y + 1, x + 1] += error * (1 / 16)

    cv2.imshow('Grayscale Image', output_image)
    cv2.waitKey()


def stucki_dithering(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    row, col = gray_image.shape
    output_image = np.zeros((row, col), dtype=np.uint8)

    for y in range(row):
        for x in range(col):
            old_pixel = gray_image[y, x]
            if old_pixel > 127:
                new_pixel = 255
            else:
                new_pixel = 0
            output_image[y, x] = new_pixel

            error = old_pixel - new_pixel
            if x + 1 < col:
                gray_image[y, x + 1] += error * (8 / 42)
            if x + 2 < col:
                gray_image[y, x + 2] += error * (4 / 42)
            if y + 1 < row:
                if x - 1 >= 0:
                    gray_image[y + 1, x - 1] += error * (2 / 42)
                gray_image[y + 1, x] += error * (8 / 42)
                if x + 1 < col:
                    gray_image[y + 1, x + 1] += error * (4 / 42)
            if y + 2 < row:
                gray_image[y + 2, x] += error * (2 / 42)

    cv2.imshow('Grayscale Image', output_image)
    cv2.waitKey()


# 8. Transforms a grayscale image in a color one
def color_image(image):
    # row, col, _ = image.shape
    # for i in range(row):
    #     for j in range(col):
    #         b, g, r = image[i, j]
    #         gray = (0.3 * r + 0.59 * g + 0.11 * b)
    #         image[i, j] = (gray, gray, gray)

    image = cv2.imread('fruits.png', cv2.IMREAD_GRAYSCALE)
    row, col = image.shape
    output_image = np.zeros((row, col,3), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            g=image[i, j]
            output_image[i, j] = [g, abs(128-g), 255 - g]
            # output_image[i, j] = (g//0.11, g//0.59, g//0.3)


    cv2.imshow('Color Image', output_image)
    cv2.waitKey()


if __name__ == '__main__':
    img = cv2.imread('fruits.png')
    # simple_averaging(img)
    # weighted_averaging(img)
    # desaturation(img)
    # decomposition_max(img)
    # decomposition_min(img)
    # single_color_chanel(img)
    # custom_shades(img)
    # custom_shades2(img)
    # floyd_steinberg_dithering(img)
    # stucki_dithering(img)
    color_image(img)

    cv2.destroyAllWindows()

#     Gamma correction is an important technique in image processing that adjusts the brightness of an image.
