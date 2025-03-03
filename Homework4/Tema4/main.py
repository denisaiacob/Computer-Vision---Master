import cv2
import numpy as np
import pytesseract


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    image = cv2.medianBlur(image, 5)

    # thresholding
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # # dilation
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)

    # # erosion
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)

    # # opening - erosion followed by dilation
    # kernel = np.ones((5, 5), np.uint8)
    # image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # # canny edge detection
    # image = cv2.Canny(image, 100, 200)

    # skew correction
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image


def add_noise1(image):
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    return cv2.add(image, gauss)


def add_noise2(image):
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    return image + image * gauss


def rotate_image(image, angle, scale):
    height, width = image.shape[:2]
    centerX, centerY = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D((centerX, centerY), angle, scale)
    return cv2.warpAffine(image, matrix, (width, height))


def horizontal_shear(image, shear_horizontal):
    rows, cols, _ = image.shape
    M_horizontal = np.array([
        [1, shear_horizontal, 0],
        [0, 1, 0]
    ], dtype=float)
    return cv2.warpAffine(image, M_horizontal, (cols, rows))


def vertical_shear(image, shear_vertical):
    rows, cols, _ = image.shape
    M_vertical = np.array([
        [1, 0, 0],
        [shear_vertical, 1, 0]
    ], dtype=float)
    return cv2.warpAffine(image, M_vertical, (cols, rows))


def resize_image_down(image, scale_down):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_down)
    new_height = int(original_height * scale_down)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def resize_image_up(image, scale_up):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_up)
    new_height = int(original_height * scale_up)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def resize_image_ratio(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def average_blur(image, size):
    return cv2.blur(image, (size, size))


def gaussian_blur(image, size, deviation):
    return cv2.GaussianBlur(image, (size, size), deviation)


def applying_transformations(image):
    # image = add_noise1(image)
    image = rotate_image(image, 45, 0.5)
    # image = horizontal_shear(image, 0.5)
    # image = resize_image_up(image, 2)
    # image = gaussian_blur(image, 3, 1)
    # cv2.imshow('transformed image', image)
    # cv2.waitKey(0)
    return image


def check_result(image,ground_truth):
    image=applying_transformations(image)
    image=preprocess_image(image)
    text= pytesseract.image_to_string(image)

    correct = 0
    wrong = 0
    max_length = max(len(text), len(ground_truth))
    padded_text = text.ljust(max_length)
    padded_ground_truth = ground_truth.ljust(max_length)

    for char1, char2 in zip(padded_text, padded_ground_truth):
        if char1 == char2:
            correct += 1
        else:
            wrong += 1

    print("Number of correct characters: ", correct)
    print("Number of wrong characters: ", wrong)



if __name__ == '__main__':
    img1 = cv2.imread('example_02.jpg')
    img2 = cv2.imread('ocr-test.png')
    img3 = cv2.imread('sample21.jpg')

    img_preprocessed = preprocess_image(img1)
    ground_truth_ex02 = "Tesseract Will Fail With Noisy Backgrounds"
    # print(pytesseract.image_to_string(img_preprocessed))

    img_preprocessed = preprocess_image(img2)
    with open('ocr_test.txt', 'r', encoding='utf-8') as file:
        ground_truth_ocr_test = ''.join(line for line in file)
    # print(pytesseract.image_to_string(img_preprocessed))

    img_preprocessed = preprocess_image(img3)
    with open('sample21.txt', 'r', encoding='utf-8') as file:
        ground_truth_sample = ''.join(line for line in file)
    # print(pytesseract.image_to_string(img_preprocessed))

    # print(ground_truth_ex02)
    # print(ground_truth_ocr_test)
    # print(ground_truth_sample)

    check_result(img1,ground_truth_ex02)

    # cv2.imshow("Image", img1)
    # cv2.waitKey()

