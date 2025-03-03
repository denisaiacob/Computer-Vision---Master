import os

import cv2
import numpy as np

import warnings


# a. Image in which the skin pixels are white and all non-skin pixels are black
def skin_pixels_rgb_methods(image):
    row, col, _ = image.shape
    output_image = np.zeros((row, col), dtype=np.uint8)

    for i in range(row):
        for j in range(col):
            b, g, r = map(float, image[i, j])
            if r > 95 and g > 40 and b > 20 \
                    and max(r, g, b) - min(r, g, b) > 15 \
                    and abs(r - g) > 15 and r > g and r > b:
                output_image[i, j] = 255
    return output_image


def skin_pixels_hsv_methods(image):
    row, col, _ = image.shape
    output_image = np.zeros((row, col), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(row):
        for j in range(col):
            h, s, v = image[i, j]
            if 0 <= h <= 50 \
                    and 59 <= s <= 173 \
                    and 89 <= v <= 255:
                output_image[i, j] = 255
    return output_image


def skin_pixels_YCbCr_methods(image):
    row, col, _ = image.shape
    output_image = np.zeros((row, col), dtype=np.uint8)

    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            Y = 0.299 * r + 0.587 * g + 0.114 * b
            Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
            Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

            if Y > 80 and 85 < Cb < 135 < Cr < 180:
                output_image[i, j] = 255
    return output_image




def accuracy_methods():
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    directory_path_face = "Face_Dataset/Pratheepan_Dataset/FacePhoto"
    directory_path_truth_face = "Face_Dataset/Ground_Truth/GroundT_FacePhoto"
    for filename in os.listdir(directory_path_truth_face):
        filename_no_ext = filename.split('.')[0]
        face_photo_name = filename
        for facePhoto in os.listdir(directory_path_face):
            if filename_no_ext in facePhoto:
                face_photo_name = facePhoto
        image_path = os.path.join(directory_path_face, face_photo_name)
        truth_path = os.path.join(directory_path_truth_face, filename)
        image = cv2.imread(image_path)
        ground_truth_image = cv2.imread(truth_path)

        output_image = skin_pixels_rgb_methods(image)
        # output_image = skin_pixels_hsv_methods(image)
        # output_image = skin_pixels_YCbCr_methods(image)

        row, col = output_image.shape
        for i in range(row):
            for j in range(col):
                if output_image[i, j] == 255 and ground_truth_image[i, j][0] == 255:
                    true_positives += 1
                elif output_image[i, j] == 255 and ground_truth_image[i, j][0] ==0:
                    false_positives += 1
                elif output_image[i, j] == 0 and ground_truth_image[i, j][0] == 255:
                    false_negatives += 1
                elif output_image[i, j] == 0 and ground_truth_image[i, j][0] == 0:
                    true_negatives += 1

    directory_path_family = "Face_Dataset/Pratheepan_Dataset/FamilyPhoto"
    directory_path_truth_family = "Face_Dataset/Ground_Truth/GroundT_FamilyPhoto"
    for filename in os.listdir(directory_path_truth_family):
        filename_no_ext = filename.split('.')[0]
        face_photo_name = filename
        for facePhoto in os.listdir(directory_path_family):
            if filename_no_ext in facePhoto:
                face_photo_name = facePhoto
        image_path = os.path.join(directory_path_family, face_photo_name)
        truth_path = os.path.join(directory_path_truth_family, filename)
        image = cv2.imread(image_path)
        ground_truth_image = cv2.imread(truth_path)

        output_image = skin_pixels_rgb_methods(image)
        # output_image = skin_pixels_hsv_methods(image)
        # output_image = skin_pixels_YCbCr_methods(image)

        row, col = output_image.shape
        for i in range(row):
            for j in range(col):
                if output_image[i, j] == 255 and ground_truth_image[i, j][0] == 255:
                    true_positives += 1
                elif output_image[i, j] == 255 and ground_truth_image[i, j][0] == 0:
                    false_positives += 1
                elif output_image[i, j] == 0 and ground_truth_image[i, j][0] == 255:
                    false_negatives += 1
                elif output_image[i, j] == 0 and ground_truth_image[i, j][0] == 0:
                    true_negatives += 1
        print(true_positives,true_negatives,false_positives,false_negatives)
    p1 = true_positives + true_negatives
    p2 = true_positives + false_positives + false_negatives + true_negatives
    accuracy = p1 / p2
    return accuracy

def face_detection(image):
    image_copy= image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    row, col, _ = image.shape

    lower_skin = np.array([0, 59, 89], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)

    output_image = cv2.inRange(image_copy, lower_skin, upper_skin)
    cv2.imshow('Face Image', output_image)
    cv2.waitKey()

    contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        side_length = max(w, h)

        center_x = x + w // 2
        center_y = y + h // 2
        square_x = center_x - side_length // 2
        square_y = center_y - side_length // 2

        square_contour = np.array([[square_x, square_y],
                                   [square_x + side_length, square_y],
                                   [square_x + side_length, square_y + side_length],
                                   [square_x, square_y + side_length]])

        cv2.rectangle(image, (square_x, square_y), (square_x + side_length, square_y + side_length), (0, 0, 255), 2)

        return image


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL.PngImagePlugin")
    img = cv2.imread('3.jpg')
    # img = cv2.imread('4.jpg')
    # img = cv2.imread('t26.jpg')
    # img = cv2.imread('group.jpg')

    # output_image = skin_pixels_rgb_methods(img)
    # output_image = skin_pixels_hsv_methods(img)
    # output_image = skin_pixels_YCbCr_methods(img)
    # cv2.imshow('Skin Image', output_image)
    # cv2.waitKey()

    print(accuracy_methods())

    # img = cv2.imread('4.jpg')
    # img = face_detection(img)
    # cv2.imshow('Face Image', img)
    # cv2.waitKey()
    # img = cv2.imread('2.jpg')
    # img = face_detection(img)
    # cv2.imshow('Face Image', img)
    # cv2.waitKey()

    cv2.destroyAllWindows()
