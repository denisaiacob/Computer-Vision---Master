import cv2
import numpy as np


def create_emoticon():
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    color=(255, 204, 204)

    # Draw the head (circle)
    cv2.circle(image, (200, 200), 100, color, -1)

    # Draw the ears (triangles)
    cv2.fillPoly(image, [np.array([[180, 110], [110, 70], [130,130]], np.int32)], color)  # Left ear
    cv2.fillPoly(image, [np.array([[220, 110], [290, 70], [275, 135]], np.int32)], color)  # Right ear

    # Draw inner ears (triangles)
    cv2.fillPoly(image, [np.array([[133, 128], [120, 85], [155, 110]], np.int32)], (255, 153, 204))  # Left inner ear
    cv2.fillPoly(image, [np.array([[247, 110], [280, 85], [272, 128]], np.int32)], (255, 153, 204))  # Right inner ear

    # Draw the eyes (ovals)
    cv2.ellipse(image, (160, 170), (15, 25), 0, 0, 180, (0, 0, 0), -1)  # Left eye
    cv2.ellipse(image, (240, 170), (15, 25), 0, 0, 180, (0, 0, 0), -1)  # Right eye

    # Draw the pupils (circles)
    cv2.circle(image, (160, 170), 7, (255, 255, 255), -1)  # Left pupil
    cv2.circle(image, (240, 170), 7, (255, 255, 255), -1)  # Right pupil

    # Draw the nose (triangle)
    nose_points = np.array([[200, 200], [190, 220], [210, 220]], np.int32)
    cv2.fillPoly(image, [nose_points], (255, 102, 178))

    # Draw the mouth (arc)
    cv2.ellipse(image, (200, 230), (30, 20), 0, 0, 180, (0, 0, 0), 2)

    # Draw whiskers
    cv2.line(image, (160, 220), (120, 220), (0, 0, 0), 2)  # Left whisker
    cv2.line(image, (160, 230), (120, 230), (0, 0, 0), 2)  # Left whisker
    cv2.line(image, (240, 220), (280, 220), (0, 0, 0), 2)  # Right whisker
    cv2.line(image, (240, 230), (280, 230), (0, 0, 0), 2)  # Right whisker

    cv2.imwrite('IacobDenisaAlexandra.jpg', image)
    cv2.imshow('Emoji', image)
    cv2.waitKey(0)


def crop_image(image, x, y, width, length):
    if (y + width > image.shape[0]) or (x + length > image.shape[1]) or (x < 0) or (y < 0):
        raise ValueError("Cropping dimensions exceed image boundaries.")

    cropped_image = image[y:y + width, x:x + length]

    return cropped_image


if __name__ == '__main__':
    # create_emoticon()

    # 2. Open an image, display its size, plot/write the image
    img = cv2.imread("lena.tif", cv2.IMREAD_COLOR)

    cv2.imshow("lena", img)
    cv2.waitKey()

    img = cv2.imread('lena.tif', cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    print("height:", height)
    print("width:", width)

    cv2.imwrite('savedImage.png', img)

    # 3. Apply filters that blur / sharpen the image.Test these functions with at least 2 values for
    # the parameters.Save the image

    kernel1 = np.array([[0, -2, 0],
                        [-2, 8, -2],
                        [0, -2, 0]])

    filter1 = cv2.filter2D(img, -1, kernel1)
    cv2.imshow('Filter 1', filter1)
    cv2.waitKey()
    cv2.imwrite('filter1.png', filter1)

    kernel2 = np.ones((6, 6), np.float32) / 36
    filter2 = cv2.filter2D(img,-1, kernel2)
    cv2.imshow('Filter 2', filter2)
    cv2.waitKey()
    cv2.imwrite('filter2.png', filter2)

    img_blur = cv2.blur(img, (6, 6))
    cv2.imshow('Blur', img_blur)
    cv2.waitKey()
    cv2.imwrite('blur.png', img_blur)

    # 5. Rotate an image using different angles, clockwise and counterclockwise.
    # How can an image rotation function be implemented?
    centerX, centerY = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D((centerX, centerY), -45, 1)

    rotated = cv2.warpAffine(img, M, (width, height))
    cv2.imshow("Rotated by -45 Degrees", rotated)
    cv2.waitKey()

    # Write a function that crops a rectangular part of an image. The parameters of this function
    # are the position of the upper, left pixel in the image, where the cropping starts, the width
    # and the length of the rectangle.

    x=200
    y=150
    wid=100
    length=200

    cropped = crop_image(img, x, y, wid, length)
    # Save or display the cropped image
    cv2.imwrite('croppedImage.png', cropped)
    cv2.imshow('Cropped Image', cropped)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
