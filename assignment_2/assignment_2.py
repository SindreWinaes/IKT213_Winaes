
import cv2
import numpy as np


def main():

    img = cv2.imread("lena-2.png", cv2.IMREAD_COLOR)

    #  1. Padding (reflection, width = 100)
    padded = padding(img, 100)
    cv2.imwrite("lena-2_padded.png", padded)

    # 2. Cropping: 80 px from left/top, 130 px from right/bottom
    h, w = img.shape[:2]
    cropped = cropping(img, 80, w - 130, 80, h - 130)
    cv2.imwrite("lena-2_cropped.png", cropped)


    # 3. Resize to 200x200
    resized = resizing(img, 200, 200)
    cv2.imwrite("lena-2_resized_200x200.png", resized)


    # 4. Manual copy into an empty array
    empty = np.zeros((h, w, 3), dtype=np.uint8)
    copied = copying(img, empty)
    cv2.imwrite("lena-2_copied.png", copied)

    # 5. Grayscale
    gray = grayscale(img)
    cv2.imwrite("lena-2_grayscale.png", gray)


    # 6. HSV
    hsv_img = HSV(img)
    cv2.imwrite("lena-2_hsv.png", hsv_img)

    # 7 Colorshift by 50
    empty_shift = np.zeros_like(img, dtype=np.uint8)
    shifted = hue_shifted(img, empty_shift, 50)
    cv2.imwrite("lena-2_shifted_50.png", shifted)

    # 8 Smoothing
    smooth = smoothing(img)
    cv2.imwrite("lena-2_smooth.png", smooth)

    # 9 Rotation:  90 clockwise and 180
    rot90 = rotation(img, 90)
    cv2.imwrite("lena-2_rot90.png", rot90)
    rot180 = rotation(img, 180)
    cv2.imwrite("lena-2_rot180.png", rot180)






def padding(image, border_width):
    return cv2.copyMakeBorder(
        image,
        top = border_width,
        bottom = border_width,
        left = border_width,
        right = border_width,
        borderType=cv2.BORDER_REFLECT_101
    )


def cropping(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1].copy()


def resizing(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def copying(image, emptyPictureArray):
    emptyPictureArray[:] = image
    return emptyPictureArray

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hue_shifted(image, emptyPictureArray, hue):
    emptyPictureArray[:] = (image.astype(np.uint16) + hue) % 256
    return emptyPictureArray.astype(np.uint8)


def HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        return image


if __name__ == "__main__":
    main()