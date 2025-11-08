
import cv2
import numpy as np



def sobel_edge_detection(image):

    # Reduce noise with Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply Sobel filter in both x and y directions
    sobelxy = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)

    # Convert from 64-bit float in both x and y directions
    sobelxy = np.uint8(np.absolute(sobelxy))

    cv2.imwrite('sobel_edges.png', sobelxy)
    return sobelxy


def canny_edge_detection(image, threshold_1, threshhold_2):

    # Reduce noise with Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply Canny edge  detection
    edges = cv2.Canny(blurred, threshold_1, threshhold_2)


    cv2.imwrite('canny_edges.png', edges)
    return edges

# Find all occurrences of template in an image
def template_match(image, template):

    # Convert both images to grayscale for template matching
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Get template dimmensions
    w, h = template_gray.shape[::-1]

    # Perform template matching
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Set match quality threshold
    threshold = 0.9

    # Find all locations where match quality exceeds threshold
    loc = np.where(res >= threshold)

    result_image = image.copy()

    # Draw red rectangle around each match
    for pt in zip(*loc[::-1]):
        cv2.rectangle(result_image,
                      pt,
                      (pt[0] + w, pt[1] + h),
                      (0, 0, 255),
                      2
        )

    cv2.imwrite('template_match_result.png', result_image)
    return result_image


def resize(image, scale_factor, up_or_down):

    result = image.copy()

    # Apply pyramid operation scale_factor
    for i in range(scale_factor):
        if up_or_down == "up":
            result = cv2.pyrUp(result,
                               dstsize=(result.shape[1] * 2,
                                        result.shape[0] * 2)
            )
        elif up_or_down == "down":
            # Downsample: halve the size
            result = cv2.pyrDown(result,
                                 dstsize=(result.shape[1] // 2,
                                          result.shape[0] // 2)
            )


    cv2.imwrite(f'resized_{up_or_down}_{scale_factor}x.png', result)
    return result

def main():

    # Load lamborghini image
    img = cv2.imread('lambo.png')

    # Task 1: Sobel edge detection
    sobel_edge_detection(img)

    # Task 2:  Canny edge detection  with threshold
    canny_edge_detection(img, 50, 50)


    # Task 3: Template mathcing with shapes
    shapes_img = cv2.imread('shapes-1.png')
    template_img = cv2.imread('shapes_template.jpg')
    template_match(shapes_img, template_img)

    resize(img, 2, "up")
    resize(img, 2, "down")

if __name__ == "__main__":
    main()