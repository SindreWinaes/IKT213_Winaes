
import cv2
import numpy as np

def main():

    ref_img = "reference_img.png"
    harris_corner_detection(ref_img)

    sift_image_alignment("align_this.jpg", "reference_img.png")



# Detect harris corners in an image
def harris_corner_detection(ref_img):

    # Read image
    img = cv2.imread(ref_img)

    # Convert to grayscale and float32
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Apply Harris Corner Dettection
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Mark corners in red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Save result
    cv2.imwrite("harris.png", img)

    return img

def sift_image_alignment(image_to_align, reference_image):

    # Read images in grayscale
    img1 = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_to_align, cv2.IMREAD_GRAYSCALE)

    # Keep color version for output
    img2_color = cv2.imread(image_to_align)

    # Create SIFT detecto and find keypoints
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN mathcer parameters
    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    mathces = flann.knnMatch(des1, des2, k=2)

    # Spply Lowe's ratio test
    good = []

    for m, n in mathces:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # Extract matched keypoint locations
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchsMask = mask.ravel().tolist()

    # Warp image to align with reference
    h, w = img1.shape
    aligned = cv2.warpPerspective(img2_color, M, (w, h))

    # Draw bounding box around detected object
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True,255, 3, cv2.LINE_AA)

    # Draw mathces
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchsMask, flags=2)
    mathces_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # save result
    cv2.imwrite("matches.png", mathces_img)
    cv2.imwrite("aligned.png", aligned)




if __name__ == "__main__":
    main()