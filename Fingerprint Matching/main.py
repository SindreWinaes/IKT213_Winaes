
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def main():

    #---- Compare UiA images using both methods ----#
    img1_path = "UiA front1.png"
    img2_path = "UiA front3.jpg"
    dataset_path = "data_check"

    # ORB + BF
    good_orb, vis_orb = match_fp_orb(img1_path, img2_path, fingerprint=False)
    # SIFT + FLANN
    good_sift, vis_sift = match_fp_sift(img1_path, img2_path, fingerprint=False)

    print(f"UiA ORB+BF: good={good_orb}")
    print(f"UiA Sift+FLANN: good={good_sift}")

    # Save match visualisation photos
    if vis_orb is not None:
        cv.imwrite("uia_orb_bf.png", vis_orb)
    if vis_sift is not None:
        cv.imwrite("uia_sift_flann.png", vis_sift)

    # ---- Evaluate fingerprint dataset using both methods ----#
    eval_data_check(dataset_path, method="orb", threshold=20, save_dir="results_orb")
    eval_data_check(dataset_path, method="sift", threshold=20, save_dir="results_sift")



#---- Read an image in grayscale ----#
def load_gray(p):
    return cv.imread(p, cv.IMREAD_GRAYSCALE)


#---- Apply Otsu Thresholdign fo fingerprint preprocessing ----#
def pre_pro_fp(p):
    img = cv.imread(p, 0)
    _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img_bin



#---- SIFT + FLANN ----#
def match_fp_sift(p1, p2, fingerprint=True, ratio=0.75):

    # Preprocess or load depending on data type
    img1 = pre_pro_fp(p1) if fingerprint else load_gray(p1)
    img2 = pre_pro_fp(p2) if fingerprint else load_gray(p2)

    # Creates SIFT detector
    sift = cv.SIFT_create(nfeatures=1000)

    # Detects keyponts anf compute descriptors
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None:
        return 0, None

    # FLANN parameters for SIFT
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks= 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # KNN match and Lowes ratio test
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]

    # Creates an image with both input pictures
    match_img = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), match_img


#---- ORB + Brute-Force matching ----#
def match_fp_orb(p1, p2, fingerprint=True, ratio=0.75):

    img1 = pre_pro_fp(p1) if fingerprint else load_gray(p1)
    img2 = pre_pro_fp(p2) if fingerprint else load_gray(p2)

    # ORB detector
    orb = cv.ORB_create(nfeatures=1000)
    kp1, d1 = orb.detectAndCompute(img1, None)
    kp2, d2 = orb.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        return 0, None

    # Bruteforce matching with Hamming distance
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    # Apply ratio test
    good = [m for m, n in matches if m.distance < ratio * n.distance]

    # Creates an mage with both input pictures with lines drawn between keypoints
    vis = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), vis


#---- Evaluate Fingerprint set ----#
def eval_data_check(root, method="orb", threshold=20, save_dir="results"):

    y = []
    yhat = []

    os.makedirs(save_dir, exist_ok=True)

    for folder in sorted(os.listdir(root)):
        fpath = os.path.join(root, folder)
        if not os.path.isdir(fpath): continue

        # Collect two images per folder
        files = [os.path.join(fpath, x) for x in sorted(os.listdir(fpath))
                 if x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        if len(files) != 2:
            continue


        # Select matching method
        if method == "orb":
            good, vis = match_fp_orb(files[0], files[1], fingerprint=True, ratio=0.7)
            out = os.path.join(save_dir, f"{folder}_orb.png")
        else:
            good, vis = match_fp_sift(files[0], files[1], fingerprint=True, ratio=0.7)
            out = os.path.join(save_dir, f"{folder}_sift.png")

        # Save match visualisation
        if vis is not None:
            cv.imwrite(out, vis)

        y.append(1 if "same" in folder.lower() else 0)
        yhat.append(1 if good >= threshold else 0)

        print(f"{folder}: good={good} -> {'MATCH' if good >= threshold else 'NO MATCH'}")

    # Display Confusion Matrix
    cm = confusion_matrix(y, yhat, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=["Different (0)", "Same(1)"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({method.upper()})");
    plt.tight_layout();
    plt.show()




if __name__ == '__main__':
    main()