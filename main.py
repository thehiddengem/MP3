from statistics import covariance
import numpy as np
import cv2
import math

max_h = 179
max_s = 255

def calculate_hs_histogram(img, bin_size, hs_hist):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hs_hist = np.zeros((math.ceil((max_h+1)/bin_size), math.ceil((max_s+1)/bin_size)))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] += 1
    return hs_hist

def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask

def calculate_matrix(h, s):
    for i in range(0,10):
        img_train1 = cv2.imread(f"./skin{i}.png")
        img_hsv = cv2.cvtColor(img_train1, cv2.COLOR_BGR2HSV) #get grey image
        
        # filtering the image
        h = img_hsv[:, :, 0]
        s = img_hsv[:, :, 1]
        
        #get vectors for the matrix
        h_vector = np.concatenate((np.asarray(h_vector), np.asarray(h).reshape(-1)))
        s_vector = np.concatenate((np.asarray(s_vector), np.asarray(s).reshape(-1)))
    #assemble matrix 
    matrix = np.matrix((h_vector, s_vector))

    return matrix



#P1
# Training
# C = cv2.imread("skin1.png")
# img_train2 = cv2.imread("skin2.png")
# img_train3 = cv2.imread("skin3.png")
# img_train4 = cv2.imread("skin4.png")
# img_train5 = cv2.imread("skin5.png")
# img_train6 = cv2.imread("skin6.png")
# img_train7 = cv2.imread("skin7.png")
# img_train8 = cv2.imread("skin8.png")
# img_train9 = cv2.imread("skin9.png")
# img_train10 = cv2.imread("skin10.png")


bin_size = 20
h=[]
s=[]
hs_hist = np.zeros((math.ceil((max_h + 1) / bin_size), math.ceil((max_s + 1) / bin_size)))
for i in range(0,10):
    img_train1 = cv2.imread(f"./skin{i}.png")
    hs_hist = calculate_hs_histogram(img_train1, bin_size, hs_hist)
hs_hist /= hs_hist.sum()

# Testing
img_test = cv2.imread("testing_image.bmp")

threshold = 0.03
mask = color_segmentation(img_test, hs_hist, bin_size, threshold)

img_seg = img_test * mask

cv2.imwrite(f".input.png", img_test)
cv2.imwrite("Mask.png", (mask*255).astype(np.uint8))
cv2.imwrite("Segmentation_HS.png", img_seg.astype(np.uint8))


#P2

matrix = calculate_matrix(h,s)
#calculate mean of the given matrix
mean = np.mean(matrix, axis=1, dtype=np.float64)
#calculate covariance of the given matrix
covariance=np.cov(matrix)
#Calculate the skin probability for each pixel in testing_image.bmp based on the
#estimated Gaussian distribution and set a threshold to perform segmentation.


#P3 Harris Corner Detector in OpenCV
img = cv2.imread("checkerboard.png")
img_hsv = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
#gray = np.float32(gray)
dst = cv2.cornerHarris(img_hsv,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite("./checkerboard_new.png",img)


img = cv2.imread("toy.png")
img_hsv = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

dst = cv2.cornerHarris(img_hsv,4,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite("./toy_new.png",img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()