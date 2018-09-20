import cv2
import numpy as np

def crack_length(start, end):
    cracks_length = []
    for i in range(len(start)):
        length = end[i]-start[i]
        cracks_length.append(length)
    return cracks_length

def crack_severity(crack_set,crack_frames):

    bgr = [40, 158, 16]
    thresh = 60
    green_pix_sum = 0
    total_pix = 0
    severity_c = []
    if(len(crack_frames)!= 0):
        crack_set.append(crack_frames)
    for frames_c in crack_set:
        for img in frames_c:
            minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
            maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
            total_pix += np.size(img)
            maskBGR = cv2.inRange(img,minBGR,maxBGR)
            resultBGR = cv2.bitwise_and(img, img, mask = maskBGR)
            green_pixel_count = np.size(np.where(resultBGR == True))    
            green_pix_sum += green_pixel_count
    severity_c.append(green_pix_sum/total_pix)
    return severity_c

    # bright = cv2.imread("/home/allahbaksh/semantic_segmentation_cracks/test.png")
    # cv2.imshow('result', resultBGR)
    # cv2.waitKey()

