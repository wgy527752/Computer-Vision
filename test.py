'''
Python script for CMPE264 Project
Team member: Guangyu Wang, Can Gao
Description: Using openCV package opencv3 to run the hdr algorithm
			 We use the Debevec algorithm and Robertson algorithm 
		     which we got the idea from the online source: 
		     http://docs.opencv.org/3.1.0/d2/df0/tutorial_py_hdr.html#gsc.tab=0 and 
		     http://docs.opencv.org/3.0-rc1/d3/db7/tutorial_hdr_imaging.html
			 And we understand the Debevec algorithm and Reinhard tone mapping alogrithm 
			 from their paper:
			 https://www.cs.utah.edu/~reinhard/cdrom/tonemap.pdf and 
			 http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
			 And we use some online pictures for testing and comparasion, you can find them from:
			 http://pages.cs.wisc.edu/~csverma/CS766_09/HDRI/hdr.html
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the images and exposure time into list
img_name = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg"]
#put the image into list
img_list = [cv2.imread(im) for im in img_name]
#save the exposure time into array
exposure_times = np.array([2, 1, 0.5, 0.25, 0.1667, 0.1, 0.0667, 0.0333, 0.1667], dtype=np.float32)


# Merge images based on Debevec Algorithm
debe1 = cv2.createMergeDebevec()
debe_hdr = debe1.process(img_list, times=exposure_times.copy())
#merge_robertson = cv2.createMergeRobertson()
#hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Do tone mapping based on Reinhard algorithm and do the gamma correction
#tune the gamma value
tonemap1 = cv2.createTonemapDurand(gamma=2.2)
res_debvec1 = tonemap1.process(debe_hdr.copy())
#tonemap2 = cv2.createTonemapReinhard(gamma=2.2)
#res_debvec2 = tonemap2.process(debe_hdr.copy())
#tonemap2 = cv2.createTonemapDurand(gamma=1.3)
#res_robertson = tonemap2.process(hdr_robertson.copy())

# Convert typp to save and display
res_debvec_8bit1 = np.clip(res_debvec1*255, 0, 255).astype('uint8')
#res_debvec_8bit2 = np.clip(res_debvec2*255, 0, 255).astype('uint8')
#res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
#res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
#cv2.imwrite("ldr_debvec24_Durand35.jpg", res_debvec_8bit1)
#cv2.imwrite("ldr_debvec24_Reinhard35.jpg", res_debvec_8bit2)
#cv2.imwrite("ldr_robertson.jpg", res_robertson_8bit)
#cv2.imwrite("fusion_mertens.jpg", res_mertens_8bit)

# Get the response function
func1_deve = cv2.createCalibrateDebevec()
func2_deve = func1_deve.process(img_list, times=exposure_times)
func3_deve = debe1.process(img_list, times=exposure_times.copy(), response=func2_deve.copy())

#func1_rob = cv2.createCalibrateRobertson()
#func2_rob = func1_rob.process(img_list, times=exposure_times)
#func3_rob = merge_robertson.process(img_list, times=exposure_times.copy(), response=func2_rob.copy())

#save the results
cv2.imwrite("response.jpg", func3_deve)
