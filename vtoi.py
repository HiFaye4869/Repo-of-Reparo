# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:03:11 2023

@author: Christine
"""

'''
https://devpress.csdn.net/aitech/6465bef577f0ca41cb317522.html?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7Eactivity-3-130009301-blog-121722837.235%5Ev38%5Epc_relevant_anti_t3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7Eactivity-3-130009301-blog-121722837.235%5Ev38%5Epc_relevant_anti_t3&utm_relevant_index=6
'''

import cv2
import os
import argparse 

# saving = False

saving=True
frame_id = 0
dir1 = ""

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, help="path to the video", default='./data.mp4')
args = parser.parse_args()

cap = cv2.VideoCapture(args.path)
dir1 = args.path
f_name = dir1.split('.')[0]   #name of the input video: eg. [file_name].mp4
os.makedirs(f_name, exist_ok=True)
saving = True
frame_id = 0
while True:

    # try:
    _, im = cap.read()
    if im is None:
        break

    cv2.imshow('name', im)
    key = cv2.waitKey(10) & 0xFF
    if saving:
        file_name = os.path.join(f_name, str(frame_id))
        cv2.imwrite(file_name + ".jpg", im)

        frame_id += 1
    if (key == ord('q')) | (key == 27):
        break


cv2.destroyAllWindows()
cap.release()
