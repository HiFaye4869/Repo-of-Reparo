# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:17:33 2023

@author: Christine
"""

import cv2
import os

#根据自己的实际情况更改目录。
#要转换的图片的保存地址，按顺序排好，后面会一张一张按顺序读取。
convert_image_path = 'reconstruct_results1'

size = (256,256)

videoWriter = cv2.VideoWriter('reconstruct_results/testvideo2.avi',cv2.VideoWriter_fourcc('I','4','2','0'),
                              24,size)
 
path_list = os.listdir(convert_image_path)
 
#image_133_0.jpg
 
#path_list.sort(key=lambda x:int(re.split(r'image_|_0.jpg')[1]))

path_list.sort(key=lambda x:int(x.split('_0.jpg')[0]))

for img in path_list :
    img = os.path.join(convert_image_path, img)
    read_img = cv2.imread(img)
    videoWriter.write(read_img)
videoWriter.release()




#原文链接：https://blog.csdn.net/Enchanted_ZhouH/article/details/77168455