# Grayscale conversion
import cv2
import os
import glob

origin_path = './train/walk/03/'
destin_path = './gray/train/walk/03/'

test_origin_path = './test/runbag/'
test_origin_path2 = './test/walk/'
test_destin_path = './gray/test/runbag/'
test_destin_path2 = './gray/test/walk/'

for filename in os.listdir(origin_path):
    file_path = origin_path + filename
    img = cv2.imread(file_path)
    # cv2.imshow('img', img)
    # print(img.shape)
    # cv2.waitKey(0)
    
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, tail = os.path.split(file_path) 
    new_name = os.path.splitext(tail)[0]
    final_path = destin_path + f"{new_name}.jpg"
    status = cv2.imwrite(final_path, gray_image[:,:,1])
    
    print('Image written to file-system :', status)
    
for filename in os.listdir(test_origin_path):
    file_path = test_origin_path + filename
    img = cv2.imread(file_path)
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, tail = os.path.split(file_path) 
    new_name = os.path.splitext(tail)[0]
    final_path = test_destin_path + f"{new_name}.jpg"
    status = cv2.imwrite(final_path, gray_image[:,:,1])
    
for filename in os.listdir(test_origin_path2):
    file_path = test_origin_path2 + filename
    img = cv2.imread(file_path)
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, tail = os.path.split(file_path) 
    new_name = os.path.splitext(tail)[0]
    final_path = test_destin_path2 + f"{new_name}.jpg"
    status = cv2.imwrite(final_path, gray_image[:,:,1])

    print('Image written to file-system :', status)
