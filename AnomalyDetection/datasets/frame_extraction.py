import cv2
import os
import argparse

def FrameCapture(input_path, output_path):
    vidObj = cv2.VideoCapture(input_path)
    video_length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(f'Number of frames: {video_length}')

    # Counter variable
    count = 0
    print("Converting Video...\n")    
    # Checks if frame was extracted successfully
    
    while vidObj.isOpened():
        succes, image = vidObj.read()
        if not succes:
            continue
        filename = output_path+"{}.png".format(count + 1)
        cv2.imwrite(filename, image)
        count = count + 1
        if (count > (video_length-1)):
            vidObj.release()
            print("Done extracting frames. \n%d frames extracted" % count)
            
if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input_path", help="path to video")
    a.add_argument("--output_path", help="path to images")
    args = a.parse_args()
    print(args)
    FrameCapture(args.input_path, args.output_path)
    