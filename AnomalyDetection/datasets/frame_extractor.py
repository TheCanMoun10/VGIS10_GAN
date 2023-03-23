import cv2
import os
from pathlib import Path
import argparse
import glob

def FrameCapture(input_path, output_path):
    folder_name = Path(input_path).stem
    folder_path = os.path.join(output_path, folder_name)
    os.mkdir(folder_path)
    print(f"Created {folder_name} directory in {output_path} ")
    

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
        filename = folder_path + "/{}.jpg".format(count + 1)
        cv2.imwrite(filename, image)
        count = count + 1
        if (count > (video_length-1)):
            vidObj.release()
            print("Done extracting frames. \n%d frames extracted" % count)
            
if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--video_path", default='./datasets/videos/training/04.avi', help="path to video")
    a.add_argument("--frames_path", default='./datasets/avenue/training/frames', help="path to frames folder")
    args = a.parse_args()
    print(args)
    FrameCapture(args.video_path, args.frames_path)