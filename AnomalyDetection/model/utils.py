import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import torch
from PIL import Image
rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, img_norm, dtype=np.float32):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1] or [0, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    # print("lalala")
    image_decoded = cv2.imread(filename)
    image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=dtype)
    if img_norm == "dyn_norm":
        image_resized = (image_resized - image_resized.min())/(image_resized.max()-image_resized.min())
    else:
        image_resized = (image_resized / 127.5) - 1
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1, img_norm="mnad_norm", dtype=np.float32):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        self.img_norm = img_norm
        self.dtype = dtype
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width, self.img_norm, self.dtype)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
        
    def __len__(self):
        return len(self.samples)
    
    
def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        noisy_image = ins + noise
        if noisy_image.max().data > 1 or noisy_image.min().data < -1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
            if noisy_image.max().data > 1 or noisy_image.min().data < -1:
                raise Exception('input image with noise has values larger than 1 or smaller than -1')
        return noisy_image
    return ins
