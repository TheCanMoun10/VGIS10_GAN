import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

log_dir = os.path.join('./evals', 'avenue', 'classifier-tests', 'flow_loss'+str(1.5))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

arr_a = [np.array([1.0, 2.2, 3.4, 4.9, 5.12]), np.array([1.1, 5.5, 5.98, 7.34, 9.56]), np.array([1.63, 2.25, 8.93, 4.89, 5.63]), np.array([1.63, 2.25, 8.93, 4.89, 5.63]) ]
arr_b = [np.array([7.7, 7.98, 8.12, 12.5, 0.0]), np.array([45.0, 23.45, 33.50, 8.0, 9.0]), np.array([0.1, 12.12, 3.89, 13.05, 11.98]), np.array([1.63, 2.25, 8.93, 4.89, 5.63])]
gt_arr = [np.array([1, 1, 0, 1, 1]), np.array([0, 1, 0, 0, 1]), np.array([0, 1, 1, 1, 0]), np.array([1, 0, 0, 1, 1])]


scores = np.array([], dtype=np.float32)
labels = np.array([], dtype=np.int8)
comb_scores = np.array([], dtype=np.float32)
class_scores = np.array([], dtype=np.float32)
video = 0
    
for i in range(len(arr_a)):
        # combine psnr and classification score, by adding them together.
        distance =arr_a[i][1:]
        classification_distance = arr_b[i][1:]
        
        comb_distance = distance + classification_distance
        comb_distance -= min(comb_distance)
        comb_distance /= max(comb_distance)
        comb_distance_smooth = gaussian_filter1d(comb_distance, sigma=3) # Smooth the curve.
        
        distance -= min(distance)  # distance = (distance - min) / (max - min)
        distance /= max(distance)
        distance_smooth = gaussian_filter1d(distance, sigma=3)
        
        classification_distance -= min(classification_distance)
        classification_distance /= max(classification_distance)
        classification_smooth = gaussian_filter1d(classification_distance, sigma=3)
        
        frames_a = gt_arr[i][1:]
        
        axis = np.arange(0, distance.size)
        
        plt.plot(axis, distance_smooth, 'r', label='psnr-score', linestyle='dashdot', zorder=1)
        plt.plot(axis, classification_smooth, 'b', label='class-score', linestyle='--', zorder=1)
        plt.plot(axis, comb_distance_smooth, 'g', label='comb-score', linewidth=2, zorder=0)
        plt.bar(axis, frames_a,  linewidth=1, width=1.0, color='red',label='ground-truth', alpha=0.3, zorder=2)
        plt.xlim(0, distance.size)
        plt.ylim(0, 1)
        
        video_len = distance.size
           
        plt.xlabel('Frame. Number of frames: ' + str(video_len) + '.')
        plt.ylabel('Anomaly score')
        plt.title('Anomaly score of each frame', x=0.5, y=1.08, ha='center', fontsize='large')
            
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode='expand', borderaxespad=0, ncol=4)

        fig = plt.gcf()
        vid_name = log_dir + '/test_video_' + str(video) + 'curve.png'
        fig.savefig(vid_name)
        plt.show()
                
        #comb_distance = (distance + classification_distance)/2
        
        scores = np.concatenate((scores, distance), axis=0)
        # comb_scores = np.concatenate((comb_scores, comb_distance), axis=0)
        class_scores = np.concatenate((class_scores, classification_distance), axis=0)
        video += 1
        plt.clf()
        # labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    # norm_scores = np.array([], dtype=np.float32)
    # abnorm_scores = np.array([], dtype=np.float32)
    # for i in range(len(cl_group_norm)):
