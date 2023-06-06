import numpy as np
import os
import time
import torch
import argparse
import cv2
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision
from scipy.ndimage import gaussian_filter1d

from config import update_config
from Dataset import Label_loader
from utils import psnr_error
import Dataset
from models.unet import UNet, d_netclassifier
import wandb

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--seq', default=1, type=int, help='seq to evaluate.')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--show_heatmap', action='store_true',
                    help='Show and save the difference heatmap real-timely, this drops fps.')
parser.add_argument('--wandb', action='store_true',
                    help='Use wandb to log the psnr curve and heatmap.')


def val(cfg, model=None, model_abn=None, model_classifier = None, flow_loss=0.6):
    if model:  # This is for testing during training.
        generator = model
        generator_abn = model_abn
        classifier = model_classifier
        generator.eval()
        generator_abn.eval()
        classifier.eval()
    else:
        generator = UNet(input_channels=12, output_channel=3).cuda().eval()
        generator_abn = UNet(input_channels=12, output_channel=3).cuda().eval()
        classifier = d_netclassifier().cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g_norm'])
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g_abn'])
        classifier.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_c'])
        print(f'The pre-trained generator and classifier has been loaded from \'weights/{cfg.trained_model}\'.\n')

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    print(video_folders)
    
    fps = 0
    psnr_group = []
    classification_group = []

    log_dir = os.path.join('./evals', args.dataset, 'classifier-tests', 'flow_loss'+str(flow_loss)) # depending on dataset and test change accordingly.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not model:
        if cfg.show_curve:
            fig = plt.figure("Image")
            manager = plt.get_current_fig_manager()
            #manager.window.setGeometry(550, 200, 600, 500)
            # This works for QT backend, for other backends, check this ⬃⬃⬃.
            # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
            plt.xlabel('frames')
            plt.ylabel('psnr')
            plt.title('psnr curve')
            plt.grid(ls='--')

            cv2.namedWindow('target frames', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('target frames', 384, 384)
            cv2.moveWindow("target frames", 100, 100)

        if cfg.show_heatmap:
            cv2.namedWindow('difference map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('difference map', 384, 384)
            cv2.moveWindow('difference map', 100, 550)
    start = time.time()
    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset = Dataset.test_dataset(cfg, folder)

            if not model:
                name = folder.split('/')[-1]
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

                if cfg.show_curve:
                    video_writer = cv2.VideoWriter(f'results/{name}_video.avi', fourcc, 30, cfg.img_size)
                    curve_writer = cv2.VideoWriter(f'results/{name}_curve.avi', fourcc, 30, (600, 430))

                    js = []
                    plt.clf()
                    ax = plt.axes(xlim=(0, len(dataset)), ylim=(30, 45))
                    line, = ax.plot([], [], '-b')

                if cfg.show_heatmap:
                    heatmap_writer = cv2.VideoWriter(f'results/{name}_heatmap.avi', fourcc, 30, cfg.img_size)

            psnrs = []
            classification = []
            
            for j, clip in enumerate(dataset):
                input_np = clip[0:12, :, :]
                target_np = clip[12:15, :, :]
                input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()
               
                G_frame = generator(input_frames)
                class_frame = classifier(target_frame)
                
                if j % 500 == 0:
                    name = folder.split('/')[-1]
                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]
                    vutils.save_image(save_target, os.path.join(log_dir, '%s_%04d_real_sample.png' % (name, j)), normalize=True)
                    vutils.save_image(save_G_frame, os.path.join(log_dir, '%s_%04d_predicted_frame.png' % (name, j)), normalize=True)
                test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))
                # classification.append(class_frame[0][1].cpu().detach().numpy()) # Append normal score.
                classification.append(class_frame[0][0].cpu().detach().numpy()) # Append abnormal score.
                
                if not model:
                    if cfg.show_curve:
                        cv2_frame = ((target_np + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                        js.append(j)
                        line.set_xdata(js)  # This keeps the existing figure and updates the X-axis and Y-axis data,
                        line.set_ydata(psnrs)  # which is faster, but still not perfect.
                        plt.pause(0.001)  # show curve

                        cv2.imshow('target frames', cv2_frame)
                        cv2.waitKey(1)  # show video

                        video_writer.write(cv2_frame)  # Write original video frames.

                        buffer = io.BytesIO()  # Write curve frames from buffer.
                        fig.canvas.print_png(buffer)
                        buffer.write(buffer.getvalue())
                        curve_img = np.array(Image.open(buffer))[..., (2, 1, 0)]
                        curve_writer.write(curve_img)

                    if cfg.show_heatmap:
                        diff_map = torch.sum(torch.abs(G_frame - target_frame).squeeze(), 0)
                        diff_map -= diff_map.min()  # Normalize to 0 ~ 255.
                        diff_map /= diff_map.max()
                        diff_map *= 255
                        diff_map = diff_map.cpu().detach().numpy().astype('uint8')
                        heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

                        cv2.imshow('difference map', heat_map)
                        cv2.waitKey(1)

                        heatmap_writer.write(heat_map)  # Write heatmap frames.
                        


                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                print(f'\r Classification: {class_frame[0][0].cpu().detach().numpy():.4f} psnr: {test_psnr:.3f}.Detecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps.', end='')
            psnr_group.append(np.array(psnrs))
            classification_group.append(np.array(classification))
            # print(classification_group)
            
            if not model:
                if cfg.show_curve:
                    video_writer.release()
                    curve_writer.release()
                if cfg.show_heatmap:
                    heatmap_writer.release()
    end_total = time.time()
    elapsed = (end_total - start)/60 # in minutes
    print(f'\nTotal frame detection time: {elapsed:.2f} minutes.')
    print('\nAll frames were detected, begin to compute AUC.')

    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    comb_scores = np.array([], dtype=np.float32)
    class_scores = np.array([], dtype=np.float32)
    video_name = 1
    for i in range(len(psnr_group)):
        # combine psnr and classification score, by adding them together.
        distance = psnr_group[i][1:]
        classification_distance = classification_group[i][1:]
 
        
        distance -= min(distance)  # distance = (distance - min) / (max - min)
        distance /= max(distance)
        distance_smooth = gaussian_filter1d(distance, sigma=3)  # Smooth the curve.
                
        classification_distance -= min(classification_distance)
        classification_distance /= max(classification_distance)
        classification_smooth = gaussian_filter1d(classification_distance, sigma=3) # Smooth the curve.
        
        comb_distance = distance + classification_distance
        comb_distance -= min(comb_distance)
        comb_distance /= max(comb_distance)
        comb_distance_smooth = gaussian_filter1d(comb_distance, sigma=3) # Smooth the curve.
        
        axis = np.arange(0, distance.size)
        frames_aplt = gt[i][4:]
        
        plt.plot(axis, distance_smooth, 'r', label='psnr-score', linestyle='dashdot', zorder=1)
        plt.plot(axis, classification_smooth, 'b', label='class-score', linestyle='--', zorder=1)
        plt.plot(axis, comb_distance_smooth, 'g', label='Pipeline', zorder=0)
        plt.bar(axis, frames_aplt,  linewidth=2, width=1.0, color='red',label='ground-truth', alpha=0.3, zorder=2)
        plt.xlim(0, distance.size)
        plt.ylim(0, 1)
        
        video_len = distance.size
           
        plt.xlabel('Frame. Number of frames: ' + str(video_len) + '.')
        plt.ylabel('Anomaly score')
        plt.title('Anomaly score of each frame')
            
        plt.title('Anomaly score of each frame', x=0.5, y=1.08, ha='center', fontsize='large')  
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode='expand', borderaxespad=0, ncol=4)

        fig = plt.gcf()
        vid_name = log_dir + '/video_' + str(video_name) + '_curve.png'
        fig.savefig(vid_name)
        
        scores = np.concatenate((scores, distance), axis=0)
        comb_scores = np.concatenate((comb_scores, comb_distance), axis=0)
        class_scores = np.concatenate((class_scores, classification_distance), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        video_name += 1
        plt.clf()
          
    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'
        
    assert comb_scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {comb_scores.shape[0]} detected frames.'

    frames_a = np.where(labels == 1)
    abnormal_frames = []
    abnormal_scores = []
    # new_abnormal_scores = []
    for i in range(frames_a[0].shape[0]):
        abnormal_frames.append(frames_a[0][i])
        abnormal_scores.append(scores[frames_a[0][i]])
        
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0) # Psnr scores.
    fpr_class, tpr_class, thresholds_class = metrics.roc_curve(labels, class_scores, pos_label=0) # Classification scores.
    comb_fpr, comb_tpr, comb_thresholds = metrics.roc_curve(labels, comb_scores, pos_label=0)

    auc = metrics.auc(fpr, tpr) # Psnr auc.
    auc_class = metrics.auc(fpr_class, tpr_class) # Classification auc.
    comb_auc = metrics.auc(comb_fpr, comb_tpr) # Combined auc.
    
    # _, predicted = torch.max(class_scores, 1) 
    # total_samples += labels.size(0) 
    # total_correct += (predicted == labels).sum().item()

    print(f'AUC (Future Frame, psnr): {auc*100:.4f}%\n')
    print(f'AUC (Proposed Pipeline): {comb_auc*100:.4f}%\n')
    print(f'AUC (Classification): {auc_class*100:.4f}%\n')

    return auc, comb_auc #, output_class


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
    # Uncomment this to test the AUC mechanism.
    # labels = [0,  0,   0,   0,   0,  1,   1,    1,   0,  1,   0,    0]
    # scores = [0, 1/8, 2/8, 1/8, 1/8, 3/8, 6/8, 7/8, 5/8, 8/8, 2/8, 1/8]
    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # print(fpr)
    # print('~~~~~~~~~~~~`')
    # print(tpr)
    # print('~~~~~~~~~~~~`')
    # print(thresholds)
    # print('~~~~~~~~~~~~`')
