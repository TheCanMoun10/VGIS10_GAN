import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.unet import d_netclassifier
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate import val
import torchvision.utils as vutils
import torchvision
import wandb

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset', default='shanghaitech', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--flownet', default='2sd', type=str, help='lite: LiteFlownet, 2sd: FlowNet2SD.')
parser.add_argument('--kldiv_loss', type=float, default=0.1, help='The weight of KL divergence loss.')
parser.add_argument('--wfl_loss', type=float, default=0.1, help='The weight of the flow loss.')
parser.add_argument('--dropRate', type=float, default=0.2, help='The drop rate of the dropout layer.')
parser.add_argument('--wandb', default=False, action='store_true', help='If True, use wandb to log the training process.')
# parser.add_argument('--generate_frames', default=False, action='store_true', help='If True, generate frames from the trained model.')

args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

if args.wandb:    
    wandb.init(project="VGIS10_AnomalyGeneration",
            
            config={
                "Iterations": args.iters,
                "Dataset": args.dataset,
                "Batch size" : args.batch_size,
                "Flow_loss_weight": args.wfl_loss,
                },
                name=f'FullNetwork_{args.dataset}_bs{args.batch_size}_iters{args.iters}'
            )

# if args.generate_frames:
#     log_dir_abnormal = os.path.join('./dataset', args.dataset+"_abnormals")
#     if not os.path.exists(log_dir_abnormal):
#         os.makedirs(log_dir_abnormal)

log_dir = os.path.join('./images', 'full_network', args.dataset, str(args.iters), 'flow_loss'+str(args.wfl_loss)+"_test2")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
generator = UNet(input_channels=12, output_channel=3).cuda()
discriminator = PixelDiscriminator(input_nc=3).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.d_lr)

# Setup classifier:
classifier = d_netclassifier().cuda()
classifier_loss = torch.nn.BCELoss().cuda()
optimizer_C = torch.optim.SGD(classifier.parameters(), lr=train_cfg.c_lr)

if train_cfg.resume:
    generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
    discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
    classifier.load_state_dict(torch.load(train_cfg.resume)['net_c'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
    optimizer_C.load_state_dict(torch.load(train_cfg.resume)['optimizer_c'])
    print(f'Pre-trained generator, discriminator and classifiers have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator, discriminator and classifiers are going to be trained from scratch.\n')

assert train_cfg.flownet in ('lite', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('FFP_Net/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('FFP_Net/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

# Losses:
adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()
kullback_loss = nn.KLDivLoss(reduction='batchmean').cuda()
perceptual_loss = Perceptual_Loss([2,7,12,21,30]).cuda()


train_dataset = Dataset.train_dataset(train_cfg)
# print(train_dataset)

# Remember to set drop_last=True, because we need to use 4 frames to predict one7 frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}_wflowloss{args.wfl_loss}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()
classifier = classifier.train()

total_samples = 0
total_correct = 0
total_samples_norm = 0
total_correct_norm = 0

accuracy_threshold = 0.0
accuracy_threshold_norm = 0.0

try:
    step = start_iter
    while training:
        for indice, clips, flow_strs in train_dataloader:
            input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
            target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
            input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss

            # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
            for index in indice:
                train_dataset.all_seqs[index].pop()
                if len(train_dataset.all_seqs[index]) == 0:
                    train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                    random.shuffle(train_dataset.all_seqs[index])

            G_frame = generator(input_frames)
            # Step 1: Classification for pseudo anomalies
            pseudo_class = classifier(G_frame)
            # print(pseudo_class)
            # Step 2: generate labels for pseudo anomalies --> tensor of [1 0] repeat 16 times
            pseudo_labels = torch.tensor([1, 0], dtype=torch.float32).repeat(args.batch_size, 1).cuda()
            # print(pseudo_labels)
            # Step 3: Apply BCE on tensors
                
            # Step 1: Classification for normals
            normal_class = classifier(target_frame)
            # print(normal_class)
            # Step 2: generate labels for normals --> tensor of [0 1] repeat number of batch size 16 times
            normal_labels = torch.tensor([0, 1], dtype=torch.float32).repeat(args.batch_size, 1).cuda()
            # print(normal_labels)
            # Step 3: Apply BCE on tensors

            if train_cfg.flownet == 'lite':
                gt_flow_input = torch.cat([input_last, target_frame], 1)
                pred_flow_input = torch.cat([input_last, G_frame], 1)
                # No need to train flow_net, use .detach() to cut off gradients.
                flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net).detach()
                flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
            else:
                gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
                pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

                flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()

            if train_cfg.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = train_cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                    print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

            inte_l = intensity_loss(G_frame, target_frame)
            grad_l = gradient_loss(G_frame, target_frame)
            fl_l   =  -args.wfl_loss*flow_loss(flow_gt, flow_pred)
            gan_l  = adversarial_loss(discriminator(G_frame))
            # kl_loss = -args.kldiv_loss*kullback_loss(nn.functional.log_softmax(target_frame, dim=1), nn.functional.softmax(G_frame, dim=1))
            p_loss = perceptual_loss(G_frame, target_frame)
            cl_loss_abn = torch.mean(classifier_loss(pseudo_class, pseudo_labels))
            cl_loss_n = torch.mean(classifier_loss(normal_class, normal_labels))
            # Add mean classifier loss.
            C_l_t = (cl_loss_abn + cl_loss_n)/2
            G_l_t = 1. * inte_l + 1. * grad_l + 0.05 * gan_l + 2*fl_l + p_loss + C_l_t
        
            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))
            # C_l_t = cl_loss_abn + cl_loss_n

            # Or just do .step() after all the gradients have been computed, like the following way:
            optimizer_D.zero_grad()
            D_l.backward()
            optimizer_G.zero_grad()
            optimizer_C.zero_grad()

            G_l_t.backward()
            optimizer_D.step()
            optimizer_G.step()
            optimizer_C.step()

            
            torch.cuda.synchronize()
            time_end = time.time()
            if step > start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end

            if step != start_iter:
                if step % 10 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    psnr = psnr_error(G_frame, target_frame)
                    lr_g = optimizer_G.param_groups[0]['lr']
                    lr_d = optimizer_D.param_groups[0]['lr']
                    lr_c = optimizer_C.param_groups[0]['lr']

                    # _, predicted_abn = torch.max(pseudo_class, 1) # predicted_abn = 0 if normal, 1 if abnormal
                    # print(predicted_abn.shape)
                    # total_samples += pseudo_labels.size(0)
                    # print(pseudo_labels.shape)
                    
                    # total_correct += (predicted_abn == pseudo_labels[0]).sum().item()

                    # accuracy_abn = total_correct / total_samples
                    
                    # if accuracy_abn >= accuracy_threshold:
                    #     classification_abn = "Abnormal"
                    #     accuracy_threshold = accuracy_abn
                    # else:
                    #     classification_abn = "Normal"

                    # _, predicted_norm = torch.max(normal_class, 1) # predicted_norm = 0 if abnormal, 1 if abnormal
                    
                    # total_samples_norm += normal_labels.size(0)
                    # total_correct_norm += (predicted_norm == normal_labels).sum().item()
                    
                    # accuracy_norm = total_correct_norm / total_samples_norm
                    
                    # if accuracy_norm >= accuracy_threshold_norm:
                    #     classification_norm = "Normal" # Normal
                    #     accuracy_threshold_norm = accuracy_norm
                    # else:
                    #     classification_norm = "Abnormal"
    
                    # print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | "
                    #       f"gan_l: {gan_l:.3f} | G_l_total: {G_l_t:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "
                    #       f"p_loss: {p_loss:.3f} | iter: {iter_t:.3f}s | ETA: {eta} | lr: {lr_g} / {lr_d} / {lr_c} | \n"
                    #       f"Class_abnormal: {classification_abn} | cl_loss_abn: {cl_loss_abn:.5f} ")
                    
                    print(f"[{step} / {int(train_cfg.iters)}] | inte_l: {inte_l:.4f} | grad_l: {grad_l:.4f} | fl_l: {fl_l:.4f}  | gan_l: {gan_l:.4f} | G_l_total: {G_l_t:.4f} | ")
                    print(f"p_loss: {p_loss:.4f} | cl_loss_abn: {cl_loss_abn:.4f} | cl_loss_n: {cl_loss_n:.4f} | C_l_t: {C_l_t:.4f} | psnr: {psnr:.4f} | ")
                    print(f"iter: {iter_t:.4f}s | ETA: {eta} | lr: {lr_g} / {lr_d} / {lr_c} | \n")

                    if args.wandb:
                        wandb.log({"psnr": psnr, "G_l_total": G_l_t, "D_l": D_l, "gan_l": gan_l, "fl_l": fl_l,
                                    "inte_l": inte_l, "grad_l": grad_l, "fl_l": fl_l, "p_loss": p_loss,
                                    "iter": iter_t, "lr_g": lr_g, "lr_d": lr_d, "lr_c": lr_c, "cl_loss_abn": cl_loss_abn, "cl_loss_n": cl_loss_n, "C_l_t": C_l_t})

                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]

                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)
                    writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
                    writer.add_scalar('G_loss_total/gan_loss', gan_l, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                    writer.add_scalar('G_loss_total/p_loss', p_loss, global_step=step)
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

                if step % int(train_cfg.iters / 100) == 0:
                    # if args.wandb:
                    #     wandb.log({"G_frame": [wandb.Image(save_G_frame, caption='%05d_G_frame' % (step))], "target_frame ": [wandb.Image(save_target, caption="%05d_target frame" % (step))]})
                    
                    vutils.save_image(save_target, os.path.join(log_dir, '%05d_target_sample.png' % (step)), normalize=True)
                    vutils.save_image(save_G_frame, os.path.join(log_dir, '%05d_G_frame.png' % (step)), normalize=True)
                    writer.add_image('image/G_frame', save_G_frame, global_step=step)
                    writer.add_image('image/target', save_target, global_step=step)
                
                if step % train_cfg.save_interval == 0:
                    if args.wandb:
                        wandb.log({"avenue_G_frame": [wandb.Image(save_G_frame, caption="%s_%05d_predicted_frame" % (args.dataset, step))], 
                                   "avenue_target_frame ": [wandb.Image(save_target, caption='%s_%05d_real_sample' % (args.dataset, step))]})
                    vutils.save_image(save_target, os.path.join(log_dir, '%s_%05d_real_sample.jpg' % (args.dataset, step)), normalize=True)
                    vutils.save_image(save_G_frame, os.path.join(log_dir, '%s_%05d_predicted_frame.jpg' % (args.dataset, step)), normalize=True)
                    # abn_acc = total_correct / total_samples
                    # norm_acc = total_correct_norm / total_samples_norm
                                    
                    # print(f"Saving model at iteration {step} \n")
                    # print(f"Best prediction accuracy: {accuracy_threshold*100:.2f} \n")
                    # print(f'Accuracy of abnormal classifier: {abn_acc*100:.2f}%')
                    # print(f"Predicted labels at iteration:  {predicted_abn.cpu().numpy()} | 0: abnormal, 1: normal")
                    # print(f"Abnormals classified as abnormal: {total_correct}")
                    # print(f"Abnormals classified as normal: {total_samples - total_correct} \n")
                                    
                    # print(f'Accuracy of normal classifier: {norm_acc*100:.2f}%')
                    # print(f"Best prediction accuracy: {accuracy_threshold_norm*100:.2f} \n")
                    # print(f"Predicted labels at iteration:  {predicted_norm.cpu().numpy()} | 0: abnormal, 1: normal")                    
                    # print(f"Normals classified as normal: {total_correct_norm}")
                    # print(f"Normals classified as abnormal: {total_samples_norm - total_correct_norm} \n")
                    
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict(),
                                  'net_c': classifier.state_dict(), 'optimizer_c': optimizer_C.state_dict()}
                    # torch.save(model_dict, f'weights/test7_{train_cfg.dataset}_{step}_klloss{args.kldiv_loss}_perceptloss.pth')
                    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth')
                    # print(f'\nAlready saved: \'{train_cfg.dataset}_{step}_klloss{args.kldiv_loss}_perceptloss.pth\'.')
                    print(f'\nAlready saved: \'{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth\'.')

                if step % train_cfg.val_interval == 0:
                    auc, auc_comb = val(train_cfg, model=generator, model_classifier=classifier, flow_loss=args.wfl_loss)
                    writer.add_scalar('results/auc', auc, global_step=step)
                    writer.add_scalar('results/auc_comb', auc_comb, global_step=step)
                    # writer.add_scalar('results/out_class', out_class, global_step=step)
                    generator.train()
                    classifier.train()

            step += 1
            if step > train_cfg.iters:
                training = False
                print('Finished training')
                                
                # print(f"Saving model at iteration {step} \n")
                # print(f'Accuracy of abnormal classifier: {abn_acc*100:.2f}%')
                # print(f"Abnormals classified as abnormal: {total_correct}")
                # print(f"Abnormals classified as normal: {total_samples - total_correct} \n")
                                
                # print(f'Accuracy of normal classifier: {norm_acc*100:.2f}%')
                # print(f"Normals classified as normal: {total_correct_norm}")
                # print(f"Normals classified as abnormal: {total_samples_norm - total_correct_norm} \n")
                
                print(f"AUC (Future Frame): {auc*100}%")
                print(f"AUC_norm: {auc_comb*100}%")
                
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                              'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict(),
                              'net_c': classifier.state_dict(), 'optimizer_c': optimizer_C.state_dict()}
                torch.save(model_dict, f'weights/fullnetwork_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth')
                print(f'\n Model saved: \'fullnetwork_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth\'.')
                break

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth\'.\n')
    # abn_acc = total_correct / total_samples
    # norm_acc = total_correct_norm / total_samples_norm
                    
    print(f"Saving model at iteration {step} \n")
 
    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict(),
                  'net_c': classifier.state_dict(), 'optimizer_c': optimizer_C.state_dict()}
    # torch.save(model_dict, f'weights/test7_latest_{train_cfg.dataset}_{step}_klloss{args.kldiv_loss}_perceptloss.pth')
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth')