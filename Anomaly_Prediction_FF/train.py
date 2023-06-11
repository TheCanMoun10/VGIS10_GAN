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

log_dir = os.path.join('./images', args.dataset, str(args.iters), str(args.batch_size), 'flow_loss'+str(args.wfl_loss))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Abnormal branch:
generator_abn = UNet(input_channels=12, output_channel=3).cuda()
discriminator_abn = PixelDiscriminator(input_nc=3).cuda()
Optimizer_G_abn = torch.optim.Adam(generator_abn.parameters(), lr=train_cfg.g_lr)
Optimizer_D_abn = torch.optim.Adam(discriminator_abn.parameters(), lr=train_cfg.d_lr)

# Normal branch:
generator_nrm = UNet(input_channels=12, output_channel=3).cuda()
discriminator_nrm = PixelDiscriminator(input_nc=3).cuda()
optimizer_G_nrm = torch.optim.Adam(generator_nrm.parameters(), lr=train_cfg.g_lr)
optimizer_D_nrm = torch.optim.Adam(discriminator_nrm.parameters(), lr=train_cfg.d_lr)

# Setup classifier:
classifier = d_netclassifier().cuda()
classifier_loss = torch.nn.BCELoss().cuda()
optimizer_C = torch.optim.SGD(classifier.parameters(), lr=train_cfg.c_lr)

if train_cfg.resume:
    generator_abn.load_state_dict(torch.load(train_cfg.resume)['net_g_abn'])
    discriminator_abn.load_state_dict(torch.load(train_cfg.resume)['net_d_abn'])
    generator_nrm.load_state_dict(torch.load(train_cfg.resume)['net_g_norm'])
    discriminator_nrm.load_state_dict(torch.load(train_cfg.resume)['net_d_norm'])
    classifier.load_state_dict(torch.load(train_cfg.resume)['net_c'])
    Optimizer_G_abn.load_state_dict(torch.load(train_cfg.resume)['optimizer_g_abn'])
    optimizer_G_nrm.load_state_dict(torch.load(train_cfg.resume)['optimizer_g_norm'])
    Optimizer_D_abn.load_state_dict(torch.load(train_cfg.resume)['optimizer_d_abn'])
    optimizer_C.load_state_dict(torch.load(train_cfg.resume)['optimizer_c'])
    print(f'Pre-trained generators, discriminators and classifiers have been loaded.\n')
else:
    generator_abn.apply(weights_init_normal)
    discriminator_abn.apply(weights_init_normal)
    discriminator_nrm.apply(weights_init_normal)
    print('Generators, discriminators and classifier are going to be trained from scratch.\n')

assert train_cfg.flownet in ('lite', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('FFP_Net/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('FFP_Net/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

# Losses abnnormal:
adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()
kullback_loss = nn.KLDivLoss(reduction='batchmean').cuda()
perceptual_loss = Perceptual_Loss([2,7,12,21,30]).cuda()


# Losses Normal:
adversarial_loss_normal = Adversarial_Loss().cuda()
discriminate_loss_normal = Discriminate_Loss().cuda()
gradient_loss_normal = Gradient_Loss(3).cuda()
flow_loss_normal = Flow_Loss().cuda()
intensity_loss_normal = Intensity_Loss().cuda()
perceptual_loss_normal = Perceptual_Loss([2,7,12,21,30]).cuda()

train_dataset = Dataset.train_dataset(train_cfg)
# print(train_dataset)

# Remember to set drop_last=True, because we need to use 4 frames to predict one7 frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}_wflowloss{args.wfl_loss}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator_abn = generator_abn.train()
discriminator_abn = discriminator_abn.train()
discriminator_nrm = discriminator_nrm.train()
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

            G_frame = generator_abn(input_frames)
            G_frame_normal = generator_nrm(input_frames)
            # Step 1: Classification for pseudo anomalies
            pseudo_class = classifier(G_frame)
            # print(pseudo_class)
            # Step 2: generate labels for pseudo anomalies --> tensor of [1 0] repeat 16 times
            pseudo_labels = torch.tensor([1, 0], dtype=torch.float32).repeat(args.batch_size, 1).cuda()
            # print(pseudo_labels)
            # Step 3: Apply BCE on tensors
                
            # Step 1: Classification for normals
            normal_class = classifier(G_frame_normal)
            # print(normal_class)
            # Step 2: generate labels for normals --> tensor of [0 1] repeat number of batch size 16 times
            normal_labels = torch.tensor([0, 1], dtype=torch.float32).repeat(args.batch_size, 1).cuda()
            # print(normal_labels)
            # Step 3: Apply BCE on tensors

            if train_cfg.flownet == 'lite':
                gt_flow_input = torch.cat([input_last, target_frame], 1)
                pred_flow_input = torch.cat([input_last, G_frame], 1) # Abnormal flow.
                pred_flow_input_nrm = torch.cat([input_last, G_frame_normal], 1) # Normal flow 
                # No need to train flow_net, use .detach() to cut off gradients.
                flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net).detach()
                flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
                flow_pred_nrm = flow_net.batch_estimate(pred_flow_input_nrm, flow_net).detach()
            else:
                gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
                pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)
                pred_flow_input_nrm = torch.cat([input_last.unsqueeze(2), G_frame_normal.unsqueeze(2)], 2)

                flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()
                flow_pred_nrm = (flow_net(pred_flow_input_nrm * 255.) / 255.).detach()

            if train_cfg.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = train_cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                    print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

            # Abnormal branch training:
            inte_l_abn = intensity_loss(G_frame, target_frame)
            grad_l_abn = gradient_loss(G_frame, target_frame)
            fl_l_abn   =  -args.wfl_loss*flow_loss(flow_gt, flow_pred)
            gan_l_abn  = adversarial_loss(discriminator_abn(G_frame))
            # kl_loss = -args.kldiv_loss*kullback_loss(nn.functional.log_softmax(target_frame, dim=1), nn.functional.softmax(G_frame, dim=1))
            p_loss_abn = perceptual_loss(G_frame, target_frame)
            cl_loss_abn = torch.mean(classifier_loss(pseudo_class, pseudo_labels))
            cl_loss_n = torch.mean(classifier_loss(normal_class, normal_labels))
            # Add mean classifier loss.
            C_l_t = (cl_loss_abn + cl_loss_n)/2
            G_l_t_abn = 1. * inte_l_abn + 1. * grad_l_abn + 0.05 * gan_l_abn + 2*fl_l_abn + C_l_t + p_loss_abn # Abnormal GAN loss + Classifier loss.

            # Normal branch training:
            inte_l_nrm = intensity_loss_normal(G_frame_normal, target_frame)
            grad_l_nrm = gradient_loss_normal(G_frame_normal, target_frame)
            fl_l_nrm = flow_loss_normal(flow_pred_nrm, flow_gt)
            gan_l_nrm = adversarial_loss_normal(discriminator_nrm(G_frame_normal))
            p_loss_nrm = perceptual_loss_normal(G_frame_normal, target_frame)
            G_l_t_nrm = 1. * inte_l_nrm + 1. * grad_l_nrm + 0.05 * gan_l_nrm + 2*fl_l_nrm + p_loss_nrm # Normal GAN loss.
            
            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_l_abn = discriminate_loss(discriminator_abn(target_frame), discriminator_abn(G_frame.detach()))
            D_l_nrm = discriminate_loss_normal(discriminator_nrm(target_frame), discriminator_nrm(G_frame_normal.detach()))
            # D_l_t = (D_l_abn + D_l_norm) / 2 
            # C_l_t = cl_loss_abn + cl_loss_n

            # Or just do .step() after all the gradients have been computed, like the following way:
            Optimizer_D_abn.zero_grad()
            optimizer_D_nrm.zero_grad()
            D_l_abn.backward(retain_graph=True)
            D_l_nrm.backward(retain_graph=True)

            # D_l_t.backward()
            Optimizer_G_abn.zero_grad()
            optimizer_G_nrm.zero_grad()
            optimizer_C.zero_grad()

            G_l_t_abn.backward(retain_graph=True)
            G_l_t_nrm.backward(retain_graph=True)


            Optimizer_D_abn.step()
            Optimizer_G_abn.step()
            optimizer_G_nrm.step()
            optimizer_C.step()

            
            torch.cuda.synchronize()
            time_end = time.time()
            if step > start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end

            if step != start_iter:
                if step % 20 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    psnr_nrm = psnr_error(G_frame_normal, target_frame)
                    psnr_abn = psnr_error(G_frame, target_frame)
                    lr_g = Optimizer_G_abn.param_groups[0]['lr']
                    lr_d = Optimizer_D_abn.param_groups[0]['lr']
                    lr_c = optimizer_C.param_groups[0]['lr']
                    
                    print(f"[{step} / {int(train_cfg.iters)}]")
                    print(f"fl_l_abn: {fl_l_abn:.3f}  | gan_l_abn: {gan_l_abn:.3f} | G_l_total_abn: {G_l_t_abn:.3f} | ")
                    print(f"fl_l_nrm: {fl_l_nrm:.3f}   | gan_l_nrm: {gan_l_nrm:.3f} | G_l_total_nrm: {G_l_t_nrm:.3f} | ")
                    print(f"cl_loss_abn: {cl_loss_abn:.3f} | cl_loss_nrm: {cl_loss_n:.3f} | C_l_t: {C_l_t:.3f} | ") 
                    print(f"psnr_abn: {psnr_abn:.3f} | psnr_nrm: {psnr_nrm:.3f}")
                    print(f"iter: {iter_t:.3f}s | ETA: {eta} | lr: {lr_g} / {lr_d} / {lr_c} | \n")

                    if args.wandb:
                        wandb.log({"psnr_abn": psnr_abn, "psnr_nrm": psnr_nrm, 
                                   "G_l_total_abn": G_l_t_abn, "G_l_total_nrm": G_l_t_nrm,
                                   "D_l_abn": D_l_abn,  "D_l_nrm": D_l_nrm, 
                                   "gan_l_nrm":gan_l_nrm, "gan_l_abn": gan_l_abn, 
                                   "fl_l_abn": fl_l_abn, "fl_l_nrm": fl_l_nrm,
                                    "cl_loss_abn": cl_loss_abn, "cl_loss_nrm": cl_loss_n, "C_l_t": C_l_t})

                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame_normal = ((G_frame_normal[0]+1)/2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_G_frame_normal = save_G_frame_normal.cpu().detach()[(2,1,0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]

                    writer.add_scalar('psnr/train_psnr_abn', psnr_abn, global_step=step)
                    writer.add_scalar('psnr/train_psnr_norm', psnr_nrm, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total_abn', G_l_t_abn, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total_norm', G_l_t_nrm, global_step=step)                    
                    writer.add_scalar('total_loss/d_loss_abn', D_l_abn, global_step=step)
                    writer.add_scalar('total_loss/d_loss_abn', D_l_nrm, global_step=step)                    
                    writer.add_scalar('G_loss_total/gan_loss_abn', gan_l_abn, global_step=step)
                    writer.add_scalar('G_loss_total/gan_loss_norm', gan_l_nrm, global_step=step)                    
                    writer.add_scalar('G_loss_total/fl_loss_abn', fl_l_abn, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss_norm', fl_l_nrm, global_step=step)


                if step % int(train_cfg.iters / 100) == 0:
                    # if args.wandb:
                    #     wandb.log({"G_frame": [wandb.Image(save_G_frame, caption='%05d_G_frame' % (step))], "target_frame ": [wandb.Image(save_target, caption="%05d_target frame" % (step))]})
                    
                    vutils.save_image(save_target, os.path.join(log_dir, '%05d_target_sample.png' % (step)), normalize=True)
                    vutils.save_image(save_G_frame, os.path.join(log_dir, '%05d_Gabn_frame.png' % (step)), normalize=True)
                    vutils.save_image(save_G_frame_normal, os.path.join(log_dir, '%05d_Gnrm_frame_pred.png' % (step)), normalize=True)
                    writer.add_image('image/G_frame', save_G_frame, global_step=step)
                    writer.add_image('image/target', save_target, global_step=step)
                    writer.add_image('image/G_frame_pred', save_G_frame_normal, global_step=step)                    
                
                if step % train_cfg.save_interval == 0:
                    if args.wandb:
                        wandb.log({"avenue_G_frame": [wandb.Image(save_G_frame, caption="%s_%05d_Gabn_abnormal_frame" % (args.dataset, step))],
                                   "avenue_G_frame_pred": [wandb.Image(save_G_frame_normal, caption="%s_%05d_Gnrm_predicted_frame" % (args.dataset, step))], 
                                   "avenue_target_frame ": [wandb.Image(save_target, caption='%s_%05d_real_sample' % (args.dataset, step))]})
                    vutils.save_image(save_target, os.path.join(log_dir, '%s_%05d_real_sample.jpg' % (args.dataset, step)), normalize=True)
                    vutils.save_image(save_G_frame, os.path.join(log_dir, '%s_%05d_Gabn_abnormal_frame.jpg' % (args.dataset, step)), normalize=True)
                    vutils.save_image(save_G_frame_normal, os.path.join(log_dir, '%s_%05d_Gnrm_predicted_frame.jpg' % (args.dataset, step)), normalize=True)

                    model_dict = {'net_g_norm': generator_nrm.state_dict(), 'optimizer_g_norm': optimizer_G_nrm.state_dict(),
                                  'net_d_norm': discriminator_nrm.state_dict(), 'optimizer_d_norm': optimizer_D_nrm.state_dict(),
                                  'net_g_abn': generator_abn.state_dict(), 'optimizer_g_abn': Optimizer_G_abn.state_dict(),
                                  'net_d_abn': discriminator_abn.state_dict(), 'optimizer_d_abn': Optimizer_D_abn.state_dict(),
                                  'net_c': classifier.state_dict(), 'optimizer_c': optimizer_C.state_dict()}
                    # torch.save(model_dict, f'weights/test7_{train_cfg.dataset}_{step}_klloss{args.kldiv_loss}_perceptloss.pth')
                    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth')
                    # print(f'\nAlready saved: \'{train_cfg.dataset}_{step}_klloss{args.kldiv_loss}_perceptloss.pth\'.')
                    print(f'\nAlready saved: \'{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth\'.')

                if step % train_cfg.val_interval == 0:
                    auc, auc_comb = val(train_cfg, model=generator_nrm, model_abn=generator_abn, model_classifier=classifier, flow_loss=args.wfl_loss)
                    writer.add_scalar('results/auc', auc, global_step=step)
                    writer.add_scalar('results/auc_comb', auc_comb, global_step=step)
                    # writer.add_scalar('results/out_class', out_class, global_step=step)
                    generator_abn.train()
                    generator_nrm.train()
                    classifier.train()

            step += 1
            if step > train_cfg.iters:
                training = False
                print('Finished training')
                                
                print(f"AUC (Future Frame): {auc*100}%")
                print(f"AUC_norm: {auc_comb*100}%")
                
                if wandb:
                    wandb.log({"AUC": auc, "AUC_norm": auc_comb})
                
                model_dict = {'net_g_norm': generator_nrm.state_dict(), 'optimizer_g_norm': optimizer_G_nrm.state_dict(),
                              'net_d_norm': discriminator_nrm.state_dict(), 'optimizer_d_norm': optimizer_D_nrm.state_dict(),
                              'net_g_abn': generator_abn.state_dict(), 'optimizer_g_abn': Optimizer_G_abn.state_dict(),
                              'net_d_abn': discriminator_abn.state_dict(), 'optimizer_d_abn': Optimizer_D_abn.state_dict(),
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

    model_dict = {'net_g_norm': generator_nrm.state_dict(), 'optimizer_g_norm': optimizer_G_nrm.state_dict(),
                  'net_d_norm': discriminator_nrm.state_dict(), 'optimizer_d_norm': optimizer_D_nrm.state_dict(),
                  'net_g_abn': generator_abn.state_dict(), 'optimizer_g_abn': Optimizer_G_abn.state_dict(),
                  'net_d_abn': discriminator_abn.state_dict(), 'optimizer_d_abn': Optimizer_D_abn.state_dict(),
                  'net_c': classifier.state_dict(), 'optimizer_c': optimizer_C.state_dict()}
    # torch.save(model_dict, f'weights/test7_latest_{train_cfg.dataset}_{step}_klloss{args.kldiv_loss}_perceptloss.pth')
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}_flloss{args.wfl_loss}_perceptloss_classifier.pth')
    
    #Thus ends the program.