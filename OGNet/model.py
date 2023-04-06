import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import numpy as np
from network import d_net, g_net
import torchvision.utils as vutils
from opts_fine_tune_discriminator import parse_opts_ft
from opts import parse_opts
from fine_tune_dicriminator import fine_tune
from utils import gaussian
from dataloader import load_data
from sklearn import metrics
import os
import wandb

def check_auc(g_model_path, d_model_path, i):
    opt_auc = parse_opts()
    opt_auc.batch_shuffle = False
    opt_auc.drop_last = False
    opt_auc.data_path = './data/avenue/test'
    dataloader = load_data(opt_auc)
    model = OGNet(opt_auc, dataloader)
    model.cuda()
    d_results, labels = model.test_patches(g_model_path, d_model_path, i)
    d_results = np.concatenate(d_results)
    labels = np.concatenate(labels)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, d_results, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    EER1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    d_f1 = np.copy(d_results)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    AUC = metrics.auc(fpr1, tpr1)
    print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(AUC, EER1,
                                                                  eer_threshold1,f1_score))
    return AUC, f1_score, d_results

class OGNet(nn.Module):
    @staticmethod
    def name():
        return 'Old is Gold: Redefining the adversarially learned one-class classification paradigm'

    def __init__(self, opt, dataloader):
        super(OGNet, self).__init__()
        self.adversarial_training_factor  = opt.adversarial_training_factor
        self.g_learning_rate = opt.g_learning_rate
        self.d_learning_rate = opt.d_learning_rate
        self.epoch = opt.epoch
        self.batch_size = opt.batch_size
        self.n_threads = opt.n_threads
        self.sigma_noise = opt.sigma_noise
        self.dataloader = dataloader
        self.c = opt.nc
        self.g = g_net(self.c).cuda()
        self.d = d_net(self.c).cuda()
        self.image_grids_numbers = opt.image_grids_numbers
        self.filename = ''
        self.n_row_in_grid = opt.n_row_in_grid
        self.wandb = opt.wandb

    def train(self, normal_class):
        
        if self.wandb:    
            wandb.init(project="VGIS10_OGNet",
            
            config={
                "g_learning_rate": self.g_learning_rate,
                "d_learning_rate" : self.d_learning_rate,
                "Sigma noise" : self.sigma_noise,
                "number of image channels" : self.c,
                "dataset": 'avenue',
                "epochs" : self.epoch,
                "batch size" : self.batch_size,
                },
                name="{0}_{1}_glr{2}_dlr{3}".format(self.epoch, self.batch_size, self.g_learning_rate, self.d_learning_rate)
            )
            
        AUC_phase1 = []
        F1_score_phase1 = []
        
        AUC_phase2 = []
        F1_score_phase2 = []
        
        self.g.train()
        self.d.train()
        print("Length of dataloader:", len(self.dataloader))

        # Set optimizators
        g_optim = optim.Adam(self.g.parameters(), lr=self.g_learning_rate)
        d_optim = optim.Adam(self.d.parameters(), lr=self.d_learning_rate)

        fake = torch.ones([self.batch_size], dtype=torch.float32).cuda()
        valid = torch.zeros([self.batch_size], dtype=torch.float32).cuda()
        print('Training until high epoch...')
        for num_epoch in range(self.epoch):
            # print("Epoch {0}".format(num_epoch))
            for i, data in enumerate(self.dataloader):
                input, gt_label = data
                input = input.cuda()
                # print("Torch input tensor min/max values: ", (torch.min(input), torch.max(input)))

                g_optim.zero_grad()
                d_optim.zero_grad()
                sigma = self.sigma_noise ** 2
                input_w_noise = gaussian(input, 1, 0, sigma)  #Noise
                # Inference from generator
                g_output = self.g(input_w_noise)

                vutils.save_image(input[0:self.image_grids_numbers, :, :, :],
                                './results/%03d_real_samples_epoch.png' % (num_epoch), nrow=self.n_row_in_grid, normalize=True)
                vutils.save_image(g_output[0:self.image_grids_numbers, :, :, :],
                                './results/%03d_fake_samples_epoch.png' % (num_epoch), nrow=self.n_row_in_grid, normalize=True)
                vutils.save_image(input_w_noise[0:self.image_grids_numbers, :, :, :],
                                './results/%03d_noise_samples_epoch.png' % (num_epoch), nrow=self.n_row_in_grid, normalize=True)

                ##############################################
                
                d_fake_output = self.d(g_output)
                d_real_output = self.d(input)
                d_fake_loss = F.binary_cross_entropy(torch.squeeze(d_fake_output), fake)
                d_real_loss = F.binary_cross_entropy(torch.squeeze(d_real_output), valid)
                d_sum_loss = 0.5 * (d_fake_loss + d_real_loss)
                d_sum_loss.backward(retain_graph=True)
                g_optim.zero_grad()

                ##############################################
                g_recon_loss = F.mse_loss(g_output, input)
                g_adversarial_loss = F.binary_cross_entropy(d_fake_output.squeeze(), valid)
                g_sum_loss = (1-self.adversarial_training_factor)*g_recon_loss + self.adversarial_training_factor*g_adversarial_loss
                g_sum_loss.backward()
                
                d_optim.step()
                g_optim.step()
                
                if self.wandb:
                    wandb.log({'Loss_G' : g_sum_loss, 'Recon_G_loss': g_recon_loss, 'Loss_D_fake': d_fake_loss, 'Loss_D_real' : d_real_loss, 'G_adversarial_loss' : g_adversarial_loss})

                high_epoch_g_model_name = 'g_high_epoch'
                high_epoch_d_model_name = 'd_high_epoch'
                g_model_save_path = './models/' + high_epoch_g_model_name
                d_model_save_path = './models/' + high_epoch_d_model_name
                                    
                if i%150 == 0 and self.wandb:
                        pixels_gen = g_output[0].detach().cpu().permute(1,2,0).numpy()
                        pixels_noise = input_w_noise[0].detach().cpu().permute(1,2,0).numpy()
                        pixels_input = input[0].detach().cpu().permute(1,2,0).numpy()
                        np.rot90(pixels_gen, k=0, axes=(1,0))
                        np.rot90(pixels_noise, k=0, axes=(1,0))
                        np.rot90(pixels_input, k=0, axes=(1,0))
                    
                        fake_image = wandb.Image(pixels_gen, caption="Generator Image")
                        noisy_image_fake = wandb.Image(pixels_noise, caption="Noisy input sample")
                        input_image = wandb.Image(pixels_input, caption="Input image")
                        
                        wandb.log({'Input images': input_image, 'Noisy image sample': noisy_image_fake, 'Train_Generator Image': fake_image})     
                                  
                if i%1 == 0:
                    opts_ft = parse_opts_ft() #opts for phase two

                    if num_epoch == opts_ft.low_epoch:
                        g_model_name = 'g_low_epoch'
                        model_save_path = './models/' + g_model_name
                        torch.save({
                            'epoch': num_epoch,
                            'g_model_state_dict': self.g.state_dict(),
                            'g_optimizer_state_dict': g_optim.state_dict(),
                        }, model_save_path)

                    if num_epoch >= opts_ft.high_epoch:
                        g_model_name = 'g_high_epoch'
                        d_model_name = 'd_high_epoch'
                        model_save_path = './models/' + g_model_name
                        torch.save({
                            'epoch': num_epoch,
                            'g_model_state_dict': self.g.state_dict(),
                            'g_optimizer_state_dict': g_optim.state_dict(),
                        }, model_save_path)

                        model_save_path = './models/' + d_model_name
                        torch.save({
                            'epoch': num_epoch,
                            'd_model_state_dict': self.d.state_dict(),
                            'd_optimizer_state_dict': d_optim.state_dict(),
                        }, model_save_path)

                        print('Epoch {0} / Iteration {1}, before phase two: '.format(num_epoch, i))                        
                        
                        auc, F1_score, results = check_auc(g_model_save_path, d_model_save_path,1)
                        AUC_phase1.append(auc)
                        # EER_phase1.append(EER_pre)
                        # EER_thres_phase1.append(eer_thres_pre)
                        F1_score_phase1.append(F1_score)
                    
                        fine_tune() #Phase two
                        print('After phase two: ')

                        auc2, F1_score2, dresults = check_auc(g_model_save_path, d_model_save_path,1)
                        AUC_phase2.append(auc2)
                        # EER_phase2.append(EER_post)
                        # EER_thres_phase2.append(eer_thres_post)
                        F1_score_phase2.append(F1_score2)
                        
                        if self.wandb:
                            wandb.log({'AUC_phase_1' : AUC_phase1 , 'F1_score_phase_1' : F1_score_phase1, 'AUC_phase_2' : AUC_phase2 , 'F1_score_phase_2' : F1_score_phase2 }, step=i)
                        #     wandb.log({'AUC_phase_2' : AUC_phase2 ,'EER1_phase_2': EER_phase2, 'EER1_thresh_phase_2' : EER_thres_phase2, 'F1_score_phase_2' : F1_score_phase2}, step=i)
                            
    def test_patches(self,g_model_path, d_model_path,i):  #test all images/patches present inside a folder on given g and d models. Returns d score of each patch
        test_opts = parse_opts()
        checkpoint_epoch_g = -1
        g_checkpoint = torch.load(g_model_path)
        self.g.load_state_dict(g_checkpoint['g_model_state_dict'])
        checkpoint_epoch_g = g_checkpoint['epoch']
        if checkpoint_epoch_g is -1:
            raise Exception("g_model not loaded")
        else:
            pass
        d_checkpoint = torch.load(d_model_path)
        self.d.load_state_dict(d_checkpoint['d_model_state_dict'])
        checkpoint_epoch_d = d_checkpoint['epoch']
        if checkpoint_epoch_g == checkpoint_epoch_d:
            pass
        else:
            raise Exception("d_model not loaded or model mismatch between g and d")

        self.g.eval()
        self.d.eval()
        labels = []
        d_results = []
        count = 0
        for input, label in self.dataloader:
            input = input.cuda()
            g_output = self.g(input)
            d_fake_output = self.d(g_output)
            
            if count%150 == 0 and test_opts.wandb:
                pixels_gen = g_output[0].detach().cpu().permute(1,2,0).numpy()
                # pixels_d_fake = d_fake_output[0].detach().cpu().permute(1,2,0).numpy()
                pixels_input = input[0].detach().cpu().permute(1,2,0).numpy()
                np.rot90(pixels_gen, k=0, axes=(1,0))
                # np.rot90(pixels_d_fake, k=0, axes=(1,0))
                np.rot90(pixels_input, k=0, axes=(1,0))
                    
                fake_image = wandb.Image(pixels_gen, caption="Generator Image")
                # noisy_image_fake = wandb.Image(pixels_d_fake, caption="Discriminator fake output")
                input_image = wandb.Image(pixels_input, caption="Input image")
                        
                wandb.log({'Test_image_real': input_image, 'Test_image_fake': fake_image}) #'Discriminator image sample': noisy_image_fake, 'Test_Generator Image': fake_image})
            
            count +=1
            d_results.append(d_fake_output.cpu().detach().numpy())

            labels.append(label)
            if test_opts.wandb:
                wandb.log({'Discriminator results': d_results, 'Labels': labels})
        return d_results, labels

