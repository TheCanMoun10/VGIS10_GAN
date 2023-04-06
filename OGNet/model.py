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

def check_auc(g_model_path, d_model_path, i):
    opt_auc = parse_opts()
    opt_auc.batch_shuffle = False
    opt_auc.drop_last = False
    opt_auc.data_path = './data/avenue/gray/test'
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
    print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
                                                                  eer_threshold1,f1_score))

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

    def train(self, normal_class):
        self.g.train()
        self.d.train()

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
                
                high_epoch_g_model_name = 'g_high_epoch'
                high_epoch_d_model_name = 'd_high_epoch'
                g_model_save_path = './models/' + high_epoch_g_model_name
                d_model_save_path = './models/' + high_epoch_d_model_name
                                
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
                        
                        check_auc(g_model_save_path, d_model_save_path,1)
                        fine_tune() #Phase two
                        print('After phase two: ')


                        check_auc(g_model_save_path, d_model_save_path,1)

    def test_patches(self,g_model_path, d_model_path,i):  #test all images/patches present inside a folder on given g and d models. Returns d score of each patch
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
            count +=1
            d_results.append(d_fake_output.cpu().detach().numpy())
            labels.append(label)
        return d_results, labels

