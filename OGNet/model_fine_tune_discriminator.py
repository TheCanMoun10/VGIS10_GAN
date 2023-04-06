import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from network import d_net, g_net

class Fine_Tune_Disc(nn.Module):
    @staticmethod
    def name():
        return 'Old is Gold: Redefining the adversarially learned one-class classification paradigm'

    def __init__(self, opt, dataloader, nc=1):
        super(Fine_Tune_Disc, self).__init__()
        self.d_learning_rate = opt.d_learning_rate
        self.epoch = opt.epoch
        self.batch_size = opt.batch_size
        self.n_threads = opt.n_threads
        self.high_epoch_fake_loss_contribution = opt.high_epoch_fake_loss_contribution
        self.psuedo_anomaly_contribution = opt.psuedo_anomaly_contribution

        self.dataloader = dataloader
        self.iterations = opt.iterations
        self.nc = opt.nc

        self.g_low_epoch = g_net(self.nc).cuda()
        self.g_high_epoch = g_net(self.nc).cuda()
        self.d_high_epoch = d_net(self.nc).cuda()

        self.filename = ''
        self.test_anomaly_threshold = opt.test_anomaly_threshold

    def train(self, load_model_epoch, model_folder_path):
        load_g_low_epoch_model = model_folder_path + 'g_low_epoch'
        g_low_epoch_checkpoint = torch.load(load_g_low_epoch_model)
        self.g_low_epoch.load_state_dict(g_low_epoch_checkpoint['g_model_state_dict'])
        checkpoint_epoch1 = g_low_epoch_checkpoint['epoch']
        load_g_high_epoch_model = model_folder_path + 'g_high_epoch'
        g_high_epoch_checkpoint = torch.load(load_g_high_epoch_model)
        self.g_high_epoch.load_state_dict(g_high_epoch_checkpoint['g_model_state_dict'])
        checkpoint_epoch2 = g_high_epoch_checkpoint['epoch']
        load_d_high_epoch_model = model_folder_path + 'd_high_epoch'
        d_high_epoch_checkpoint = torch.load(load_d_high_epoch_model)
        self.d_high_epoch.load_state_dict(d_high_epoch_checkpoint['d_model_state_dict'])
        checkpoint_epoch3 = d_high_epoch_checkpoint['epoch']
        self.g_low_epoch.eval()
        self.g_high_epoch.eval()
        self.d_high_epoch.train()

        # Set optimizator(s)
        d_optim = optim.Adam(self.d_high_epoch.parameters(), lr=self.d_learning_rate)
        label_real_input = torch.ones([self.batch_size], dtype=torch.float32).cuda() * 0
        label_fake_high_quality = torch.ones([self.batch_size], dtype=torch.float32).cuda() * 0
        label_fake_low_quality = torch.ones([self.batch_size], dtype=torch.float32).cuda() * 1
        label_fake_augmented = torch.ones([self.batch_size], dtype=torch.float32).cuda() * 1


        it = 0
        for num_epoch in range(self.epoch):
            train_loader_iter = iter(self.dataloader)
            total_batches_pulled = 0
            for i, data in enumerate(train_loader_iter):
                input, gt_label = data
                input = input.cuda()
                d_optim.zero_grad()
                input_w_noise = input

                # Inference from generators
                g_low_epoch_output = self.g_low_epoch(input_w_noise)
                g_high_epoch_output = self.g_high_epoch(input_w_noise)
                total_batches_pulled += 3           # to avoid the iterator to crash
                if total_batches_pulled >= len(self.dataloader):
                    # print('breaking because the batches are finished')
                    break
                b1 = next(train_loader_iter)
                b2 = next(train_loader_iter)
                low_fake_augmented_1 , gar = b1
                low_fake_augmented_2 , gar = b2

                g_low_fake_augmented_1 = self.g_low_epoch(low_fake_augmented_1.cuda())
                g_low_fake_augmented_2 = self.g_low_epoch(low_fake_augmented_2.cuda())
                augmented_image = (g_low_fake_augmented_1 + g_low_fake_augmented_2) / 2
                g_augmented_data = self.g_high_epoch(augmented_image)

                #######################
                #######################
                d_low_epoch_fake_output = self.d_high_epoch(g_low_epoch_output)    #low epoch recon
                d_high_epoch_fake_output = self.d_high_epoch(g_high_epoch_output)  #high epoch recon
                d_real_output = self.d_high_epoch(input)   #real image
                d_augmented_g = self.d_high_epoch(g_augmented_data)  #psuedo anomaly
                loss_1 = F.binary_cross_entropy(torch.squeeze(d_low_epoch_fake_output), label_fake_low_quality)  * self.psuedo_anomaly_contribution + F.binary_cross_entropy(torch.squeeze(d_augmented_g), label_fake_augmented) * (1-self.psuedo_anomaly_contribution)
                loss_2_1 = F.binary_cross_entropy(torch.squeeze(d_high_epoch_fake_output), label_fake_high_quality)
                loss_2_2 = F.binary_cross_entropy(torch.squeeze(d_real_output), label_real_input)
                loss_2 = (loss_2_1 * self.high_epoch_fake_loss_contribution) + (loss_2_2 * (1-self.high_epoch_fake_loss_contribution))
                d_sum_loss = loss_1 + loss_2
                d_sum_loss.backward()
                d_optim.step()
                
                high_epoch_g_model_name = 'g_high_epoch'
                high_epoch_d_model_name = 'd_high_epoch'
                g_model_save_path = './models/' + high_epoch_g_model_name
                d_model_save_path = './models/' + high_epoch_d_model_name
                
                if it == self.iterations:

                    torch.save({
                        'epoch': num_epoch,
                        'g_model_state_dict': self.g_high_epoch.state_dict(),
                    }, g_model_save_path)


                    torch.save({
                        'epoch': num_epoch,
                        'd_model_state_dict': self.d_high_epoch.state_dict(),
                        'd_optimizer_state_dict': d_optim.state_dict(),
                    }, d_model_save_path)

                if it == self.iterations:
                    return
                it += 1
        return d_model_save_path, g_model_save_path