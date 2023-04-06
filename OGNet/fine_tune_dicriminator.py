

from model_fine_tune_discriminator import Fine_Tune_Disc

from opts_fine_tune_discriminator import parse_opts_ft
from dataloader import load_data_train



def fine_tune():  #Phase two

    opt = parse_opts_ft()
    low_epoch = opt.low_epoch
    high_epoch = opt.high_epoch
    for i in range(high_epoch, high_epoch+1, 1):

        dataloader = load_data_train(opt)

        model = Fine_Tune_Disc(opt, dataloader)
        model.cuda()

        model_folder_path = './models/'
        load_model_epoch = ['epoch_{0}'.format(low_epoch), 'epoch_{0}'.format(i)]
        model.train(load_model_epoch, model_folder_path)
