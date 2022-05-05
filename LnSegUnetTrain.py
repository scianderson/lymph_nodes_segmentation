import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset import *
from model import *
# from loss import *
import os
import SimpleITK as sitk


if __name__ == "__main__":
    mode='gpu'
    if mode=='gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # after switch device, you need restart the kernel
        # torch.cuda.set_device(1)
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_dtype(torch.float64)
        
    resume = True
    save_model = True
    print(f'resume:{resume}, save_model:{save_model}')
    output_dir = 'Models/UNet1024'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    epoch_loss_list = []
    epoch_num = 1001
    start_epoch_num = 27
    batch_size = 5
    learning_rate = 5e0

    model = UNet1024()
    model.train()
    if mode=='gpu':
        model.cuda()
    net = torch.nn.DataParallel(model, device_ids=[0, 1])
    # criterion = DiceLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = UnetDataset(root_dir='/home/sci/hdai/Projects/Dataset/LymphNodes', patch_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if resume:
        checkpoint = torch.load(f'{output_dir}/epoch_{start_epoch_num-1}_checkpoint.pth.tar')    
        model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with open(f'{output_dir}/loss.txt', 'a') as f:
            f.write(f'From {start_epoch_num} to {epoch_num+start_epoch_num}\n')
            f.write(f'BCE; Adadelta, lr={learning_rate}; batch size: {batch_size}\n')
    else:
        start_epoch_num = 0  

        with open(f'{output_dir}/loss.txt', 'w+') as f:
            f.write(f'From {start_epoch_num} to {epoch_num+start_epoch_num}\n')
            f.write(f'BCE; Adadelta: lr={learning_rate}; batch size: {batch_size}\n')

    print(f'Starting from iteration {start_epoch_num} to iteration {epoch_num+start_epoch_num}')

    for epoch in range(start_epoch_num, start_epoch_num+epoch_num):
        epoch_loss = 0

        for i, batched_sample in tqdm(enumerate(dataloader)):
            '''innerdomain backpropagate'''
    #         print(i)
            input_data = batched_sample['img'].double()#.to(device)
    #         print(input.shape)
            input_data.requires_grad = True
            # u_pred: [batch_size, *data_shape, feature_num] = [1, 5, ...]
            output_pred = net(input_data)
            # output_pred = model(input)
            output_true = batched_sample['mask']

            optimizer.zero_grad()
            loss = criterion(output_pred, output_true.double())
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        with open(f'{output_dir}/loss.txt', 'a') as f:
            f.write(f'{epoch_loss}\n')

        print(f'epoch {epoch} loss: {epoch_loss}')#, norm: {torch.norm(f_pred,2)**2}
        epoch_loss_list.append(epoch_loss)
        if epoch%1==0:       
            if save_model:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, f'{output_dir}/epoch_{epoch}_checkpoint.pth.tar')

    plt.figure(figsize=(7,5))
    plt.title('Innerdomain loss')
    plt.xlabel('epoch')
    plt.ylabel('BCE loss')
    plt.plot(epoch_loss_list)
    plt.savefig(f'{output_dir}/adadelta_loss_{learning_rate}.png')