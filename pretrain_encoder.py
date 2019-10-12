import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import time
import numpy as np
import argparse
import os

from models.models_encoder import CNNDecoderSimple, CNNEncoderSimple, CNNEncoderSimpleForContextAttention
from utils.utils import ensure_dir, save_cnn_plots


def run_epoch(epoch, dataloader, encoder, decoder, optimizer, criterion, is_eval=False, save_plots=False, dir_plots=""):
    losses = []
    if is_eval:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()
    for i, data in enumerate(dataloader):
        if i % (10000 // 16) == 0:
            print(f"{i}/{len(dataloader)}")
        
        z, _ = data
        z = z * 2 - 1
        z = z.to(device="cuda:0")
        # ===================forward=====================
        h = encoder(z)
        output = decoder(h)
        loss = criterion(output, z)
        losses.append(loss.detach())
        # ===================backward====================
        if not is_eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # =====================plot======================
        if save_plots:
            save_cnn_plots(epoch + 1, i, z.detach(), output.detach(), plot_each=1, dir_plots=dir_plots)
    
    loss = torch.FloatTensor(losses).mean()
    return loss


def main():
    # =====================parse args=======================
    print(ARGS)
    num_epochs = ARGS.epochs
    batch_size = ARGS.batch_size
    learning_rate = ARGS.lr_rate
    dataset_path = ARGS.dataset_path + f"/{ARGS.step}"
    max_time = ARGS.max_time
    context_attention_string = "_for_context_attention" if ARGS.context_attention else ""
    dir_checkpoint = f'./output_cnn/CNN_autoencoder/checkpoints{context_attention_string}'
    dir_plots = f'./output_cnn/CNN_autoencoder/plots{context_attention_string}'
    ensure_dir(dir_checkpoint)
    ensure_dir(dir_plots)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError
    device = torch.device(ARGS.device)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    if device != "cpu":
        torch.cuda.manual_seed(ARGS.seed)
    
    # =====================import data======================
    
    print("\nLoading datasets ...")
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    dataset_train = dset.ImageFolder(root=f'{dataset_path}/train', transform=transform)
    dataset_valid = dset.ImageFolder(root=f'{dataset_path}/valid', transform=transform)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    
    print("Dataset splits -> Train: {} | Valid: {}".format(len(dataset_train), len(dataset_valid)))
    
    # =====================init models======================
    # input and output of the encoder-decoder are 64x64 images
    
    if ARGS.context_attention:
        encoder = CNNEncoderSimpleForContextAttention().to(device=device)  # no linear layers, but conv2D 1x1
    else:
        encoder = CNNEncoderSimple().to(device=device)  # two linear layers to output visual features
    decoder = CNNDecoderSimple().to(device=device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate,
                                 weight_decay=1e-5)
    
    min_loss_valid = 1000000
    start_time = time.time()
    
    # ========================train=========================
    print("Starting training ...\n")
    for epoch in range(num_epochs):
        if time.time() - start_time > max_time:
            break
        
        loss_train = run_epoch(epoch, dataloader_train, encoder, decoder, optimizer, criterion, is_eval=False,
                               save_plots=False)
        loss_valid = run_epoch(epoch, dataloader_valid, encoder, decoder, optimizer, criterion, is_eval=True,
                               save_plots=True, dir_plots=dir_plots)
        print('Epoch {}/{} |, Train loss: {:.4f} | Valid loss: {:.4f}'
              .format(epoch + 1, num_epochs, loss_train.item(), loss_valid.item()))
        
        # =====================save models======================
        if min_loss_valid > loss_valid:
            min_loss_valid = loss_valid
            torch.save(encoder.state_dict(), dir_checkpoint + '/CNN_encoder.pth')
            torch.save(decoder.state_dict(), dir_checkpoint + '/CNN_decoder.pth')
    
    print("\nTraining Completed!")
    print("Minimum loss on validation set:", min_loss_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--max_time', default=36000, type=int,
                        help='maximum time in seconds for training')
    parser.add_argument('--epochs', default=15, type=int,
                        help='max number of epochs')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='size of each batch')
    parser.add_argument('--lr_rate', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--dataset_path', default="./data", type=str,
                        help='data path')
    parser.add_argument('--step', default=0.001, type=float,
                        help='Step used to generate the data')
    parser.add_argument('--context_attention', default=True, type=bool,
                        help='choose between CNNEncoderSimple and CNNEncoderSimpleForContextAttention')
    
    ARGS = parser.parse_args()
    
    main()
