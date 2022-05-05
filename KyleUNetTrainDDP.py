import argparse
import os
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split, Subset

import scipy

import model as md
import Kyle_model as kmd
import dataset as dtst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch', default=4, type=int, metavar='N',
                        help='number of samples in a batch')
    parser.add_argument('-lr', '--learn_rate', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('-c', '--checkpoint_path', default='checkpoint.tar', type=str,
                        help='path to the checkpoint file for the training')
    parser.add_argument('-lc', '--load_checkpoint', action='store_true',
                        help='whether to load the model from a checkpoint file')
    parser.add_argument('-l', '--log_path', default=None, type=str,
                        help='path to the log file for the training')
    args = parser.parse_args()

    args.world_size = args.nodes * args.gpus
    os.environ["MASTER_ADDR"] = "155.98.19.220"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(train, nprocs=args.gpus, args=(args,))

    train(0, args)
    
    
def train(gpu, args):
    rank = args.nr*args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(0)
    model = kmd.UNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([62.9]).cuda(gpu)).cuda(gpu)
    optimizer = torch.optim.Adadelta(model.parameters(),
                                     lr=args.learn_rate)

    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])

    start_epoch = None
    if args.load_checkpoint:
        map_location = {"cuda:0" : f"cuda:{gpu}"}
        checkpoint = torch.load(args.checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if gpu == 0:
            print(f"Loaded model from {args.checkpoint_path} "
                  + f"at epoch {checkpoint['epoch']} "
                  + f"with loss: {checkpoint['train_loss']:.4f}")
    
    dataset = dtst.UnetDataset(
        root_dir="/home/sci/kyle.anderson/lymph_nodes/Dataset",
        patch_size=128,
        min_probability=0.1
    )
    all_indices = list(range(len(dataset)))
    random.Random(13).shuffle(all_indices)
    # split1: test is last 20%
    train_set = Subset(dataset, all_indices[:int(0.8*len(all_indices))])
    test_set = Subset(dataset, all_indices[int(0.8*len(all_indices)) - len(all_indices):])
    # split2: test is first 20%
    # train_set = Subset(dataset, all_indices[len(all_indices)-int(0.8*len(all_indices)):])
    # test_set = Subset(dataset, all_indices[:len(all_indices)-int(0.8*len(all_indices))])
    print(len(train_set)+len(test_set) == len(dataset))

    """
    train_set, test_set = random_split(
        dataset,
        [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)],
        generator=torch.Generator().manual_seed(0)
    )
    """
    if gpu == 0:
        print(f"Training on a training split of {len(train_set)} samples ({len(test_set)} test samples).")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=args.world_size,
        rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set,
        num_replicas=args.world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=test_sampler
    )

    st = datetime.now()
    total_step = len(train_loader)
    
    test_losses = []
    start_epoch = start_epoch if start_epoch else 0
    end_epoch = start_epoch + args.epochs if start_epoch else args.epochs
    for epoch in range(start_epoch, end_epoch):
        et = datetime.now()
        epoch_loss = 0.0
        for i, samples in enumerate(train_loader):
            optimizer.zero_grad()
            img = torch.from_numpy(scipy.ndimage.zoom(samples["img"], [1.0, 1.0, 0.5, 0.5, 0.5]))
            mask = torch.from_numpy(scipy.ndimage.zoom(samples["mask"], [1.0, 1.0, 0.5, 0.5, 0.5]))
            preds = model(img.cuda(non_blocking=True).type(torch.cuda.FloatTensor))
            loss = criterion(preds, mask.cuda(non_blocking=True).type(torch.cuda.FloatTensor))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i+1) % int(total_step / 5) == 0 and gpu == 0:
                print(
                    f"Epoch [{epoch+1:3d}/{end_epoch:3d}], "
                    + f"Step [{i+1:2d}/{total_step}], "
                    + f"Loss: {loss.item():6.4f}"
                    )

        test_loss = test(model, criterion, test_loader)
        test_losses.append(test_loss)
        
        if gpu == 0:
            print(f"Epoch duration: {datetime.now() - et}")
            print(f"Epoch test loss: {test_loss:.4f}\n")
            torch.save(
                {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss / len(train_loader),
                "test_loss": test_loss
                },
                args.checkpoint_path
            )
            if args.log_path:
                try:
                    with open(args.log_path, 'a') as f:
                        f.write(f"Time: {datetime.now()}\t"
                                + f"Epoch {epoch+1:3d} "
                                + f"Train Loss: {epoch_loss/len(train_loader):.4f}\t"
                                + f"Test Loss: {test_loss:.4f}\n")
                except:
                    print(f"Unable to open {args.log_path}.")
            if len(test_losses) >= 2:
                if test_losses[-1] >= test_losses[-2]:
                    print(f"Test loss did not decrease. Ending training.")
                    break

    if gpu == 0:
        print(f"Training completed in {datetime.now() - st}.")
        
    dist.destroy_process_group()

def test(model, loss_fn, test_loader):
    num_batches = len(test_loader)
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            img = torch.from_numpy(scipy.ndimage.zoom(batch["img"], [1.0, 1.0, 0.5, 0.5, 0.5]))
            mask = torch.from_numpy(scipy.ndimage.zoom(batch["mask"], [1.0, 1.0, 0.5, 0.5, 0.5]))
            pred = model(img.cuda(non_blocking=True).type(torch.cuda.FloatTensor))
            test_loss += loss_fn(pred, mask.cuda(non_blocking=True).type(torch.cuda.FloatTensor)).item()

    return test_loss / num_batches

if __name__=="__main__":
    main()
