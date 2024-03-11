# -*- coding: utf-8 -*-

import argparse
import os
import torch.optim as optim
from models import  VGG9
import dataloader
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

dir(torch.utils)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='TAB Training')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N', help='number of workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of training epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch number for resume models')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='number of batch size')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('-T', '--time_steps', default=4, type=int, metavar='N', help='inference time-step')
parser.add_argument('--out_dir', default='./logs/', type=str, help='log dir')
parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
parser.add_argument('--method', default='TTBN', type=str, help='BN method')
parser.add_argument('--tau', type=float, default=0.9, help='tau value of LIF neuron')

args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        mean_out = outputs.mean(1)
        loss = criterion(mean_out, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


def main():
    # set manual seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset, val_dataset = dataloader.Cifar10(download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = VGG9(tau=args.tau)

    model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0

    out_dir = os.path.join(args.out_dir, f'method_{args.method}_tau_{args.tau}_T_{args.time_steps}')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    best_acc = 0
    best_epoch = 0

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):

        loss, acc = train(model, device, train_loader, criterion, optimizer, epoch, args)
        print('Epoch {}/{} train loss={:.5f} train acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        test_acc = test(model, test_loader, device)
        print('Epoch {}/{} test acc={:.3f}'.format(epoch, args.epochs, test_acc))
        writer.add_scalar('test_acc', test_acc, epoch)
        scheduler.step()

        save_max = False
        if best_acc < test_acc:
            best_acc = test_acc
            save_max = True
            best_epoch = epoch + 1
        print('Best Test acc={:.3f}'.format(best_acc))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))


if __name__ == '__main__':
    main()
    print("666 DONE!")
