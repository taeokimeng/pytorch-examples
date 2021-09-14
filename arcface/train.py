from __future__ import print_function
import os
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils import data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# from torchsummary import summary
from config.config import Config
from margin import ArcMarginProduct
from models.iresnet import iresnet18
from models.resnet import resnet18
from models.net import Net


def plot_3d_features(features, labels, num_classes, epoch, prefix, path):
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plot = ax.plot(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            features[labels==label_idx, 2],
            '.',
            alpha=0.8,
            c=colors[label_idx]
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], bbox_to_anchor=(1.04, 1), loc='upper left', 
               handler_map={type(plot): HandlerPathCollection(update_func=update_prop)}) # prop={'size': 6}
    # dirname = osp.join(args.save_dir, prefix)
    # if not osp.exists(dirname):
    #     os.mkdir(dirname)
    save_path = os.path.join(path, prefix + '_epoch_' + str(epoch) + '.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_angular_space(features, labels, num_classes, epoch, prefix):
    plt.figure(figsize=(10, 10))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        sc = plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], bbox_to_anchor=(1.04, 1), loc='upper left', 
               handler_map={type(sc): HandlerPathCollection(update_func=update_prop)}) # prop={'size': 6}
    # dirname = osp.join(args.save_dir, prefix)
    # if not osp.exists(dirname):
    #     os.mkdir(dirname)
    save_name = prefix + '_epoch_' + str(epoch) + '.png'
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        sc = plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], bbox_to_anchor=(1.04, 1), loc='upper left', 
               handler_map={type(sc): HandlerPathCollection(update_func=update_prop)}) # prop={'size': 6}
    # dirname = osp.join(args.save_dir, prefix)
    # if not osp.exists(dirname):
    #     os.mkdir(dirname)
    save_name = prefix + '_epoch_' + str(epoch) + '.png'
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def update_prop(handle, orig):
    """
    Update marker size
    """
    marker_size = 12
    handle.update_from(orig)
    handle.set_sizes([marker_size])


if __name__ == '__main__':
    opt = Config()
    device = torch.device(opt.cuda)

    transform=transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                        batch_size=opt.train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                        batch_size=opt.test_batch_size, shuffle=True)

    if opt.loss == 'softmax':
        criterion = nn.CrossEntropyLoss()

    if opt.backbone == 'iresnet18':
        model = iresnet18()
    elif opt.backbone == 'resnet18':
        model = resnet18()
    elif opt.backbone == 'net':
        model = Net()

    s = 10
    m = 0.25
    plot_folder = '' # f's{s}_m{m}'
    plot_save_path = os.path.join(opt.image_save_path, plot_folder)

    if not os.path.isdir(opt.image_save_path):
        os.makedirs(opt.image_save_path)

    if not os.path.isdir(plot_save_path):
        os.makedirs(plot_save_path)

    if opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=s, m=m, easy_margin=opt.easy_margin)

    # print(model)
    model = model.to(device=device)
    # print(summary(model, input_size=(1, 28, 28)))
    # model = DataParallel(model)
    metric_fc = metric_fc.to(device)
    # metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.8)

    start = time.time()
    for epoch in range(1, opt.max_epoch + 1):
        # scheduler.step()
        model.train()
        all_features, all_labels = [], []
        all_norm_features = []
        for batch_idx, data in enumerate(train_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature, final_output = model(data_input)
            output = metric_fc(final_output, label) # feature
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_features.append(feature.data.cpu().numpy()) # feature
            all_labels.append(label.data.cpu().numpy())

            all_norm_features.append(F.normalize(feature).data.cpu().numpy())
            # all_norm_labels.append(F.normalize(label).data.cpu().numpy())

            iters = epoch * len(train_loader) + batch_idx

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, epoch, batch_idx, speed, loss.item(), acc))
                start = time.time()

        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        all_norm_features = np.concatenate(all_norm_features, 0)
        # all_norm_labels = np.concatenate(all_norm_labels, 0)
        plot_3d_features(all_norm_features, all_labels, opt.num_classes, epoch, prefix=f'arcface_3d_{opt.backbone}_s_{s}_m_{m}', path=plot_save_path)
        # plot_features(all_features, all_labels, opt.num_classes, epoch, prefix=f'arcface_{opt.backbone}_s_{s}_m_{m}')
        # plot_angular_space(all_norm_features, all_labels, opt.num_classes, epoch, prefix=f'arcface_ang_{opt.backbone}_s_{s}_m_{m}')
        
        scheduler.step()
        # if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
        #     save_model(model, opt.checkpoints_path, opt.backbone, epoch)

        # model.eval()

