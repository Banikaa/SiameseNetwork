import os
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps

from siamese import SiameseNetwork
from libs.dataset import Dataset
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()



    # parser.add_argument(
    #     '-v',
    #     '--val_path',
    #     type=str,
    #     help="Path to directory containing validation dataset.",
    #     required=True
    # )
    # parser.add_argument(
    #     '-o',
    #     '--out_path',
    #     type=str,
    #     help="Path for saving prediction images.",
    #     required=True
    # )
    # parser.add_argument(
    #     '-c',
    #     '--checkpoint',
    #     type=str,
    #     help="Path of model checkpoint to be used for inference.",
    #     required=True
    # )

    val_path = '/media/internal/DATA/FYPStudents/Andrei-Internship/SIAMESE/data/dataset_1_index_cpp/test'
    out_path ='./outputs_9'
    checkpoint = out_path + '/epoch_50.pth'

    os.makedirs(out_path, exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset     = Dataset(val_path, shuffle_pairs=False, augment=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=1)

    # criterion = torch.nn.BCELoss()

    checkpoint = torch.load(checkpoint)
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    losses = []
    correct = 0
    total = 0

    inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])


    def get_match(img1, img2):
        finger1 = img1[12:-4]
        finger2 = img2[12:-4]

        if finger1 == finger2:
            return 1
        else:
            return 0

    def contrastive_loss(output, target):
        # output is the similarity score from the model
        # target is the ground truth label (1 for matching pair, 0 for non-matching pair)
        margin = 0.5
        loss = target * F.relu(1 - output) + (1 - target) * F.relu(output - margin)
        return loss.mean()
    
    for i, ((img1, img2), y, (class1, class2), (path1, path2)) in enumerate(val_dataloader):
        print("[{} / {}]".format(i, len(val_dataloader)))

        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
        if class1 == class2:
            fname1 = path1[0].split('/')[-1]
            fname2 = path2[0].split('/')[-1]
            match = get_match(fname1, fname2)
            target = torch.Tensor([match]).unsqueeze(1).to(device)
        else:
            target = torch.Tensor([0]).unsqueeze(1).to(device)

        prob = model(img1, img2)
        loss = contrastive_loss(prob, target)

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (prob > 0.5)).item()
        total += len(y)

        fig = plt.figure("class1={}\tclass2={}".format(class1, class2))
        plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(class1, prob[0][0].item(), class2))

        print(class1, class2)

        # Apply inverse transform (denormalization) on the images to retrieve original images.
        img1 = inv_transform(img1).cpu().numpy()[0]
        img2 = inv_transform(img2).cpu().numpy()[0]

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(img1[0])
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(img2[0])
        plt.axis("off")

        # show the plot
        plt.savefig(os.path.join(out_path, '{}.png').format(str(i) + '_' +class1 + '_' + class2))

    print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))