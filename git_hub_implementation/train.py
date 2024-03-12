import os
import argparse
import csv
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.cuda.empty_cache()

from siamese_network import SiameseNetwork
from dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    train_path = '/media/internal/DATA/FYPStudents/Andrei-Internship/SIAMESE/data/dataset_1_kajor_knuckle/train'
    # train_path = '/media/internal/DATA/FYPStudents/Andrei-Internship/SIAMESE/data/index_train_multic/'
    # val_path = '/media/internal/DATA/FYPStudents/Andrei-Internship/SIAMESE/data/dataset_1_index_cpp_only_fingernail/val'
    out_path = './outputs_14'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    train_csv =  open(os.path.join(out_path, 'train.csv'), 'w')
    val_csv = open(os.path.join(out_path, 'val.csv'), 'w')
    csv_t_extra = 'epoch,train_loss,train_acc,\n'
    t_writer = csv.writer(train_csv)
    t_writer.writerow(['epoch', 'train_loss', 'train_acc'])
    v_writer = csv.writer(val_csv)
    csv_v_extra = 'epoch,val_loss,val_acc,\n'
    v_writer.writerow(['epoch', 'val_loss', 'val_acc'])

    # parser.add_argument(
    #     '--train_path',
    #     type=str,
    #     help="Path to directory containing training dataset.",
    #     required=True
    # )
    # parser.add_argument(
    #     '--val_path',
    #     type=str,
    #     help="Path to directory containing validation dataset.",
    #     required=True
    # )
    # parser.add_argument(
    #     '-o',
    #     '--out_path',
    #     type=str,
    #     help="Path for outputting model weights and tensorboard summary.",
    #     required=True
    # )
    parser.add_argument(
        '-b',
        '--backbone',
        type=str,
        help="Network backbone from torchvision.models to be used in the siamese network.",
        default="resnet50"
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help="Learning Rate",
        default=1e-4
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Number of epochs to train",
        default=50
    )
    parser.add_argument(
        '-s',
        '--save_after',
        type=int,
        help="Model checkpoint is saved after each specified number of epochs.",
        default=5
    )

    args = parser.parse_args()

    # os.makedirs(args.out_path, exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train/ val dataset creation
    train_dataset   = Dataset('train', train_path, shuffle_pairs=False, augment=True)
    val_dataset = train_dataset
    # val_dataset     = Dataset('val', val_path, shuffle_pairs=True, augment=False)

    # tran/val dataloader creation
    train_dataloader = DataLoader(train_dataset, batch_size=1, drop_last=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=1)

    model = SiameseNetwork(backbone=args.backbone)
    model.to(device)
    print(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(out_path, "summary"))

    best_val = 100000000000

    # see if same finger same person
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

    for epoch in range(args.epochs):


        print("[{} / {}]".format(epoch, args.epochs))
        model.train()

        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2), y, (class1, class2), (path1, path2) in train_dataloader:

            # print(path1, path2)
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)

            if class1 == class2:
                fname1 = path1[0].split('/')[-1]
                fname2 = path2[0].split('/')[-1]
                match = get_match(fname1, fname2)
                target = torch.Tensor([match]).unsqueeze(1).to(device)
            else:
                target = torch.Tensor([0]).unsqueeze(1).to(device)

            loss = contrastive_loss(prob, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # print(prob.item(), target.item())
            # print(target.item() == (prob.item() > 0.5))
            correct += torch.count_nonzero(target == (prob > 0.5)).item()
            total += len(target)

        train_loss = sum(losses)/len(losses)
        train_acc = correct / total

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        t_writer.writerow([epoch, train_loss, train_acc])
        csv_t_extra += '{},{},{}\n'.format(epoch, train_loss, train_acc)
        #
        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))
        # , correct / total
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2), (path1, path2) in val_dataloader:
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

            # torch.set_grad_enabled(False)
            losses.append(loss.item())
            correct += torch.count_nonzero(target == (prob > 0.5)).item()
            total += len(target)

        val_loss = sum(losses)/max(1, len(losses))
        val_acc = correct / total

        v_writer.writerow([epoch, val_loss, val_acc])
        csv_v_extra += '{},{},{}\n'.format(epoch, val_loss, val_acc)

        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "best.pth")
            )

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "epoch_{}.pth".format(epoch + 1))
            )


    train_csv.close()
    val_csv.close()

    with open(os.path.join(out_path, 'train_extra.csv'), 'w') as f:
        f.write(csv_t_extra)
    with open(os.path.join(out_path, 'val_extra.csv'), 'w') as f:
        f.write(csv_v_extra)

    print("Training Complete!")

    # Save graphs of training and validation loss and accuracy

    # train_df = pd.read_csv(os.path.join(out_path, 'train.csv'), names = ['epoch', 'train_loss', 'train_acc'])
    # val_df = pd.read_csv(os.path.join(out_path, 'val.csv'), names = ['epoch', 'val_loss', 'val_acc'])

    # Load CSV files into pandas dataframes
    file1 = 'train.csv'
    file2 = 'val.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(df1['epoch'], df1['train_loss'], label='train')
    plt.plot(df2['epoch'], df2['val_loss'], label='val')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(df1['epoch'], df1['train_acc'], label='train')
    plt.plot(df2['epoch'], df2['val_acc'], label='val')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig('/Users/banika/Desktop/outputs_9.png')




    # train_df.set_index('epoch')
    # plt.plot(train_df['epoch'], train_df['train_loss'], train_df['train_acc'], label='Train')
    # plt.savefig(os.path.join(out_path, 'train.png'))
    #
    # val_df.set_index('epoch')
    # plt.plot(val_df['epoch'], val_df['val_loss'], val_df['val_acc'], label='Val')
    # plt.savefig(os.path.join(out_path, 'val.png'))

