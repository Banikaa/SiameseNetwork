import os
import glob
import time

import numpy as np
from PIL import Image
import pandas as pd
import math

import pickle

import torch
from torchvision import transforms



# i1 = pickle.load(open('/Users/banika/H-UNIQUE/SIAMESE/extra/indices1.pkl', 'rb'))
# i2 = pickle.load(open('/Users/banika/H-UNIQUE/SIAMESE/extra/indices2.pkl', 'rb'))
# print(i1)


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, tr_stage, path, shuffle_pairs=True, augment=False):

        print('Dataset init')

        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.

            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images


        '''
        self.path = path
        self.tr_stage = tr_stage
        self.feed_shape = [3, 224, 224]
        self.shuffle_pairs = shuffle_pairs

        self.unique_pairs = []

        self.sname = 'pairs_major_knuckle_only'

        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])

        self.create_pairs()


    def is_unique(self, p):
        if p[0] > p[1]:
            return self.is_unique((p[1], p[0]))
        if p in self.unique_pairs:
            return False
        else:
            self.unique_pairs.append(p)
            return True


    # this method returns 2 lists: indeices1 and indices2
    # each list contains the indices of the images that will form a pair
    # i.e. indices1[i] and indices2[i] will form a pair
    def create_pairs(self):

        # if os.path.exists(self.tr_stage + '_' + self.sname + '.pkl'):
        #     self.pairs = pickle.load(open(self.tr_stage + '_' + self.sname + '.pkl', 'rb'))
        #     return

        self.image_paths = glob.glob(os.path.join(self.path, "*/*.png"))


        self.image_classes = []
        self.class_indices = {}

        # create a dictionary with all the classes and the img in them
        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []

        self.list1 = []
        self.list2 = []

        match_nb = 1
        non_match_nb = 1


        def get_anatomy_from_fname(fname):
            return fname.split('_')[2].split('.')[0]

        for img_path in self.image_paths:
            match_class = img_path.split(os.path.sep)[-2]
            img_fname = img_path.split(os.path.sep)[-1]

            img_type = get_anatomy_from_fname(img_fname)

            matches = os.listdir(os.path.join(self.path, match_class))


            for i in range(match_nb):
                match = np.random.choice(matches)
                match_type = get_anatomy_from_fname(match)

                while match_type != img_type and match == img_fname:
                    match = np.random.choice(matches)
                    match_type = get_anatomy_from_fname(match)

                self.list1.append(img_path)
                self.list2.append(os.path.join(self.path, match_class, match))

            for i in range(non_match_nb):

                non_match_pick = np.random.choice([match, img_fname])
                non_match_class = np.random.choice(list(set(self.class_indices.keys()) - {match_class}))
                non_match = np.random.choice(os.listdir(os.path.join(self.path, non_match_class)))
                non_match_type = get_anatomy_from_fname(non_match)

                while non_match_type != img_type and non_match != img_fname:
                    non_match_class = np.random.choice(list(set(self.class_indices.keys()) - {match_class}))
                    non_match = np.random.choice(os.listdir(os.path.join(self.path, non_match_class)))
                    non_match_type = get_anatomy_from_fname(non_match)

                # print('non_match_type', non_match_type, 'img_type', img_type)
                self.list1.append(os.path.join(self.path, match_class, non_match_pick))
                self.list2.append(os.path.join(self.path, non_match_class, non_match))

        self.pairs = list(zip(self.list1, self.list2))

        print(len(self.pairs))

        if self.shuffle_pairs:
            np.random.shuffle(self.pairs)

        pickle.dump(self.pairs, open(self.tr_stage + '_' + self.sname + '.pkl', 'wb'))


    def __iter__(self):
        self.create_pairs()

        for [idx, idx2] in self.pairs:
            image_path1 = idx
            image_path2 = idx2

            class1 = image_path1.split(os.path.sep)[-2]
            class2 = image_path2.split(os.path.sep)[-2]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            if self.transform:
                image1 = self.transform(image1).float()
            image2 = self.transform(image2).float()

            yield (image1, image2), torch.FloatTensor([class1==class2]), (class1, class2), (image_path1, image_path2)
        
    def __len__(self):
        return len(self.image_paths)
