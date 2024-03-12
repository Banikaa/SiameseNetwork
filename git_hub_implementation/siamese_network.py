import torch
import torch.nn as nn
import pandas as pd

from torchvision import models


similarity_csv = '/media/internal/DATA/FYPStudents/Andrei-Internship/SIAMESE/AI_tools/11k_annotations_full.csv'
similarity_csv = pd.read_csv(similarity_csv)

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        backbone = 'resnet50'

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models 
        self.backbone = models.__dict__[backbone]( weights='ResNet50_Weights.DEFAULT' , progress=True)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


    def forward(self, img1, img2):

        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # print(feat1.shape)

        # Multiply (element-wise) the feature vectors of the two images together,
        # to generate a combined feature vector representing the similarity between the two.
        # combined_features = feat1 * feat2

        # TODO: cosine distance and l2 distance
        # (feat1*feat2)/(torch.norm(feat1,p=2)*torch.norm(feat2,p=2))

        # TODO: EUCLIDEAN DISTANCE
        euclidean_distance = torch.norm(feat1 - feat2, p=2)
        similarity = 1 / (1 + euclidean_distance)

        # TODO: COSINE SIMILARITY
        # cosine_similarity = torch.nn.functional.cosine_similarity(feat1, feat2, dim=0)
        # similarity = cosine_similarity
        # cosine_distance = 1 - cosine_similarity
        # normalised_cosine_distance = (cosine_distance + 1) / 2
        # similarity = normalised_cosine_distance

        cos_similarity = torch.nn.functional.cosine_similarity(feat1, feat2, dim=1)


        # cos_similarity = 0.5 * (cos_similarity + 1)
        # similarity = cos_similarity
        # print(cos_similarity)

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        # output = self.cls_head(combined_features)
        return similarity