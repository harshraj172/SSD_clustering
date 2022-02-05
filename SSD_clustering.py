#!/usr/bin/env python
# coding: utf-8

# # Project
# * **Aim:** To validate the performance of **Single-Shot-Detection** when the data is clustered and each cluster is used to train a separate model. 
# * **Process overview:**
#     1. Download PascalVOC Dataset of the year 2012(you can download others). 
#     1. Use Pretrained *VGG-16 model* to get the feature vector of images in the data.
#     2. Cluster the images based on their feature vectors by MiniBatchKMeans(to avoid RAM Crash).
#     3. Initialize num_clusters no. of SSD models, training each with a different set of data(data from each clusters).
#     4. Train a SSD model with the whole data.
#     5. Validate the difference in performance between the two approaches
# 

# In[1]:


#get_ipython().system(' git clone https://github.com/harshraj172/SSD_clustering.git')


# In[39]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xml.etree.ElementTree as ET

# For commands
import os
import json
import requests
#os.chdir('/content/')
import time
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# For visualization
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import cv2
import imageio as io
from pylab import *
from sklearn.manifold import TSNE

#For model performance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import joblib
from sklearn.cluster import MiniBatchKMeans

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from torchvision import datasets

# For array manipulation
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import os
import random
from math import sqrt


# In[40]:


from SSD_clustering.data.PascalVOC.Dataset import SSDDataset
from SSD_clustering.utils.utils import *
from SSD_clustering.utils import AuxiliaryConvolutions, PredictionConvolutions, Loss
from SSD_clustering.model import ssd, base_model
from SSD_clustering.utils.torchutils import *

# In[41]:

# In[44]:


def downloadVOC(save_path='./data/', year='2012', download=True):
  """downloads the PascalVOC Dataset"""
  datasets.VOCDetection(root=save_path, year=year, download=download, transform=transforms.ToTensor())


# In[45]:


def transformIMG(imgsize=300, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
  """Resize the raw image, Normalize it"""
  tsfm = transforms.Compose([
      transforms.Resize([imgsize, imgsize]),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
  ])
  return tsfm


# In[46]:


def SampleFromData(img_folder_path, n:int):
  """Sample img path from the full list of images"""
  imgFile_names = []
  for file_ in os.listdir(img_folder_path):
    imgFile_names.append(file_)
  
  # return random.sample(imgFile_names, n)
  return imgFile_names[:n]


# In[47]:


def readURL(url):
  resp = requests.get(url)
  data = json.loads(resp.text)
  return data


# **Auxiliary Convolutions**
# 
# It is used as the additional transformation of the feature vector generated through base model which is then used in concatenation with the former.

# In[48]:


def extractFeatures(
    img,
    model,
    model_name='vgg16',
    method='m1',
    ):
    """
  Different ways for feature extraction to be used in Clustering
  Arguments:
  img((int, int, int)): batch of images(batch_size, imgsize, imgsize)
  model: loaded model for feature extraction
  model_name(str, optional): the model to use for feature extraction
  method(str, optional): different methods tried for feature extraction
                         m1-> the features are extracted in the same way as is done by the base_model in SSD
                         m2-> takes the last layer output of the model, does avgPooling and passes through 
                              the first layer of the classifier architecture
  """

    if model_name == 'vgg16':

        if method == 'm1':
            aux_convs =                 AuxiliaryConvolutions.AuxiliaryConvolutions().to(device)
            rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1,
                    1)).to(device)  # there are 512 channels in conv4_3_feats
            (conv4_3_feats, conv7_feats) = model(img)

            # Rescale conv4_3 after L2 norm

            norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
            conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
            conv4_3_feats = conv4_3_feats * rescale_factors  # (N, 512, 38, 38)

            # (PyTorch autobroadcasts singleton dimensions during arithmetic)

            # Run auxiliary convolutions (higher level feature map generators)

            (conv8_2_feats, conv9_2_feats, conv10_2_feats,
             conv11_2_feats) = aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

            # flatten feature vectors obtained at different layers

            conv4_3_feats = torch.flatten(conv4_3_feats, start_dim=1)
            conv7_feats = torch.flatten(conv7_feats, start_dim=1)
            conv8_2_feats = torch.flatten(conv8_2_feats, start_dim=1)
            conv9_2_feats = torch.flatten(conv9_2_feats, start_dim=1)
            conv10_2_feats = torch.flatten(conv10_2_feats, start_dim=1)
            conv11_2_feats = torch.flatten(conv11_2_feats, start_dim=1)

            # Concatenate the feature vectors to obtain a final feature representation of the image

            x = torch.cat([
                conv4_3_feats,
                conv7_feats,
                conv8_2_feats,
                conv9_2_feats,
                conv10_2_feats,
                conv11_2_feats,
                ], dim=1)
             
        elif method == 'm2':

            # Get features part of the network

            model_features = model.features

            x = model_features(img)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier[0](x)  # only first classifier layer
    return x


# In[49]:


def cluster(X, n_clusters, algo='kmeans'):
    if algo=='kmeans':
        kmeans = MiniBatchKMeans(n_clusters, random_state=0, batch_size=128).fit(X)
        return kmeans.labels_, kmeans.cluster_centers_ 


# In[50]:


def reduce_dim(X, method='TSNE', dim=2):
    if method=='TSNE':
        transform = TSNE
        trans = transform(n_components=dim) 
        Xreduced = trans.fit_transform(X) 
    return Xreduced


# In[51]:


def plot_(x,y1,y2,row,col,ind,title,xlabel,ylabel,label,isimage=False,color='b'):

    """
    This function is used for plotting images and graphs (Visualization of end results of model training)
    Arguments:
    x - (np.ndarray or list) - an image array
    y1 - (list) - for plotting graph on left side.
    y2 - (list) - for plotting graph on right side.
    row - (int) - row number of subplot 
    col - (int) - column number of subplot
    ind - (int) - index number of subplot
    title - (string) - title of the plot 
    xlabel - (list) - labels of x axis
    ylabel - (list) - labels of y axis
    label - (string) - for adding legend in the plot
    isimage - (boolean) - True in case of image else False
    color - (char) - color of the plot (prefered green for training and red for testing).
    """
    
    plt.subplot(row,col,ind)
    if isimage:
        plt.imshow(x)
        plt.title(title)
        plt.axis('off')
    else:
        plt.plot(y1,label=label,color='g'); plt.scatter(x,y1,color='g')
        if y2!='': plt.plot(y2,color=color,label='validation'); plt.scatter(x,y2,color=color)
        plt.grid()
        plt.legend()
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)


# In[52]:


def ShowClusterIMG(img_folder_path, img_file_paths, clusterID, cluster_labels, n_images=5, save_img=False):
  iter=0
  plt.figure(figsize=(13,3))
  for i,iterator in enumerate(cluster_labels):
      if iterator == clusterID:
          img = cv2.imread(img_folder_path+'/'+img_file_paths[i])
          img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
          plot_(img,"","",1,n_images,iter+1,"cluster="+str(clusterID),"","","",True)
          iter+=1
      if iter>=n_images: break
  if save_img:
    plt.savefig(f'clustered{clusterID}_images.png', bbox_inches='tight')
  plt.show()


# In[53]:


def train(model, criterion, optimizer, train_dl, valid_dl, EPOCH, print_feq):

    for epoch in range(1, EPOCH + 1):
        model.train()
        train_loss = []

        for step, (img, boxes, labels) in enumerate(train_dl):
            time_1 = time.time()
            img = img.cuda()
            
            # boxes = torch.cat((boxes), dim=0)
            boxes = [box.cuda() for box in boxes]
            # labels = torch.cat((labels), dim=0)
            labels = [label.cuda() for label in labels]

            pred_loc, pred_sco = model(img)

            loss = criterion(pred_loc, pred_sco, boxes, labels)

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # losses.update(loss.item(), images.size(0))
            train_loss.append(loss.item())
            if step % print_feq == 0:
                print(
                    "epoch:",
                    epoch,
                    "\tstep:",
                    step + 1,
                    "/",
                    len(train_dl) + 1,
                    "\ttrain loss:",
                    "{:.4f}".format(loss.item()),
                    "\ttime:",
                    "{:.4f}".format((time.time() - time_1) * print_feq),
                    "s",
                )

        model.eval()
        valid_loss = []
        for step, (img, boxes, labels) in enumerate(tqdm(valid_dl)):
            img = img.cuda()
            boxes = [box.cuda() for box in boxes]
            labels = [label.cuda() for label in labels]
            pred_loc, pred_sco = model(img)
            loss = criterion(pred_loc, pred_sco, boxes, labels)
            valid_loss.append(loss.item())

        print(
            "epoch:",
            epoch,
            "/",
            EPOCH + 1,
            "\ttrain loss:",
            "{:.4f}".format(np.mean(train_loss)),
            "\tvalid loss:",
            "{:.4f}".format(np.mean(valid_loss)),
        )

    return valid_loss


# #MAIN

# In[56]:


def main(
    download=True,
    n_clusters=22,
    n_data=500,
    method="m1",
    img_folder_path="/content/data/VOCdevkit/VOC2012/JPEGImages",
    annotation_folder_path="/content/data/VOCdevkit/VOC2012/Annotations",
    cluster_visualization=False,
    ToTrain=True,
    TrainWithClustering=True,
    min_cluster_size = 5,
    split_size=0.8,
    n_classes=21,
    EPOCH=5,
    print_feq=100,
    LR=1e-3,
    BS=4,
    momentum=0.9,
    weight_decay=5e-4,
):
    """
    main function 
    
    Arguments:
        download - (bool, default=True) - To download the data
        n_clusters - (int, default=22) - No. of clusters
        n_data - (int, default=500) -  No. of data to sample from whole set
        method - (str, default='m1') -  different methods for feature extraction
        img_folder_path - (str) -  path to directory of images
        annotation_folder_path - (str) - path to directory of annotations
        cluster_visualization - (bool, default=False): to perform visualization of clusters
        train - (bool, default=True) - to perform training
        TrainWithClustering - (bool, default=True) - to train with partitioning the data  
        min_cluster_size - (int, default=5) - minimum cluster size for training to be performed
        split_size - (float, default=0.8) - split % for train & val
        n_classes - (int, default=21) - No. of classes in the data labels
        EPOCH - (int, default=5) - no. of epochs 
        LR - (float, default=1e-3) - Learning Rate
        BS - (float, default=4) - batch size 
        momentum - (float, default=0.9) - momentum while optimizing through Adam
        weight_decay - (float, default=5e-4) - weight decay 
    """

    # --------------DATASET PREP--------------
    downloadVOC(download=download)  # downloads the VOCDataset

    # label map dict
    label_map = readURL(
        "https://raw.githubusercontent.com/harshraj172/SSD_clustering/main/data/PascalVOC/label_map.json"
    )
    rev_label_map = readURL(
        "https://raw.githubusercontent.com/harshraj172/SSD_clustering/main/data/PascalVOC/rev_label_map.json"
    )

    # sample n file(imgs) from the Dataset
    img_file_paths = SampleFromData(img_folder_path=img_folder_path, n=n_data)

    # Dataset & Dataloader
    ds = SSDDataset(
        file_folder=img_file_paths,
        img_folder_path=img_folder_path,
        annotation_folder_path=annotation_folder_path,
        label_map=label_map,
        transform=transformIMG(),
    )
    dl = DataLoader(ds, batch_size=BS, collate_fn=ds.collate_fn)


    # --------------EXTRACT FEATURES--------------
    X_encoded = []
    if method == "m1":
        model = base_model.VGGBase().to(device)
    elif method == "m2":
        model = models.vgg16(pretrained=True).to(device)

    for i, (img, boxes, label) in enumerate(dl):
        img = img.to(device)
        x = extractFeatures(img, model)
        X_encoded.extend(x.cpu().detach().numpy())
    X_encoded = np.array(X_encoded)


    # --------------CLUSTERING & VISUALIZATION--------------
    X_reduced = reduce_dim(X_encoded)  # reduce dim
    cluster_labels, centroids = cluster(
        X=X_encoded, n_clusters=n_clusters
    )  # clustering

    # Visualization
    if cluster_visualization:
        print("if Number of clusters: " + str(n_clusters))
        print("-------------------------------")
        print("-------------------------------")

        # Clustering
        cluster_labels, centroids = cluster(X=X_encoded, n_clusters=n_clusters)

        # Scatter Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 1, 1)
        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=cluster_labels.astype(float),
            s=50,
            alpha=0.5,
        )
        # plt.scatter(centroids[:, 0], centroids[:, 1], c=None, s=50)
        plt.show()

        # Show atmost n_images images per cluster
        for row in range(n_clusters):
            ShowClusterIMG(
                img_folder_path=img_folder_path,
                img_file_paths=img_file_paths,
                clusterID=row,
                cluster_labels=cluster_labels,
            )
            print()


    # --------------TRAIN SSD--------------
    if ToTrain:

        img_file_paths = np.array(img_file_paths)
        cluster_labels = np.array(cluster_labels)
        valid_loss_lst = []
      
        if TrainWithClustering:

            # define the list models with each cluster data passed to different model
            model_list = nn.ModuleList(
                [ssd.SSD(n_classes).to(device) for i in range(n_clusters)]
            )

            for cluster_id in np.unique(cluster_labels):

                img_name = img_file_paths[(cluster_labels == cluster_id)]
                print(f"Number of images in cluster {cluster_id} = {len(img_name)}")

                # if number of data in cluster is more than min_cluster_size
                if len(img_name) > min_cluster_size:

                    model = model_list[cluster_id]
                    criterion = Loss.MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(
                        device
                    )
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=LR,
                        momentum=momentum,
                        weight_decay=weight_decay,
                    )

                    # partition data after clustering
                    train_img_name = img_name[: int(len(img_name) * split_size)]
                    valid_img_name = img_name[int(len(img_name) * split_size) :]

                    # train dataset
                    train_ds = SSDDataset(
                        train_img_name,
                        img_folder_path=img_folder_path,
                        annotation_folder_path=annotation_folder_path,
                        label_map=label_map,
                        transform=transformIMG(),
                    )
                    train_dl = DataLoader(
                        train_ds, batch_size=BS, collate_fn=train_ds.collate_fn
                    )

                    # valid dataset
                    valid_ds = SSDDataset(
                        valid_img_name,
                        img_folder_path=img_folder_path,
                        annotation_folder_path=annotation_folder_path,
                        label_map=label_map,
                        transform=transformIMG(),
                    )
                    valid_dl = DataLoader(
                        valid_ds, batch_size=BS, collate_fn=valid_ds.collate_fn
                    )

                    # start training
                    valid_loss_lst.append(train(
                        model, criterion, optimizer, train_dl, valid_dl, EPOCH, print_feq
                    ))

                    print()
                    print(f"Finished Training for model number {cluster_id}")
                    print(f"-------------------------------------------------")
                    print(f"-------------------------------------------------")
                    print()
                    print()
            print(f"**Average Valid Loss = {np.mean(valid_loss_lst)}**")    

        else:
            model = ssd.SSD(n_classes).to(device)
            criterion = Loss.MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=LR,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            # partition data after clustering
            train_img_name = img_file_paths[: int(len(img_file_paths) * split_size)]
            valid_img_name = img_file_paths[int(len(img_file_paths) * split_size) :]

            # train dataset
            train_ds = SSDDataset(
                train_img_name,
                img_folder_path=img_folder_path,
                annotation_folder_path=annotation_folder_path,
                label_map=label_map,
                transform=transformIMG(),
            )
            train_dl = DataLoader(
                train_ds, batch_size=BS, collate_fn=train_ds.collate_fn
            )

            # valid dataset
            valid_ds = SSDDataset(
                valid_img_name,
                img_folder_path=img_folder_path,
                annotation_folder_path=annotation_folder_path,
                label_map=label_map,
                transform=transformIMG(),
            )
            valid_dl = DataLoader(
                valid_ds, batch_size=BS, collate_fn=valid_ds.collate_fn
            )

            # start training
            valid_loss_lst = train(
                model, criterion, optimizer, train_dl, valid_dl, EPOCH, print_feq
            )
            print()
            print(f"Finished Training for model")
            print(f"-------------------------------------------------")
            print(f"-------------------------------------------------")
            print()
            print()


# In[ ]:


# __name__
if __name__=="__main__":
  device = onceInit(kCUDA=True)  # get the device and init random seed

  # run 10 iter to track non-deterministism
  for _ in range(10):
    main(download=True, cluster_visualization=True, n_data=300)

