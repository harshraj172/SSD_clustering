import argparse
import numpy as np 
import pandas as pd 
import pandas.util.testing as tm
import random
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, KMeans

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from torchvision import datasets

from data.PascalVOC.Dataset import SSDDataset
from utils.utils import *
from utils.torchutils import *
from utils import AuxiliaryConvolutions, PredictionConvolutions, Loss
from model import ssd, base_model
from train import train
import random

def_data_folder = "./data"
def_img_folder = f"{def_data_folder}/VOCdevkit/VOC2012/JPEGImages" 
def_annotation_folder = f"{def_data_folder}/VOCdevkit/VOC2012/Annotations"

# Arguments
def myArgs():
    parser = argparse.ArgumentParser()

    # init 
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cudadevice', type=str, default='cuda:0')

    # Data
    parser.add_argument('--save_path', type=str, default='./data/')
    parser.add_argument('--year', type=str, default='2012')
    parser.add_argument('--data', type=str, default=def_data_folder, metavar="data folder")
    parser.add_argument('--img_folder_path', type=str, default=def_img_folder)
    parser.add_argument('--annotation_folder_path', type=str, default=def_annotation_folder)
    parser.add_argument('--download', type=eval, default=True)
    parser.add_argument('--n_classes', type=int, default=21)
    parser.add_argument('--imgsize', type=int, default=300)
    parser.add_argument('--n_data', type=int, default=500)

    # Clustering
    parser.add_argument('--TrainWithClustering', type=eval, default=True)
    parser.add_argument('--model_name', type=str, default='vgg16')
    parser.add_argument('--method', type=str, default='m2')
    parser.add_argument('--clustering_algo', type=str, default='kmeans')
    parser.add_argument('--reduce_dim_method', type=str, default='TSNE') 
    parser.add_argument('--cluster_visualization', type=eval, default=True)
    parser.add_argument('--n_clusters', type=int, default=22)
    parser.add_argument('--save_img', type=bool, default=False) 

    # Training
    parser.add_argument('--min_cluster_size', type=int, default=5)
    parser.add_argument('--split_size', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--print_feq', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    return args

args = myArgs()

def extractFeatures(
    imgs,
    model,
    model_name=args.model_name,
    method=args.method,
    ):
    """
  Different ways for feature extraction to be used in Clustering
  Arguments:
  imgs((int, int, int)): batch of images(batch_size, imgsize, imgsize)
  model: loaded model for feature extraction
  model_name(str, optional): the model to use for feature extraction
  method(str, optional): different methods tried for feature extraction
                         m1-> the features are extracted in the same way as is done by the base_model in SSD
                         m2-> takes the last layer output of the model, does avgPooling and passes through 
                              the first layer of the classifier architecture
  """

    if model_name == 'vgg16':

        if method == 'm1':
            aux_convs = \
                AuxiliaryConvolutions.AuxiliaryConvolutions().to(device)
            rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1,
                    1)).to(device)  # there are 512 channels in conv4_3_feats
            (conv4_3_feats, conv7_feats) = model(imgs)

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

            x = model_features(imgs)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier[0](x)  # only first classifier layer
    return x


def cluster(X, n_clusters, algo=args.clustering_algo):
    if algo=='kmeans':
        kmeans = MiniBatchKMeans(n_clusters, random_state=0, batch_size=128).fit(X)
        return kmeans.labels_, kmeans.cluster_centers_ 

def reduce_dim(X, method=args.reduce_dim_method, dim=2):
    if method=='TSNE':
        transform = TSNE
        trans = transform(n_components=dim) 
        Xreduced = trans.fit_transform(X) 
    return Xreduced


def data_prep(download, img_folder_path, annotation_folder_path, 
			 n_data=args.n_data, method=args.method):
  """
  download - (bool, default=True) - To download the data
  img_folder_path - (str) -  path to directory of images
  annotation_folder_path - (str) - path to directory of annotations
  n_data - (int, default=500) -  No. of data to sample from whole set
  docluster - (bool, default=True) - to train with partitioning the data
  method - (str, default='m1') -  different methods for feature extraction
  """
  # --------------DATASET PREP--------------
  downloadVOC(args.save_path, year=args.year, download=download)  # downloads the VOCDataset

  # label map dict
  label_map = readURL(
      "https://raw.githubusercontent.com/harshraj172/SSD_clustering/main/data/PascalVOC/label_map.json"
  )

  # sample n file(imgs) from the Dataset
  img_file_paths = SampleFromData(img_folder_path=img_folder_path, n=n_data)  # constant order

  # Dataset & Dataloader
  ds = SSDDataset(
      file_folder=img_file_paths,
      img_folder_path=img_folder_path,
      annotation_folder_path=annotation_folder_path,
      label_map=label_map,
      transform=transformIMG(args.imgsize),
  )
  dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate_fn)

  # --------------EXTRACT FEATURES--------------
  X_encoded = []
  if method == "m1":
      model = base_model.VGGBase().to(device)
  elif method == "m2":
      model = models.vgg16(pretrained=True).to(device)

  class_labels = []
  for i, (imgs, boxes, labels) in enumerate(dl):
      class_labels.extend(
          [float(max(set(label), key=list(label).count)) for label in labels]
      )  # among various labels take label with highest frequency
      imgs = imgs.to(device)
      x = extractFeatures(imgs=imgs, model=model, method=method)
      X_encoded.extend(x.cpu().detach().numpy())

  class_labels = np.array(class_labels)
  X_encoded = np.array(X_encoded)

  return img_file_paths, label_map, class_labels, X_encoded


def clusterANDvisual(X_encoded, img_file_paths, img_folder_path, 
					n_clusters=args.n_clusters, cluster_visualization=args.cluster_visualization):

  """
  img_folder_path - (str) -  path to directory of images
  """

  # --------------CLUSTERING & VISUALIZATION--------------
  X_reduced = reduce_dim(X_encoded)  # reduce dim
  cluster_labels, centroids = cluster(
      X=X_encoded, n_clusters=n_clusters
  )  # clustering
  cluster_labels = np.array(cluster_labels)

  # Visualization
  if cluster_visualization:
      print("if Number of clusters: " + str(n_clusters))
      print("-------------------------------")
      print("-------------------------------")

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
      plt.scatter(centroids[:, 0], centroids[:, 1], c=None, s=50)
      plt.show()

      # Show atmost n_images images per cluster
      for row in range(n_clusters):
          ShowClusterIMG(
              img_folder_path=img_folder_path,
              img_file_paths=img_file_paths,
              clusterID=row,
              cluster_labels=cluster_labels,
	      save_img=args.save_img
          )
          print()
  return cluster_labels


def trainSSD(
    device,
    img_file_paths, 
    class_labels,
    cluster_labels,
    label_map,
    img_folder_path,
    annotation_folder_path,
    method=args.method,
    n_clusters=args.n_clusters,
    TrainWithClustering=args.TrainWithClustering,
    min_cluster_size=args.min_cluster_size,
    split_size=args.split_size,
    n_classes=args.n_classes,
    EPOCH=args.epochs,
    print_feq=args.print_feq,
    LR=args.learning_rate,
    BS=args.batch_size,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
): 

  # --------------TRAIN SSD--------------
  img_file_paths = np.array(img_file_paths)
  valid_loss_per_clus = []  

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
                  transform=transformIMG(args.imgsize),
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
                  transform=transformIMG(args.imgsize),
              )
              valid_dl = DataLoader(
                  valid_ds, batch_size=BS, collate_fn=valid_ds.collate_fn
              )

              # start training
              valid_loss_per_clus.append(
                  train(
                      model,
                      criterion,
                      optimizer,
                      train_dl,
                      valid_dl,
                      EPOCH,
                      print_feq,
                      device
                  )
              )

              print()
              print(f"Finished Training for model number {cluster_id}")
              print(f"-------------------------------------------------")
              print(f"-------------------------------------------------")
              print()
              print()

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
          transform=transformIMG(args.imgsize),
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
          transform=transformIMG(args.imgsize),
      )
      valid_dl = DataLoader(
          valid_ds, batch_size=BS, collate_fn=valid_ds.collate_fn
      )

      # start training
      valid_loss_per_clus.append(train(
          model, criterion, optimizer, train_dl, valid_dl, EPOCH, print_feq, device
      ))

      print()
      print(f"Finished Training for model")
      print(f"-------------------------------------------------")
      print(f"-------------------------------------------------")
      print()
      print()

  return valid_loss_per_clus


# __name__
if __name__=="__main__":
    device = onceInit(kCUDA=True, cudadevice=args.cudadevice, seed=args.seed)  # get the device and init random seed
    os.makedirs(os.path.dirname(args.img_folder_path), exist_ok=True)  
    os.makedirs(os.path.dirname(args.annotation_folder_path), exist_ok=True)  

    img_file_paths, label_map, class_labels, X_encoded = data_prep(
        download=args.download, 
        img_folder_path=args.img_folder_path, 
        annotation_folder_path=args.annotation_folder_path, 
        method=args.method
    )
      
    cluster_labels = clusterANDvisual(X_encoded=X_encoded, img_file_paths=img_file_paths, img_folder_path=args.img_folder_path, n_clusters=args.n_clusters, cluster_visualization=args.cluster_visualization)
      
    del X_encoded

    valid_loss_per_clus = trainSSD(
        device=device,
        img_file_paths=img_file_paths,
        class_labels=class_labels, 
        cluster_labels=cluster_labels,
        label_map=label_map,
        img_folder_path=args.img_folder_path,
        annotation_folder_path=args.annotation_folder_path,
        TrainWithClustering=args.TrainWithClustering
    )
      
    if args.TrainWithClustering:
        print("class_labels")
        print(class_labels)

        print("cluster_labels") 
        print(cluster_labels)

    print("valid_loss_per_clus")
    print(valid_loss_per_clus)
