# SSD-with-Clustering

## How To Use:

> git clone https://github.com/harshraj172/SSD_clustering.git
> 
> python /content/SSD_clustering/main.py --img_folder_path /content/data/VOCdevkit/VOC2012/JPEGImages --annotation_folder_path /content/data/VOCdevkit/VOC2012/Annotations --cluster_visualization False


### TO DO:
1) Make the process Deterministic.
   Possible Reasons of Non-Determinism:
   > os.listdir() - Yes
   > 
   > Feature Vector - Yes
   > 
   > TSNE
   > 
   > Mini-Batch KMeans - No

2) Train with more data.

3) Study better Clustering.

### Related Papers:

https://antonylam.github.io/papers/Adaptive_Spatial-Spectral_Dictionary_Learning_for_Hyperspectral_Image_Restoration.pdf

https://arxiv.org/abs/1811.01753
