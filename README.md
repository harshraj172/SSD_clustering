# SSD-with-Clustering

## Pseudo-Code:
```
def sample_objects(image):
    """
    Sample out the regions in the image containing 
    objects using foreground/background algrorithm
    Arguments:
        image: a RGB image
    Returns:
        sampled_objects: regions containing objects
    """


def extract_features(image):
    """
    Use pretrained model architecture to 
    get the features vectors for the image
    Arguments:
        image: a RGB image
    Returns:
        feature_vector: features of the image 
                        as a vector
    """


def cluster(feature_vectors):
    """
    cluster the features_vector using a
    unsupervised learning algorithm
    Arguments:
        feature_vectors: feature vector of the images as a matrix
    Retruns:
        cluster_labels: cluster labels for each image
    """

class SSD():
    def __init__(self,):
    self.image = image
    """
    Perform Single-Shot Object Detection
    as described in the paper(https://arxiv.org/abs/1512.02325)
    Arguments:
        image: a RGB image
    Returns:
        bbox_locs: predicted location of bounding boxes
        class_scores: class scores for predicted bounding box
    """


def main(images, n_clusters):
    """
    => sample_objects: For every image in the dataset sample the regions with objects
    => extract_features: Extract features for these object images
    => cluster: cluster these feature vectors to assign the object images labels
    => SSD: initialize a separate SSD model for each cluster
    Arguments:
        images: all images in the dataset 
        n_clusters: no.of clusters in clustering
    """

if __name__=="__main__":
    images = Input
    main(images, n_cluster=22)
```

## How To Use:

#### 1. Clone this Github Repository

#### 2. Use the Below command to call the main.py 

` python main_manny.py --img_folder_path /content/data/VOCdevkit/VOC2012/JPEGImages --annotation_folder_path /content/data/VOCdevkit/VOC2012/Annotations` 
