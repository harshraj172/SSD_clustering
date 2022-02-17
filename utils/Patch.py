import numpy as np
from collections import namedtuple
from skimage import segmentation

class patch:

    def __init__(self, imgs, n_segments, compactness):
        self.imgs = imgs   
        self.n_segments = n_segments
        self.compactness = compactness

    def get_labels(self, img, n_segments, compactness):
        labels = segmentation.slic(img, n_segments, compactness)
        return labels

    def get_patches(self):
        all_patches = []
        img_patch = namedtuple('img_patch', ['img', 'patch'])
        for img in self.imgs:
            labels = self.get_labels(img, self.n_segments, self.compactness)
            patches = []

            for id in np.unique(labels):
                img_copy = img.copy()
                mask = np.zeros(labels.shape)
                mask[labels!=id] = 1
                img_copy[mask==1] = 0
                patch = img_patch(img, img_copy)
                patches.append(patch)

            all_patches.extend(patches)
        
        return all_patches
