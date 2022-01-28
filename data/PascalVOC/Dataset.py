from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset

class SSDDataset(Dataset):
    def __init__(self, file_folder, img_folder_path, annotation_folder_path, label_map, imgsize=300, is_test=False, transform=None):
        self.img_folder_path = img_folder_path
        self.annotation_folder_path = annotation_folder_path
        self.file_folder = file_folder
        self.label_map = label_map
        self.imgsize = imgsize
        self.transform = transform
        self.is_test = is_test
        
    def __getitem__(self, idx):
        file_ = self.file_folder[idx]
        img_path = f"{self.img_folder_path}/{file_}"
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        if not self.is_test:
            annotation_path = f"{self.annotation_folder_path}/{file_.split('.')[0]}.xml"

            xy = self.get_xy(annotation_path)
            box = torch.FloatTensor(list(xy))

            new_box = self.box_resize(box, img)
            if self.transform is not None:
                img = self.transform(img)
              
            label = torch.FloatTensor(self.get_label(annotation_path))

            return img, new_box, label
        else:
            return img
    
    def __len__(self):
        return len(self.file_folder)
        
    def get_xy(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        Xmin = [int(float(x.text)) for x in root.findall('./object/bndbox/xmin')] 
        Xmax = [int(float(x.text)) for x in root.findall('./object/bndbox/xmax')]
        Ymin = [int(float(x.text)) for x in root.findall('./object/bndbox/ymin')]
        Ymax = [int(float(x.text)) for x in root.findall('./object/bndbox/ymax')]
        
        ret = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in zip(Xmin, Ymin, Xmax, Ymax)]
        return ret
    
    def get_label(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        label = [self.label_map[x.text] for x in root.findall('./object/name')]
  
        return label

    def show_box(self):
        file_ = random.choice(self.file_folder)
        annotation_path = f"{self.annotation_folder_path}/{file_.split('.')[0]}.xml"
        
        img_box = Image.open(self.img_folder_path + file_)
        with open(annotation_path) as f:
            annotation = f.read()
            
        draw = ImageDraw.Draw(img_box)
        xy = self.get_xy(annotation)
        print('bbox:', xy)
        draw.rectangle(xy=[xy[:2], xy[2:]])
        
        return img_box
        
    def box_resize(self, box, img):
        dims = (self.imgsize, self.imgsize)
        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        new_box = box / old_dims
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        
        return new_box
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  
