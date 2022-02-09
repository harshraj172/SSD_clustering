# -*- coding: utf-8 -*-
"""

Title: torch utilities.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko
"""
import numpy as np
import torch

def initSeeds(seed=1):
	print(f"initSeeds({seed})")
	random.seed(seed)
	torch.manual_seed(seed) 	#turn on this 2 lines when torch is being used
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)


def get_cuda(cudadevice='cuda:0'):
	""" return the best Cuda device """
	devid = cudadevice
	#print ('Current cuda device ', devid, torch.cuda.get_device_name(devid))
	#device = 'cuda:0'	#most of the time torch choose the right CUDA device
	return torch.device(devid)		#use this device object instead of the device string

def dumpModelSize(model, details=True):
	total = sum(p.numel() for p in model.parameters())
	if details:
		for name, param in model.named_parameters():
			if param.requires_grad:
				num_params = sum(p.numel() for p in param)
				print(f"name: {name}, num params: {num_params} ({(num_params/total) *100 :.2f}%)")

	print(f"total params: {total}, ", end='')
	print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def downloadVOC(save_path=args.save_path, year=args.year, download=args.download):
    """downloads the PascalVOC Dataset"""
    datasets.VOCDetection(root=save_path, year=year, download=download, transform=transforms.ToTensor())

def transformIMG(imgsize=args.imgsize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Resize the raw image, Normalize it"""
    tsfm = transforms.Compose([
      transforms.Resize([imgsize, imgsize]),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
    return tsfm

def SampleFromData(img_folder_path, n:int):
    """Sample img path from the full list of images"""
    imgFile_names = []
    for file_ in os.listdir(img_folder_path):
    imgFile_names.append(file_)

    imgFile_names.sort()

    # return random.sample(imgFile_names, n)
    return imgFile_names[:n]

def readURL(url):
    resp = requests.get(url)
    data = json.loads(resp.text)
    return data
