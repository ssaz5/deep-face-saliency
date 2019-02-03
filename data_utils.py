"""
Created on Sun Jan 21 21:20:32 2018
@author: Monkey-PC

Modified for needs of ConvAge project
"""

"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import pickle as pickle
from torch.utils.data.sampler import Sampler
import scipy.stats as st


class FacePlacedData(data.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_paths = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        else:
            key = int(key)
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)


    def __len__(self):
        return len(self.image_paths)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_path = self.image_paths[index]

        grayscale = transforms.Grayscale()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
        topil = transforms.ToPILImage()
        res = transforms.Resize([240, 320])
        
        img = Image.open(os.path.join(self.root_dir_name, 'face_placed/' + \
                                      img_path))
        img = res(img)        
        #img = grayscale(img)
        img = to_tensor(img)
#         img = norm(img)
        #img = img.squeeze()

        target = np.load(os.path.join(self.root_dir_name, 'face_placed/'+ \
                                        img_path[:-4] + '_gnd.npy'))
        target = np.uint8(target//255)
        target = torch.from_numpy(target)
        target = target.unsqueeze(0)
        target = topil(target)
        target = res(target)
        target = grayscale(target)
        target = to_tensor(target)
        target /= torch.max(target)
        target = target.squeeze()
        #print(torch.max(target), torch.min(target))
        return img, target

class ClusterRandomSampler(Sampler):
    r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for cluster_indices in train_dataset.cluster_indices:
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)       
        
        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)        
        return iter(lst)

    def __len__(self):
        return len(self.data_source)

        return img, target

    
class FaceVAE(data.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_paths = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        else:
            key = int(key)
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)


    def __len__(self):
        return len(self.image_paths)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_path = self.image_paths[index]

        grayscale = transforms.Grayscale()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
        topil = transforms.ToPILImage()
        res = transforms.Resize([128, 128])
        
        img = Image.open(os.path.join(self.root_dir_name, \
                                      img_path)).convert('RGB')
        img = res(img)        
        #img = grayscale(img)
        mask = gkern(img.size[1], img.size[0], 2, 2.5)
        mask = mask/np.max(mask)
        th = (np.mean(mask) - 1*np.std(mask)) 
        mask[mask<th] = 0
        mask[mask>=th] = 1
#         print(mask.shape)
        mask = np.tile(mask,[3,1,1])
        

#         print("Image size:" , mask.shape)
        
        
        img = to_tensor(img).float()
        mask = torch.from_numpy(mask).float()
#         print("Image size:" , img.size())
#         img = norm(img)
      
        #img = img.squeeze()
        
#         print(img, mask)
        
        
        img = torch.mul(img,mask)
        target = img
#         target = to_tensor(mask_img)
#         target.unsqueeze(0)
#         print("MASK SIZE: ", target.size())
        return img, target

def gkern(kernlen_x=21,kernlen_y=21, nsig_x=3, nsig_y=4):
    """Returns a 2D Gaussian kernel array."""

    interval_x = (2*nsig_x+1.)/(kernlen_x)
    interval_y = (2*nsig_y+1.)/(kernlen_y)
    x = np.linspace(-nsig_x-interval_x/2., nsig_x+interval_x/2., kernlen_x+1)
    y = np.linspace(-nsig_y-interval_y/2., nsig_y+interval_y/2., kernlen_y+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.diff(st.norm.cdf(y))
    kernel_raw = np.sqrt(np.outer(kern1d, kern2d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel    


def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Leibler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.astype(np.float64).prod()
    dqv = qv.astype(np.float64).prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))   


def get_indices_from_name(sub_name, indices, indices_subjects, num_images= 5):
    out_indices =  [i for i, j in zip(indices,indices_subjects) if j == sub_name]
    out = np.array([out_indices[i] for i in np.random.randint(len(out_indices),size=num_images)])
    return out

def get_top_subjects(dist, subjects):
    idx = np.argsort(dist)
    for i, id in enumerate(idx):
        if id < 5:
            break
    idx = idx[:i]


    subject_above = [subjects[i-1] for i in np.floor((idx)/5).astype(np.uint16)]
    return len(np.unique(subject_above))+1, len(idx)