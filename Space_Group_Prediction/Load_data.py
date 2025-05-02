import os
import json
import torchvision
from skimage import io
from Utils import gray2rgb
from torch.utils.data import DataLoader, Dataset


class Dataset_SG(Dataset):
    def __init__(self, json_paths, image_folder, transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        gray2rgb
    ])):
        self.paths = json_paths
        self.folder = image_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        folder = self.folder
        path = self.paths[index]
        file = open(path)
        data = json.load(file)
        SG = data['space_group'] - 1
        ID = data['ID']
        im_path = os.path.join(folder, ID + '.png')
        image = io.imread(im_path)
        x = self.transforms(image)
        return x, SG
    
def SG_Dataloaders(batch_size, json_paths_fold1, json_paths_fold2, image_folder, json_paths_test = None):
    if json_paths_test != None:
        dataset = Dataset_SG(json_paths_test, image_folder)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
        return dataloader
    else:
        dataset_fold1 = Dataset_SG(json_paths_fold1, image_folder)
        dataset_fold2 = Dataset_SG(json_paths_fold2, image_folder)
        dataloader_fold1 = DataLoader(dataset_fold1, batch_size = batch_size, shuffle=True)
        dataloader_fold2 = DataLoader(dataset_fold2, batch_size = batch_size, shuffle=True)
        return dataloader_fold1, dataloader_fold2