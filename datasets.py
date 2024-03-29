from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pdb
from torchvision import datasets, transforms
import numpy as np
class RadarDataset(Dataset):
    def __init__(self, txtpath, transform=None):
        self.file_list = []
        with open(txtpath, 'r') as f:
            # pdb.set_trace()
            for line in f.readlines():
                line = line.rstrip().split(',')
                self.file_list.append(line)
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        angle1_img_path = self.file_list[idx][0]
        angle2_img_path = self.file_list[idx][1]
        angle3_img_path = self.file_list[idx][2]
        doppler_img_path = self.file_list[idx][3]
        range_img_path = self.file_list[idx][4]

        angle1_img = Image.open(angle1_img_path)
        if self.transform:
            angle1_img_transformed = self.transform(angle1_img)
        angle2_img = Image.open(angle2_img_path)
        if self.transform:
            angle2_img_transformed = self.transform(angle2_img)
        angle3_img = Image.open(angle3_img_path)
        if self.transform:
            angle3_img_transformed = self.transform(angle3_img)
        doppler_img = Image.open(doppler_img_path)
        if self.transform:
            doppler_img_transformed = self.transform(doppler_img)
        range_img = Image.open(range_img_path)
        if self.transform:
            range_img_transformed = self.transform(range_img)
        label = self.file_list[idx][-1]
        dict = {"daosanjiao" :0,
                "N"          :1,
                "shanghua"   :2,
                "W"          :3,
                "xieshangtui":4,
                "xiexiala"   :5,
                "Z"          :6
                }
        range_time = range_img_transformed
        doppler_time = doppler_img_transformed
        angle_time = np.concatenate((angle1_img_transformed, angle2_img_transformed,angle3_img_transformed),axis = 0)
        if self.transform:
            return range_time,doppler_time,angle_time,dict[label]
        else:
            return range_time,doppler_time,angle_time,dict[label]


def get_dataloader(file_list, batch_size=32):
    # Image Augmentation
    Transforms = transforms.Compose(
        [
            transforms.Resize((128, 50)),
            transforms.RandomResizedCrop((128, 50)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = RadarDataset(file_list, transform=Transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(data_loader)
    return data_loader
