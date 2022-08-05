#%% imports
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule

#%% Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% data module class
class CIFAR10_dm(LightningDataModule) :
    def __init__(self, data_dir: str, batch_size:int, num_workers:int = 2):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # dataset has PILImage images of range [0, 1]. 
        # We transform them to Tensors of normalized range [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
        self.data_dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        #download CIFAR10 dataset to data directory
        CIFAR10(self.data_dir , train=True, download=True)
        CIFAR10(self.data_dir , train=False, download=True)

    
    def setup(self, stage=None) :

        if stage == 'fit' or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train = True, transform = self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000,5000])

        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train = False, transform = self.transform)
    

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size= self.batch_size, num_workers= self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size= self.batch_size, num_workers= self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size= self.batch_size, num_workers= self.num_workers)