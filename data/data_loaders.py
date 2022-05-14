# from ..base import BaseDataLoader
# 상대경로 importerror
from .datasets import *
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from base import BaseDataLoader
from torchvision import transforms

class MaskDataLoader(BaseDataLoader):
    """
    Image mask dataloader using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1):

        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



if __name__ == '__main__':
    transform = transforms.Compose([
                    transforms.Resize((224,224)), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5), (0.5)) # -1 ~ 1
                ])
    dataset = MaskDataset(transform, training=True)
    dataloader = MaskDataLoader(dataset, 
                                batch_size=16, 
                                shuffle=True, 
                                validation_split=0.2, 
                                training=True)
    for e, (img, label) in enumerate(dataloader):
        print(img,label)
        if e > 10: break
