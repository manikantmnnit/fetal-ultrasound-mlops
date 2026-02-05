import logging
from pathlib import Path
from typing import override
from lightning.fabric.utilities.data import _set_sampler_epoch
import torch
import numpy as np

# for image 
from torchvision import datasets, transforms
import cv2 as cv
from PIL import Image

from torch.utils.data import DataLoader,WeightedRandomSampler

import json
from dataclasses import dataclass

# lightning 
import lightning.pytorch as pl

# import utility functions
from utlis.transforms import ApplyCLAHE,CroppedImage
from config import SEED, batch_size

current_dir=Path(__file__).resolve()

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(current_dir.parent.parent/"logs"/"data_preprocess.log"),
        logging.StreamHandler()
    ]
)


current_dir=Path(__file__).resolve()



class MixedFetalDataModule(pl.LightningDataModule):
    def __init__(self,
            mixed_data_dir:str,
            top_side:int,
            left_side:int,
            batch_size:int=batch_size,
            seed:int=SEED
                      ):
        super().__init__()
        self.mixed_data_dir=Path(mixed_data_dir)
        self.batch_size=batch_size
        self.top_side= top_side
        self.left_side =left_side
        self.seed=seed
        logging.info(f"Data dir: {self.mixed_data_dir}")
        logging.info(f"Batch size: {self.batch_size} | Seed: {self.seed}")
        logging.info(f"Crop params: top={self.top_side}, left={self.left_side}")

        

        # placeholder ( to be defnied these values soon)
        self.train_dataset=None
        self.val_dataset=None
        self.test_dataset=None
        self.mean=None
        self.std=None

        # json path
        self.train_stats_path=current_dir.parent.parent/'reports'/'train_stats.json'
        self.split_path = current_dir.parent.parent/"reports/train_val_test_split.json"
       

    # setup is used here for data processing, and transformation
    def setup(self, stage=None):
        # seed for everything
        pl.seed_everything(self.seed, workers=True)
        logging.info("Global seed set")

        train_dir=self.mixed_data_dir/'train'
        val_dir=self.mixed_data_dir/"val"
        test_dir=self.mixed_data_dir/"test"
        logging.info(f"Train dir: {train_dir}")
        logging.info(f"Val dir: {val_dir}")
        logging.info(f"Test dir: {test_dir}")

        
        #find mean and std of train folder
        transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
        ])

        temp_train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transform
        )

        temp_loader = DataLoader(
            temp_train_dataset ,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        mean = torch.zeros(3)
        std = torch.zeros(3)
        total = 0
        train_stats_path=current_dir.parent.parent/'reports'/'train_stats.json'
        train_stats_path.parent.mkdir(parents=True,exist_ok=True)

          # save mean and std in json format
        if train_stats_path.exists():
            with open(train_stats_path,'r') as f:
                stats_train = json.load(f)
               
        else:
            stats_train ={}
        
        # compute only if current batch_size is not stored
        if f"batch_{self.batch_size}" not in stats_train:
            logging.info("Computing mean/std for training dataset")
            for images, _ in temp_loader:

                b, c, h, w = images.shape
                images = images.view(b, c, -1)  # ( batch,channel, pixels)

                mean += images.mean(2).sum(0)  # find mean for each chennel (2 represents chennel)
                std += images.std(2).sum(0) # find std for each channel
                total += b

            
            self.mean=(mean/total).tolist()
            self.std=(std/total).tolist()

            
            # update stats for current self.batch_size
            stats_train[f"batch_{self.batch_size}"] = {
                "mean": self.mean,
                "std": self.std
            }
            logging.info(f"Computed mean and std: {stats_train[f"batch_{self.batch_size}"]}")
            

            # save stats_train
            with open(train_stats_path, "w") as f:
                json.dump(stats_train, f, indent=4)
            
            logging.info("Saved stats to JSON")
        else:
            logging.info("Loaded cached mean/std from JSON")
            
            self.mean = stats_train[f"batch_{self.batch_size}"]["mean"]
            self.std = stats_train[f"batch_{self.batch_size}"]["std"]
            logging.info(f"Mean: {self.mean}")
            logging.info(f"Std: {self.std}")
            

        normalize=transforms.Normalize(mean=self.mean,std=self.std)

        # train_ tranformation
        train_tranf=transforms.Compose([
            
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            ApplyCLAHE(),
            transforms.Grayscale(num_output_channels=3),
            CroppedImage(top=self.top_side, left=self.left_side),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)

        ])

        # val_transf
        val_test_tfms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            ApplyCLAHE(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # datset
        self.train_dataset=datasets.ImageFolder(train_dir,transform=train_tranf)
        self.val_dataset=datasets.ImageFolder(val_dir,transform=val_test_tfms)
        self.test_dataset=datasets.ImageFolder(test_dir,transform=val_test_tfms)
        logging.info(f"Train dataset size: {len(self.train_dataset)}")
        logging.info(f"Val dataset size: {len(self.val_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

        logging.info(f"Classes: {self.train_dataset.classes}")

        # dataloders
    def train_dataloader(self):
            logging.info("Building train DataLoader with WeightedRandomSampler")
            labels = [s[1] for s in self.train_dataset.samples]
            class_counts = torch.bincount(torch.tensor(labels))
            logging.info(f"Class counts: {class_counts.tolist()}")
            class_weights = 1. / class_counts.float()
            sample_weights = [class_weights[l] for l in labels]

            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
            return DataLoader(self.train_dataset,batch_size=self.batch_size,sampler=sampler)


    def val_dataloader(self):
            logging.info("Building validation DataLoader")
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
            logging.info("Building test DataLoader")
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
            




    
if __name__=="__main__":
    pl.seed_everything(SEED, workers=True)
    mixed_data_dir=current_dir.parent.parent/'data'/'mixed_images'
    dm=MixedFetalDataModule(mixed_data_dir=mixed_data_dir,top_side=20,left_side=20,batch_size=batch_size)
    dm.setup()    # preprocess and transoformation
    # print("Train classes:", dm.train_dataset.classes)
    labels = [s[1] for s in dm.train_dataset.samples]
    from collections import Counter
    print(Counter(labels))
  
    print(f"Train dataset length:{len(dm.train_dataset)},Val dataset length:{len(dm.val_dataset)},Test dataset length:{len(dm.test_dataset)}")

    #  # Iterate one batch
    train_loader = dm.train_dataloader() 
    sample_labels=[]
    for _, labels in train_loader:
        sample_labels.extend(labels.tolist())
    print("Sampled distribution (one epoch):")
    print(Counter(sample_labels))

    print(f" mean and std of current batch size of train dataset")
    train_stats_path=current_dir.parent.parent/'reports'/'train_stats.json'
    with open(train_stats_path,'r') as f:
        json_file=json.load(f)
    print(json_file[f'batch_{batch_size}'])
    



