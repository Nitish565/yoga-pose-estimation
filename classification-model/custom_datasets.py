import torch
import pandas as pd
from PIL import Image


class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, img_dir, transforms):
        super(PascalDataset, self).__init__()
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transforms = transforms
        
        self.df = pd.read_csv(csv_dir)
        self.len = len(self.df.index)
        categories = self.df['category'].unique()
        
        self.cat_to_id = {}
        self.id_to_cat = {}
        #to generate category ids from 0 -> len(categories)-1
        #need this while calculating loss on preds in training loop
        for i, cat in enumerate(categories):
            self.cat_to_id[cat] = i         
            self.id_to_cat[i] = cat
           
    def __getitem__(self, index):
        img = Image.open(self.img_dir/self.df.loc[index]['file_name'])
        img = self.transforms(img)
        
        label = self.cat_to_id[self.df.loc[index]['category']]
        return (img, label)
    
    def __len__(self):
        return self.len    
    
    def get_category_label(self, id):
        return self.id_to_cat[id]

    
