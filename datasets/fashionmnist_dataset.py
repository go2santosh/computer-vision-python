import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Define a custom dataset for FashionMNIST using a CSV file
class FashionMNISTDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

        label, image = [], []
        
        for i in self.data:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        
        if self.transform is not None:
            # transfrom the numpy array to PIL image before the transform function
            pil_image = Image.fromarray(np.uint8(image)) 
            image = self.transform(pil_image)
            
        return image, label