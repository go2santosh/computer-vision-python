import os
import re
from torch.utils.data import Dataset
from PIL import Image

class SOTSDataset(Dataset):
    def __init__(self, root_dir, subset='outdoor', transform=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            subset (str): Specify 'outdoor' or 'indoor' to load respective data.
            transform (callable, optional): Transformations to apply to the images.
        """
        if subset not in ['outdoor', 'indoor']:
            raise ValueError("Subset must be 'outdoor' or 'indoor'")
        
        self.root_dir = os.path.abspath(root_dir)  # Use absolute path
        self.subset = subset
        self.transform = transform
        self.image_pairs = []
        
        # Validate directory structure
        required_subdirs = [
            os.path.join(self.root_dir, subset, 'hazy'),
            os.path.join(self.root_dir, subset, 'clear')
        ]
        for subdir in required_subdirs:
            if not os.path.exists(subdir):
                raise FileNotFoundError(f"Required directory not found: {subdir}")
        
        # Collect valid image pairs for the specified subset
        hazy_dir = os.path.join(self.root_dir, subset, 'hazy')
        clear_dir = os.path.join(self.root_dir, subset, 'clear')
        
        # File name can end with .png or .jpg
        hazy_images = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.jpg'))])
        clear_images = sorted([f for f in os.listdir(clear_dir) if f.lower().endswith(('.png', '.jpg'))])

        clear_image_map = {os.path.splitext(f)[0]: f for f in clear_images}
        
        for hazy_file in hazy_images:
            match = re.match(r"(\d+)[._]", hazy_file)
            if match:
                base_name = match.group(1)
                if base_name in clear_image_map:
                    hazy_path = os.path.join(hazy_dir, hazy_file)
                    clear_path = os.path.join(clear_dir, clear_image_map[base_name])
                    self.image_pairs.append((hazy_path, clear_path))

        if not self.image_pairs:
            raise RuntimeError(f"No valid hazy-clear image pairs found for {subset}!")
        print(f"Loaded {len(self.image_pairs)} valid {subset} image pairs")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        hazy_path, clear_path = self.image_pairs[index]
        
        if not os.path.exists(hazy_path):
            raise FileNotFoundError(f"Hazy image not found: {hazy_path}")
        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"Clear image not found: {clear_path}")
            
        hazy_image = Image.open(hazy_path).convert('RGB')
        clear_image = Image.open(clear_path).convert('RGB')

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        return hazy_image, clear_image