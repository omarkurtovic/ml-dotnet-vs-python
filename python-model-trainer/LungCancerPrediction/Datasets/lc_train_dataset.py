import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class LungCancerTrainDataset(Dataset):
    def __init__(self, with_flips, data_directory, max_images_per_category=None):
        self.images = []
        self.labels = []
        
        categories = ["Bengin cases", "Malignant cases", "Normal cases"]
        
        for i, category in enumerate(categories):
            path = os.path.join(data_directory, category)
            files = os.listdir(path)
            
            # Replicating your 75% split logic
            category_image_count = int(len(files) * 0.75)
            
            if max_images_per_category is not None:
                category_image_count = min(category_image_count, max_images_per_category)
                
            for j in range(category_image_count):
                file_path = os.path.join(path, files[j])
                
                # Load image and convert to PyTorch Tensor
                image = Image.open(file_path).convert('L') # Assuming Grayscale CT scan
                tensor = TF.to_tensor(image)
                
                self.images.append(tensor)
                self.labels.append(i)
                
                if with_flips:
                    # flip([2]) in C# usually maps to horizontal flip
                    self.images.append(TF.hflip(tensor).clone())
                    self.labels.append(i)
                    
                    # flip([1]) in C# usually maps to vertical flip
                    self.images.append(TF.vflip(tensor).clone())
                    self.labels.append(i)

    # Equivalent to public override long Count => _images.Count;
    def __len__(self):
        return len(self.images)

    # Equivalent to public override Dictionary<string, Tensor> GetTensor(long index)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def get_class_weights(self):
        total = len(self.labels)
        num_classes = 3
        
        # Count occurrences of each label
        counts = {label: self.labels.count(label) for label in set(self.labels)}
        
        weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
        return torch.tensor(weights, dtype=torch.float)