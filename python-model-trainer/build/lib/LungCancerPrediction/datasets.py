import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from .services import ImageLoader

class LungCancerTestDataset(Dataset):
    def __init__(self, data_directory):
        self.images = []
        self.labels = []
        
        categories = ["Bengin cases", "Malignant cases", "Normal cases"]
        
        for i, category in enumerate(categories):
            path = os.path.join(data_directory, category)
            files = os.listdir(path)
            
            category_image_count = int(len(files) * 0.75)
            
            for j in range(category_image_count, len(files)):

                self.images.append(ImageLoader.image_path_to_tensor(os.path.join(path, files[j])))
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

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
        
        counts = {label: self.labels.count(label) for label in set(self.labels)}
        
        weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
        return torch.tensor(weights, dtype=torch.float)


class LungCancerTrainDataset(Dataset):
    def __init__(self, with_flips, data_directory, max_images_per_category=None):
        self.images = []
        self.labels = []
        self.categories = ["Bengin cases", "Malignant cases", "Normal cases"]

        if data_directory is None or not os.path.exists(data_directory):
            return
        
        for i, category in enumerate(self.categories):
            path = os.path.join(data_directory, category)
            files = os.listdir(path)
            
            category_image_count = int(len(files) * 0.75)
            
            if max_images_per_category is not None:
                category_image_count = min(category_image_count, max_images_per_category)
                
            for j in range(category_image_count):

                tensor = ImageLoader.image_path_to_tensor(os.path.join(path, files[j]))
                self.images.append(tensor)
                self.labels.append(i)
                
                if with_flips:
                    self.images.append(TF.hflip(tensor).clone())
                    self.labels.append(i)
                    
                    self.images.append(TF.vflip(tensor).clone())
                    self.labels.append(i)

    @staticmethod
    def from_params(images, labels, categories):
        dataset = LungCancerTrainDataset(with_flips=False, data_directory="")
        dataset.images = images
        dataset.labels = labels
        dataset.categories = categories
        return dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def get_class_weights(self):

        if self.categories is None or len(self.categories) == 0:
            return []

        if self.labels is None or len(self.labels) == 0:
            return [0] * len(self.categories)

        total_labels = len(self.labels)
        class_weights = [0] * len(self.categories)

        for i in range(len(class_weights)):
            class_count = self.labels.count(i)
            if class_count > 0:
                class_weights[i] = total_labels / (len(self.categories) * class_count)
            else:
                class_weights[i] = 0

        return class_weights