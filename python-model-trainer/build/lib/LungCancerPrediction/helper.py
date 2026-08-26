
import os
import torchvision
from .datasets import LungCancerTrainDataset

# for running the helper script to transform images and save them to a new directory
# python-model-trainer> python -m LungCancerPrediction.helper

data_directory = r"C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\data\lung-cancer-prediction"
transformation_directory = r"C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\data\lung-cancer-prediction\transformed_images"

if not os.path.exists(transformation_directory):
    os.makedirs(transformation_directory)

for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)


train_dataset = LungCancerTrainDataset(data_directory=data_directory, with_transforms=True)


for i in range(10):
    sample = train_dataset[0]
    image = sample["image"]

    torchvision.utils.save_image(image, os.path.join(transformation_directory, f"transformed_image_{i}.png"))