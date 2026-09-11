
import os
import torchvision
from .datasets import LungCancerTrainDataset

# for running the helper script to show the model summary or transform images
# python-model-trainer> python -m LungCancerPrediction.helper




def print_model_summary():
    from .lc_controller import LungCancerNN
    import torchinfo

    model = LungCancerNN()
    torchinfo.summary(model, input_size=(1, 1, 256, 256))

def transform_images():
    data_directory = r"C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\data\lung-cancer-prediction"
    transformation_directory = r"C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\data\lung-cancer-prediction\transformed_images"

    if not os.path.exists(transformation_directory):
        os.makedirs(transformation_directory)

    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


    train_dataset = LungCancerTrainDataset(data_directory=data_directory, with_augmentation=True)


    for i in range(10):
        sample = train_dataset[0]
        image = sample["image"]

        torchvision.utils.save_image(image, os.path.join(transformation_directory, f"transformed_image_{i}.png"))



while True:
    print("Please select an option:")
    print("1. Transform images")
    print("2. Show model summary")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        print("Transforming images...")
        transform_images()
    elif choice == "2":
        print("Showing model summary...")
        print_model_summary()
    elif choice == "3":
        exit()
    else:
        print("Invalid choice. Please try again.")
