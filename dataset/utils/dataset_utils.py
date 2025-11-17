import os
import ujson
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

batch_size = 32
train_ratio = 0.8

def check(config_path, train_path, test_path):

    # Check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config["batch_size"] == batch_size:

            print("\nDataset already generated.\n")
            return True
    
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data):

    dataset_image, dataset_label = data

    X_train, X_test, y_train, y_test = train_test_split(
        dataset_image,
        dataset_label,
        train_size=train_ratio,
        stratify=dataset_label,
        random_state=42
    )
    
    print("Total number of samples:", len(dataset_image))
    print("Number of training samples:", len(X_train))
    print("Number of testing samples:", len(X_test))
    
    return dataset_image, dataset_label

def visualize_data_train_test_plot(dir_path, train_data, test_data, num_classes):

    os.makedirs(os.path.join(dir_path, 'figures'), exist_ok=True)
    X_train, y_train = train_data
    X_test, y_test   = test_data

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=np.arange(num_classes + 1) - 0.5, rwidth=0.8)
    plt.title("Train Set Distribution")
    plt.xticks(range(num_classes))
    plt.xlabel("Class Label")
    plt.ylabel("Samples")

    plt.subplot(1, 2, 2)
    plt.hist(y_test, bins=np.arange(num_classes + 1) - 0.5, rwidth=0.8, color='orange')
    plt.title("Test Set Distribution")
    plt.xticks(range(num_classes))
    plt.xlabel("Class Label")
    plt.ylabel("Samples")

    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'figures', 'data_distribution.png'))
    plt.close()


def save_file(config_path, train_path, test_path, train_data, test_data, num_classes):
    X_train, y_train = train_data
    X_test, y_test = test_data
    # Save config file
    config = {
        "batch_size": batch_size,
        "num_classes": num_classes
    }
    print("Saving to disk.\n")

    np.savez_compressed(train_path + "data.npz", images=X_train, labels=y_train)
    np.savez_compressed(test_path + "data.npz",  images=X_test, labels=y_test)
    
    with open(config_path, 'w') as f:
        ujson.dump(config, f)
    
    print("Finish generating dataset.\n")

class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label