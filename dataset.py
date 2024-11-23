import os 
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from skimage.io import imread
from torchvision import transforms

class CelebFacesDataset(Dataset):
    def __init__(self, df, root_dir,transform=None):
        
        '''
        Creates the Celebrities Faces Dataset
        -------------------------------------
        Parameters:
            df: pandas DataFrame
                Dataframe with the image paths and labels
            root_dir: str
                Directory where the images are stored
            transform: torchvision.transforms
                Transforms to apply to the images
        -------------------------------------
        Returns:
            image: torch.Tensor
            label: int
        '''
        
        self.data = df     
        self.transform = transform
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = imread(img_path)
        image = Image.fromarray(image)  
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def dataloader_stats(dataloader):
    '''
    Extracts the mean and std of the dataset
    -------------------------------------
    Parameters:
        dataloader: torch.utils.data.DataLoader
            DataLoader object
    -------------------------------------
    Returns:
        mean: float
        std: float
    '''
    
    mean = 0
    std = 0
    
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    
    return mean, std

    
    
if __name__ == '__main__':
    this_file_path = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file_path)
    
    data_dir = os.path.join(this_dir, 'data')
    data_csv = os.path.join(data_dir, 'face_recognition', 'dataset_faces.csv')
    df = pd.read_csv(data_csv, sep=',', header=None)
    root_dir = os.path.join(data_dir, 'face_recognition', 'Faces')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_split = 0.8
    train_size = int(train_split * len(df))
    test_size = len(df) - train_size
    train_df, test_df = random_split(df, [train_size, test_size])
    
    train_dataset = CelebFacesDataset(df, root_dir, transform)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print('Dataset length: {}'.format(len(train_dataset)))
    first_image, first_label = train_dataset[0]
    print('First image shape: {}'.format(first_image.shape))
    print('First label: {}'.format(first_label))
        
    mean, std = dataloader_stats(loader)
    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))
    
    
    