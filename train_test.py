import os 
from tqdm import tqdm
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb

from dataset import CelebFacesDataset
from model import get_resnet18_model, get_resnet101_model, get_vgg16_model, ResNet, ResidualBlock
from argument_parser import parser

# Parse the arguments
args = parser.parse_args()

# Check if CUDA is available and set the device accordingly
use_cuda = torch.cuda.is_available()
print(use_cuda)
torch.manual_seed(42)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Initialize a new W&B run if specified
if args.use_wandb:
    wandb.init(project='celeb_faces_final', config=args)

# Define paths
this_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
print(this_dir)
data_dir = os.path.join(this_dir, 'data')  # Define the data directory
img_dir = os.path.join(data_dir, 'face_recognition', 'Faces')  # Define the image directory
data_path = os.path.join(data_dir, 'face_recognition','dataset_faces.csv')  # Define the path to the dataset CSV file
data_df = pd.read_csv(data_path)  # Load the dataset CSV file

# Define a set of transformations to be applied to the images
transforms_set = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.6389, 0.4773, 0.4056],  # Normalize images with the given mean and std
                         std=[0.2343, 0.1996, 0.1831]),
])

# Load the dataset
dataset = CelebFacesDataset(data_df, img_dir, transforms_set)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # Define the size of the training set
test_size = len(dataset) - train_size  # Define the size of the validation set
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Split the dataset

# Create DataLoaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # DataLoader for the training set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # DataLoader for the validation set

# Define the model
num_classes = 31  # Define the number of classes

# Create the model based on the specified architecture
if args.model == 'resnet18':
    model = get_resnet18_model(num_classes, device)  # Create a ResNet-18 model
elif args.model == 'resnet101':
    model = get_resnet101_model(num_classes, device)  # Create a ResNet-101 model
elif args.model == 'resnet':
    model = ResNet(ResidualBlock, [3, 4, 6,  3, 2], num_classes).to(device=device)  # Create a ResNet model

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Define the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Define the optimizer

# Function to train the model
def train(dataloader, model, criterion, optimizer, epoch, adjust_lr_fn=None):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.clone().detach().to(device) 
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    
    if args.use_wandb:
        wandb.log({"train_loss": avg_loss}, step=epoch)

    # Adjust learning rate after each epoch, if specified
    if adjust_lr_fn:
        adjust_lr_fn(optimizer, args.gamma, epoch, args.lr)

# Function to test the model
def test(dataloader, model, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.clone().detach().to(device) 
            output = model(data)
            test_loss += criterion(output, target).item()  # accumulate loss for the whole dataset
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
    test_loss /= len(dataloader)  # Now it's per batch, not per sample
    test_accuracy = 100. * correct / len(dataloader.dataset)
    
    # Calculate evaluation metrics
    precision_score_ = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_score_ = recall_score(all_targets, all_preds, average='macro')
    f1_score_ = f1_score(all_targets, all_preds, average='macro')
    accuracy_score_ = accuracy_score(all_targets, all_preds)
    

    print(f'\nPrecision: {precision_score_:.2f}')
    print(f'Recall: {recall_score_:.2f}')
    print(f'F1 Score: {f1_score_:.2f}')
    print(f'Accuracy: {accuracy_score_:.2f}')
    print(f'====> Test set loss: {test_loss:.4f}')
    print(f'====> Test set accuracy: {test_accuracy:.2f}%')
    
    if args.use_wandb:
        wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy, "precision": precision_score_, "recall": recall_score_, "f1_score": f1_score_, "accuracy": accuracy_score_}, step=epoch)

    return test_loss, test_accuracy, precision_score_, recall_score_, f1_score_, accuracy_score_

# Function to adjust the learning rate
def adjust_lr(optimizer, gamma, step, lr):
    lr = lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Main training loop
if __name__ == '__main__':
    best_accuracy = 0
    save_model_dir = os.path.join(this_dir, 'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    pth_path = os.path.join(save_model_dir, f'{args.model}_celeb_faces.pth')
    
    print('Training model with this configuration:')
    print('Device:', device)
    print('Model:', args.model)
    print('Batch size:', args.batch_size)
    print('Epochs:', args.epochs)
    print('Learning rate:', args.lr)
    print('Gamma:', args.gamma)
    print('Wandb:', args.use_wandb)
    
    print(args)
    
    try:
        for epoch in tqdm(range(1, args.epochs + 1)):
            train(train_loader, model, criterion, optimizer, epoch, adjust_lr_fn=adjust_lr)
            test_loss, test_accuracy, precision_score_, recall_score_, f1_score_, accuracy_score_ = test(test_loader, model, criterion, epoch)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), pth_path)
    except KeyboardInterrupt:
        print('Training interrupted')
