import argparse

parser = argparse.ArgumentParser(description='Training parser.')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for learning rate scheduler')
parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
parser.add_argument('--use_wandb', type=bool, default=False, help='Use wandb')
