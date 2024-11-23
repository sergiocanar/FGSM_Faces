import torch
print(torch.__version__)
print(torch.version.cuda)

#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print(use_cuda)
torch.manual_seed(42)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
