import numpy as np
import torch
import torch.utils.data
from torchvision import datasets,transforms
import torchvision
import torchvision.transforms as transforms

def load_dog_cat_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    def is_dog_or_cat(x):
        return x[1] == 3 or x[1] == 5  # 3: cat, 5: dog
    
    trainset = torch.utils.data.Subset(trainset, [i for i in range(len(trainset)) if is_dog_or_cat(trainset[i])])
    testset = torch.utils.data.Subset(testset, [i for i in range(len(testset)) if is_dog_or_cat(testset[i])])
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

train_loader, test_loader = load_dog_cat_data()import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class RBM(nn.Module):
    def __init__(self, n_vis=3072, n_hin=500, k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
    def sample_from_p(self, p):
        return torch.bernoulli(p)
    
    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h
    
    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v
    
    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)
        
        return v, v_
    
    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -hidden_term - vbias_term

# Function to show and save images
def show_and_save(file_name, img):
    npimg = img.cpu().numpy()
    f = f"./{file_name}.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training loop
rbm = RBM(n_vis=3072, n_hin=500, k=1).to(device)
optimizer = optim.Adam(rbm.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
num_epochs = 5500

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 3072)
        sample_data = data.bernoulli().to(device)
        
        v, v1 = rbm(sample_data)
        loss = torch.mean(rbm.free_energy(v) - rbm.free_energy(v1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rbm.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    if epoch % 10 == 0:
        # Visualize some reconstructed images
        with torch.no_grad():
            test_data = next(iter(test_loader))[0][:64].view(-1, 3072).to(device)
            _, recon = rbm(test_data)
            show_and_save(f"reconstruct_epoch_{epoch}", make_grid(recon.view(-1, 3, 32, 32).data))

# Note: Ensure that train_loader and test_loader are defined before running this code