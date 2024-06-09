import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 3
batch_size = 64
learning_rate = 1e-3

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.dec3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        enc1 = F.relu(self.enc1(x))
        enc2 = F.relu(self.enc2(self.pool(enc1)))
        enc3 = F.relu(self.enc3(self.pool(enc2)))
        
        up1 = F.relu(self.up1(enc3))
        dec3 = F.relu(self.dec3(torch.cat([up1, enc2], dim=1)))
        up2 = F.relu(self.up2(dec3))
        dec2 = F.relu(self.dec2(torch.cat([up2, enc1], dim=1)))
        dec1 = torch.tanh(self.dec1(dec2))
        
        return dec1

# Define the diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        
        self.unet = UNet()
        
    def forward_diffusion_sample(self, x_0, t, noise):
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1).expand_as(x_0)
        return alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * noise
    
    def reverse_diffusion_step(self, x_t, t):
        noise_pred = self.unet(x_t)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1).expand_as(x_t)
        beta_t = self.betas[t].view(-1, 1, 1, 1).expand_as(x_t)
        
        return (x_t - beta_t * noise_pred) / alpha_t.sqrt()

# Initialize model, loss and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffusionModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        noise = torch.randn_like(images).to(device)
        t = torch.randint(0, model.num_timesteps, (images.size(0),), device=device).long()
        
        x_t = model.forward_diffusion_sample(images, t, noise)
        
        optimizer.zero_grad()
        noise_pred = model.unet(x_t)
        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training Complete")

# Saving the model
torch.save(model.state_dict(), 'diffusion_model.pth')

# Generate some images
model.eval()
with torch.no_grad():
    for i in range(10):
        noise = torch.randn(1, 1, 28, 28).to(device)
        generated_image = noise
        for t in reversed(range(model.num_timesteps)):
            generated_image = model.reverse_diffusion_step(generated_image, t)
        plt.subplot(2, 5, i + 1)
        plt.imshow(generated_image.cpu().squeeze(), cmap='gray')
    plt.show()
