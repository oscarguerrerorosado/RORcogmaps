import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_AE(nn.Module):
    def __init__(self, n_hidden=500):
        super().__init__()

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 15, 20

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden)
        

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  
        x = x.view(-1, 64 * self.dim1 * self.dim2)  
        x = F.relu(self.fc1(x))
        
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x)) 
        x = x.view(-1, 64, self.dim1, self.dim2) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv6(x))  
        return x

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true,C_factor, alpha=0):
        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        # Whitening loss (batch whitening).
        hidden_constraint_loss = 0
        batch_size, hidden_dim = hidden.shape

        # SSCP matrix
        M = torch.mm(hidden.t(), hidden)
    
        I = torch.eye(hidden_dim, device='cuda')
        C = C_factor*I - M    # C = I - M    
        hidden_constraint_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)
        
        loss = recon_loss + hidden_constraint_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()