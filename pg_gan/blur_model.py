import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class blur(nn.Module):
    def __init__(self):
        super(blur, self).__init__()
        self.dense = nn.Linear(50,980)
        self.upsamp1 = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv2Dt1 = nn.ConvTranspose2d(20,20,2) 
        self.upsamp2 = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv2Dt2 = nn.ConvTranspose2d(20,20,2)
        self.conv2Dt3 = nn.ConvTranspose2d(20,1,2)

    def forward(self, x):
        out = F.relu(self.dense(x))
        out = out.view(1,7,7,20)
        out = out.permute(0,3,1,2)
        out = F.relu(self.conv2Dt1(self.upsamp1(out)))
        out = F.relu(self.conv2Dt2(self.upsamp2(out)))
        out = F.relu(self.conv2Dt3(out))
        return out


if __name__ == "__main__":
    blur_gen = torch.load("./weights/blur_generator.pt")
    blur_gen = blur_gen.cuda()
    # Visualizing Blur Generator
    N = 10
    f, ax  = plt.subplots(1,10, figsize=(15,10))
    for i in range(N):
        data = np.random.normal(0,1,(1,50))
        data = torch.Tensor(data).cuda()
        blur = blur_gen(data).data.cpu().numpy()
        ax[i].imshow(blur[0,0,:,:],cmap='gray')
    plt.show()     
