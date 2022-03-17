import numpy as np
import torch
import sys
import torch.nn.functional as F


class HighPass(torch.nn.Module):
    def __init__(self, sigma, gray=True, device='cpu'):
        super(HighPass, self).__init__()
        self.device = device
        self.gray = gray
        self.lowpass = GaussianBlur(sigma, device)


    def forward(self, x):
        if self.gray:
            w = torch.Tensor([.30,.59,.11])[None,:,None,None]
            w = w.to(x.device)
            x = torch.sum(x * w, 1, keepdims=True)

        return x - self.lowpass(x)


class GaussianBlur(torch.nn.Module):
    def __init__(self, sigma, device='cpu'):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.device = device


    def gauss_kernel(self, size, sigma, n_chan):
        xx,yy = torch.meshgrid(torch.arange(size), torch.arange(size))
        c = size//2

        kernel = torch.exp( (-(xx - c)**2 - (yy - c)**2) /(2*sigma**2) )
        kernel = kernel/torch.sum(kernel)
        kernel = kernel.repeat(n_chan, 1, 1, 1)
        kernel = kernel.to(self.device)

        return kernel


    def gauss_conv(self, x, sigma, k):
        p = k//2
        x = F.pad(x, (p, p, p, p), mode='reflect')
        kern = self.gauss_kernel(k, sigma, x.shape[1])
        return F.conv2d(x, kern, groups=x.shape[1])


    def blur(self, x):
        shape = x.shape[2:4]
        n_levels = max(0, int(np.log2(self.sigma))+1)


        for i in range(n_levels - 1):
            x = self.gauss_conv(x, 1.0, 7)
            x = self.gauss_conv(x, 1.0, 7)
            x = self.gauss_conv(x, np.sqrt(2), 9)
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', 
                    align_corners=False) #x[:, :, 0::2, 0::2]


        sigma_cur = 2**(n_levels-1) if n_levels > 1 else 0
        sigma_final = np.sqrt(self.sigma**2 - sigma_cur**2) / (2**n_levels)

        if sigma_final > 1e-2:
            x = self.gauss_conv(x, sigma_final, max(7, 6*sigma_final+1))

        return F.interpolate(x, size=shape, mode='bilinear', align_corners=False)


    def forward(self, x):
        return self.blur(x)
        #return y.view(y.shape[0], -1)
