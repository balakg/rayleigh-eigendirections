import sys
import numpy as np
import cv2
import pickle as pkl
sys.path.append('./BigGAN-PyTorch')
import bgutils
import torch


class BigGAN(object):
    def __init__(self, config_path, out_size):

        with open(config_path, 'rb') as handle:
            config = pkl.load(handle)

        experiment_name = '138k'
        model = __import__(config['model'])
        self.device = 'cuda'
        self.out_size = out_size
        G = model.Generator(**config)
        G.to(self.device)
        bgutils.load_weights(G if not (config['use_ema']) else None, None, {}, 
                             './BigGAN-PyTorch/weights', experiment_name, config['load_weights'],
                             G if config['ema'] and config['use_ema'] else None,
                             strict=False, load_optim=False)
        G.eval()
        self.G = G


    def generateImage(self, z, c):
        z = torch.FloatTensor(z).to(self.device, torch.float32)
        c = torch.FloatTensor([c]).to(self.device, torch.int64)
        c = c.repeat(z.shape[0])

        images = self.G(z, self.G.shared(c))

        #batch = z.shape[0]
        #images = []
        #for i in range(z.shape[0]):
        #    im = self.G(z[i:i+1,...], self.G.shared(c))
        #    images.append(im)
        #images = torch.cat(images, 0)
        x = (images.detach().cpu().numpy() + 1) / 2. 
        x = np.transpose(x, (0, 2, 3, 1)) 
       
        return self.resize(x)


    def resize(self, img):
        img_out = [ cv2.resize(img[j, ...], (self.out_size, self.out_size)) for j in range(img.shape[0]) ]
        return np.stack(img_out, 0)
