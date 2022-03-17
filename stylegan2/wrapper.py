import dnnlib.tflib as tflib
import pretrained_networks
import numpy as np
import cv2


class StyleGAN2(object):
    def __init__(self, path, out_size=256):

        tflib.init_tf()
        _, _, Gs = pretrained_networks.load_networks(path)

        self.Gs = Gs
        self.w_avg = Gs.get_var('dlatent_avg')
        self.out_size = out_size
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


    def getStyle(self, z, psi):
        # Z must be of shape (batch, 512)
        if z.ndim != 2 or z.shape[1] != 512:
            raise ValueError('Incorrect latent code shape.')

        w = self.Gs.components.mapping.run(z, None)[:, 0, :]
        w = self.w_avg + (w - self.w_avg) * psi
        return w


    def generateImageFromStyle(self, w):
        # W must be (batch, 512) or (batch, 18, 512)
 
        if w.ndim < 2 or w.ndim > 3:
            raise ValueError('Incorrect style code shape.')

        #if w.ndim == 1:
        #    w = np.expand_dims(w, 0)

        if w.ndim == 2:
            w = np.tile(np.expand_dims(w, 1), (1, 18, 1))

        img = self.Gs.components.synthesis.run(w, randomize_noise=False, output_transform=self.fmt)
        return self.resize_and_float(img)


    def generateImageFromLatents(self, z, psi):
        # z must be of shape (512,) or (batch, 512)

        if z.shape[-1] != self.Gs.input_shape[1]:
            raise ValueError('Latent codes must have last dimension = %d' % self.Gs.input_shape[1])
        
        if z.ndim == 1:
            z = np.expand_dims(z,0)

        if z.ndim != 2:
            raise ValueError("Incorrect number of dimensions!")


        img = self.Gs.run(z, None, randomize_noise=False, output_transform=self.fmt, truncation_psi=psi)
        return self.resize_and_float(img)


    def resize_and_float(self, img):
        img_out = [ cv2.resize(img[j, ...], (self.out_size, self.out_size)) for j in range(img.shape[0]) ]
        return np.stack(img_out, 0)/255.0
