import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import imageio


# ----- MODEL LOADING FUNCS ----- #

# Stylegan2 loader
def load_stylegan2(out_size):
    sys.path.append('./stylegan2')
    import wrapper

    return wrapper.StyleGAN2(
        path='./stylegan2/stylegan2-ffhq-config-f.pkl', 
        out_size=out_size)


def load_biggan(out_size):
    sys.path.append('./BigGAN-PyTorch')
    import wrapper

    return wrapper.BigGAN(
        config_path='./BigGAN-PyTorch/config.pkl', 
        out_size=out_size)


# Arcface (facial recognition) model loader
def load_faceid_model(device):
    from insightface import get_model
    net = get_model('r50', fp16=False)
    net.load_state_dict(torch.load('./insightface/r50-weights.pth'))
    net.eval()
    net.to(device)
    return net


# Face segmentation model loader
def load_face_segmenter(device):
    sys.path.append('./face-parsing')
    from bisenet_model import BiSeNet

    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load('./face-parsing/79999_iter.pth'))
    net.eval()
    net.to(device)
    return net


# Face landmark model loader
def load_facemesh_model():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
                   static_image_mode=True,
                   max_num_faces=1,
                   min_detection_confidence=0.5)


# Load attribute classifier
def load_attribute_model(name, device):
    C = tv.resnet50(pretrained=False, progress=True, num_classes = 1)
    params = './classifiers/%s.weights' % name
    C.load_state_dict(torch.load(params, map_location=lambda storage, loc: storage))
    C.eval()
    C.to(device)
    return C


# ----- Feature Funcs ----- #

def im_to_vec(size=None, mask_func=None, filt_func=None):
    def func(im):
        if mask_func is not None:
            im = mask_func(im)

        if filt_func is not None:
            im = filt_func(im)

        if size is not None:
            im = [ cv2.resize(im[j, ...], (size,size)) for j in range(im.shape[0]) ]
            im = np.stack(im, 0)

        return im.reshape(im.shape[0], -1)

    return func


def face_to_id(net, device):

    @torch.no_grad()
    def func(im):
        im = npy_to_floattensor(im, device)
        im = F.interpolate(im, size=112, mode='bilinear', align_corners=True) 
        im.sub_(0.5).div_(0.5)
        y = net(im).detach().cpu().numpy()
        return y / np.linalg.norm(y, axis=1, keepdims=True)

    return func


def face_to_mesh(region, face_mesh):

    n_points = 468

    if region == 'all':
        idx = np.arange(n_points)
    elif region == 'mouth':
        idx = [13, 14, 62, 292, 81, 82, 312, 311, 
               310, 318, 402, 317, 87, 178, 80, 88]
    else:
        raise ValueError('Not a defined region!')


    def func(im):
        P = []
        for i in range(im.shape[0]):
            results = face_mesh.process( np.uint8(im[i,...]*255) )

            if (results is None 
                or results.multi_face_landmarks is None 
                or len(results.multi_face_landmarks) == 0): 
                p_i = np.zeros(468*3)
                p_i[:] = np.nan                
            else:
                results = results.multi_face_landmarks[0]

                x_i, y_i, z_i = [], [], []
                for j in idx:
                    l = results.landmark[j]
                    x_i.append(l.x)
                    y_i.append(l.y)
                    z_i.append(l.z)

                x_i = np.array(x_i) - np.median(x_i)
                y_i = np.array(y_i) - np.median(y_i)
                z_i = np.array(z_i) - np.median(z_i)

                p_i = np.concatenate((x_i, y_i, z_i))
                #bad_vals = np.logical_or(np.isinf(p_i), np.isnan(p_i))
                #bad_idx = np.where(bad_vals == 1)[0]
                #p_i[bad_idx] = 0

            P.append(p_i)

        return np.stack(P, axis=0)

    return func



def mask_by_coordinates(region=None):

    negate = False
    if ('not-' in region):
        region = region.split("-")[1]
        negate = True
 

    # Coordinates copied from LowRankGAN paper (based on 256x256 image)
    region_to_coords = {'left_eye': [120, 95, 20, 38],
                   'right_eye': [120, 159, 20, 38],
                   'eyes': [120, 128, 20, 115],
                   'nose': [142, 131, 40, 46],
                   'mouth': [184, 127, 30, 70],
                   'chin': [217, 130, 42, 110],
                   'left_region': [128, 74, 128, 64],
                   'point_eye': [120, 95, 2, 2],
                   'point_mouth': [184, 127, 2, 2],
                   }

    coordinate = region_to_coords[region]

    def func(im_in):
        s = im_in.shape[2]//256 # Scale coordinates based on image size.
        mask = np.zeros_like(im_in)

        center_x, center_y = s*coordinate[0], s*coordinate[1]
        crop_x, crop_y = s*coordinate[2], s*coordinate[3]
        xx = center_x - crop_x // 2
        yy = center_y - crop_y // 2
        mask[:, xx:xx + crop_x, yy:yy + crop_y, :] = 1.

        if negate:
            mask = 1 - mask

        return im_in * mask

    return func


def mask_by_segmentation(seg_net, device, region=None):

    def norm(im):
        mean = torch.Tensor([.485,.456,.406])[None,:,None,None].to(im.device)
        std = torch.Tensor([.229,.224,.225])[None,:,None,None].to(im.device)
        return (im - mean) / std

    negate = False
    if ('not-' in region):
        region = region.split("-")[1]
        negate = True


    def func(im_in):
        im_in = npy_to_floattensor(im_in, device)
        im = norm(im_in)
        im = F.interpolate(im, size=512, mode='bilinear', align_corners=True)
        ind = torch.argmax(seg_net(im)[0], dim=1, keepdims=True).float()

        if region == 'hair':
            mask = ind == 17
        elif region == 'skin':
            mask = ind == 1
        elif region == 'face':
            mask = torch.logical_and(ind >= 1, ind <= 13)
        elif region == 'head':
            mask = torch.logical_or( torch.logical_and(ind >= 1, ind <= 13), ind == 17 )
        elif region == 'mouth':
            mask = torch.logical_and(ind >= 11, ind <= 13)
        elif region == 'eyes':
            mask = torch.logical_and(ind >= 2, ind <= 5)
        elif region == 'nose':
            mask = ind == 10
        else:
            raise ValueError('Unidentified region!')


        mask = 1.0 * mask
        if negate:
            mask = 1 - mask

        # Move segmentation boundary away from edges a little to soften constraints.
        kernel = torch.ones((1, 1, 5, 5))
        kernel /= torch.sum(kernel)
        kernel = kernel.to(mask.device)
        mask = F.pad(mask, (4, 4, 4, 4), mode='replicate')
        mask = F.conv2d(mask, kernel)
        mask = 1.0 * (mask > 0.999)
        mask = F.interpolate(mask, im_in.shape[2], mode='bilinear')

        return floattensor_to_npy(im_in * mask)

    return func


def filter(filter_net, device):

    def func(im_in):
        im = npy_to_floattensor(im_in, device)
        im = filter_net(im)
        return floattensor_to_npy(im)

    return func


def im_to_attr(model, device):

    @torch.no_grad()
    def func(im):
        im = npy_to_floattensor(im, device)
        return model(im).cpu().detach().numpy()

    return func


def npy_to_floattensor(im, device):
    """
    im: numpy array of uint images (batch, *, *, 3)
    device: location of output tensor
    """

    im = np.transpose(im, (0, 3, 1, 2))
    im = torch.FloatTensor(im).to(device)

    return im


def floattensor_to_npy(im):
    return im.permute(0, 2, 3, 1).detach().cpu().numpy()


# ----- REDs ALGORITHM FUNCS ----- #

def get_jacobian(x, step, batch, generator, funcs, two_sided=True):

    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError('latent point has incorrect shape!')

    dim = x.shape[1]

    def latent_to_features(latent):
        im = generator(latent)
        return [ func(im) for func in funcs ]

    J = [ [] for i in range(len(funcs)) ] 

    x1 = np.tile(x, (dim, 1)) + np.eye(dim)*step
    if two_sided:
        x0 = np.tile(x, (dim, 1)) - np.eye(dim)*step
        delta = step*2
    else:
        y0 = latent_to_features(x)
        delta = step

    for j in range(0, dim, batch):
        #print("%2d" % j, end='\r')
        idx = [ j+i for i in range(min(batch, dim - j)) ]

        y1 = latent_to_features(x1[idx,...])
        if two_sided:
            y0 = latent_to_features(x0[idx,...])

        for k in range(len(funcs)):
            diff = np.transpose( (y1[k] - y0[k])/delta )
            diff[ np.logical_not(np.isfinite(diff)) ] = 0 # Take care of bad values 
            J[k].append( diff )

    return [ np.concatenate(a, 1) for a in J ]


def eigs(A):
    E,V = np.linalg.eig(A)
    idx = np.argsort(np.real(E))[::-1]
    return np.real(V[:, idx]), E[idx]


def get_rank_by_explained_variance(E, var_total=0.99999):
    var = E**2/np.sum(E**2)
    #print(var*100)
    return min(E.size, 
               np.sum(np.cumsum(var) <= var_total) + 1 
           )


def split_space_by_explained_variance(A, beta):
    V,E = eigs(A)
    #print(E)
    rank = get_rank_by_explained_variance(E, beta)
    return V[:, 0:rank], V[:, rank:]


def get_nullspace_intersection(A_list, beta_list):
    # Get combined column space first.
    C = [split_space_by_explained_variance(A, beta)[0] for A,beta in zip(A_list,beta_list)]
    C = np.concatenate(C, axis=1)
    Q,_ = np.linalg.qr(C) # Get set of orthonormal vectors spanning C

    # Nullspace is what remains after removing Q from full basis.
    d = Q.shape[0]
    B = np.eye(d) # Full basis
    for i in range(d):
        # Gram-Schmidt orthogonalization procedure
        v = np.copy(B[:,i])
        s = 0
        for j in range(Q.shape[1]):
            s += proj(Q[:,j], v)
        B[:,i] = v - s

    # Only return the significant vectors of B
    u,s,vh = np.linalg.svd(B) 
    rank = np.sum(s > 1e-3) 
    return u[:, 0:rank] 
 

def proj(u, v):
    return u * np.sum(u * v) / np.sum( u * u)


def get_REDs(A_f, A_c, beta_f, beta_c):

    if len(A_c) > 1:
       raise ValueError('Only supporting one changing attribute.')

    A_f_null = get_nullspace_intersection(A_f, beta_f)
    A_c_proj = np.matmul( np.transpose(A_f_null), 
                          np.matmul( A_c[0], A_f_null )
                        )

    V,_ = split_space_by_explained_variance(A_c_proj, beta_c)
    return np.matmul(A_f_null, V)


def project_onto_subspace(x, V):
    Q, _ = np.linalg.qr(V) # d x n
    Q = np.transpose(Q) # n x d
    x = np.expand_dims(x, 0) # 1 x d
    y = np.sum(x * Q, 1, keepdims=True) # n x 1
    y = np.sum(y * Q, 0) # d,
    return y/np.linalg.norm(y)


"""
Class for storing a grid of images together as one big image. Used for storing
traversal strips.
"""
class ImageGrid:
    def __init__(self, el_size, n_rows, n_cols):
        self.el_size = el_size
        self.size = (n_rows, n_cols)
        self.I = np.zeros((el_size*n_rows, el_size*n_cols, 3)) 


    def insert(self, im, row, col):
        assert (len(im.shape) == 3 
                and im.shape[0] == self.el_size 
                and im.shape[1] == self.el_size)

        self.I[row*self.el_size:(row+1)*self.el_size, 
               col*self.el_size:(col+1)*self.el_size, :] = im 
              

    def fliplr(self):
        """
        Flips grid elements left-to-right.
        """
  
        for j in range(self.size[1]//2): # for each column
            col_left = np.copy(self.I[:, j*self.el_size : (j+1) * self.el_size, :]) 

            jr = self.size[1] - j - 1
            col_right = np.copy(self.I[:, jr*self.el_size : (jr+1) * self.el_size, :])

            self.I[:, j*self.el_size : (j+1) * self.el_size, :] = col_right
            self.I[:, jr*self.el_size : (jr+1) * self.el_size, :] = col_left

    
    def insert_grid(self, J, row, col):
       """
       Insert another grid into this one at specified location (row,col). 
       """

       if J.el_size != self.el_size:
           raise ValueError("Element sizes of grids do not match!")

       if row < 0 or col < 0:
           raise ValueError('row, col must be >= 0')

       J_r, J_c = J.size

       if row + J_r > self.size[0] or col + J_c > self.size[1]:
           raise ValueError('Array is too big for this image grid')

       self.I[row*self.el_size:(row+J_r)*self.el_size, 
               col*self.el_size:(col+J_c)*self.el_size, :] = J.I 


    def save(self, name):
        imageio.imwrite(name, np.uint8(self.I))
