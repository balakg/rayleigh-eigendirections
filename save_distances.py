import os
import sys 
from argparse import ArgumentParser
import numpy as np
import imageio
import torch
import utils
import pickle


def main(config):

    # Make GPUs visible
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config.gpu])

    gan = utils.load_stylegan2(out_size = config.im_size)
    generator = gan.generateImageFromStyle

    # Get all feature functions
    funcs = []
    dev = 'cuda:1'

    # id, region, landmark points, attribute
    for name in config.save_funcs:
        spec = name.split("_")
        feat_type = spec[0]

        if feat_type == 'id':
            faceid_model = utils.load_faceid_model(dev)
            func = utils.face_to_id(faceid_model, dev)

        elif feat_type == 'r':
            region = spec[1]           

            mask_func = None
            if region == 'all':
                spec = spec[2:]
            else:
                if spec[2] == 'coord':
                    mask_func = utils.mask_by_coordinates(region)

                elif spec[2] == 'seg':
                    seg_net = utils.load_face_segmenter(dev)
                    mask_func = utils.mask_by_segmentation(seg_net, dev, region)

                else:
                    raise ValueError('Invalid segmentation algorithm!')

                spec = spec[3:]

            func = utils.im_to_vec(mask_func = mask_func) 

        elif feat_type == 'l':
            mesh_model = utils.load_facemesh_model()
            func = utils.face_to_mesh(spec[1], mesh_model)

        elif feat_type == 'a':
            func = utils.im_to_attr( utils.get_attribute_model(spec[1]) )

        else:
            raise ValueError('Invalid feature name: %s' % name)

        funcs.append( func )

    if not os.path.exists("%s/dists" % config.save_dir):
        os.makedirs("%s/dists" % config.save_dir)

    Z = np.load("%s/%.2d_Z.npy" % (config.save_dir, config.seed))
    save(Z, generator, funcs, config, config.seed)


def save(Z, generator, funcs, config, seed):
    im0 = generator(Z[0, 0:1, :])
    F = []
    D = []
    for k,f in enumerate(funcs):           
         y0 = f( im0 )
         F.append( y0 )
         dists = np.zeros((Z.shape[0], Z.shape[1]))
         D.append(dists)


    for i in range(Z.shape[0]): # For each RED at seed point (pos/neg)
        for j in range(Z.shape[1]):

            z0 = Z[i, j:j+1, :]
            im_cur = generator(z0)

            # Compute distances
            for k,f in enumerate(funcs):           
                y1 = f(im_cur)
                D[k][i,j] = np.linalg.norm(y1 - F[k])

    for i, f_name in enumerate(config.save_funcs):
        np.save('%s/dists/%.2d_%s_dist.npy' % (config.save_dir, seed, f_name), D[i])



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--gpu', type=int, nargs='+', help='list of gpus to use')
    parser.add_argument('--batch', type=int, default=10) 
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--save_funcs', type=str, nargs='+')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--seed', type=int)

    config = parser.parse_args()
    main(config)
