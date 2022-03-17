import os
import sys 
from argparse import ArgumentParser
import numpy as np
import imageio
import torch
import utils
from functools import partial


def main(config):

    # Make GPUs visible
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config.gpu])

    # Load seed style code
    seed_file = np.load(config.seed_file)

    # Get generator
    if config.syn_alg == "stylegan2":
        gan = utils.load_stylegan2(out_size = config.im_size)
        generator = gan.generateImageFromStyle
        z = seed_file[np.newaxis,config.seed,:]
    elif config.syn_alg == "biggan":
        gan = utils.load_biggan(out_size = config.im_size)
        z = seed_file['z'][np.newaxis,config.seed,:] 
        c = seed_file['y'][config.seed]
        generator = partial(gan.generateImage, c=c)
    else:
        raise ValueError('Invalid synthesis model!')


    # Get all feature functions
    funcs = []
    dev = 'cuda:1'

    # id, region, landmark points, attribute
    for name in config.f + [config.c]:
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

            filt_func = None
            if len(spec) > 0:
                import freq_models
                filt_type, sigma = spec[0], float(spec[1])

                if filt_type == 'lo':
                    filt = freq_models.GaussianBlur(sigma, dev)
                elif filt_type == 'hi':
                    filt = freq_models.HighPass(sigma, True, dev)
                else:
                    raise ValueError('Invalid frequency option')

                filt_func = utils.filter(filt, dev) 

            func = utils.im_to_vec(mask_func=mask_func, filt_func=filt_func) 

        elif feat_type == 'l':
            mesh_model = utils.load_facemesh_model()
            func = utils.face_to_mesh(spec[1], mesh_model)

        elif feat_type == 'a':
            func = utils.im_to_attr( utils.get_attribute_model(spec[1]) )

        else:
            raise ValueError('Invalid feature name: %s' % name)

        funcs.append( func )

    generate_traversals(z, generator, funcs, config)
    #filt = freq_models.HighPass(2, True, 'cuda:1')
    #filt_func = utils.filter(filt, 'cuda:1')

    #im = generator(z) 
    #im_filt = filt_func(im)

    #imageio.imwrite('test.jpg', im_filt[0,...])


def get_local_directions(z, generator, funcs, config, two_sided=True):

    def jacobian_to_grammian(J, eps=1e-9):
        A = np.matmul(J.T, J)
        A += np.random.randn(*A.shape) * eps # To avoid any singular issues
        return A/np.linalg.norm(A)

    if config.dir_alg == 'rand':
        V = np.random.rand(z.shape[1], z.shape[1]) - 0.5
        V = V/np.linalg.norm(V, axis=0, keepdims=True)
    else:
        J = utils.get_jacobian( z, config.step, config.batch, generator, funcs, two_sided)
        Af = [ jacobian_to_grammian(Ji) for Ji in J[0:len(config.f)] ]
        Ac = [ jacobian_to_grammian(Ji) for Ji in J[len(config.f):] ]

        if config.dir_alg == 'maxc':
            V = utils.split_space_by_explained_variance(Ac[0], config.beta_c)[0]

        elif config.dir_alg == 'minf':
            V = utils.get_nullspace_intersection(Af, config.beta_f)

        elif config.dir_alg == 'reds':
            V = utils.get_REDs(Af, Ac, config.beta_f, config.beta_c)
        else:
            raise ValueError('Not a valid algorithm.')

    return V


def generate_traversals(z0, generator, funcs, config):

    # Get directions at seed point
    V0 = get_local_directions(z0, generator, funcs, config, True)
    config.n_eig = min(config.n_eig, V0.shape[1])

    # Image and latent-storing arrays
    I = utils.ImageGrid(config.im_size, 2*config.n_eig, config.n_step+1)
    Z = np.zeros((2*config.n_eig, config.n_step+1, z0.shape[-1]))

    # Get seed image
    im0 = generator(z0)

    # Make paths
    for t in range(config.n_eig*2): # For each RED at seed point (pos/neg direction)
        I.insert(255*im0[0,...], t, 0) #Add starting image
        Z[t, 0, :] = z0

        v = V0[:, t//2] #Starting eigenvector
        if t%2 == 1: # positive or negative direction
            v *= -1

        for j in range(config.n_step):
            #print(t, j)
            if j > 0 and config.path_alg == 'local':
                Vi = get_local_directions(Z[t, j:j+1, :], generator, funcs, config, False)
                v_new = utils.project_onto_subspace(v, Vi)
                v = v_new if np.sum(v_new * v) > 0 else -v_new

            # Move to new location
            Z[t, j+1, :] = Z[t, j, :] + config.step * v
            im_cur = generator(Z[t, j+1:j+2, :])
            I.insert(255*im_cur[0,...], t, j+1) 

    # Save data 
    I.save('%s/%.2d.jpg' % (config.save_dir, config.seed))
    np.save('%s/%.2d_Z.npy' % (config.save_dir, config.seed), Z)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='+', help='list of gpus to use')
    parser.add_argument('--seed_file', type=str, help='path to latent codes file')
    parser.add_argument('--batch', type=int, default=10) 
    parser.add_argument('--im_size', type=int)
    parser.add_argument('--f', type=str, nargs='+', required=True)
    parser.add_argument('--c', type=str, required=True) 
    parser.add_argument('--step', type=float, required = True)
    parser.add_argument('--beta_f', type=float, nargs='+', required = True)
    parser.add_argument('--beta_c', type=float, required = True)
    parser.add_argument('--n_eig', type=int, required = True)
    parser.add_argument('--n_step', type=int, required = True)
    parser.add_argument('--syn_alg', type=str, choices=['stylegan2', 'biggan'], required=True) 
    parser.add_argument('--dir_alg', type=str, choices=['minf', 'maxc', 'reds', 'rand'], required=True)
    parser.add_argument('--path_alg', type=str, choices=['linear', 'local'], required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--results_dir', type=str)
    config = parser.parse_args()

    if len(config.beta_f) != len(config.f):
        raise ValueError('betas should be same length as fixed fucntions')

    config.save_dir = "%s/%s/%s_%s/" % (config.results_dir, config.syn_alg, config.dir_alg, config.path_alg)

    if config.dir_alg == 'maxc':
        config.save_dir += config.c
    elif config.dir_alg == "minf":
        config.save_dir += ",".join(config.f)
    elif config.dir_alg == "reds":
        config.save_dir += ",".join([config.c] + config.f)

    if config.dir_alg not in ['rand', 'maxc']:
        for beta in config.beta_f:
            config.save_dir += ",%.7f" % beta 

    # Save directory name
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    np.random.seed(4321)

    #if not os.path.exists('%s/%.2d.jpg' % (config.save_dir, config.seed)):
    main(config)
