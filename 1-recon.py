"""Reconstruct phantoms."""

import os.path
import tomopy
import numpy as np
import matplotlib.pyplot as plt
import xdesign as xd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(data, params, dynamic_range=1.0, max_iter=200):
    """Reconstruct data using given params.

    Parameters
    ----------
        data : dictionary
            Contains three keys:
                original : the original image
                sinogram : the projections
                angles : the angles of each of the projections
        params : dictionary
            Contains parameters to use for recostructing the data using
            tomopy.recon().
        dynamic_range : float
            The expected dynamic range of the reconstructed image. This param
            is used to scale a png image of the reconstruction
        max_iter : int
            The maximum number iterations if the algorithm is iterative

    """
    # padding was added to keep square image in the field of view
    pad = (data['sinogram'].shape[2] - data['original'].shape[2]) // 2
    # initial reconstruction guess; use defaults unique to each algorithm
    recon = None
    end = 1 if 'num_iter' not in params else max_iter
    step = 1 if 'num_iter' not in params else params['num_iter']
    for i in range(1, end+step, step):
        # name the output file
        if 'filter_name' in params:
            filename = "peppers/{}.{}.{:03d}".format(params['algorithm'],
                                                     params['filter_name'], i)
        else:
            filename = "peppers/{}.{:03d}".format(params['algorithm'], i)
        # look for the ouput; only reconstruct if it doesn't exist
        if os.path.isfile(filename + '.npz'):
            existing_data = np.load(filename + '.npz')
            recon = existing_data['recon']
        else:
            recon = tomopy.recon(
                init_recon=recon,
                tomo=data['sinogram'],
                theta=data['angles'],
                **params,
            )
        # compute quality metrics
        scales, msssim, quality_maps = xd.msssim(
            data['original'][0],
            recon[0, pad:recon.shape[1]-pad, pad:recon.shape[2]-pad],
            L=dynamic_range,
        )
        # save all information
        logger.info("{} : ms-ssim = {:05.3f}".format(filename, msssim))
        np.savez(
            filename + '.npz',
            recon=recon,
            msssim=msssim,
        )
        plt.imsave(
            filename + '.png',
            recon[0, pad:recon.shape[1]-pad, pad:recon.shape[2]-pad],
            format='png',
            cmap=plt.cm.cividis,
            vmin=0, vmax=1.1*dynamic_range,
        )


if __name__ == '__main__':
    data = np.load('peppers/data.npz')
    dynamic_range = np.max(data['original'])
    for params in [
        {'algorithm': 'art', 'num_iter': 10},
        {'algorithm': 'grad', 'num_iter': 10, 'reg_par': -1},
        {'algorithm': 'gridrec'},
        {'algorithm': 'gridrec', 'filter_name': None},
        {'algorithm': 'gridrec', 'filter_name': 'none'},
        {'algorithm': 'gridrec', 'filter_name': 'butterworth'},
        {'algorithm': 'gridrec', 'filter_name': 'cosine'},
        {'algorithm': 'gridrec', 'filter_name': 'hamming'},
        {'algorithm': 'gridrec', 'filter_name': 'hann'},
        {'algorithm': 'gridrec', 'filter_name': 'parzen'},
        {'algorithm': 'gridrec', 'filter_name': 'ramlak'},
        {'algorithm': 'gridrec', 'filter_name': 'shepp'},
        {'algorithm': 'mlem', 'num_iter': 10},
        {'algorithm': 'sirt', 'num_iter': 10},
        {'algorithm': 'tv', 'num_iter': 10},
    ]:
        main(data, params, dynamic_range=dynamic_range)
