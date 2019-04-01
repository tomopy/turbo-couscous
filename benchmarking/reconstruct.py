"""Reconstruct phantoms."""

import os.path
import logging

import click
import tomopy
import numpy as np
import matplotlib.pyplot as plt
import xdesign as xd

logger = logging.getLogger(__name__)


def reconstruct(data, params, dynamic_range=1.0, max_iter=200, phantom='peppers'):
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
    phantom : string
        The name of the phantom
    """
    logger.info('{}'.format(params))
    os.makedirs('{}/{}'.format(phantom, params['algorithm']), exist_ok=True)
    # padding was added to keep square image in the field of view
    pad = (data['sinogram'].shape[2] - data['original'].shape[2]) // 2
    # initial reconstruction guess; use defaults unique to each algorithm
    recon = None
    end = 1 if 'num_iter' not in params else max_iter
    step = 1 if 'num_iter' not in params else params['num_iter']
    for i in range(1, end+step, step):
        # name the output file
        if 'filter_name' in params:
            filename = "{0}/{1}/{1}.{2}.{3:03d}".format(phantom, params['algorithm'],
                                                     params['filter_name'], i)
        else:
            filename = "{0}/{1}/{1}.{2:03d}".format(phantom, params['algorithm'], i)
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
        msssim = np.empty(len(recon))
        for z in range(len(recon)):
            # compute the reconstructed image quality metrics
            scales, msssim[z], quality_maps = xd.msssim(
                data['original'][z],
                recon[z, pad:recon.shape[1]-pad, pad:recon.shape[2]-pad],
                L=dynamic_range,
            )
        # save all information
        logger.info("{} : ms-ssim = {:05.3f}".format(filename, np.mean(msssim)))
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

@click.command()
@click.option('-p', '--phantom', default='peppers', help='Name of a phantom.')
@click.option('-i', '--num_iter', default=1, help='Number of iterations between saves.', type=int)
@click.option('-m', '--max-iter', default=5, help='Total number of iterations.', type=int)
def main(phantom, num_iter):
    """Reconstruct data using TomoPy."""
    data = np.load('{}/simulated_data.npz'.format(phantom))
    dynamic_range = np.max(data['original'])
    for params in [
        {'algorithm': 'art', 'num_iter': num_iter},
        {'algorithm': 'grad', 'num_iter': num_iter, 'reg_par': -1},
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
        {'algorithm': 'mlem', 'num_iter': num_iter},
        {'algorithm': 'sirt', 'num_iter': num_iter},
        {'algorithm': 'tv', 'num_iter': num_iter},
    ]:
        try:
            reconstruct(data, params, dynamic_range=dynamic_range, max_iter=5)
        except ValueError as e:
            logger.warn(e)

# TODO: Add 'fbp', 'bart', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
# 'pml_quad'

if __name__ == '__main__':
    main()
