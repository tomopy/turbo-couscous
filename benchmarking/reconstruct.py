"""Reconstruct phantoms."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import logging

import click
import tomopy
import numpy as np
import matplotlib.pyplot as plt
import xdesign as xd

logger = logging.getLogger(__name__)


def reconstruct(
        data,
        params,
        dynamic_range=1.0,
        max_iter=200,
        phantom='peppers',
        output_dir='',
):
    """Reconstruct data using given params.

    Resumes from previous reconstruction if exact files already exist.

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
        is used to scale a jpg image of the reconstruction
    max_iter : int
        The maximum number iterations if the algorithm is iterative
    phantom : string
        The name of the phantom
    """
    logger.info('{}'.format(params))
    base_path = os.path.join(output_dir, phantom, params['algorithm'])
    # padding was added to keep square image in the field of view
    pad = (data['sinogram'].shape[2] - data['original'].shape[2]) // 2
    # initial reconstruction guess; use defaults unique to each algorithm
    recon = None
    end = 1 if 'num_iter' not in params else max_iter
    step = 1 if 'num_iter' not in params else params['num_iter']
    for i in range(1, end + step, step):
        # name the output file
        if 'filter_name' in params:
            filename = os.path.join(
                base_path, "{}.{}.{:03d}".format(params['algorithm'],
                                                 params['filter_name'], i))
        else:
            filename = os.path.join(base_path, "{}.{:03d}".format(
                params['algorithm'], i))
        # look for the ouput; only reconstruct if it doesn't exist
        if os.path.isfile(filename + '.npz'):
            existing_data = np.load(filename + '.npz')
            recon = existing_data['recon']
            msssim = existing_data['msssim']
        else:
            try:
                recon = tomopy.recon(
                    init_recon=recon,
                    tomo=data['sinogram'],
                    theta=data['angles'],
                    **params,
                )
            except ValueError as e:
                logger.warn(e)
                return
            # compute quality metrics
            msssim = np.empty(len(recon))
            for z in range(len(recon)):
                # compute the reconstructed image quality metrics
                scales, msssim[z], quality_maps = xd.msssim(
                    data['original'][z],
                    recon[z, pad:recon.shape[1] - pad, pad:recon.shape[2] -
                          pad],
                    L=dynamic_range,
                )
            os.makedirs(base_path, exist_ok=True)
            # save all information
            np.savez(
                filename + '.npz',
                recon=recon,
                msssim=msssim,
            )
            plt.imsave(
                filename + '.jpg',
                recon[0, pad:recon.shape[1] - pad, pad:recon.shape[2] - pad],
                format='jpg',
                cmap=plt.cm.cividis,
                vmin=0,
                vmax=1.1 * dynamic_range,
            )
        logger.info("{} : ms-ssim = {:05.3f}".format(filename,
                                                     np.mean(msssim)))


@click.command()
@click.option(
    '-p',
    '--phantom',
    default='peppers',
    help='Name of a phantom.',
)
@click.option(
    '-i',
    '--num-iter',
    default=1,
    help='Number of iterations between saves.',
    type=int,
)
@click.option(
    '-m',
    '--max-iter',
    default=5,
    help='Total number of iterations.',
    type=int,
)
@click.option(
    '-o',
    '--output-dir',
    default=os.path.join('local', tomopy.__version__),
    help='Folder to put data inside',
    type=click.Path(exists=False),
)
@click.option(
    '--ncore',
    default=1,
    help='Number of CPU cores to use,',
    type=int,
)
def main(phantom, num_iter, max_iter, output_dir, ncore):
    """Reconstruct data using TomoPy."""
    data = np.load(os.path.join(output_dir, phantom, 'simulated_data.npz'))
    dynamic_range = np.max(data['original'])
    for params in [
        {'algorithm': 'gridrec'},
        {'algorithm': 'art', 'num_iter': num_iter},
        {'algorithm': 'grad', 'num_iter': num_iter, 'reg_par': -1},
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
    ]:  # yapf: disable
        params.update({'ncore': ncore})
        reconstruct(
            data,
            params,
            dynamic_range=dynamic_range,
            max_iter=max_iter,
            output_dir=output_dir,
        )


# TODO: Add 'fbp', 'bart', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
# 'pml_quad'

if __name__ == '__main__':
    main()
