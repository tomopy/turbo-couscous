"""A CLI for reconstructing simulated data using TomoPy and rating the quality
of the reconstructions using XDesign."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ast
import logging
import os.path
import time

import click
import tomopy
import numpy as np
import matplotlib.pyplot as plt
import xdesign as xd

logger = logging.getLogger(__name__)


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
    default=5,
    help='Number of iterations between saves.',
    type=int,
)
@click.option(
    '-m',
    '--max-iter',
    default=300,
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
    default=16,
    help='Number of CPU cores to use',
    type=int,
)
@click.option(
    '--parameters',
    default=None,
    help='Python List of Python Dict of tomopy parameters as string',
)
def main(phantom, num_iter, max_iter, output_dir, ncore, parameters):
    """Reconstruct data using TomoPy.

    Chooses which algorithms to run based on the name of the conda environment.
    Expects three environments: astra, 1.1, and 1.5 which has the tomopy
    API of those versions.
    """
    data = np.load(os.path.join(output_dir, phantom, 'simulated_data.npz'))
    dynamic_range = np.max(data['original'])
    if parameters is not None:
        parameters = ast.literal_eval(parameters)
    elif 'astra' in os.environ['CONDA_PREFIX']:
        import astra
        parameters = [
            {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA'}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options': {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': num_iter}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options': {'proj_type': 'cuda', 'method': 'CGLS_CUDA', 'num_iter': num_iter}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options': {'proj_type': 'cuda', 'method': 'EM_CUDA', 'num_iter': num_iter}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options': {'proj_type': 'cuda', 'method': 'SART_CUDA', 'num_iter': num_iter}},
        ]
    elif '1.1' in os.environ['CONDA_PREFIX']:
        parameters = [
            {'algorithm': 'art', 'num_iter': num_iter},
            {'algorithm': 'bart', 'num_iter': num_iter},
            {'algorithm': 'fbp'},
            # {'algorithm': 'fbp', 'filter_name': None},
            # {'algorithm': 'fbp', 'filter_name': 'none'},
            # {'algorithm': 'fbp', 'filter_name': 'butterworth'},
            # {'algorithm': 'fbp', 'filter_name': 'cosine'},
            # {'algorithm': 'fbp', 'filter_name': 'hamming'},
            # {'algorithm': 'fbp', 'filter_name': 'hann'},
            # {'algorithm': 'fbp', 'filter_name': 'parzen'},
            # {'algorithm': 'fbp', 'filter_name': 'ramlak'},
            # {'algorithm': 'fbp', 'filter_name': 'shepp'},
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
            {'algorithm': 'osem', 'num_iter': num_iter},
            {'algorithm': 'ospml_hybrid', 'num_iter': num_iter},
            {'algorithm': 'ospml_quad', 'num_iter': num_iter},
            {'algorithm': 'pml_hybrid', 'num_iter': num_iter},
            {'algorithm': 'pml_quad', 'num_iter': num_iter},
            {'algorithm': 'sirt', 'num_iter': num_iter},
        ]
    elif '1.5' in os.environ['CONDA_PREFIX']:
        parameters = [
            {'algorithm': 'grad', 'num_iter': num_iter, 'reg_par': -1},
            {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'},
            {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'LINEAR'},
            {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'NN'},
            {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'LINEAR'},
            {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'LINEAR'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'NN'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'LINEAR'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'tv', 'num_iter': num_iter},
        ]
    else:
        raise ValueError("Test environment not recognized.")
    for params in parameters:  # yapf: disable
        if 'device' in params and params['device'] == 'gpu':
            params.update({'ncore': 1, 'pool_size': ncore})
        elif 'device' in params and params['device'] == 'cpu':
            params.update({'ncore': 2, 'pool_size': 8})
        else:
            params.update({'ncore': ncore})
        reconstruct(
            data,
            params,
            dynamic_range=dynamic_range,
            max_iter=max_iter,
            output_dir=output_dir,
        )


def reconstruct(
        data,
        params,
        dynamic_range=1.0,
        max_iter=200,
        phantom='peppers',
        output_dir='',
        term_crit=0.01,
):
    """Reconstruct data using given params.

    Resume from previous reconstruction if exact files already exist.
    Save files to file named by as:
    output_dir/algorithm/algorithm.filter_name.device.INTERPOLATION.[npz jpg]

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
    term_crit : float
        Benchmark ends early if reconstruction quality increases less than this
        amount.
    """
    logger.info('{}'.format(params))
    if params['algorithm'] is tomopy.astra:
        algorithm = params['options']['method']
    else:
        algorithm = params['algorithm']
    base_path = os.path.join(output_dir, phantom, algorithm)
    # padding was added to keep square image in the field of view
    pad = (data['sinogram'].shape[2] - data['original'].shape[2]) // 2
    # initial reconstruction guess; use defaults unique to each algorithm
    recon = None
    peak_quality = 0
    end = 1 if 'num_iter' not in params else max_iter
    step = 1 if 'num_iter' not in params else params['num_iter']
    for i in range(step, end + step, step):
        # name the output file by combining the algorithm name with some
        # important (key) input parameters
        filename = algorithm
        for key_param in ['filter_name', 'device', 'interpolation']:
            if key_param in params:
                filename = ".".join([filename, params[key_param]])
        filename = os.path.join(base_path, "{}.{:03d}".format(filename, i))
        # look for the ouput; only reconstruct if it doesn't exist
        if os.path.isfile(filename + '.npz'):
            existing_data = np.load(filename + '.npz')
            recon = existing_data['recon']
            msssim = existing_data['msssim']
            wall_time = existing_data['time']
        else:
            try:
                start = time.perf_counter()
                recon = tomopy.recon(
                    init_recon=recon,
                    tomo=data['sinogram'],
                    theta=data['angles'],
                    **params,
                )
                stop = time.perf_counter()
                wall_time = stop - start
            except Exception as e:
                logger.warning(e)
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
                time=wall_time
            )
            plt.imsave(
                filename + '.jpg',
                recon[0, pad:recon.shape[1] - pad, pad:recon.shape[2] - pad],
                format='jpg',
                cmap=plt.cm.cividis,
                vmin=0,
                vmax=1.1 * dynamic_range,
            )
        logger.info("{} : ms-ssim = {:05.3f} : time = {:05.3f}s".format(
            filename, np.mean(msssim), wall_time))
        if i > 1 and np.mean(msssim) - peak_quality < term_crit:
            logger.info("Early termination at {} iterations".format(i))
            break
        peak_quality = max(np.mean(msssim), peak_quality)


if __name__ == '__main__':
    main()
