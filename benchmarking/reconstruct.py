"""A CLI for reconstructing simulated data using TomoPy and rating the quality
of the reconstructions using XDesign."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ast
import logging
import os.path
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import tomopy
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
    '-m',
    '--max-iter',
    default=500,
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
@click.option(
    '--algorithm',
    default=None,
    help='Name of algorithm to use given as a string',
)
def main(phantom, max_iter, output_dir, ncore, parameters, algorithm):
    """Reconstruct data using TomoPy.

    Automatically chooses which algorithms to run based the TomoPy
    version number and whether third-party modules are importable.
    Alternatively, a string representing a python list of dictionaries
    of parameters can be provided to run a custom subset of tests.
    """
    logging.basicConfig(level=logging.INFO)

    data = np.load(os.path.join(output_dir, phantom, 'simulated_data.npz'))
    dynamic_range = np.max(data['original'])
    if parameters is not None:
        parameters = ast.literal_eval(parameters)

    elif algorithm is not None:
        default_parameters = {
            'gridrec': [
                {'algorithm': 'gridrec', 'filter_name': 'butterworth'},
                {'algorithm': 'gridrec', 'filter_name': 'cosine'},
                {'algorithm': 'gridrec', 'filter_name': 'hamming'},
                {'algorithm': 'gridrec', 'filter_name': 'hann'},
                {'algorithm': 'gridrec', 'filter_name': 'parzen'},
                {'algorithm': 'gridrec', 'filter_name': 'ramlak'},
                {'algorithm': 'gridrec', 'filter_name': 'shepp'},
            ],
            'sirt': [
                {'algorithm': 'sirt'},
#                 {'algorithm': 'sirt', 'accelerated': True,
#                  'device': 'cpu', 'interpolation': 'LINEAR'},
#                 {'algorithm': 'sirt', 'accelerated': True,
#                  'device': 'cpu', 'interpolation': 'CUBIC'},
            ],
            'sirt_gpu': [
                {'algorithm': 'sirt', 'accelerated':
                 True, 'device': 'gpu', 'interpolation': 'NN'},
                # {'algorithm': 'sirt', 'accelerated':
                #     True, 'device': 'gpu', 'interpolation': 'LINEAR'},
                # {'algorithm': 'sirt', 'accelerated':
                #     True, 'device': 'gpu', 'interpolation': 'CUBIC'}
            ]
        }

        if algorithm in default_parameters:
            parameters = default_parameters[algorithm]
        else:
            parameters = [{'algorithm': algorithm}]

    else:
        parameters = [
            # {'algorithm': 'gridrec', 'filter_name': 'none'},  # none isn't none, it's ramlak?
            {'algorithm': 'gridrec', 'filter_name': 'butterworth'},
            {'algorithm': 'gridrec', 'filter_name': 'cosine'},
            {'algorithm': 'gridrec', 'filter_name': 'hamming'},
            {'algorithm': 'gridrec', 'filter_name': 'hann'},
            {'algorithm': 'gridrec', 'filter_name': 'parzen'},
            {'algorithm': 'gridrec', 'filter_name': 'ramlak'},
            {'algorithm': 'gridrec', 'filter_name': 'shepp'},
            # {'algorithm': 'fbp', 'filter_name': 'butterworth'},
            # {'algorithm': 'fbp', 'filter_name': 'cosine'},
            # fbp is currenlty broken, it doesn't
            # take filters into consideration
            # {'algorithm': 'fbp', 'filter_name': 'hamming'},
            # {'algorithm': 'fbp', 'filter_name': 'hann'},
            # {'algorithm': 'fbp', 'filter_name': 'parzen'},
            # {'algorithm': 'fbp', 'filter_name': 'ramlak'},
            # {'algorithm': 'fbp', 'filter_name': 'shepp'},
            {'algorithm': 'art'},
            {'algorithm': 'bart'},
            {'algorithm': 'mlem'},
            {'algorithm': 'osem'},
            {'algorithm': 'ospml_hybrid'},
            {'algorithm': 'ospml_quad'},
            {'algorithm': 'pml_hybrid'},
            {'algorithm': 'pml_quad'},
            {'algorithm': 'sirt'},
            # {'algorithm': 'tikh'},
            {'algorithm': 'tv'},
            # {'algorithm': 'grad'},
            # {'algorithm': 'mlem', 'accelerated':
            #     True, 'device': 'gpu', 'interpolation': 'NN'},
            # {'algorithm': 'mlem', 'accelerated':
            #     True, 'device': 'gpu', 'interpolation': 'LINEAR'},
            # {'algorithm': 'mlem', 'accelerated':
            #     True, 'device': 'gpu', 'interpolation': 'CUBIC'},
            # {'algorithm': 'mlem', 'accelerated':
            #     True, 'device': 'cpu', 'interpolation': 'NN'},
            # {'algorithm': 'mlem', 'accelerated':
            #     True, 'device': 'cpu', 'interpolation': 'LINEAR'},
            # {'algorithm': 'mlem', 'accelerated':
            #     True, 'device': 'cpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'sirt', 'accelerated':
                True, 'device': 'gpu', 'interpolation': 'NN'},
            {'algorithm': 'sirt', 'accelerated':
                True, 'device': 'gpu', 'interpolation': 'LINEAR'},
            {'algorithm': 'sirt', 'accelerated':
                True, 'device': 'gpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'sirt', 'accelerated':
                True, 'device': 'cpu', 'interpolation': 'NN'},
            {'algorithm': 'sirt', 'accelerated':
                True, 'device': 'cpu', 'interpolation': 'LINEAR'},
            {'algorithm': 'sirt', 'accelerated':
                True, 'device': 'cpu', 'interpolation': 'CUBIC'},
        ]
        try:
            import astra
            parameters += [
                {'algorithm': tomopy.astra,
                 'options': {'proj_type': 'cuda',
                             'method': 'FBP_CUDA', 'FilterType': 'ram-lak'}},
                # FIXME: Not confident that these filter parameters are being passed to astra
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'shepp-logan'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'cosine'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'hamming'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'hann'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'none'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'tukey'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'lanczos'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'triangular'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'gaussian'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'barlett-hann'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'blackman'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'nuttall'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'blackman-harris'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'blackman-nuttall'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'flat-top'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'kaiser'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'parzen'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'projection'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'sinogram'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'rprojection'}},
                # {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'rsinogram'}},
                {'algorithm': tomopy.astra, 'options':
                    {'proj_type': 'cuda', 'method': 'SIRT_CUDA'}},
                {'algorithm': tomopy.astra, 'options':
                    {'proj_type': 'cuda', 'method': 'CGLS_CUDA'}},
                {'algorithm': tomopy.astra, 'options':
                    {'proj_type': 'cuda', 'method': 'EM_CUDA'}},
                {'algorithm': tomopy.astra, 'options':
                    {'proj_type': 'cuda', 'method': 'SART_CUDA'}},
            ]
        except ImportError:
            tomopy.astra = None
        try:
            import lprec
            parameters += [
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'parzen'},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'ramp'},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'shepp-logan'},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'cosine'},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'cosine2'},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'hamming'},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'fbp', 'filter_name': 'hann'},
                {'algorithm': tomopy.lprec, 'lpmethod': 'cg', 'num_iter': num_iter},
                {'algorithm': tomopy.lprec, 'lpmethod': 'em', 'num_iter': num_iter},
                {'algorithm': tomopy.lprec, 'lpmethod': 'grad',
                    'num_iter': num_iter},  # broken
                {'algorithm': tomopy.lprec, 'lpmethod': 'tv', 'num_iter': num_iter},
                # {'algorithm': tomopy.lprec, 'lpmethod': 'tve', 'num_iter': num_iter},  # broken
                # {'algorithm': tomopy.lprec, 'lpmethod': 'tvl1', 'num_iter': num_iter},  # broken
            ]
        except ImportError:
            tomopy.lprec = None
    for params in parameters:  # yapf: disable
        if 'device' in params and params['device'] == 'gpu':
            params.update({'ncore': 1, 'pool_size': ncore})
        elif 'device' in params and params['device'] == 'cpu':
            params.update({'ncore': 2, 'pool_size': 8})
        elif params['algorithm'] is tomopy.lprec:
            params.update({'ncore': 1})
        elif params['algorithm'] is tomopy.astra:
            params.update({'ncore': 1})
            try:
                params['filter_name'] = params['options']['FilterType']
            except KeyError:
                pass
        else:
            params.update({'ncore': ncore})
        reconstruct(
            data,
            params,
            dynamic_range=dynamic_range,
            max_iter=max_iter,
            output_dir=output_dir,
            phantom=phantom,
        )


def reconstruct(
        data,
        params,
        dynamic_range,
        max_iter,
        phantom,
        output_dir,
        max_time=3600,
):
    """Reconstruct data using given params.

    Resume from previous reconstruction if exact files already exist.
    Save files to file named by as:
    output_dir/algorithm/algorithm.filter_name.device.INTERPOLATION.[npz png]

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
    max_time : float
        The maximum wall time per slice before stopping (seconds).
    phantom : string
        The name of the phantom
    """
    logger.info('{}'.format(params))

    # padding was added to keep square image in the field of view
    pad = (data['sinogram'].shape[2] - data['original'].shape[2]) // 2

    # Determine the algorithm name in the filesystem
    if params['algorithm'] is tomopy.astra:
        algorithm = 'astra-' + params['options']['method'].lower()
    elif params['algorithm'] is tomopy.lprec:
        algorithm = 'lprec-' + params['lpmethod'].lower()
    else:
        algorithm = params['algorithm'].lower()
    base_path = os.path.join(output_dir, phantom, algorithm)
    if 'device' in params and params['device'] == 'gpu':
        base_path = base_path + '_cuda'

    # initial reconstruction guess; use defaults unique to each algorithm
    recon = None
    peak_quality = 0
    total_time = 0

    # Create evenly spaced samples across a log plot
    if 'gridrec' in algorithm or 'fbp' in algorithm:
        iters, steps = [1], [1]
    else:
        iters = np.unique(np.logspace(
            0, np.log10(max_iter), num=16, dtype=int))
        steps = [iters[0]] + np.diff(iters).tolist()
        np.testing.assert_array_equal(np.cumsum(steps), iters)

    for i in range(len(iters)):
        # name the output file by combining the algorithm name with some
        # important (key) input parameters
        filename = algorithm
        for key_param in ['filter_name', 'device', 'interpolation']:
            if key_param in params:
                filename = ".".join([filename, str(params[key_param])])
        filename = os.path.join(base_path, "{}.{:03d}".format(
            filename, iters[i]
        ))

        # look for the ouput; only reconstruct if it doesn't exist
        if os.path.isfile(filename + '.npz'):
            logger.info("{} exists!".format(filename))
            existing_data = np.load(filename + '.npz')
            recon = existing_data['recon']
            wall_time = existing_data['time']
            total_time += wall_time
        else:
            if 'gridrec' in algorithm or 'fbp' in algorithm:
                pass
            elif params['algorithm'] is tomopy.astra:
                params['options']['num_iter'] = steps[i]
            else:
                params['num_iter'] = steps[i]
            try:
                start = time.perf_counter()
                # Do reconstruction in chunks because GPU memory is small
                # FIXME: It's not fair to include all of this GPU memory
                # allocation and destruction in the wall_time. In practice,
                # you wouldn't check the answer every few iterations?
                chunk_size = 8
                shape = data['sinogram'].shape
                if (
                    shape[1] > chunk_size and 'device' in params
                    and params['device'] == 'gpu'
                ):
                    if recon is None:
                        recon = np.empty((shape[1], shape[2], shape[2]))
                        for j in range(0, 32, chunk_size):
                            recon[j:j+chunk_size] = tomopy.recon(
                                init_recon=None,
                                tomo=data['sinogram'][:, j:j+chunk_size, :],
                                theta=data['angles'],
                                **params,
                            )
                    else:
                        for j in range(0, 32, chunk_size):
                            recon[j:j+chunk_size] = tomopy.recon(
                                init_recon=recon[j:j+chunk_size],
                                tomo=data['sinogram'][:, j:j+chunk_size, :],
                                theta=data['angles'],
                                **params,
                            )
                elif params['algorithm'] is tomopy.lprec:
                    recon = tomopy.recon(
                        init_recon=recon,
                        tomo=data['sinogram'] / np.sqrt(1500 * 2048),
                        theta=data['angles'],
                        **params,
                    )
                else:
                    recon = tomopy.recon(
                        init_recon=recon,
                        tomo=data['sinogram'],
                        theta=data['angles'],
                        **params,
                    )
                stop = time.perf_counter()
                wall_time = stop - start
                total_time += wall_time
            except Exception as e:
                logger.warning(e)
                return
            os.makedirs(base_path, exist_ok=True)
            # save all information
            np.savez_compressed(
                filename + '.npz',
                recon=recon,
                time=wall_time,
                total_time=total_time,
            )
            plt.imsave(
                filename + '.png',
                recon[0, pad:recon.shape[1] - pad, pad:recon.shape[2] - pad],
                format='png',
                cmap=plt.cm.cividis,
                vmin=0,
                vmax=1.1 * dynamic_range,
            )
        logger.info(
            "{} : time = {:05.3f}s, total time = {:05.3f}s".format(
                filename, wall_time, total_time)
        )
        if total_time > max_time * recon.shape[0]:
            logger.info(f"Terminate early due to {max_time}s time limit.")
            break


if __name__ == '__main__':
    main()
