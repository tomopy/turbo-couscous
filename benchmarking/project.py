"""A CLI for generating simulated data from TomoPy phantoms."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import logging
import click
import tomopy
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '-p',
    '--phantom',
    default='peppers',
    help='Name of a phantom.',
)
@click.option(
    '-w',
    '--width',
    default=1446,
    help='Pixel width of phantom before padding.',
    type=int,
)
@click.option(
    '-a',
    '--num-angles',
    default=1500,
    help='Number of projection angles.',
    type=int,
)
@click.option(
    '-t',
    '--trials',
    default=32,
    help='Number of phantom repetitions.',
    type=int,
)
@click.option(#edit
    '-p_n',
    '--poisson_noise',
    default=0,
    help='Whether to add poisson noise, and how much.',
)
@click.option(#edit
    '-guass?',
    '--guassian_tf',
    default=False,
    help='Whether to add gaussian distortions',
)
@click.option(#edit
    '-guass_m',
    '--guassian_mean',
    default=0,
    help='The mean of the gaussian distortions',
)
@click.option(#edit
    '-guass_std',
    '--guassian_std',
    default=0,
    help='The standard deviation of the guassian distortions.',
)
@click.option(#edit
    '-s_p?',
    '--salt_pepper_tf',
    default=False,
    help='Whether to add salt_pepper noise.',
)
@click.option(#edit
    '-s_p_prob',
    '--salt_pepper_prob',
    default=0,
    help='The probabilty of the salt_pepper noise.',
)
@click.option(#edit
    '-s_p_val',
    '--salt_pepper_val',
    default=None,
    help='The value of the salt_pepper noise.',
)
@click.option(
    '--emission/--transmission',
    default=True,
    help='Specify a transmission or emission noise model.',
)
@click.option(
    '-o',
    '--output-dir',
    default=os.path.join('local', tomopy.__version__),
    help='Folder to put data inside.',
    type=click.Path(exists=False),
)
def project(num_angles, width, phantom, trials, poisson_noise, guassian_tf, guassian_mean, guassian_std, salt_pepper_tf, salt_pepper_prob, salt_pepper_val, emission, output_dir):
    """Simulate data acquisition for tomography using TomoPy.

    Reorder the projections according to opitmal projection ordering and save
    a numpyz file with the original, projections, and angles to the disk.
    """
    simdata_file = os.path.join(output_dir, phantom, 'simulated_data.npz')
    if os.path.isfile(simdata_file):
        logger.warning('Simulated data already exists!')
        return
    if phantom == 'coins':
        pad = (2048 - width) // 2
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, 'images/coins_2048.tif')
        original = plt.imread(filename)[np.newaxis, pad:2048-pad, pad:2048-pad]
    else:
        original = tomopy.peppers(width)
    os.makedirs(os.path.join(output_dir, phantom), exist_ok=True)
    dynam_range = np.max(original)
    plt.imsave(
        os.path.join(output_dir, phantom, 'original.png'),
        original[0, ...],
        format='png',
        cmap=plt.cm.cividis,
        vmin=0,
        vmax=1.1 * dynam_range,
    )
    angles = tomopy.angles(num_angles)
    # Reorder projections optimally
    p = multilevel_order(len(angles)).astype(np.int32)
    # angles = angles[p, ...]
    sinogram = tomopy.project(original, angles, pad=True)
    if trials > 1:
        original = np.tile(original, reps=(trials, 1, 1))
        sinogram = np.tile(sinogram, reps=(1, trials, 1))
    
    if guassian_tf:
        sinogram = tomopy.sim.project.add_gaussian(sinogram, mean = float(guassian_mean), std = float(guassian_std))
    if poisson_noise>0:
        if emission is True:
            sinogram = np.random.poisson(sinogram / poisson_noise) * poisson_noise
        else:
            norm = np.max(sinogram)
            sinogram = -np.log(np.random.poisson(np.exp(-sinogram / norm) * poisson_noise) / poisson_noise) * norm
    if salt_pepper_tf:
        sinogram = tomopy.sim.project.add_salt_pepper(sinogram, prob = float(salt_pepper_prob), val = float(salt_pepper_val))
    logger.info('Original shape: {}, Padded Shape: {}'.format(
        original.shape, sinogram.shape))
    np.savez_compressed(
        simdata_file, original=original, angles=angles, sinogram=sinogram
    )


def fft_order(x):
    """Reorder x according to the 1D Cooley-Tukey FFT access pattern."""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 2:  # this cutoff should be optimized
        return x
    else:
        X_even = fft_order(x[::2])
        X_odd = fft_order(x[1::2])
        return np.concatenate([X_even, X_odd])


def multilevel_order(L):
    """Return integers 0...L ordered by Guan and Gordon multilevel scheme.

    H. Guan and R. Gordon, “A projection access order for speedy convergence
    of ART (algebraic reconstruction technique): a multilevel scheme for
    computed tomography,” Phys. Med. Biol., vol. 39, no. 11, pp. 2005–2022,
    Nov. 1994.
    """
    if L % 2 > 0:
        raise ValueError("L ({}) must be a power of 2".format(L))
    N = 2
    order = list()
    order.append(np.array([0, 1]) / 2)
    level = 4
    while N < L:
        order.append(fft_order(np.arange(1, level, 2)) / level)
        N += level / 2
        level *= 2
    return (np.concatenate(order) * L).astype('int')

if __name__ == '__main__':
    project()
