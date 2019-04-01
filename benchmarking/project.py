"""Generate sinograms from phantoms."""

import os
import logging

import click
import tomopy
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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

@click.command()
@click.option('-p', '--phantom', default='peppers', help='Name of a phantom.')
@click.option('-w', '--width', default=256, help='Pixel width of phantom before padding.', type=int)
@click.option('-a', '--num_angles', default=256, help='Number of projection angles.', type=int)
@click.option('-t', '--trials', default=1, help='Number of phantom repeitions.', type=int)
@click.option('-n', '--noise', help='Whether to add noise.', is_flag=True)
def project(num_angles, width, phantom, trials, noise):
    """Simulate data acquisition for tomography using TomoPy.

    Reorder the projections according to opitmal projection ordering and save
    a numpyz file with the original, projections, and angles to the disk.
    """
    original = tomopy.peppers(width)
    os.makedirs(phantom, exist_ok=True)
    dynam_range = np.max(original)
    plt.imsave(
        '{}/original.png'.format(phantom), original[0, ...],
        format='png',
        cmap=plt.cm.cividis,
        vmin=0, vmax=1.1*dynam_range,
        )
    if trials > 1:
        original = np.tile(original, reps=(trials, 1, 1))
    angles = tomopy.angles(num_angles)
    # Reorder projections optimally
    p = multilevel_order(len(angles)).astype(np.int32)
    angles = angles[p, ...]
    sinogram = tomopy.project(original, angles, pad=True)
    if noise:
        sinogram = np.random.poisson(sinogram)
    logger.info('Original shape: {}, Padded Shape: {}'.format(original.shape, sinogram.shape))
    np.savez(
        '{}/simulated_data.npz'.format(phantom),
        original=original, angles=angles, sinogram=sinogram
    )


if __name__ == '__main__':
    project()
