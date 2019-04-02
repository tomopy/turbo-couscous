"""Functions for creating a plot of image quality vs reconstruction time."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import json
import logging
import os
import re

import click
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '-p',
    '--phantom',
    default='peppers',
    help='Name of a phantom.',
)
@click.option(
    '-o',
    '--output-dir',
    default='',
    help='Folder to put data inside',
    type=click.Path(exists=False),
)
def summarize(phantom, output_dir, summary_file=None):
    """Scrape reconstructions data and summarize it in a JSON.

    If the JSON exists already, it will be updated instead of replaced.
    """
    base_path = os.path.join(output_dir, phantom)
    # Load data from file or make empty dictionary
    if summary_file is None:
        summary_file = os.path.join(base_path, 'summary.json')
    if os.path.isfile(summary_file):
        with open(summary_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = dict()
    logger.info("Summary is located at {}".format(summary_file))
    # Search the phantom folder for results and summarize them
    for folder in glob.glob(os.path.join(base_path, "*")):
        if os.path.isdir(folder):
            algo_results = scrape_image_quality(folder)
            algo = os.path.basename(folder)
            all_results[algo] = algo_results
            logger.info("Found results for {}".format(algo))
    # Save the results as a JSON
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)


def image_quality_vs_time_plot(
        plot_name,
        json_filename,
        algo_folder_dir,
):
    """Create a lineplot with errorbars of image quality vs time.

    The vertical axis is the MS-SSIM metric from 0 to 1 and the horizontal axis
    is the reconstruction wall time for each algorithm in seconds. The plot is
    saved to the disk.

    Parameters
    ----------
    plot_name : file path
        The output path and filename including the file extension.
    json_filename : file path
        The name of the timemory json to scrape for wall time data.
    algo_folder_dir: folder path
        The folder to look in for each of the directories named after each
        algorithm. There should be one folder for each algorithm named in the
        timemory JSON. Each of these folders contains a series of numbered
        npz files.

    """
    raise NotImplementedError()

    results = scrape_algorithm_times(json_filename)

    for algo in results.keys():
        algo_folder = os.path.join(algo_folder_dir, algo)
        results[algo].update(scrape_image_quality(algo_folder))

    plt.figure(dpi=600)

    for algo in results.keys():
        # Normalize the iterations to the range [0, total_wall_time]
        time_steps = (
            np.array(results[algo]["num_iter"]) / results[algo]["num_iter"][-1]
            * results[algo]["wall time"])
        plt.errorbar(
            x=time_steps,
            y=results[algo]["quality"],
            yerr=results[algo]["error"],
            fmt='-o')

    plt.ylim([0, 1])

    plt.legend(results.keys())
    plt.xlabel('time [s]')
    plt.ylabel('MS-SSIM Index')

    plt.savefig(plot_name, dpi=600, pad_inches=0.0)


def scrape_image_quality(algo_folder):
    """Scrape the quality std error and iteration numbers from the files.

    {algo_folder} is a folder containing files {algo}.{num_iter}.npz with
    keyword "mssim" pointing to an array of quality values.

    Return a dictionary for the folder containing three lists for the
    concatenated image quality at each iteration, std error at each iteration,
    and the number of each iteration. The length of the lists is the number of
    files in {algo_folder}.
    """
    quality = list()
    error = list()
    num_iter = list()

    for file in glob.glob(os.path.join(algo_folder, "*.npz")):
        data = np.load(file)
        # get the iteration number from the filename sans extension
        i = os.path.basename(file).split(".")[1]
        try:
            i = int(i)
        except ValueError:
            pass  # Gridrec and FBP have filter names instead
        num_iter.append(i)
        quality.append(np.mean(data['msssim']))
        error.append(np.std(data['msssim']))

    logger.debug("num_iter: {}".format(num_iter))
    logger.debug("quality: {}".format(quality))
    logger.debug("error: {}".format(error))

    return {
        "quality": quality,
        "error": error,
        "num_iter": num_iter,
    }


if __name__ == '__main__':
    summarize()
