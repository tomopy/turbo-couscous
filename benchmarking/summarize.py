"""A CLI for creating a line plot of reconstructed image quality vs
reconstruction time."""

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
import tomopy
import xdesign as xd

from .linestyles import linestyles

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
    default=os.path.join('local', tomopy.__version__),
    help='Folder to put data inside',
    type=click.Path(exists=False),
)
@click.option(
    '-t',
    '--trials',
    default=1,
    help='Number of phantom repetitions.',
    type=int,
)
@click.option(
    '--title',
    default=None,
    help='A title for the summary plot.',
    type=str,
)
@click.option('-v', '--verbose', is_flag=True)
def summarize(
    phantom,
    output_dir,
    trials,
    title,
    summary_file=None,
    verbose=False,
):
    """Scrape reconstructions data and summarize it in a JSON.

    If the JSON exists already, it will be updated instead of replaced.
    """
    if verbose:
        logger.setLevel(level=logging.DEBUG)
    base_path = os.path.join(output_dir, phantom)

    # Load data from file or make empty dictionary
    if summary_file is None:
        summary_file = os.path.join(base_path, 'summary.json')
        summary_plot = os.path.join(base_path, 'summary.svg')
    if os.path.isfile(summary_file):
        with open(summary_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = dict()
    logger.info("Summary is located at {}".format(summary_file))

    # Load the original for comparison
    original = np.load(os.path.join(output_dir, phantom,
                                    'simulated_data.npz'))['original']

    # Search the phantom folder for results and summarize them
    for folder in glob.glob(os.path.join(base_path, "*")):
        if os.path.isdir(folder):
            algo_results = scrape_image_quality(folder, original)
            algo = os.path.basename(folder)
            if algo in all_results:
                all_results[algo].update(algo_results)
            else:
                all_results[algo] = algo_results
            concat_recons(folder, base_path)
            logger.info("Found results for {}".format(algo))

    # Save the results as a JSON
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)
    image_quality_vs_time_plot(summary_plot, all_results, trials, title)


def image_quality_vs_time_plot(
    plot_name,
    results,
    trials=1,
    title=None,
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
    plt.figure(dpi=600, figsize=(8., 4.5))
    xlabel = "number of iterations"
    for algo in results.keys():
        if 'gridrec' in algo.lower() or 'fbp' in algo.lower():
            # These methods are categorical instead of sequential
            time_steps = np.ones(len(results[algo]["num_iter"]))
            if "wall_time" in results[algo]:
                time_steps = results[algo]["wall_time"]
                xlabel = "wall time per slice"
            plt.errorbar(
                x=np.array(time_steps) / trials,
                y=results[algo]["quality"],
                yerr=results[algo]["error"],
                **linestyles[algo.lower()],
            )
            for i, filter_name in enumerate(results[algo]["num_iter"]):
                dataxy = np.array(
                    [time_steps[i] / trials, results[algo]["quality"][i]])
                plt.annotate(
                    filter_name,
                    xy=dataxy,
                    xytext=dataxy,
                    # va='center',
                    # arrowprops={"arrowstyle": '-'},
                )
        else:
            time_steps = np.array(results[algo]["num_iter"])
            if "wall_time" in results[algo]:
                time_steps = results[algo]["wall_time"]
                xlabel = "wall time per slice"
            plt.errorbar(
                x=np.array(time_steps) / trials,
                y=results[algo]["quality"],
                yerr=results[algo]["error"],
                **linestyles[algo.lower()],
            )

    plt.ylim([0, 1])
    plt.xlim([0.01, 45*60])
    plt.semilogx(basex=2)
    plt.xticks(
        [0.01, 1, 5, 10, 30, 60, 5 * 60, 10 * 60, 30 * 60, 45 * 60],
        ['0.01s', '1s', '5s', '10s', '30s', '1m', '5m', '10m', '30m', '45m'],
    )

    plt.legend(
        results.keys(),
        ncol=1,
        handlelength=3,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.,
    )
    plt.xlabel(xlabel)
    plt.ylabel('MS-SSIM Index')
    if title is None:
        plt.title(os.path.dirname(os.path.realpath(plot_name)))
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=600, pad_inches=0.0)


def concat_recons(algo_folder, base_path):
    recons = list()
    for file in glob.glob(os.path.join(algo_folder, "*.png")):
        print(file)
        recons.append(plt.imread(file))

    # split into chunks of five images
    for i in range(0, np.ceil(len(recons) / 5).astype(int)):
        combined = np.concatenate(
            recons[5 * i:min(len(recons), 5 * (i + 1))],
            axis=1,
        )
        plt.imsave(algo_folder + '-{}.png'.format(i), combined)


def scrape_image_quality(algo_folder, original):
    """Scrape the quality std error and iteration numbers from the files.

    {algo_folder} is a folder containing files {algo}.{num_iter}.npz which
    contain key 'recon' which is the recosntructed images for comparison and
    key 'total_time' which is the total reconstruction time for the image in
    the file.

    Return a dictionary for the folder containing lists for the concatenated
    image quality at each iteration, std error at each iteration, the number of
    each iteration, and the total wall time for each iteration. The length of
    the lists is the number of files in {algo_folder}.
    """
    dynamic_range = np.max(original)

    quality = list()
    error = list()
    num_iter = list()
    wall_time = list()

    for file in glob.glob(os.path.join(algo_folder, "*.npz")):
        data = np.load(file)
        filename = os.path.basename(file)
        keywords = filename.split(".")
        if "gridrec" in filename or "fbp" in filename:
            # get the filter name from the filename sans extension
            if len(keywords) > 3:
                i = keywords[-3]
            else:
                i = ""
        else:
            # get the iteration number from the filename sans extension
            i = keywords[-2]
            i = int(i)

        try:
            # compute quality metrics
            recon = data['recon']
            pad = (recon.shape[-1] - original.shape[-1]) // 2
            msssim = np.empty(len(recon))
            for z in range(len(recon)):
                # compute the reconstructed image quality metrics
                scales, msssim[z], quality_maps = xd.msssim(
                    original[z],
                    recon[z, pad:recon.shape[1] - pad,
                          pad:recon.shape[2] - pad],
                    L=dynamic_range,
                    sigma=11,
                )
            if np.any(np.isnan(msssim)):
                logger.error("Quality rating contains NaN!")
            quality.append(np.nanmean(msssim))
            error.append(np.nanstd(msssim))
            num_iter.append(i)
            wall_time.append(data['total_time'].item())
        except (KeyError, FileNotFoundError):
            logger.warning("MSSSIM data missing from {}".format(file))
            pass  # The data was missing from the file

    num_iter, quality, error, wall_time = zip(
        *sorted(zip(num_iter, quality, error, wall_time)))

    logger.debug("num_iter: {}".format(num_iter))
    logger.debug("quality: {}".format(quality))
    logger.debug("error: {}".format(error))
    logger.debug("wall_time: {}".format(wall_time))

    return {
        "quality": quality,
        "error": error,
        "num_iter": num_iter,
        "wall_time": wall_time,
    }


def scrape_algorithm_times(json_filename):
    """Scrape wall times from the timemory json.

    Search for timer tags containing "algorithm='{algorithm}'" then extract
    the wall times. Return a new dictionary of dictionaries
    where the first key is the algorithm name and the second key is "wall time".
    """
    results = {}

    if os.path.isfile(json_filename):
        with open(json_filename, "r") as file:
            data = json.load(file)
    else:
        return results

    for timer in data["ranks"][0]["manager"]["timers"]:
        # only choose timer with "algorithm in the tag
        if "algorithm" in timer["timer.tag"]:
            # find the part that contains the algorithm name
            m = re.search("'.*'", timer["timer.tag"])
            # strip off the single quotes
            clean_tag = m.group(0).strip("'")
            # convert microseconds to seconds
            wtime = timer["timer.ref"]["wall_elapsed"] * 1e-6

            print("{tag:>10} had a wall time of {wt:10.3g} s".format(
                tag=clean_tag,
                wt=wtime,
            ))

            results[clean_tag] = {
                "wall_time": wtime,
            }

    return results


if __name__ == '__main__':
    summarize()
