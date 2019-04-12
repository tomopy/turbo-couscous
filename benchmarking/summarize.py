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
import tomopy

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
def summarize(phantom, output_dir, summary_file=None):
    """Scrape reconstructions data and summarize it in a JSON.

    If the JSON exists already, it will be updated instead of replaced.
    """
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
    # Search the phantom folder for results and summarize them
    for folder in glob.glob(os.path.join(base_path, "*")):
        if os.path.isdir(folder):
            all_results.update(
                scrape_algorithm_times(
                    os.path.join(folder, 'run_tomopy.json')))
            algo_results = scrape_image_quality(folder)
            algo = os.path.basename(folder)
            all_results[algo].update(algo_results)
            logger.info("Found results for {}".format(algo))
    # Save the results as a JSON
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)
    image_quality_vs_time_plot(summary_plot, all_results)


def image_quality_vs_time_plot(
        plot_name,
        results,
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
    plt.figure(dpi=600)

    for algo in results.keys():
        if algo in ['gridrec', 'fbp']:
            # These methods are categorical instead of sequential
            time_steps = np.ones(len(results[algo]["num_iter"]))
            time_steps = time_steps * results[algo]["wall_time"]
            plt.errorbar(
                x=time_steps,
                y=results[algo]["quality"],
                yerr=results[algo]["error"],
                fmt='o',
            )
            for i, filter_name in enumerate(results[algo]["num_iter"]):
                plt.annotate(
                    filter_name,
                    (time_steps[i], results[algo]["quality"][i]),
                    va='center',
                )
        else:
            time_steps = np.array(results[algo]["num_iter"])
            time_steps = time_steps * results[algo]["wall_time"] / 32
            logger.warning("Assumed wall_time is for 32 iterations.")
            plt.errorbar(
                x=time_steps,
                y=results[algo]["quality"],
                yerr=results[algo]["error"],
                fmt='-o',
            )

    plt.ylim([0, 1])
    plt.xlim([0, 50000])

    plt.legend(results.keys())
    plt.xlabel('wall time [s]')
    plt.ylabel('MS-SSIM Index')
    plt.title(os.path.dirname(os.path.realpath(plot_name)))

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
        try:
            quality.append(np.mean(data['msssim']))
            error.append(np.std(data['msssim']))
            num_iter.append(i)
        except KeyError:
            logger.warning("MSSSIM data missing from {}".format(file))
            pass  # The data was missing from the file

    logger.debug("num_iter: {}".format(num_iter))
    logger.debug("quality: {}".format(quality))
    logger.debug("error: {}".format(error))

    return {
        "quality": quality,
        "error": error,
        "num_iter": num_iter,
    }


def scrape_algorithm_times(json_filename):
    """Scrape wall times from the timemory json.

    Search for timer tags containing "algorithm='{algorithm}'" then extract
    the wall times. Return a new dictionary of dictionaries
    where the first key is the algorithm name and the second key is "wall time".
    """
    with open(json_filename, "r") as file:
        data = json.load(file)

    results = {}

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
