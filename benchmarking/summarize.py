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
@click.option('-v', '--verbose', is_flag=True)
def summarize(phantom, output_dir, trials, summary_file=None, verbose=False):
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
    # Search the phantom folder for results and summarize them
    for folder in glob.glob(os.path.join(base_path, "*")):
        if os.path.isdir(folder):
            # Now using python wall times instead of pyctest times
            # all_results.update(
            #     scrape_algorithm_times(
            #         os.path.join(folder, 'run_tomopy.json')))
            algo_results = scrape_image_quality(folder)
            algo = os.path.basename(folder)
            if algo in all_results:
                all_results[algo].update(algo_results)
            else:
                all_results[algo] = algo_results
            logger.info("Found results for {}".format(algo))
    # Save the results as a JSON
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)
    image_quality_vs_time_plot(summary_plot, all_results, trials)


cm = plt.cm.plasma


linestyles = {
    'gridrec': {'linestyle': '', 'marker': 'o', 'color': cm(0.0)},
    'fbp': {'linestyle': '', 'marker': 'X', 'color': cm(0.1)},
    'art': {'linestyle': '-', 'marker': 'o', 'color': cm(0.4)},
    'bart': {'linestyle': '-', 'marker': '>', 'color': cm(0.45)},
    'sirt': {'linestyle': '-', 'marker': '<', 'color': cm(0.5),
             'fillstyle': 'none', },
    'osem': {'linestyle': '-', 'marker': 's', 'color': cm(0.65),
             'fillstyle': 'none', },
    'ospml_hybrid': {'linestyle': ':', 'marker': 's', 'color': cm(0.7)},
    'ospml_quad': {'linestyle': '--', 'marker': 's', 'color': cm(0.75)},
    'mlem': {'linestyle': '-', 'marker': 'o', 'color': cm(1.0)},
    'pml_hybrid': {'linestyle': ':', 'marker': 'o', 'color': cm(0.95)},
    'pml_quad': {'linestyle': '--', 'marker': 'o', 'color': cm(0.9)},
    'grad': {'linestyle': '-', 'marker': 's', 'color': cm(0.6)},
    'tv': {'linestyle': '-', 'marker': 'd', 'color': cm(0.8)},
}


def image_quality_vs_time_plot(
        plot_name,
        results,
        trials=1,
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
        if algo.lower() in ['gridrec', 'fbp']:
            # These methods are categorical instead of sequential
            time_steps = np.ones(len(results[algo]["num_iter"]))
            xlabel = "number of iterations"
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
                dataxy = np.array([time_steps[i] / trials, results[algo]["quality"][i]])
                plt.annotate(
                    filter_name,
                    xy=dataxy,
                    xytext=dataxy,
                    # va='center',
                    # arrowprops={"arrowstyle": '-'},
                )
        else:
            time_steps = np.array(results[algo]["num_iter"])
            xlabel = "number of iterations"
            if "wall_time" in results[algo]:
                time_steps = results[algo]["wall_time"]
                xlabel = "wall time per slice"
            plt.errorbar(
                x=np.array(time_steps) / trials,
                y=results[algo]["quality"],
                yerr=results[algo]["error"],
                **linestyles[algo.lower()],
            )

    plt.ylim([0, .5])
    plt.semilogx(basex=2)
    plt.xticks(
        [0.1, 1, 5, 10, 30, 60, 5*60, 10*60, 30*60, 3600,],# 3*3600, 6*3600, 12*3600, 24*3600],
        ['0.1s', '1s', '5s', '10s', '30s', '1m', '5m', '10m', '30m', '1h', '3h',],# '6h', '12h', '24h'],
    )


    plt.legend(results.keys(), ncol=3, handlelength=3)
    plt.xlabel(xlabel)
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
            if np.any(np.isnan(data['msssim'])):
                logger.error("Quality rating contains NaN!")
            quality.append(np.nanmean(data['msssim']))
            error.append(np.nanstd(data['msssim']))
            num_iter.append(i)
            wall_time.append(data['time'].item())
        except KeyError:
            logger.warning("MSSSIM data missing from {}".format(file))
            pass  # The data was missing from the file

    num_iter, quality, error, wall_time = zip(*sorted(zip(
        num_iter, quality, error, wall_time)))

    logger.debug("num_iter: {}".format(num_iter))
    logger.debug("quality: {}".format(quality))
    logger.debug("error: {}".format(error))
    logger.debug("wall_time: {}".format(wall_time))

    if "gridrec" in algo_folder or "fbp" in algo_folder:
        pass
    else:
        # Don't add the time of the one iteration to the rest
        wall_time = list(wall_time[0:1]) + list(np.cumsum(list(wall_time[1:])))

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
