"""Provides a command line interface for the benchmarking module."""

import click
import logging

from benchmarking.project import project
from benchmarking.reconstruct import main
from benchmarking.summarize import summarize


@click.group()
@click.version_option()
def benchmark():
    """Reconstruction quality benchmarking for TomoPy and its wrappers."""
    logging.basicConfig(level=logging.INFO)


benchmark.add_command(project)
benchmark.add_command(main)
benchmark.add_command(summarize)
