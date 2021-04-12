"""Defines linstyles for all possible tomopy algorithms"""

from collections import defaultdict

import matplotlib.pyplot as plt

cm = plt.cm.plasma

linestyles = defaultdict(
    lambda: {
        'linestyle': '-',
        'marker': '',
        'color': cm(0)
    },
    {
        # direct methods
        'gridrec': {
            'linestyle': '',
            'marker': 'o',
            'color': cm(0.0)
        },
        'fbp': {
            'linestyle': '',
            'marker': 'X',
            'color': cm(0.1)
        },
        'astra-fbp_cuda': {
            'linestyle': '',
            'marker': 'X',
            'color': cm(0.1),
            'fillstyle': 'none',
        },
        # iterative methods
        'art': {
            'linestyle': '-',
            'marker': '^',
            'color': cm(0.4)
        },
        'bart': {
            'linestyle': '-',
            'marker': '>',
            'color': cm(0.45)
        },
        'astra-sart_cuda': {
            'linestyle': '-',
            'marker': '>',
            'color': cm(0.45),
            'fillstyle': 'none',
        },
        'sirt': {
            'linestyle': '-',
            'marker': '<',
            'color': cm(0.5)
        },
        'sirt_cuda': {
            'linestyle': '-',
            'marker': '<',
            'color': cm(0.5),
            'fillstyle': 'none',
        },
        'astra-sirt_cuda': {
            'linestyle': '--',
            'marker': '<',
            'color': cm(0.5),
            'fillstyle': 'none',
        },
        'osem': {
            'linestyle': '-',
            'marker': 's',
            'color': cm(0.65)
        },
        'ospml_hybrid': {
            'linestyle': ':',
            'marker': 's',
            'color': cm(0.7)
        },
        'ospml_quad': {
            'linestyle': '--',
            'marker': 's',
            'color': cm(0.75)
        },
        'mlem': {
            'linestyle': '-',
            'marker': 'o',
            'color': cm(1.0)
        },
        'lprec-em': {
            'linestyle': '-',
            'marker': 'o',
            'color': cm(1.0),
            'fillstyle': 'none',
        },
        'astra-em_cuda': {
            'linestyle': '--',
            'marker': 'o',
            'color': cm(1.0),
            'fillstyle': 'none',
        },
        'pml_hybrid': {
            'linestyle': ':',
            'marker': 'o',
            'color': cm(0.95)
        },
        'pml_quad': {
            'linestyle': '--',
            'marker': 'o',
            'color': cm(0.9)
        },
        # gradient methods
        'grad': {
            'linestyle': '-',
            'marker': 's',
            'color': cm(0.6)
        },
        'lprec-grad': {
            'linestyle': '-',
            'marker': 's',
            'color': cm(0.6),
            'fillstyle': 'none',
        },
        'lprec-cg': {
            'linestyle': '--',
            'marker': 's',
            'color': cm(0.6),
            'fillstyle': 'none',
        },
        'astra-cgls_cuda': {
            'linestyle': ':',
            'marker': 's',
            'color': cm(0.6),
            'fillstyle': 'none',
        },
        'tv': {
            'linestyle': '-',
            'marker': 'd',
            'color': cm(0.8)
        },
        'lprec-tv': {
            'linestyle': '-',
            'marker': 'd',
            'color': cm(0.8),
            'fillstyle': 'none',
        },
        'lprec-tve': {
            'linestyle': '--',
            'marker': 'd',
            'color': cm(0.8),
            'fillstyle': 'none',
        },
        'lprec-tvl1': {
            'linestyle': ':',
            'marker': 'd',
            'color': cm(0.8),
            'fillstyle': 'none',
        },
    },
)
