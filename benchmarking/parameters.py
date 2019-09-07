

def choose_recon_parameters():
    """Choose reconstruction parameters modules based on available modules."""
    # Gridrec is available in all versions
    parameters = [
        # none isn't none, it's ramlak?
        # {'algorithm': 'gridrec', 'filter_name': 'none'},
        {'algorithm': 'gridrec', 'filter_name': 'butterworth'},
        {'algorithm': 'gridrec', 'filter_name': 'cosine'},
        {'algorithm': 'gridrec', 'filter_name': 'hamming'},
        {'algorithm': 'gridrec', 'filter_name': 'hann'},
        {'algorithm': 'gridrec', 'filter_name': 'parzen'},
        {'algorithm': 'gridrec', 'filter_name': 'ramlak'},
        {'algorithm': 'gridrec', 'filter_name': 'shepp'},
    ]
    if float(tomopy.__version__[:3]) < 1.5:
        parameters += [
            # {'algorithm': 'fbp'},  # broken
            {'algorithm': 'art', 'num_iter': num_iter},
            {'algorithm': 'bart', 'num_iter': num_iter},
            {'algorithm': 'mlem', 'num_iter': num_iter},
            {'algorithm': 'osem', 'num_iter': num_iter},
            {'algorithm': 'ospml_hybrid', 'num_iter': num_iter},
            {'algorithm': 'ospml_quad', 'num_iter': num_iter},
            {'algorithm': 'pml_hybrid', 'num_iter': num_iter},
            {'algorithm': 'pml_quad', 'num_iter': num_iter},
            {'algorithm': 'sirt', 'num_iter': num_iter},
        ]
    if float(tomopy.__version__[:3]) >= 1.5:
        parameters += [
            {'algorithm': 'grad', 'num_iter': num_iter, 'reg_par': -1},
            # {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'},
            # {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'LINEAR'},
            # {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'CUBIC'},
            # {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'NN'},
            # {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'LINEAR'},
            # {'algorithm': 'mlem', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'},
            # {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'LINEAR'},
            # {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'gpu', 'interpolation': 'CUBIC'},
            # {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'NN'},
            # {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'LINEAR'},
            # {'algorithm': 'sirt', 'num_iter': num_iter, 'accelerated': True, 'device': 'cpu', 'interpolation': 'CUBIC'},
            {'algorithm': 'tv', 'num_iter': num_iter},
        ]
    try:
        import astra
        parameters += [
            {'algorithm': tomopy.astra, 'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA', 'FilterType': 'ram-lak'}},
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
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options':
                {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': num_iter}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options':
                {'proj_type': 'cuda', 'method': 'CGLS_CUDA', 'num_iter': num_iter}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options':
                {'proj_type': 'cuda', 'method': 'EM_CUDA', 'num_iter': num_iter}},
            {'algorithm': tomopy.astra, 'num_iter': num_iter, 'options':
                {'proj_type': 'cuda', 'method': 'SART_CUDA', 'num_iter': num_iter}},
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
            {'algorithm': tomopy.lprec, 'lpmethod': 'grad', 'num_iter': num_iter}, # broken
            {'algorithm': tomopy.lprec, 'lpmethod': 'tv', 'num_iter': num_iter},
            # {'algorithm': tomopy.lprec, 'lpmethod': 'tve', 'num_iter': num_iter},  # broken
            # {'algorithm': tomopy.lprec, 'lpmethod': 'tvl1', 'num_iter': num_iter},  # broken
        ]
    except ImportError:
        tomopy.lprec = None
    return parameters
