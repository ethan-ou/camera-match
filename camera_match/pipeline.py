from colour import cctf_decoding, cctf_encoding

def curves_pipeline(source_gamma, target_gamma, curves='RBF'):
    return [
        {'transform': 'curves', 'type': curves},
        {'transform': 'gamma_decode', 'type': source_gamma},
        {'transform': 'gamma_encode', 'type': target_gamma},
        {'solver': 'curves'},
    ]

def matrix_pipeline(source_gamma, target_gamma, matrix='RPCC'):
    return [
        {'transform': 'gamma_decode', 'type': source_gamma},
        {'transform': 'matrix', 'type': matrix},
        {'transform': 'gamma_encode', 'type': target_gamma},
        {'solver': 'matrix'}
    ]

def nonlinear_matrix_pipeline(source_gamma, target_gamma, matrix='RPCC'):
    return [
        {'transform': 'gamma_decode', 'type': source_gamma},
        {'transform': 'gamma_encode', 'type': target_gamma},
        {'transform': 'matrix', 'type': matrix},
        {'solver': 'matrix'}
    ]

def interpolation_pipeline(source_gamma, target_gamma, interpolation='RBF'):
    return [
        {'transform': 'gamma_decode', 'type': source_gamma},
        {'transform': 'gamma_encode', 'type': target_gamma},
        {'transform': 'interpolation', 'type': interpolation},
        {'solver': 'interpolation'}
    ]

PIPELINE_STEPS = {
    'gamma_decode': cctf_decoding,
    'gamma_encode': cctf_encoding,
    'matrix': None,
    'curves': None,
    'interpolation': None,
}

def pipeline_creator(source, target, source_gamma, target_gamma, pipeline_type='interpolation',
    curves='RBF', matrix='RPCC', interpolation='RBF'):

    pass

def pipeline_reader(source, target, source_gamma, target_gamma, pipeline):
    for step in pipeline:
        if 'transform' in step:
            if step['transform'] == 'curves':



            pass
        elif 'solver' in step:
            pass
