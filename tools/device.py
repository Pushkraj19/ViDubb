import torch


def select_device(requested='auto'):
    available = {'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available(), 'cpu': True}
    if requested == 'auto':
        return next(name for name, supported in available.items() if supported)
    if requested not in available or not available[requested]:
        raise ValueError(f'Device {requested!r} is unavailable; choose auto or cpu.')
    return requested


def whisper_options(device):
    # faster-whisper/CTranslate2 has no MPS backend.
    if device == 'cuda':
        return {'device': 'cuda', 'compute_type': 'float16'}
    return {'device': 'cpu', 'compute_type': 'int8'}
