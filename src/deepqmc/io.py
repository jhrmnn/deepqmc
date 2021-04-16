import importlib
import logging

import torch
import yaml

from .errors import InputError
from .extra.debug import NestedDict
from .molecule import Molecule
from .wf import ANSATZES

log = logging.getLogger(__name__)

__all__ = ()


def validate_params(params):
    REQUIRED = {'system', 'ansatz'}
    OPTIONAL = {'train_kwargs', 'evaluate_kwargs'} | {f'{a}_kwargs' for a in ANSATZES}
    params = set(params or [])
    missing = REQUIRED - params
    if missing:
        raise InputError(f'Missing keywords: {missing}')
    unknown = params - REQUIRED - OPTIONAL
    if unknown:
        raise InputError(f'Unknown keywords: {unknown}')


def import_fullname(fullname):
    module_name, qualname = fullname.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, qualname)


def wf_from_file(workdir, extra_params=None):
    params = (
        yaml.safe_load((workdir / 'param.yaml').read_text())
        if (workdir / 'param.yaml').exists()
        else {}
    )
    if extra_params:
        params = NestedDict(params)
        for p in extra_params:
            k, v = p.split('=', 1)
            v = yaml.safe_load(v)
            params[k] = v
    if not params and (workdir / 'param.toml').exists():
        import warnings

        import toml

        params = toml.loads((workdir / 'param.toml').read_text())
        warnings.warn('TOML input files will be deprecated', FutureWarning)
    validate_params(params)
    state_file = workdir / 'state.pt'
    state = torch.load(state_file) if state_file.is_file() else None
    if state:
        log.info(f'State loaded from {state_file}')
    system = params.pop('system')
    if isinstance(system, str):
        name, system = system, {}
    else:
        name = system.pop('name', None)
    if name is None:
        mol = Molecule(**system)
    elif ':' in name:
        mol = import_fullname(name)(**system)
    else:
        mol = Molecule.from_name(name, **system)
    ansatz = params.pop('ansatz')
    ansatz = ANSATZES[ansatz]
    kwargs = params.pop(f'{ansatz.name}_kwargs', {})
    if ansatz.uses_workdir:
        assert 'workdir' not in kwargs
        kwargs['workdir'] = workdir
        workdir.mkdir(parents=True, exist_ok=True)
    wf = ansatz.entry(mol, **kwargs)
    return wf, params, state
