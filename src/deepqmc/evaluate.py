from itertools import count
from pathlib import Path

import h5py
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from uncertainties import unumpy as unp

from .sampling import LangevinSampler, sample_wf
from .utils import H5LogTable

__version__ = '0.1.0'
__all__ = ['evaluate']


def evaluate(
    wf,
    molecules=None,
    store_steps=False,
    workdir=None,
    log_dicts=None,
    *,
    n_steps=500,
    sample_size=1_000,
    sample_kwargs=None,
    sampler_kwargs=None,
):
    r"""Evaluate a wave function model.

    This is a top-level API function that rigorously evaluates a trained wave
    function model. It initializes a :class:`~deepqmc.sampling.LangevinSampler`,
    sets up a Tensorboard writer, and calls :func:`~deepqmc.sampling.sample_wf`.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function model to be evaluated
        store_steps (bool): whether to store individual sampled electron configuraitons
        workdir (str): path where to store Tensorboard event file and HDF5 file with
            sampling block energies
        n_steps (int): number of sampling steps
        sample_size (int): number of Markov-chain walkers
        n_decorrelate (int): number of extra steps between samples included
            in the expectation value averaging
        sampler_kwargs (dict): extra arguments passed to
            :class:`~deepqmc.sampling.LangevinSampler`
        sample_kwargs (dict): extra arguments passed to
            :func:`~deepqmc.sampling.sample_wf`

    Returns:
        dict: Expectation values with standard errors.
    """
    molecules = molecules or {'wf.mol': wf.mol}
    log_dicts = log_dicts or {name: None for name in molecules}
    tables_blocks = {name: None for name in molecules}
    tables_steps = {name: None for name in molecules}
    writers = {name: None for name in molecules}
    if workdir:
        workdir = Path(workdir)
        h5file = h5py.File(workdir / 'sample.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        for name, molecule in molecules.items():
            writers[name] = SummaryWriter(log_dir=f'{workdir}/{name}', flush_secs=15)
            group = h5file.require_group(f'{name}')
            group.attrs.create('geometry', molecule.coords.tolist())
            tables_blocks[name] = H5LogTable(group.require_group('blocks'))
            if store_steps:
                tables_steps[name] = H5LogTable(group.require_group('steps'))
    samplers = {
        name: LangevinSampler.from_wf(
            wf,
            mol=mol,
            sample_size=sample_size,
            writer=writer,
            n_discard=0,
            **{'n_decorrelate': 4, **(sampler_kwargs or {})},
        )
        for (name, mol), writer in zip(molecules.items(), writers.values())
    }
    for name, sampler in samplers.items():
        with tqdm(count(), desc=f'equilibrating sampler {name}', disable=None) as steps:
            next(sample_wf(wf, sampler, steps, equilibrate=True))
    steps = trange(0, n_steps, total=n_steps, desc='evaluating', disable=None)
    blocks = []
    try:
        for ((step, *_), energies) in map(
            lambda x: zip(*x),
            zip(
                *(
                    sample_wf(
                        wf,
                        sampler,
                        steps,
                        blocks=blocks,
                        log_dict=log_dict
                        if log_dict is not None
                        else table_steps.row
                        if workdir and store_steps
                        else None,
                        writer=writer,
                        equilibrate=False,
                        **(sample_kwargs or {}),
                    )
                    for (sampler, writer, log_dict, table_steps) in zip(
                        samplers.values(),
                        writers.values(),
                        log_dicts.values(),
                        tables_steps.values(),
                    )
                )
            ),
        ):
            if step == 0:
                continue
            if all(energies):
                energies_rep = '|'.join([f'{energy:S}' for energy in energies])
                steps.set_postfix(E=f'{energies_rep}')
            if workdir:
                for table_blocks in tables_blocks.values():
                    if len(blocks) > len(table_blocks['energy']):
                        block = blocks[-1]
                        table_blocks.row['energy'] = np.stack(
                            [unp.nominal_values(block), unp.std_devs(block)], -1
                        )
                h5file.flush()
    finally:
        steps.close()
        if workdir:
            for writer in writers.values():
                writer.close()
            h5file.close()
    return {f'energy {name}': energy for name, energy in zip(molecules, energies)}
