#!/usr/bin/env python3
"""
Test cell models

File: testcells.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import glob
import logging

import numpy
from pyneuroml.analysis import generate_current_vs_frequency_curve
from pyneuroml.io import read_neuroml2_file

logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.propagate = False


def test_cells() -> None:
    """Test cells with input currents"""
    cellfiles = glob.glob("GoC_0*.cell.nml")
    logger.debug(f"Cell files are: {cellfiles}")

    # TODO: what current do Golgi cells receive in reality?
    for cellfile in cellfiles[:1]:
        cell = read_neuroml2_file(cellfile).cells[0]
        cell_id = cell.id
        generate_current_vs_frequency_curve(
            cellfile,
            cell_id=cell_id,
            start_amp_nA=0.0,
            custom_amps_nA=numpy.arange(-50.0, 10.0, 5.0, dtype=float),
            pre_zero_pulse=200,
            analysis_duration=500,
            plot_voltage_traces=True,
            num_processors=8,
            simulator="jNeuroML_NEURON",
            dt=0.01,
        )


if __name__ == "__main__":
    test_cells()
