#!/usr/bin/env python3
"""
Generate membrane potential plots for cell variants with different step
currents to characterise them.

File: analyse_cells.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import glob
import logging

import matplotlib as mpl
from pyneuroml.analysis import generate_current_vs_frequency_curve
from pyneuroml.io import read_neuroml2_file
from pyneuroml.plot.PlotTimeSeries import plot_time_series_from_lems_file

mpl.rcParams["figure.dpi"] = "300"
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

    for cellfile in cellfiles:
        cell = read_neuroml2_file(cellfile).cells[0]
        cell_id = cell.id
        generate_current_vs_frequency_curve(
            cellfile,
            cell_id=cell_id,
            start_amp_nA=-10.0e-3,
            end_amp_nA=200.0e-3,
            step_nA=10e-3,
            pre_zero_pulse=200,
            analysis_duration=500,
            num_processors=8,
            simulator="jNeuroML_NEURON",
            dt=0.01,
            plot_if=False,
            show_plot_already=False,
        )

        lems_file = f"LEMS_iv_{cell_id}.xml"
        plot_time_series_from_lems_file(
            lems_file,
            offset=True,
            title=cell_id,
            show_plot_already=False,
            labels=False,
            save_figure_to=f"{lems_file.replace('.xml', '.png')}",
            bottom_left_spines_only=True,
            title_above_plot=True,
            close_plot=True,
        )


if __name__ == "__main__":
    test_cells()
