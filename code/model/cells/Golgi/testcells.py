#!/usr/bin/env python3
"""
Test cell models

File: testcells.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import glob
import logging

import neuroml
from neuroml.utils import component_factory
from pyneuroml.io import read_neuroml2_file, write_neuroml2_file
from pyneuroml.lems import generate_lems_file_for_neuroml

logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.propagate = False


def test_cells():
    """Test cells with input currents"""
    neuroml_file = "GoC_cells_test.net.nml"
    cellfiles = glob.glob("*.cell.nml")
    logger.debug(f"Cell files are: {cellfiles}")

    newdoc = component_factory(neuroml.NeuroMLDocument, id="test_golgi_cells")
    net = newdoc.add(neuroml.Network, id="test_golgi_cell_network", validate=False)

    pg = newdoc.add(
        neuroml.PulseGenerator,
        id="pg",
        delay="5ms",
        duration="1000ms",
        amplitude="0.07nA",
    )

    ctr = 0
    for afile in cellfiles:
        newdoc.add(neuroml.IncludeType, href=afile)
        acell = read_neuroml2_file(afile).cells[0]
        pop = net.add(
            neuroml.Population, id=f"golgi_cell_pop_{ctr}", size=1, component=acell.id
        )
        exp_input = net.add(
            neuroml.ExplicitInput, target=f"{pop.id}[{ctr}]", input=pg.id
        )
        ctr += 1

    logger.debug(newdoc)
    write_neuroml2_file(newdoc, neuroml_file)

    quantities, sim = generate_lems_file_for_neuroml(
        sim_id="test_golgi_cells",
        neuroml_file=neuroml_file,
        target=net.id,
        duration="1500 ms",
        dt="0.01",
        lems_file_name="LEMS_test_Golgi_cells.xml",
        target_dir=".",
    )


if __name__ == "__main__":
    test_cells()
