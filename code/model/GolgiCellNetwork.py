#!/usr/bin/env python3
"""
NeuroML model of cerebellar Golgi cell network

File: code/model/GolgiCellNetwork.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import typing

import neuroml
import typer
from neuroml.utils import component_factory
from pyneuroml.io import write_neuroml2_file
from pyneuroml.lems import generate_lems_file_for_neuroml
from pyneuroml.runners import run_lems_with


class GolgiCellNetwork(object):
    """Cerebellar Golgi Cell network model in NeuroML"""

    network_name = "Golgi_cell_network"
    nml_document = component_factory(neuroml.NeuroMLDocument, id=network_name)
    network = nml_document.add(neuroml.Network, id="Golgi_cell_network", validate=False)

    def __init__(
        self,
        neuroml_file: typing.Optional[str] = None,
        seed: typing.Optional[str] = None,
        lems_file: typing.Optional[str] = None,
        logging_level: typing.Optional[int] = None,
    ):
        """Initialise the model from a parameter file.

        :param neuroml_file: name of NeuroML file to serialise model to
        :type neuroml_file: str
        :param seed: model/simulation seed
        :type seed: str
        :param lems_file: name of LEMS simulation file
        :type lems_file: str
        """
        with open("parameters/general.json") as f:
            general_params = json.load(f)

        self.seed = general_params.get("seed", seed if seed else "1234")
        self.neuroml_file = general_params.get(
            "neuroml_file",
            neuroml_file if neuroml_file else f"{self.network_name}.net.nml",
        )
        self.lems_file = general_params.get(
            "lems_file",
            lems_file if lems_file else f"LEMS_test_Golgi_cells_{self.seed}.xml",
        )

        # set up a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level if logging_level else logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging_level if logging_level else logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

        # set up cli
        self.app = typer.Typer()
        self.app.command()(self.create_model)
        self.app.command()(self.create_simulation)
        self.app.command()(self.simulate)

    def create_model(self):
        """Create the model"""
        self.logger.info("Creating model")
        # create and add more methods here as required
        self.__create_network()

        # write to file
        write_neuroml2_file(self.nml_document, self.neuroml_file)

    def __create_network(self):
        """Create network"""

    def create_simulation(self, lems_file: typing.Optional[str] = None):
        """Create simulation
        :param lems_file: name of LEMS file to serialise simulation to
        :type lems_file: str
        """
        if lems_file:
            self.lems_file = lems_file
        else:
            self.lems_file = "some sane default"

        quantities, sim = generate_lems_file_for_neuroml(
            sim_id="test_golgi_cells",
            neuroml_file=self.neuroml_file,
            target=self.network,
            duration="1500 ms",
            dt="0.01",
            lems_file_name=self.lems_file,
            target_dir=".",
        )

    def simulate(
        self,
        lems_file: typing.Optional[str] = None,
        skip_run: bool = True,
        only_generate_scripts: bool = False,
    ):
        """Simulate the model"""
        if lems_file:
            self.lems_file = lems_file
        else:
            self.lems_file = "some sane default"

        # https://pyneuroml.readthedocs.io/en/development/pyneuroml.runners.html#pyneuroml.runners.run_lems_with
        # you can also use `pynml ..` from the command line to do this
        run_lems_with(
            engine="jneuroml_neuron", lems_file_name=self.lems_file, skip_run=skip_run
        )


if __name__ == "__main__":
    model = GolgiCellNetwork()
    model.app()
