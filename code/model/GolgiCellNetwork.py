#!/usr/bin/env python3
"""
NeuroML model of cerebellar Golgi cell network

File: code/model/GolgiCellNetwork.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import glob
import json
import logging
import typing

import neuroml
import numpy
import scipy
import typer
from neuroml.utils import component_factory
from pyneuroml.annotations import create_annotation
from pyneuroml.io import read_neuroml2_file, write_neuroml2_file
from pyneuroml.lems import generate_lems_file_for_neuroml
from pyneuroml.runners import run_lems_with
from pyneuroml.utils.units import split_nml2_quantity


class GolgiCellNetwork(object):
    """Cerebellar Golgi Cell network model in NeuroML"""

    network_name = "Golgi_cell_network"
    nml_document = component_factory(neuroml.NeuroMLDocument, id=network_name)
    annotation = create_annotation(
        subject="Golgi_cell_network",
        abstract="Cerebellar Golgi Cell network model",
        title="Cerebellar Golgi Cell network model",
        annotation_style="miriam",
        xml_header=False,
        keywords=["Golgi cell", "Cerebellum"],
        creation_date="2025-06-21",
        authors={
            "Ankur Sinha": {
                "ankur.sinha@ucl.ac.uk": "email",
                "https://orcid.org/0000-0001-7568-7167": "orcid",
            }
        },
        sources={"https://github.com/sanjayankur31/GolgiCellNetwork/": "GitHub"},
        is_version_of={
            "https://github.com/harshagurnani/GoC_Network_Sim_BehInputs": "Gurnani 2021"
        },
        references={
            "https://doi.org/10.1016/j.neuron.2021.03.027": "Gurnani and Silver, 2021, Neuron 109, 1-15",
            "https://doi.org/10.1016/j.neuron.2010.06.028": "Vervaeke et al, 2010, Neuron 67, 435 - 451",
        },
    )
    nml_document.annotation = neuroml.Annotation([annotation])
    network = nml_document.add(neuroml.Network, id="Golgi_cell_network", validate=False)

    def __init__(
        self,
    ):
        """Initialise the model"""
        # set up cli
        self.app = typer.Typer(help="Cerebellar Golgi Cell network model in NeuroML")
        self.app.command()(self.create_model)
        self.app.command()(self.create_simulation)
        self.app.command()(self.simulate)
        self.app.callback()(self.configure)

    def configure(
        self,
        code_config_file: str = "parameters/general.json",
        model_parameters_file: str = "parameters/model.json",
        neuroml_file: typing.Optional[str] = None,
        seed: typing.Optional[str] = None,
        label: typing.Optional[str] = None,
        lems_file: typing.Optional[str] = None,
        logging_level: typing.Optional[str] = None,
    ):
        """Configure model

        :param neuroml_file: name of NeuroML file to serialise model to
        :type neuroml_file: str
        :param seed: model/simulation seed
        :type seed: str
        :param lems_file: name of LEMS simulation file
        :type lems_file: str

        """
        self.code_config_file = code_config_file
        self.model_parameters_file = model_parameters_file
        with open(self.code_config_file) as f:
            self.general_params = json.load(f)

        if seed:
            self.seed = seed
        else:
            self.seed = self.general_params.get("seed", "1234")

        if label:
            provided_label = label
        else:
            provided_label = self.general_params.get("label")
        self.label = f"_{provided_label.replace(' ', '_')}" if provided_label else ""

        if neuroml_file:
            self.neuroml_file = neuroml.file
        else:
            self.neuroml_file = self.general_params.get(
                "neuroml_file",
                f"{self.network_name}{self.label}_{self.seed}.net.nml",
            )

        if lems_file:
            self.lems_file = lems_file
        else:
            self.lems_file = self.general_params.get(
                "lems_file",
                f"LEMS_test_Golgi_cells{self.label}_{self.seed}.xml",
            )

        if logging_level:
            self.logging_level = logging_level
        else:
            self.logging_level = self.general_params.get(
                "logging_level",
                "DEBUG",
            )

        # set up a logger
        self.logger = logging.getLogger(self.network_name)
        self.logger.setLevel(getattr(logging, self.logging_level))
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self.logging_level))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

    def create_model(self):
        """Create the model"""
        self.logger.info("Creating model")

        # load model parameters
        with open(self.model_parameters_file) as f:
            self.model_params = json.load(f)

        # do not use dict.get for the final because we want errors to be thrown
        # if these are missing
        self.network_xyz = self.model_params.get("Golgi_cells")["xyz"]
        self.golgi_cell_density = self.model_params.get("Golgi_cells")["density"]
        self.homogeneous_populations = self.model_params.get("Golgi_cells")[
            "homogeneous"
        ]
        self.num_golgi_populations = (
            1
            if self.homogeneous_populations
            else self.model_params.get("Golgi_cells")["num_populations"]
        )
        self.golgi_cell_files = glob.glob("./cells/Golgi/GoC_*.cell.nml")
        self.golgi_cell_variants = numpy.random.choice(
            self.golgi_cell_files, self.num_golgi_populations, replace=False
        )
        self.logger.debug(f"VAR: {self.golgi_cell_variants = }")

        self.__get_golgi_cell_locations()

        self.__create_network()
        self.__create_gap_junctions()

        # write to file
        write_neuroml2_file(self.nml_document, self.neuroml_file)

    def __get_golgi_cell_locations(self):
        """Get locations of Golgi cells that fit in the given volume

        Also calculate the total number of Golgi cells, and the number of cells
        per variant.
        """
        x, y, z = list(map(lambda x: split_nml2_quantity(x)[0], self.network_xyz))
        self.num_golgi_cells = int(
            split_nml2_quantity(self.golgi_cell_density)[0] * (x * y * z)
        )
        # round off to nearest multiple of num_golgi_populations so that we can
        # distribute the different Golgi variants uniformly
        self.num_golgi_cells = self.num_golgi_populations * round(
            self.num_golgi_cells / self.num_golgi_populations
        )
        self.logger.debug(f"VAR: {self.num_golgi_cells = }")

        self.golgi_cell_locations: typing.List[typing.List[float]] = (
            numpy.random.random_sample((self.num_golgi_cells, 3)) * [x, y, z]
        )

        self.num_cells_per_golgi_cell_variant = int(
            self.num_golgi_cells / self.num_golgi_populations
        )
        self.logger.debug(f"VAR: {self.num_cells_per_golgi_cell_variant = }")

    def __create_network(self):
        """Create network"""
        location_ctr = 0
        for pnum in range(0, self.num_golgi_populations):
            golgi_cell_variant = self.golgi_cell_variants[pnum]
            golgi_cell_component = read_neuroml2_file(golgi_cell_variant).cells[0]
            self.nml_document.add(neuroml.IncludeType, href=golgi_cell_variant)
            pop = self.network.add(
                neuroml.Population,
                id=f"{golgi_cell_component.id}",
                component=golgi_cell_component.id,
                type="populationList",
            )

            for ins in range(0, self.num_cells_per_golgi_cell_variant):
                x = self.golgi_cell_locations[location_ctr][0]
                y = self.golgi_cell_locations[location_ctr][1]
                z = self.golgi_cell_locations[location_ctr][2]

                pop.add(
                    neuroml.Instance, id=ins, location=neuroml.Location(x=x, y=y, z=z)
                )
                location_ctr += 1

    def __create_gap_junctions(self):
        """Create gap junctions between cells"""
        self.nml_document.add(
            neuroml.GapJunction,
            id="Gj_0",
            conductance=self.model_params.get("Gap_junctions")["conductance"],
        )
        self.logger.debug(f"VAR: {self.golgi_cell_locations.shape = }")
        # returns the matrix in condensed form where it is read in row major
        # form
        distance_bw_cells = scipy.spatial.distance.pdist(
            self.golgi_cell_locations, metric="euclidean"
        )
        self.logger.debug(f"VAR: {distance_bw_cells = }")

        # Only convert to square form when required, but until then, use
        # condensed form. The matrix will remain symmetric from here.
        # distance_bw_cells_matrix = scipy.spatial.distance.squareform(distance_bw_cells)
        # self.logger.debug(f"VAR: {distance_bw_cells_matrix.shape = }")

        # Vervaeke et al 2010, Figure 7
        connection_probability = 1e-2 * (
            -1745 + 1836 / (1 + numpy.exp((distance_bw_cells - 267) / 39))
        )

        random_matrix = numpy.random.random(size=distance_bw_cells.shape)
        connection_probability -= random_matrix

        # ensure minimum probability is 0
        connection_probability = numpy.maximum(connection_probability, 0)
        self.logger.debug(f"VAR: {connection_probability = }")

        weights_matrix = self.__get_gap_junction_weights_vervaeke2010(distance_bw_cells)
        self.logger.debug(f"VAR: {weights_matrix = }")

        # pop_ctr = 0
        # for pop in self.network.populations:
        #     for cell_instance in pop.instances:
        #         cell_index = (
        #             pop_ctr * self.num_cells_per_golgi_cell_variant
        #         ) + cell_instance.id
        #         self.logger.debug(f"{cell_index = }")
        #
        #         # handle connection probability
        #
        #     pop_ctr += 1

    def __get_gap_junction_weights_vervaeke2010(self, dist_matrix, dist_k=1):
        """Get weights of gap junctions as a function of distances between the
        cell somas
        """
        coupling_coefficient = -2.3 + 29.7 * numpy.exp(
            (-1 * dist_matrix) / (70.4 * dist_k)
        )
        weights = (
            0.576 * numpy.exp(coupling_coefficient / 12.4)
            + 0.00059 * numpy.exp(coupling_coefficient / 2.79)
            - 0.564
        )

        return weights

    def create_simulation(self, lems_file: typing.Optional[str] = None):
        """Create simulation
        :param lems_file: name of LEMS file to serialise simulation to
        :type lems_file: str
        """
        if lems_file:
            self.lems_file = lems_file
        else:
            # if not already set, use a default
            if not hasattr(self, "lems_file"):
                self.logger.error(
                    "No file name set for lems_file before, please pass a value"
                )
                return

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
        """Simulate the model

        :param skip_run: only parse file but do not generate scripts or run
        :type skip_run: bool
        :param only_generate_scripts: toggle whether only the runner script
            should be generated
        :type only_generate_scripts: bool

        """
        if lems_file:
            self.lems_file = lems_file
        else:
            # if not already set, use a default
            if not hasattr(self, "lems_file"):
                self.logger.error(
                    "No file name set for lems_file before, please pass a value"
                )
                return

        # https://pyneuroml.readthedocs.io/en/development/pyneuroml.runners.html#pyneuroml.runners.run_lems_with
        # you can also use `pynml ..` from the command line to do this
        run_lems_with(
            engine="jneuroml_neuron", lems_file_name=self.lems_file, skip_run=skip_run
        )


if __name__ == "__main__":
    model = GolgiCellNetwork()
    model.app()
