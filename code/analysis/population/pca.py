#!/usr/bin/env python3
"""
PCA analysis of population membrane voltages

File: pca.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import numpy
import typer
from pyneuroml.plot import generate_plot
from pyneuroml.utils.simdata import load_sim_data_from_lems_file
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("task-split")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(name)s (%(levelname)s): %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger.addHandler(handler)


def runner(lems_file: str, quantity: str = "v", num_components: int = 25):
    """Run PCA on population membrane potential data to investigate contributions of PCs"""
    simdata = load_sim_data_from_lems_file(
        lems_file, get_events=False, get_traces=True, remove_dat_files_after_load=False
    )

    timeseries_data = []
    for afile, data in simdata.items():
        for key, value in data.items():
            if key.split("/")[-1] == quantity:
                timeseries_data.append(value)

    logger.info(f"{len(timeseries_data) = }")

    pca = PCA(num_components)
    pca.fit_transform(timeseries_data)

    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    projection_1st_component = numpy.dot(timeseries_data, pca.components_[0])
    projection_2nd_component = numpy.dot(timeseries_data, pca.components_[1])

    generate_plot(
        xvalues=[range(1, len(eigenvalues) + 1)],
        yvalues=[eigenvalues],
        title="Eigenvalues",
        xaxis="Components",
        yaxis="Eigen values",
        markers=["."],
    )

    generate_plot(
        xvalues=[range(1, len(explained_variance_ratio) + 1)],
        yvalues=[explained_variance_ratio],
        title="Explained variance ratio",
        xaxis="Components",
        yaxis="% of explained variance",
        markers=["."],
    )

    cumulative_explained_variance = numpy.cumsum(explained_variance_ratio)

    generate_plot(
        xvalues=[range(1, len(explained_variance_ratio) + 1)],
        yvalues=[cumulative_explained_variance],
        title="Cumulative explained variance ratio",
        xaxis="Components",
        yaxis="Cumulative % of explained variance",
        ylim=[0, 1],
        markers=["."],
    )

    generate_plot(
        xvalues=[projection_1st_component],
        yvalues=[projection_2nd_component],
        title="Projection in PC1/PC2 space",
        xaxis="PC1",
        yaxis="PC2",
        markers=["o"],
        linestyles=[""],
    )


if __name__ == "__main__":
    typer.run(runner)
