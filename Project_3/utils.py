from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class CriterionType(StrEnum):
    GAIN = auto()
    COST = auto()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads dataset from csv file

    :param dataset_path: Path to dataset directory
    :return: Pandas Dataframe where every row represents single alternative, while every column represents single criterion
    """

    dataset = pd.read_csv(dataset_path / "dataset.csv", index_col=0)

    return dataset


def load_preference_information(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    preferences = pd.read_csv(dataset_path / "preference.csv", index_col=0)
    preferences.type = pd.Categorical(preferences.type, list(CriterionType))

    return preferences

def test_single_marginal_preference_function(
        function: Callable[[float, float, float], float],
        p: float,
        q: float,
        ax,
        x_min: float = -2,
        x_max: float = 3,
) -> None:
    _function = np.vectorize(partial(function, indifference_threshold=q, preference_threshold=p), otypes=[float])

    x = np.linspace(x_min, x_max, 300)

    ax[0].plot(x, _function(x))
    ax[1].plot([x_min, q, p, x_max], [0, 0, 1, 1])

    ticks = [0]
    ticks_labels = ["0"]

    if q == 0:
        ticks_labels[0] = "q=0"
    else:
        ticks.append(q)
        ticks_labels.append("q")

    if p==q:
        ticks_labels[-1] = "q=p" + ticks_labels[-1][1:]
    else:
        ticks.append(p)
        ticks_labels.append("p")

    ax[0].set_xticks(ticks, ticks_labels)
    ax[1].set_xticks(ticks, ticks_labels)

def test_marginal_preference_function(function: Callable[[float, float, float], float]) -> float:
    fig, ax = plt.subplots(4, 2)

    ax[0, 0].set_title("Computed")
    ax[0, 1].set_title("Expected")

    test_single_marginal_preference_function(function, q=0, p=2, ax=ax[0])
    test_single_marginal_preference_function(function, q=1, p=2, ax=ax[1])
    test_single_marginal_preference_function(function, q=1, p=1, ax=ax[2])
    test_single_marginal_preference_function(function, q=0, p=0, ax=ax[3])

    fig.tight_layout()


def find_nodes_groups(ranking: pd.DataFrame) -> list[list[str]]:
    nodes = ranking.index.tolist()

    indifference_matrix = ranking & ranking.T
    edges = [
        (nodes[i], nodes[j])
        for i, j in np.stack(np.nonzero(indifference_matrix)).T.tolist()
        if i != j
    ]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    return list(nx.clique.find_cliques(g))


def display_ranking(ranking: pd.DataFrame, title: str) -> None:
    nodes_groups = find_nodes_groups(ranking)

    nodes = ranking.index.tolist()
    edges = [
        (nodes[i], nodes[j])
        for i, j in np.stack(np.nonzero(ranking)).T.tolist()
        if i != j
    ]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    names_mapping = {}

    for node_group in nodes_groups:
        first, *others = node_group

        for i in others:
            g.remove_node(i)

        names_mapping[first] = "\n".join(node_group)

    g = nx.relabel_nodes(g, names_mapping)
    g = nx.transitive_reduction(g)

    layout = graphviz_layout(g, prog="dot")
    plt.title(title)
    nx.draw(
        g,
        layout,
        with_labels=True,
        arrows=False,
        node_shape="s",
        node_color="none",
        bbox=dict(facecolor="white", edgecolor="black"),
    )
    plt.show()
