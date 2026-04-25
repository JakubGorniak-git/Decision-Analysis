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


def load_boundary_profiles(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads boundary profiles information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    boundary_profiles = pd.read_csv(dataset_path / "boundary_profiles.csv", index_col=0)

    return boundary_profiles


def load_preference_information(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    preferences = pd.read_csv(dataset_path / "preference.csv", index_col=0)
    preferences.type = pd.Categorical(preferences.type, list(CriterionType))

    return preferences
