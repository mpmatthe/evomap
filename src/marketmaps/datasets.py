"""
Sample data for demonstration purpose.

Author: Maximilian Matthe <matthe@wiwi.uni-frankfurt.de>
"""
from importlib import resources
import numpy as np
import os
import pandas as pd

def load_tnic_snapshot():
    """ Load example data. Original data comes from 
    https://hobergphillips.tuck.dartmouth.edu/.  

    Returns
    -------
    dict
        Dictionary containing the similarity matrix, firm labels and 
        cluster assignments (based on community detection).
    """
    with resources.path("marketmaps.data.tnic_snapshot", "sim_mat.npy") as f:
        sim_mat = np.load(f)

    with resources.path("marketmaps.data.tnic_snapshot", "cluster.npy") as f:
        cluster = np.load(f)

    with resources.path("marketmaps.data.tnic_snapshot", "labels.npy") as f:
        labels = np.load(f)

    TNIC_testdata = {'matrix': sim_mat, 'labels': labels, 'cluster': cluster}
    return TNIC_testdata

def load_tnic_snapshot_small():
    """ Load example data. Original data comes from 
    https://hobergphillips.tuck.dartmouth.edu/.  

    Returns
    -------
    dict
        Dictionary containing the similarity matrix, firm labels and 
        cluster assignments (based on community detection).
    """
    with resources.path("marketmaps.data.tnic_snapshot_small", "sim_mat.npy") as f:
        sim_mat = np.load(f)

    with resources.path("marketmaps.data.tnic_snapshot_small", "cluster.npy") as f:
        cluster = np.load(f)

    with resources.path("marketmaps.data.tnic_snapshot_small", "labels.npy") as f:
        labels = np.load(f)

    TNIC_testdata = {'matrix': sim_mat, 'labels': labels, 'cluster': cluster}
    return TNIC_testdata

def load_tnic_sample():
    """ Load example data. Original data comes from 
    https://hobergphillips.tuck.dartmouth.edu/.  

    Returns
    -------
    dict
        Dictionary containing the similarity matrix, firm labels and 
        cluster assignments (based on community detection).
    """
    with resources.path("marketmaps.data.tnic_sample", "tnic_sample.csv") as f:
        df_tnic = pd.read_csv(f)

    return df_tnic


def load_tnic_sample_small():
    """ Load example data. Original data comes from 
    https://hobergphillips.tuck.dartmouth.edu/.  

    Returns
    -------
    dict
        Dictionary containing the similarity matrix, firm labels and 
        cluster assignments (based on community detection).
    """
    with resources.path("marketmaps.data.tnic_sample_small", "tnic_sample_small.csv") as f:
        df_tnic = pd.read_csv(f)

    return df_tnic