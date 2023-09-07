"""
Sample data for demonstration purpose.
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
    with resources.path("evomap.data.tnic_snapshot", "sim_mat.npy") as f:
        sim_mat = np.load(f)

    with resources.path("evomap.data.tnic_snapshot", "cluster.npy") as f:
        cluster = np.load(f)

    with resources.path("evomap.data.tnic_snapshot", "label.npy") as f:
        label = np.load(f)

    with resources.path("evomap.data.tnic_snapshot", "size.npy") as f:
        size = np.load(f)

    TNIC_testdata = {
        'matrix': sim_mat, 
        'label': label, 
        'cluster': cluster,
        'size': size}

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
    with resources.path("evomap.data.tnic_snapshot_small", "sim_mat.npy") as f:
        sim_mat = np.load(f)

    with resources.path("evomap.data.tnic_snapshot_small", "cluster.npy") as f:
        cluster = np.load(f)

    with resources.path("evomap.data.tnic_snapshot_small", "labels.npy") as f:
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
    with resources.path("evomap.data.tnic_sample", "tnic_sample.csv") as f:
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
    with resources.path("evomap.data.tnic_sample_small", "tnic_sample_small.csv") as f:
        df_tnic = pd.read_csv(f)
        df_tnic['sic1'] = df_tnic['sic1'].map(lambda x: str(x))
        df_tnic['sic2'] = df_tnic['sic2'].map(lambda x: str(x))

    return df_tnic

def load_tnic_sample_tech(unbalanced = False):
    if unbalanced:
        with resources.path('evomap.data.tnic_sample_tech', 'tnic_sample_technology_with_netflix.csv') as f:
            df_tnic = pd.read_csv(f)
    else:
        with resources.path('evomap.data.tnic_sample_tech', 'tnic_sample_technology.csv') as f:
            df_tnic = pd.read_csv(f)

    return df_tnic


def load_car_data():
    """ Load car dataset, containing perceptual and preference ratings of competing cars. 

    Data is taken from the following book, with some modifications: 

    Lilien, G. L., & Rangaswamy, A. (2004). Marketing engineering: computer-assisted marketing analysis and planning (2nd revised edition). DecisionPro.

    Returns
    -------
    dict
        Dictionary containing the perceptual ratings and preference ratings.
    """

    with resources.path("evomap.data.cars", "customer_preference_ratings.csv") as f:
        df_preferences = pd.read_csv(f)

    with resources.path("evomap.data.cars", "perceptual_attribute_ratings.csv") as f:
        df_attributes = pd.read_csv(f, index_col = 0)

    return {'preferences': df_preferences, 'attributes': df_attributes}


