"""
Sample data for demonstration purpose.

Author: Maximilian Matthe <matthe@wiwi.uni-frankfurt.de>
"""
from importlib import resources
import numpy as np
import os

def load_tnic_sample():
    """ Load example data. Original data comes from 
    https://hobergphillips.tuck.dartmouth.edu/.  

    Returns
    -------
    dict
        Dictionary containing the similarity matrix, firm labels and 
        cluster assignments (based on community detection).
    """
    with resources.path("marketmaps.data.tnic_sample", "sim_mat.npy") as f:
        sim_mat = np.load(f)

    with resources.path("marketmaps.data.tnic_sample", "cluster.npy") as f:
        cluster = np.load(f)

    with resources.path("marketmaps.data.tnic_sample", "tnic_sample") as f:
        labels = np.load(f)

    TNIC_testdata = {'matrix': sim_mat, 'labels': labels, 'cluster': cluster}
    return TNIC_testdata

