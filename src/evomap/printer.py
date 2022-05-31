"""
Useful functions to draw maps.
"""

from multiprocessing.sharedctypes import Value
from unittest.mock import DEFAULT
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
#from adjustText import adjust_text

DEFAULT_BUBBLE_SIZE = 25
DEFAULT_FONT_SIZE = 10

def draw_map(Y, c = None, size = None, labels = None, highlight_labels = None, inclusions = None, 
    zoom_on_cluster = None, annotate = None, filename = None, ax = None, 
    fig_size = None, show_box = True, cmap = None, rcparams = None, title_str = None,
    fontdict = None, title_fontdict = None, scatter_kws = {}):
    """ Draw a single (static) map. Can take additional scatterplot arguments
    as kwargs, see https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter

    Parameters
    ----------
    Y : ndarray of shape (n_samples, d)
        Map coordinates. If d = 1, the resultant map is a line. If d > 2, 
        only the first two dimensions are visualized.
    c : ndarray of shape (n_samples,), optional
        Cluster assignments used for coloring (int), by default None
    labels : ndarray of shape (n_samples,), optional
        Object labels (str), by default None
    highlight_labels: list, optional
        Labels which should be highligted on the map, by default None
    inclusions : ndarray of shape (n_samples,), optional
        Inlcusion array. A 0 indicates that the object should be excluded on the map, 
        a 1 indicates that the object should be included in the map , by default None
    zoom_on_cluster : int, optional
        Cluster number on which the map should be zoomed-in to, by default None
    annotate : str, optional
        Annotations, either None, 'labels', or 'clusters', by default None
    filename : str, optional
        Filename to which the map is saved, by default None
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot., by default None
    fig_size : tuple (int,int), optional
        Figure size, by default None
    show_box : bool, optional
        If true, a box is drawn around the map, by default True
    cmap : str, optional
        The colormap for coloring the cluster assignments. Should be a valid
        colormap Name, or a custom colormap function by default None
    rcparams : dict, optional
        Dictionary of matplotlib rcParams, by default None
    fontdict : dict, optional
        Dictionary defining the label font, by default None

    """

    # Check inputs:
    if not annotate in [None, 'clusters', 'labels']:
        raise ValueError("'annotate' should be 'clusters' or 'labels'")

    if annotate == 'labels' and labels is None:
        labels = [str(i) for i in range(Y.shape[0])]

    if not labels is None:
        labels = np.array(labels)
        annotate = 'labels'

    if not highlight_labels is None:
        # In case a single label is given, put it into a list
        if type(highlight_labels) == str:
            highlight_labels = [highlight_labels]
        if labels is None and len(highlight_labels) > 0:
            raise ValueError('Need to provide labels.')
        if not all(highlighted_label in labels for highlighted_label in highlight_labels):
            raise ValueError('All highlighted labels need to be contained in the labels array.')
    else:
        highlight_labels = []

    n_samples = len(Y)
    if len(Y.shape) == 1:
        Y = Y.reshape((n_samples,1))
        Y = np.concatenate([Y, np.ones((n_samples,1))], axis = 1)

    if Y.shape[1] > 2:
        print('Input array is not two-dimensional. Only first two dimensions will be plotted')

    # Data preparation:
    if fontdict is None:
        if annotate == 'clusters':
            fontdict = {'family': 'arial', 'size': DEFAULT_FONT_SIZE*2}
        else:
            fontdict = {'family': 'arial', 'size': DEFAULT_FONT_SIZE}

    if c is None:
        c = np.zeros((n_samples, 1))
    else:
        c = np.array(c)
    c_labels = np.unique(c)
    c = np.array([np.where(c_labels == clust)[0][0] for clust in c]).reshape((n_samples, 1))

    df_data = pd.DataFrame(
        data = np.hstack([Y, c]), 
        columns = ['x','y','cluster'])

    df_data['cluster'] = df_data['cluster'].map(int)

    df_data['label'] = labels
    if not inclusions is None: 
        df_data = df_data[inclusions == 1]
    
    if not size is None:
        df_data['size'] = size
    else:
        df_data['size'] = DEFAULT_BUBBLE_SIZE

    # Zoom in on cluster - if necessary:
    if not zoom_on_cluster is None:
        df_data = df_data[df_data['cluster'] == zoom_on_cluster]

    # Check if only valid cluster indices are included
    if len(df_data[df_data['cluster']<0]) >0:
        raise ValueError("Some cluster indices are smaller than 0")

    # Explicitly calculate colors to avoid erroneous coloring
    if cmap is None:
        if len(np.unique(c)) <= 10:
            cmap = "tab10"
        elif len(np.unique(c)) <= 15:
            cmap = mpl.cm.get_cmap('tab10')
            hex = []
            for i in range(10):
                hex.append(mpl.colors.rgb2hex(cmap(i)))

            hex.append('#ffff33')
            hex.append('#b9ff66')
            hex.append('#cdb7f6')

            cmap = ListedColormap(hex) 

        else:
            cmap = "tab20"
            print("More than 20 clusters. Will include duplicate colors unless custom colormap is provided.")

    if type(cmap) == str:
        cmap = mpl.cm.get_cmap(cmap)
    
    df_data['color'] = df_data['cluster'].map(cmap)

    # Draw map
    init_params(rcparams)

    if ax is None:
        return_fig = True
        if fig_size is None:
            fig_size = (5,5)
        fig, ax = plt.subplots(figsize = fig_size)
    else:
        return_fig = False

    scatter_kws.update(
        {'facecolors': df_data['color'], 'edgecolors': df_data['color']})

    scatter_kws['s'] = df_data['size']

    p = sns.regplot(
            x = 'x', 
            y = 'y',
            data = df_data, 
            fit_reg= False, 
            scatter_kws = scatter_kws,
            ax = ax)

    if title_fontdict is None:
        title_fontdict = {'weight': 'normal', 'size': fontdict['size']*1.2}
    if len(highlight_labels) > 0:
        fontdict.update({'size': fontdict['size']*0.8})
        highlighted_fontdict = fontdict.copy()
        highlighted_fontdict.update({'weight': 'bold', 'size': fontdict['size']*1.2})

    if annotate == 'clusters':
        cluster_means = df_data.groupby('cluster').mean()
        for i in range(len(cluster_means)):
            clust = cluster_means.iloc[i].name
            shift = 0.01*np.std(df_data, axis = 0)

            p.text(cluster_means.iloc[i][0] + shift[0],
                cluster_means.iloc[i][1] + shift[1], 
                c_labels[clust],
                color = cmap(clust),
                alpha = .6,fontdict = fontdict)

    elif annotate == 'labels' or annotate == 'points':

        if len(highlight_labels) > 0:
            # Only print highlighted labels
            for i in range(len(df_data)):
                if labels[i] in highlight_labels:
                    p.text(df_data['x'].iloc[i], df_data['y'].iloc[i], df_data['label'].iloc[i], alpha = 1, fontdict = highlighted_fontdict)
                else:
                    continue
        else:
            # Print all labels
            for i in range(len(df_data)):
                p.text(df_data['x'].iloc[i], df_data['y'].iloc[i], df_data['label'].iloc[i], alpha = .6, fontdict = fontdict)

    for highlighted_label in highlight_labels:
        df_i = df_data.query('label == @highlighted_label')        
        if len(df_i)>0:
            p.text(df_i['x'].iloc[0], df_i['y'].iloc[0], df_i['label'].iloc[0], alpha = 1, fontdict = highlighted_fontdict)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    ax.set_xticks(np.linspace(xmin, xmax, 7))
    ax.set_yticks(np.linspace(ymin, ymax, 7))

    if not title_str is None:
        ax.set_title(title_str, title_fontdict)

    if show_box == False:
        plt.box(on = None)
    else:
        plt.box(on = True)

    # Save or show plot 
    if not filename == None:
        mydpi = 300
        fig.savefig(filename, dpi = mydpi, format = 'png')

    plt.close()
    if return_fig:
        return fig

def draw_map_sequence(Y_ts, c_ts = None, n_cols = 4, time_labels = [], **kwargs):
    """ Draw a sequence of static maps next to each other. Can use the same
    arguments as the 'draw_map' function as dictionary.

    Parameters
    ----------
    Y_ts : list of ndarrays, each of shape (n_samples, d)
        Sequence of map coordinates.
    c_ts : list of ndarray, each of shape (n_samples,), optional
        Sequence of cluster assignments used for coloring (int), by default None
        If the cluster assignments are constant, one can simply provide a single 
        array as kwarg.
    n_cols : int, optional
        Max. number of maps shown in one row, by default 4
    map_kws : dict, optional
        Additional arguments for the 'draw_map' function, by default None
    time_labels: list of str, optional
        When given, use these labels as title strings
    """
    n_periods = len(Y_ts)
    n_rows = int(np.ceil(n_periods/n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (4*n_cols, 4*n_rows))
        

    if len(time_labels) == 0:
        time_labels = ["Period " + str(t+1) for t in range(n_periods)]

    draw_map_kws = {}
    draw_map_args = draw_map.__code__.co_varnames[:draw_map.__code__.co_argcount]
    for key, value in kwargs.items():
        if key in draw_map_args:
            draw_map_kws.update({key: value})

    row = 0
    col = 0
    for t in range(n_periods):
        if n_rows > 1:
            draw_map_kws.update({
                'Y': Y_ts[t],
                'ax': axs[row, col]
            })
        else:
            draw_map_kws.update({
                'Y': Y_ts[t],
                'ax': axs[col]
            })
        if not c_ts is None:
            draw_map_kws.update({'c': c_ts[t]})
        if not time_labels is None:
            draw_map_kws.update({'title_str': time_labels[t]})

        draw_map(**draw_map_kws)            
        if col < n_cols - 1:
            col += 1
        else: 
            row += 1
            col = 0

    return fig

def fit_attribute(coords, attribute_label, attribute_values, map):
    """ Fit an attribute to the map and display the resultant vector.

    To do so, regress the attribute value on map coordinates and use
    the coefficients as arrow coordinates.

    Parameters
    ----------
    coords : ndarray of shape (n_samples, n_dims)
        Map coordinates.
    attribute_label : string
        Attribute label (displayed next to vector).
    attribute_values : ndarray of shape (n_samples,)
        Attribute values for each sample.
    map : matplotlib.figure
        Figure containing the map (i.e., output of draw_map function)

    Returns
    -------
    matplotlib.figure
        Figure containing the map with property vector added.
    """
    import statsmodels.api as sm
    SCALE = 10
    ax = map.axes[0]
    X = coords
    y = attribute_values
    est=sm.OLS(y, X)
    result = est.fit().params
    result['x1']
    result['x2']
    ax.arrow(0,0,result['x1']*SCALE,result['x2']*SCALE, linestyle = '--', lw = .25, alpha = .75, width = .001, color = 'grey',  head_width = 0.1)
    ax.text(result['x1']*1.1*SCALE, result['x2']*1.1*SCALE, attribute_label, fontdict= {'size': 8, 'color': 'darkblue', 'weight': 'normal'})
    return map

def fit_attributes(map_coords, df_attributes, map):
    """ Fit multiple attributes and display their vectors in the map.

    Parameters
    ----------
    map_coords : ndarray of shape (n_samples, n_dims)
        Map coordinates.
    df_attributes : pd.DataFrame
        Dataframe containing the attributes. Each column is expected to 
        correspond to one attribute. Make sure to label colums and 
        that the number of rows equals n_samples.
    map : matplotlib.figure
        Figure containing the map (i.e., output of draw_map function)

    Returns
    -------
    matplotlib.figure
        Figure containing the map with property vectors added.
    """

    for attribute in df_attributes.columns:
        map = fit_attribute(
            coords = map_coords, 
            attribute_label = attribute,
            attribute_values = df_attributes[attribute], 
            map = map)
    return map

def draw_dynamic_map(Y_ts, c_ts = None, incl_ts = None, show_arrows = False, 
    show_last_positions_only = False, highlight_trajectories = None, 
    transparency_start = 0.1, transparency_end = 0.4, transparency_final = 1., 
    **kwargs):
    """ Draw a dynamic map, jointly visualizing all object's map coordinates
    over time. Can take all arguments of 'draw_map' as kwargs. 

    Parameters
    ----------
    Y_ts : list of ndarrays, each of shape (n_samples, d)
        Sequence of map coordinates.
    c_ts : list of ndarray, each of shape (n_samples,), optional
        Sequence of cluster assignments used for coloring (int), by default None
        If the cluster assignments are constant, one can simply provide a single 
        array as kwarg.
    incl_ts : list of ndarrays, each of shape (n_samples, ), optional
        Sequence of inclusion arrays, each indicating if an object is present
        in period t (via 0 and 1s), by default None
    show_arrows : bool, optional
        If true, objects' subsequent positions are connected by arrows, by default False
    show_last_positions_only : bool, optional
        If true, only the map positions for the last period are displayed. Should
        be combined with 'show arrows = True', by default False
    highlight_trajectories : list, optional
        Labels for which trajectories should be highlighted, by default None
    transparency_start : float, optional
        Transparency of bubbles for first period, by default 0.1
    transparency_end : float, optional
        Transparency of bubbles for second-last period, by default 0.4
    transparency_final : float, optional
        transparency of bubbles for last period, by default 1.

    """

    # Check inputs
    n_periods = len(Y_ts)
    n_samples = Y_ts[0].shape[0]
    if np.any([Y_t.shape != Y_ts[0].shape for Y_t in Y_ts]):
        raise ValueError('All input arrays need to be of similar shape.')

    if not c_ts is None:
        if np.any([c_t.shape[0] != n_samples for c_t in c_ts]):
            raise ValueError('Misshaped class arrays.')

    if not incl_ts is None:
        if np.any([incl_t.shape[0] != n_samples for incl_t in incl_ts]):
            raise ValueError('Misshaped inclusion arrays.')

    else:
        incl_ts = [np.repeat(1,n_samples)]*n_periods

    # Data preparation
    transparencies = np.linspace(transparency_start, transparency_end, n_periods-1).tolist()
    transparencies.append(transparency_final)
    if not highlight_trajectories is None:
        highlighted_transparencies = np.linspace(transparency_end, transparency_final,n_periods).tolist()
  
    highlight_colors = ['darkred', 'orange', 'darkgreen', 'slategrey']
#   highlight_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

#    if show_arrows_subset is None:
 #       show_arrows_subset = np.repeat(1, len(coords[0]))

    if not 'labels' in kwargs.keys():
        labels = np.array([str(i+1) for i in range(n_samples)])
    else:
        labels = np.array(kwargs['labels'])

    # Highlight labels are only shown in last period - so safe them now for later
    labels_to_highlight = []
    if not highlight_trajectories is None:
        labels_to_highlight = np.concatenate((labels_to_highlight, highlight_trajectories))

    if 'highlight_labels' in kwargs.keys():
        labels_to_highlight = np.concatenate((labels_to_highlight, kwargs['highlight_labels']))

    if len(labels_to_highlight) >0:
        labels_to_highlight = np.unique(labels_to_highlight)
    
    if 'annotate' in kwargs.keys():
        annotate = kwargs['annotate']
    else: 
        annotate = None

    # Draw map
    draw_map_kws = {}
    draw_map_args = draw_map.__code__.co_varnames[:draw_map.__code__.co_argcount]
    for key, value in kwargs.items():
        if key in draw_map_args:
            draw_map_kws.update({key: value})    

    # Prepare figure
    if not 'ax' in kwargs.keys():
        if 'fig_size' in kwargs.keys():
            fig_size = kwargs['fig_size']
        else:
            fig_size = (5,5)
            
        fig, ax = plt.subplots(figsize = fig_size)
        draw_map_kws.update({'ax': ax})    
    
    # Plot each period
    for t in range(n_periods):

        draw_map_kws.update({
            'Y': Y_ts[t], 
            'inclusions': incl_ts[t], 
            'filename': None, 
            'title_str': None,
            'scatter_kws': {'alpha': transparencies[t]}})

        # Only show labels for the last period
        if t < n_periods - 1:
            draw_map_kws.update({
                'annotate': None, 
                'highlight_labels': None})
        else:
            draw_map_kws.update({
                'annotate': annotate, 
                'highlight_labels': labels_to_highlight})

            if 'title_str' in kwargs.keys():
                draw_map_kws.update({'title_str': kwargs['title_str']})
            
        if not c_ts is None:
            draw_map_kws.update({'c' : c_ts[t]})

        if not (t < n_periods -1  and show_last_positions_only):
            p = draw_map(**draw_map_kws)

            if not highlight_trajectories is None:
                highlight_count = 0
                highlight_indices = [np.where(labels == label)[0][0] for label in highlight_trajectories]
                for i in highlight_indices:
                    x = Y_ts[t][:,0][i]
                    y = Y_ts[t][:,1][i]
                    sns.regplot(
                        x = [x], 
                        y = [y],
                        fit_reg=False, 
                        scatter_kws={
                            'zorder': 10,
                            'alpha': transparencies[t], 
                            'facecolors': highlight_colors[highlight_count], 
                            'edgecolor': highlight_colors[highlight_count]})
                    highlight_count += 1


        if (t > 0) and show_arrows:

            arrow_starts_x = Y_ts[t-1][:,0]
            arrow_starts_y = Y_ts[t-1][:,1]
            deltas_x = Y_ts[t][:,0] - Y_ts[t-1][:,0]
            deltas_y = Y_ts[t][:,1] - Y_ts[t-1][:,1]

            highlight_count = 0
            if not highlight_trajectories is None:
                arrow_indices = [np.where(labels == label)[0][0] for label in highlight_trajectories]
            else:
                arrow_indices = np.array([i for i in range(n_samples)])

            for i in arrow_indices:
                # In case of zooming, only print arrows for objects within this cluster
                if 'zoom_on_cluster' in kwargs.keys():
                    if not kwargs['zoom_on_cluster'] is None:
                        if str(draw_map_kws['c'][i]) != str(kwargs['zoom_on_cluster']):
                            continue

                if (incl_ts[t][i] == 1) and (incl_ts[t-1][i] == 1):
                    if not (deltas_x[i] * deltas_y[i]) == 0:
                        if not highlight_trajectories is None:
                            col = highlight_colors[highlight_count]
                            arrow_size = .005
                            alpha = transparencies[t-1]
                            highlight_count +=1
                        else:
                            col = 'grey'
                            alpha = transparencies[t-1]
                            arrow_size = .005

                        ax.arrow(
                            arrow_starts_x[i],
                            arrow_starts_y[i],
                            deltas_x[i],
                            deltas_y[i],
                            width = arrow_size,
                            head_width = arrow_size,
                            head_length = arrow_size,
                            fc = col,
                            ec = col,
                            length_includes_head = True,
                            alpha = alpha)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    ax.set_xticks(np.linspace(xmin, xmax, 7))
    ax.set_yticks(np.linspace(ymin, ymax, 7))

    if 'filename' in kwargs.keys():
        fig.savefig(kwargs['filename'], dpi = 300, format = "png")

    plt.close()
    return fig

def draw_trajectories(Y_ts, labels, selected_labels = None, title_str = None, 
    annotate_periods = True, period_labels = None, ax = None, figsize = None):
    """ Draw the trajectories of selected objects.

    Parameters
    ----------
    Y_ts : list of ndarrays, each of shape (n_samples, d)
        Sequence of map coordinates.
    labels : ndarray of shape (n_samples,)
        Object labels (str)
    selected_labels : ndarray of shape (n_selected,), optional
        Selected object labels (str), by default None
    title_str : str, optional
        Figure title, by default None
    annotate_periods : bool, optional
        If true, labels for each period are shown next to each pair of map 
        coordinates, by default True
    period_labels : ndarray of shape (n_periods,), optional
        Period labels (str), by default None
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot, by default None
    figsize : tuple, optional
        Figure size, by default (12,12)
    """

    n_periods = len(Y_ts)
    n_firms = Y_ts[0].shape[0]
    if selected_labels == None:
        selected_labels = labels

    if ax is None:
        if figsize is None:
            figsize = (5,5)
        fig, ax = plt.subplots(figsize = figsize)   

    annotations = []

    if period_labels is None and annotate_periods == True:
        period_labels = ["Period " + str(t+1) for t in range(n_periods)]

    for i in range(n_firms):
        if not labels[i] in selected_labels:
            continue
        xs = []
        ys = []
        # Plot the points
        for t in range(n_periods):
            alpha = 1 - (n_periods - t) / n_periods
            alpha = alpha * .5
            x = Y_ts[t][i,0]
            y = Y_ts[t][i,1]
            c = 'black'
            c_line = 'grey'
            label = labels[i]
            plt.scatter(x,y , c = c, alpha = alpha)
            xs.append(x)
            ys.append(y)
            if t == n_periods - 1:
                label = plt.text(x ,y , label, c = c, alpha = .7, fontsize = DEFAULT_FONT_SIZE)
                annotations.append(label)

            elif annotate_periods:
                label = plt.text(x ,y , period_labels[t], c = c_line, alpha = .5, fontsize = DEFAULT_FONT_SIZE * 0.8)
#               texts.append(label)

        # Plot the trajectory
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.plot(xs, ys, c = c_line, alpha = .4)

    if not title_str is None:
        ax.set_title(title_str, fontsize = DEFAULT_FONT_SIZE)
#    adjust_text(texts, force_points = 0.15,  arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))
    plt.close()
    return fig

def init_params(custom_params = None):
    """
    Set default aesthetic styles here. References: Matplotlib RC params

    """

    mpl.rcParams.update(
        {"axes.prop_cycle": cycler('color', [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf"]),
        "font.size": 14,
        "axes.linewidth": 0,
        "axes.titlesize": 22,
        "axes.labelsize": 16,

        "axes.edgecolor": "black",
        "axes.linewidth": 1,

        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.size": 0.1,
        "xtick.minor.size": 0.05,
        "ytick.major.size": 0.1,
        "ytick.minor.size": 0.05,

        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.color": "black",

        "savefig.facecolor": "w",
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "savefig.format": "png"
        }
        )

    if not custom_params is None:
        mpl.rcParams.update(custom_params)


