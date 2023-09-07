"""
Functions to draw maps.
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

from cycler import cycler

DEFAULT_BUBBLE_SIZE = 25
DEFAULT_FONT_SIZE = 10

title_fontdict_large = {'size': 14, 'family': 'Arial'}
title_fontdict = {'size': 14, 'family': 'Arial'}
text_fontdict = {'size': 10, 'family': 'Arial'}
axis_label_fontdict = {'size': 12, 'family': 'Arial'}

def format_tick_labels(x, pos):
    return '{0:.2f}'.format(x)

def init_params(custom_params = None):
    """
    Set default aesthetic styles here.
    """
    return None

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

#TODO: Show grid without axes not working
def style_axes(ax, show_axes, show_box, show_grid, axes_at_origin):
    """Style the axes of a map."""

    ax.set_ylabel("Dimension 2", fontdict = axis_label_fontdict)
    ax.set_xlabel("Dimension 1", fontdict = axis_label_fontdict)
    
    if show_grid:
        ax.grid(True)
    else:
        ax.grid(False)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    # Make axes square
    max_range = np.max([xmax-xmin, ymax-ymin]) * 1.1
    x_mean = xmin + (xmax - xmin)/2
    y_mean = ymin + (ymax - ymin)/2
    
    n_ticks = 5
    n_decimals = 2

    xmin = x_mean - max_range/2
    xmax = x_mean + max_range/2
    ymin = y_mean - max_range/2
    ymax = y_mean + max_range/2

    if xmin < 0 and xmax > 0:
        # Make sure that 0 is one of the ticks
        max_xtick = np.max([np.abs(xmin), np.abs(xmax)])
        xmin = -max_xtick
        xmax = max_xtick

    if ymin < 0 and ymax > 0:
        # Make sure that 0 is one of the ticks
        max_ytick = np.max([np.abs(ymin), np.abs(ymax)])
        ymin = -max_ytick
        ymax = max_ytick

    y_ticks = np.linspace(ymin, ymax, n_ticks)
    x_ticks = np.linspace(xmin, xmax, n_ticks)

    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis = 'both', labelsize = 10)
    ax.xaxis.set_major_formatter(FuncFormatter(format_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick_labels))

    if not show_axes:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if show_axes:
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

    if axes_at_origin:
        # set the x-spine (see below for more info on `set_position`)
        ax.spines['left'].set_position('zero')

        # turn off the right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # set the y-spine
        ax.spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()

        ax.set_xlabel('')
        ax.set_ylabel('')

    if show_box == False:
        plt.box(on = False)
    else:
        plt.box(on = True)

def draw_map(X, label = None, color = None, size = None,  
             inclusions = None, zoom_on_cluster = None, highlighted_labels = None, 
             show_box = True, show_grid = False, show_axes = False, axes_at_origin = False, show_legend = False,
             cmap = None, filename = None, ax = None, fig_size = None, 
             title = None, rotate_labels = 0, scatter_kws = {}, fontdict = None, rcparams = None):

    # Validate input and make sure it is in the right format
    if not label is None:
        label = np.array(label)        

    if not highlighted_labels is None:
        # In case a single label is given, put it into a list
        if type(highlighted_labels) == str:
            highlighted_labels = [highlighted_labels]
        if label is None:
            raise ValueError('Need to provide labels.')
        if not all(highlighted_label in label for highlighted_label in highlighted_labels):
            raise ValueError('All highlighted labels need to be contained in the labels array.')
    else:
        highlighted_labels = []

    # By default, rotate labels for 1D data
    if X.shape[1] == 1 and rotate_labels == 0:
        rotate_labels = 45

    # If input is two dimension, add a constant second dimension for plotting
    n_samples = len(X)
    if len(X.shape) == 1:
        X = X.reshape((n_samples,1))
        X = np.concatenate([X, np.ones((n_samples,1))], axis = 1)

    if X.shape[1] == 1:
        X = np.concatenate([X, np.zeros_like(X)], axis = 1)

    if X.shape[1] > 2:
        print('Input array is more than two-dimensional. Only first two dimensions will be plotted')
    
    if color is None:
        color = np.zeros((n_samples, 1))
    else:
        color = np.array(color)

    color_label = np.unique(color)

    # Translate color values into indices
    cluster_label = color.copy().reshape((n_samples, 1))
    cluster = np.array([np.where(color_label == clust)[0][0] for clust in color]).reshape((n_samples, 1))

    # Prepare dataframe for plottings
    df_data = pd.DataFrame(
        data = np.hstack([X, cluster]), 
        columns = ['x','y','cluster'])
    
    df_data['cluster_label'] = color

    df_data['cluster'] = df_data['cluster'].map(int)

    if not label is None:
        df_data['label'] = label

    # Check if inclusions are provided. If so, filter only included objects
    if not inclusions is None: 
        df_data = df_data[inclusions == 1]
    
    if not size is None:
        df_data['size'] = size
    else:
        df_data['size'] = DEFAULT_BUBBLE_SIZE

    # Zoom in on cluster - if necessary:
    if not zoom_on_cluster is None:
        df_data = df_data[df_data['cluster'] == zoom_on_cluster]

    # Explicitly calculate colors to avoid erroneous coloring
    if cmap is None:
        if len(np.unique(color)) <= 10:
            cmap = "tab10"
        elif len(np.unique(color)) <= 13:
            cmap = mpl.cm.get_cmap('tab10')
            hex = []
            for i in range(10):
                hex.append(mpl.colors.rgb2hex(cmap(i)))

            hex.append('#ffff33')
            hex.append('#b9ff66')
            hex.append('#cdb7f6')

            cmap = ListedColormap(hex) 
        elif len(np.unique(color)) <= 20:
            cmap = "tab20"
        else:
            cmap = "tab20"
            print("Warning: More than 20 clusters. Will include duplicate colors unless custom colormap is provided.")

    if type(cmap) == str:
        cmap = mpl.cm.get_cmap(cmap)
    
    df_data['color'] = df_data['cluster'].map(cmap)
    if len(df_data['color'].unique()) == 1:
        df_data["color"] = 'white'

    init_params(rcparams)

    # If not ax is provided, return the whole FIgure. Else, only draw the plot on the provided axes
    if ax is None:
        return_fig = True
        if fig_size is None:
            fig_size = (5,5)
        fig, ax = plt.subplots(figsize = fig_size)
    else:
        return_fig = False

    n_clusters = len(df_data['cluster'].unique())
    for cluster in df_data['cluster'].unique():
        df_data_this = df_data[df_data['cluster'] == cluster]
        if n_clusters > 1:
            scatter_kws.update({
                'facecolors': df_data_this['color'], 
                'edgecolors': df_data_this['color'],
                's': df_data_this['size']}
                )
            
        else:
            scatter_kws.update({
                'facecolors': df_data_this['color'], 
                'edgecolors': 'black',
                's': df_data_this['size']}
                )

        scatter_kws.update({
            'label': df_data_this['cluster_label'].iloc[0],
        })
        ax.scatter(df_data_this.x,df_data_this.y, **scatter_kws) 
        
    if fontdict is None:
        fontdict = text_fontdict.copy()

    # Add highlights
    if len(highlighted_labels) > 0:
        fontdict.update({'size': fontdict['size']*0.8})
        highlighted_fontdict = fontdict.copy()
        highlighted_fontdict.update({'weight': 'bold', 'size': fontdict['size']*1.2})
    
    # Only print highlighted labels
    if len(highlighted_labels) > 0:
        for i in range(len(df_data)):
            if label[i] in highlighted_labels:
                ax.text(
                    df_data['x'].iloc[i], 
                    df_data['y'].iloc[i], 
                    df_data['label'].iloc[i], 
                    alpha = 1,
                    rotation = rotate_labels, 
                    fontdict = highlighted_fontdict)
            else:
                continue
    else:
        if not label is None:
            # Print all labels
            for i in range(len(df_data)):
                ax.text(
                    df_data['x'].iloc[i], 
                    df_data['y'].iloc[i], 
                    df_data['label'].iloc[i], 
                    alpha = .9,
                    rotation = rotate_labels, 
                    fontdict = fontdict)

    style_axes(ax, show_axes, show_box, show_grid, axes_at_origin)

    if not color is None:
        if show_legend:
            lgnd = ax.legend(loc = "center right", bbox_to_anchor = (1.2, 0.5), fancybox = True, shadow = True)
            for handle in lgnd.legend_handles:
                handle.set_sizes([12.0])

    if not title is None:
        ax.set_title(title, fontdict = title_fontdict)

    # Save or show plot 
    if not filename is None:
        mydpi = 300
        fig.savefig(filename, dpi = mydpi, format = 'png')
    plt.close()
    if return_fig:
        return fig

def draw_shepard_diagram(X, D, ax = None, show_grid = False, show_rank_correlation = True):
    """Draw a shepard diagram of input dissimilarities vs map distances.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_dims)
        configuration of objects on the map
    D : np.ndarray of shape (n_samples, n_samples)
        Dissimilarity matrix
    ax : Axes, optional
        Axes to draw the shepard diagram on, by default None
    show_grid : bool, optional
        If true, grid lines will be drawn, by default False
    show_rank_correlation : bool, optional
        If true, the Spearman rank correlation coefficient will be displayed below the plot, by default True
    """
    def normalize_dhat(d_hat, n_samples):
        return d_hat * np.sqrt((n_samples * (n_samples - 1) / 2) / (d_hat**2).sum())
    
    from sklearn.isotonic import IsotonicRegression
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr

    D = D.copy()

    if type(X) == list and type(D) == list:
        all_distances = np.array([])
        all_disparities = np.array([])
        n_periods = len(X)
        for t in range(n_periods):
            distances = cdist(X[t], X[t], metric = 'euclidean')
            distances_flat = distances[np.tril_indices(len(distances),-1)]
            disparities_flat = D[t][np.tril_indices(len(D[t]),-1)]
            all_disparities = np.concatenate(all_disparities, disparities_flat)
            all_distances = np.concatenate(all_distances, distances_flat)        

    else:
        distances = cdist(X, X, metric = 'euclidean')
        distances_flat = distances[np.tril_indices(len(distances),-1)]
        disparities_flat = D[np.tril_indices(len(D),-1)]        

    ir = IsotonicRegression()

    disp_hat = ir.fit_transform(y = distances_flat, X = disparities_flat)
    disp_hat = normalize_dhat(disp_hat, X.shape[0])
    df = pd.DataFrame({'disp': disparities_flat, 'dist': distances_flat, 'disp_hat' : disp_hat})
    df = df.sort_values('disp')

    from matplotlib.ticker import FuncFormatter

    #TODO: Check if style_axes function can replace this part

    if ax is None:
        fig, ax = plt.subplots(figsize = (5,5))

    ax.plot(df['disp'], df['dist'], "C0.", markersize = 12)
    ax.plot(df['disp'], df['disp_hat'], "C1.-", markersize = 12)
    ax.set_xlabel('Input Dissimilaritiy', fontdict= axis_label_fontdict)
    ax.set_ylabel('Map Distance', fontdict = axis_label_fontdict)
    y_min = 0
    y_max = df['dist'].max()
    y_max *= 1.1
    x_min = df['disp'].min()
    x_max = df['disp'].max()
    x_min -= 0.1 * (x_max - x_min)
    x_max += 0.1 * (x_max - x_min)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    y_ticks = np.linspace(y_min, y_max, 5)
    x_ticks = np.linspace(x_min, x_max, 5)

    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis = 'both', labelsize = 10)
    ax.xaxis.set_major_formatter(FuncFormatter(format_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick_labels))

    if show_rank_correlation:
        ax.text(0.5, -0.15, 
            "Rank Correlation: {0:.2f}".format(spearmanr(df['disp'], df['dist'])[0]), 
            ha = 'center', 
            transform = ax.transAxes, 
            fontdict = axis_label_fontdict)
    
    if show_grid:
        ax.grid(True)
    else:
        ax.grid(False)

def draw_map_sequence(X_t, color_t = None, incl_t = None, n_cols = 4, time_labels = [], show_axes = False, **kwargs):
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
    draw_map_kws = {}
    draw_map_args = draw_map.__code__.co_varnames[:draw_map.__code__.co_argcount]
    for key, value in kwargs.items():
        if key in draw_map_args:
            draw_map_kws.update({key: value})
            
    n_periods = len(X_t)
    n_rows = int(np.ceil(n_periods/n_cols))

    if "fig_size" in draw_map_kws.keys():
        fig_size = draw_map_kws["fig_size"]
    else:
        fig_size = (4*n_cols, 4*n_rows)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize = fig_size)

    if len(time_labels) == 0:
        time_labels = ["Period " + str(t+1) for t in range(n_periods)]

    row = 0
    col = 0
    ymin, ymax , xmin, xmax = np.inf,-np.inf,np.inf,-np.inf

    for t in range(n_periods):
        if n_rows > 1:
            draw_map_kws.update({
                'X': X_t[t],
                'ax': axs[row, col]
            })
        else:
            draw_map_kws.update({
                'X': X_t[t],
                'ax': axs[col]
            })
        if not color_t is None:
            draw_map_kws.update({'c': color_t[t]})
        if not time_labels is None:
            draw_map_kws.update({'title': time_labels[t]})

        if not incl_t is None:
            draw_map_kws.update({'inclusions': incl_t[t]})

        _ = draw_map(**draw_map_kws)            
        if not incl_t is None:            
            ymin_i, ymax_i = np.min(X_t[t][incl_t[t] == 1, 1]), np.max(X_t[t][incl_t[t] == 1, 1])
            xmin_i, xmax_i = np.min(X_t[t][incl_t[t] == 1, 0]), np.max(X_t[t][incl_t[t] == 1, 0])
        else:
            ymin_i, ymax_i = np.min(X_t[t][:, 1]), np.max(X_t[t][:, 1])
            xmin_i, xmax_i = np.min(X_t[t][:, 0]), np.max(X_t[t][:, 0])
            
        if ymin_i < ymin:
            ymin = ymin_i
        if xmin_i < xmin:
            xmin = xmin_i
        if ymax_i > ymax:
            ymax = ymax_i
        if xmax_i > xmax:
            xmax = xmax_i
        
        if col < n_cols - 1:
            col += 1
        else: 
            row += 1
            col = 0

    # Make axes square
    max_range = np.max([xmax-xmin, ymax-ymin]) * 1.1
    x_mean = xmin + (xmax - xmin)/2
    y_mean = ymin + (ymax - ymin)/2

    xmin = x_mean - max_range/2
    xmax = x_mean + max_range/2
    ymin = y_mean - max_range/2
    ymax = y_mean + max_range/2

    if 'show_box' in draw_map_kws.keys():
        show_box = draw_map_kws['show_box']
    else:
        show_box = True

    if 'show_grid' in draw_map_kws.keys():
        show_grid = draw_map_kws['show_grid']
    else:
        show_grid = False
    
    if 'axes_at_origin' in draw_map_kws.keys():
        axes_at_origin = draw_map_kws['axes_at_origin']
    else:
        axes_at_origin = False
    
    if 'show_axes' in draw_map_kws.keys():
        show_axes = draw_map_kws['show_axes']
    else:
        show_axes = False

    row = 0
    col = 0
    for t in range(n_periods):
        if n_rows > 1:       
            style_axes(axs[row,col], show_axes=show_axes, show_box = show_box, show_grid = show_grid, axes_at_origin = axes_at_origin)

        else:
            style_axes(axs[col], show_axes=show_axes, show_box = show_box, show_grid = show_grid, axes_at_origin = axes_at_origin)

        if col < n_cols - 1:
            col += 1
        else: 
            row += 1
            col = 0
    
    plt.close() #Prevent jupyter notebooks from displaying empty figures
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

def draw_dynamic_map(X_t, color_t = None, size_t = None, incl_t = None, show_arrows = False, 
    show_last_positions_only = False, time_labels = None, 
    transparency_start = 0.1, transparency_end = 0.4, transparency_final = 1.,**kwargs):

    # Check inputs
    n_periods = len(X_t)
    n_samples = X_t[0].shape[0]
    if np.any([X.shape != X_t[0].shape for X in X_t]):
        raise ValueError('All input arrays need to be of similar shape.')

    if not color_t is None:
        if np.any([color.shape[0] != n_samples for color in color_t]):
            raise ValueError('Misshaped class arrays.')

    if not incl_t is None:
        if np.any([incl.shape[0] != n_samples for incl in incl_t]):
            raise ValueError('Misshaped inclusion arrays.')

    else:
        incl_t = [np.repeat(1,n_samples)]*n_periods

    # Data preparation
    transparencies = np.linspace(transparency_start, transparency_end, n_periods-1).tolist()
    transparencies.append(transparency_final)
  
    highlight_colors = ['darkred', 'orange', 'darkgreen', 'slategrey']

    if not 'label' in kwargs.keys():
        labels = np.array([str(i+1) for i in range(n_samples)])
    else:
        labels = np.array(kwargs['label'])

    if not time_labels is None:
        labels = [label + " " + time_labels[i] for i, label in enumerate(time_labels)]

    # Highlight labels are only shown in last period - so safe them now for later
    labels_to_highlight = []

    if 'highlighted_labels' in kwargs.keys():
        labels_to_highlight = np.concatenate((labels_to_highlight, kwargs['highlighted_labels']))

    if len(labels_to_highlight) >0:
        labels_to_highlight = np.unique(labels_to_highlight)

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
        return_fig = True
    else:
        ax = kwargs['ax']
        return_fig = False
    
    # Plot each period
    for t in range(n_periods):

        draw_map_kws.update({
            'X': X_t[t], 
            'inclusions': incl_t[t], 
            'filename': None, 
            'title': None,
            'scatter_kws': {'alpha': transparencies[t]}})

        # Only show labels for the last period
        if t < n_periods - 1:
            draw_map_kws.update({
                'highlighted_labels': None,
                'show_legend': False})
        else:
            draw_map_kws.update({
                'label': labels})

            if 'title' in kwargs.keys():
                draw_map_kws.update({'title': kwargs['title']})

            if 'show_legend' in kwargs.keys():
                draw_map_kws.update({'show_legend': kwargs['show_legend']})
            
        if not color_t is None:
            draw_map_kws.update({'color' : color_t[t]})

        if not size_t is None:
            draw_map_kws.update({'size' : size_t[t]})

        if t < n_periods-1:
            draw_map_kws.update({'label': None})

        # Plot the map positions
        if not (t < n_periods -1  and show_last_positions_only):
            p = draw_map(**draw_map_kws)
    
            # Highlight map positions
            if not labels_to_highlight is None:
                highlight_count = 0
                highlight_indices = [np.where(labels == label)[0][0] for label in labels_to_highlight]
                for i in highlight_indices:
                    x = X_t[t][:,0][i]
                    y = X_t[t][:,1][i]
                    sns.regplot(
                        x = [x], 
                        y = [y],
                        fit_reg=False, 
                        ax = ax,
                        scatter_kws={
                            'zorder': 10,
                            'alpha': transparencies[t], 
                            'facecolors': highlight_colors[highlight_count], 
                            'edgecolor': highlight_colors[highlight_count]})
                    highlight_count += 1

        # Plot the movement paths
        if (t > 0) and show_arrows:

            arrow_starts_x = X_t[t-1][:,0]
            arrow_starts_y = X_t[t-1][:,1]
            deltas_x = X_t[t][:,0] - X_t[t-1][:,0]
            deltas_y = X_t[t][:,1] - X_t[t-1][:,1]

            highlight_count = 0
            if len(labels_to_highlight) > 0:
                arrow_indices = [np.where(labels == label)[0][0] for label in labels_to_highlight]
            else:
                arrow_indices = np.array([i for i in range(n_samples)])

            for i in arrow_indices:
                # In case of zooming, only print arrows for objects within this cluster
                if 'zoom_on_cluster' in kwargs.keys():
                    if not kwargs['zoom_on_cluster'] is None:
                        if str(draw_map_kws['c'][i]) != str(kwargs['zoom_on_cluster']):
                            continue

                if (incl_t[t][i] == 1) and (incl_t[t-1][i] == 1):
                    if not (deltas_x[i] * deltas_y[i]) == 0:
                        if len(labels_to_highlight) > 0:
                            col = highlight_colors[highlight_count]
                            arrow_size = .05
                            alpha = transparencies[t-1]
                            highlight_count +=1
                        else:
                            col = 'grey'
                            alpha = transparencies[t-1]
                            arrow_size = .05
                        ax.plot(
                            [X_t[t-1][i,0], X_t[t][i,0]],
                            [X_t[t-1][i,1], X_t[t][i,1]],
                            color = 'grey', alpha = alpha,
                            linewidth = 1)


    if 'show_box' in draw_map_kws.keys():
        show_box = draw_map_kws['show_box']
    else:
        show_box = True

    if 'show_grid' in draw_map_kws.keys():
        show_grid = draw_map_kws['show_grid']
    else:
        show_grid = False
    
    if 'axes_at_origin' in draw_map_kws.keys():
        axes_at_origin = draw_map_kws['axes_at_origin']
    else:
        axes_at_origin = False
    
    if 'show_axes' in draw_map_kws.keys():
        show_axes = draw_map_kws['show_axes']
    else:
        show_axes = False

    style_axes(ax, show_axes=show_axes, show_box = show_box, show_grid = show_grid, axes_at_origin = axes_at_origin)

    if 'filename' in kwargs.keys():
        fig.savefig(kwargs['filename'], dpi = 300, format = "png")

    plt.close()
    if return_fig:
        return fig
    
def draw_trajectories(Y_ts, labels, selected_labels = None, title = None, 
    show_axes = False, show_box = True, show_grid = False, axes_at_origin = False,
    annotate_periods = True, period_labels = None, ax = None, fig_size = None):
    """ Draw the trajectories of selected objects.

    Parameters
    ----------
    Y_ts : list of ndarrays, each of shape (n_samples, d)
        Sequence of map coordinates.
    labels : ndarray of shape (n_samples,)
        Object labels (str)
    selected_labels : ndarray of shape (n_selected,), optional
        Selected object labels (str), by default None
    title : str, optional
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

    # If not ax is provided, return the whole FIgure. Else, only draw the plot on the provided axes
    if ax is None:
        return_fig = True
        if fig_size is None:
            fig_size = (5,5)
        fig, ax = plt.subplots(figsize = fig_size)
    else:
        return_fig = False  

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
            ax.scatter(x,y , c = c, alpha = alpha)
            xs.append(x)
            ys.append(y)
            if t == n_periods - 1:
                label = ax.text(x ,y , label, c = c, alpha = .7, fontsize = DEFAULT_FONT_SIZE)
                annotations.append(label)

            elif annotate_periods:
                label = ax.text(x ,y , period_labels[t], c = c_line, alpha = .5, fontsize = DEFAULT_FONT_SIZE * 0.8)
#               texts.append(label)

        # Plot the trajectory
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.plot(xs, ys, c = c_line, alpha = .4)

    style_axes(ax = ax, show_axes= show_axes, show_box = show_box, show_grid = show_grid, axes_at_origin = axes_at_origin)

    if not title is None:
        ax.set_title(title, fontdict = title_fontdict)

    plt.close()
    if return_fig:
        return fig
