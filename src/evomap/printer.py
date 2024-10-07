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
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from evomap.mapping._regression import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from cycler import cycler
import warnings
import statsmodels.api as sm

DEFAULT_BUBBLE_SIZE = 50
DEFAULT_FONT_SIZE = 12

title_fontdict_large = {'size': 20, 'family': 'Arial'}
title_fontdict = {'size': 18, 'family': 'Arial'}
text_fontdict = {'size': DEFAULT_FONT_SIZE, 'family': 'Arial'}
axis_label_fontdict = {'size': 12, 'family': 'Arial'}

def format_tick_labels(x, pos):
    return '{0:.2f}'.format(x)

def init_params(custom_params=None):
    """
    Initialize plot aesthetics.
    """
    base_style = {
        "axes.prop_cycle": cycler('color', ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                                            "#bcbd22", "#17becf"]),
        "axes.linewidth": 1,
        "axes.titlesize": 22,
        "axes.labelsize": 16,
        "font.family": 'Arial',
        "axes.edgecolor": "black",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.size": 0.2,
        "xtick.minor.size": 0.1,
        "ytick.major.size": 0.2,
        "ytick.minor.size": 0.1,
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

    if custom_params:
        base_style.update(custom_params)

    mpl.rcParams.update(base_style)


def style_axes(ax, show_axes=True, show_box=True, show_grid=False, axes_at_origin=False):
    ax.xaxis.set_visible(show_axes)
    ax.yaxis.set_visible(show_axes)

    ax.grid(show_grid and show_axes)

    if show_box:
        for spine in ax.spines.values():
            spine.set_visible(True)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(show_axes)
        ax.spines['bottom'].set_visible(show_axes)

    if axes_at_origin and show_axes:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    elif not axes_at_origin:
        ax.spines['left'].set_position(('outward', 0))
        ax.spines['bottom'].set_position(('outward', 0))

    # Setting labels
    if show_axes:
        ax.set_xlabel("Dimension 1", fontdict=axis_label_fontdict)
        ax.set_ylabel("Dimension 2", fontdict=axis_label_fontdict)

    ax.tick_params(axis='x', which='both', bottom=show_axes, labelbottom=show_axes)
    ax.tick_params(axis='y', which='both', left=show_axes, labelleft=show_axes)

    # Set equal ticks on both axes
    set_equal_ticks(ax)

    # Explicitly setting aspect to 'auto' so the axis scales adjust dynamically to the data
    ax.set_aspect('auto')
    ax.autoscale_view()

def set_equal_ticks(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Define the number of ticks you want
    num_ticks = 5

    # Create ticks based on the range of each axis
    x_ticks = np.linspace(x_min, x_max, num_ticks)
    y_ticks = np.linspace(y_min, y_max, num_ticks)

    # Set these ticks to the axis
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Optionally, you can format the ticks to show a specific number of decimal places or in a particular format
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

def draw_map(X, label=None, color=None, size=None, inclusions=None, zoom_on_cluster=None, highlighted_labels=None, 
             show_box=True, show_grid=False, show_axes=True, axes_at_origin=False, show_legend=False,
             cmap=None, filename=None, ax=None, fig_size=None, 
             title=None, rotate_labels=0, scatter_kws={}, fontdict=None, rcparams=None):
    """
    Plot a scatter map with optional labels, coloring, and sizing.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points to plot. n_features should be 1 or 2.
    label : array-like, optional
        Labels for each data point.
    color : array-like, optional
        Colors or group identifiers for each data point. If None, all points will have the same color.
    size : array-like, optional
        Sizes for each data point. If None, a default size is used.
    inclusions : array-like of bool, optional
        Boolean array to select which points are included in the plot.
    zoom_on_cluster : int or string, optional
        Cluster identifier to zoom in on specific cluster data points.
    highlighted_labels : list, optional
        Labels to be highlighted on the plot.
    show_box : bool, optional
        If True, show a box around the plot. Default is True.
    show_grid : bool, optional
        If True, show grid lines on the plot. Default is False.
    show_axes : bool, optional
        If True, show the axes of the plot. Default is False.
    axes_at_origin : bool, optional
        If True, draw axes lines through the origin. Default is False.
    show_legend : bool, optional
        If True, display a legend on the plot. Default is False.
    cmap : str or Colormap, optional
        Colormap to use for coloring the points. If None, a default colormap is used.
    filename : str, optional
        Path to save the figure file. If None, the figure is not saved.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes are created.
    fig_size : tuple, optional
        Size of the figure to create. Ignored if `ax` is not None.
    title : str, optional
        Title of the plot.
    rotate_labels : int, optional
        Angle to rotate the labels. Default is 0.
    scatter_kws : dict, optional
        Additional keyword arguments to pass to the scatter plot function.
    fontdict : dict, optional
        Font dictionary for the labels. If None, a default fontdict is used.
    rcparams : dict, optional
        Dictionary to update matplotlib's rcParams for customizing plots.

    Returns
    -------
    matplotlib.figure.Figure
        Only if `ax` is None, the figure containing the plot is returned.
    """
    n_samples = len(X)
    X = np.atleast_2d(X)
    if X.shape[1] == 1:
        X = np.hstack([X, np.zeros_like(X)])

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        cmap = plt.get_cmap('tab10')

    df_data = pd.DataFrame(X, columns=['x', 'y'])
    if color is None:
        df_data['color'] = ['white'] * n_samples  # Use the first color in the colormap for all points
        edge_colors = ['black'] * n_samples
    else:
        color = np.asarray(color)
        unique_colors = np.unique(color)
        if len(unique_colors) > cmap.N:
            warnings.warn("Number of color labels exceeds the number of available colors in the colormap; colors will repeat.")
        color_indices = pd.Categorical(color, categories=unique_colors).codes
        colors = cmap(color_indices % cmap.N)  # Use modulo to avoid index out of bounds
        df_data['color'] = [colors[i] for i in range(len(color_indices))]  # Assign colors as a list of RGBA values
        edge_colors = ['black'] * n_samples

    if label is not None:
        df_data['label'] = label
    if size is not None:
        df_data['size'] = size
    else:
        df_data['size'] = DEFAULT_BUBBLE_SIZE

    if inclusions is not None:
        df_data = df_data[inclusions == 1]

    if zoom_on_cluster is not None:
        df_data = df_data[df_data['color'] == zoom_on_cluster]

    init_params(rcparams)

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size or (6,6))
        return_fig = True
    else:
        return_fig = False

    scatter_args = {'edgecolors': edge_colors, 'alpha': 0.75}
    scatter_args.update(scatter_kws)
    ax.scatter(df_data['x'], df_data['y'], c=df_data['color'], s=df_data['size'], **scatter_args)
    
    if label is not None:
        for _, row in df_data.iterrows():
            if highlighted_labels is None or row['label'] in highlighted_labels:
                ax.text(row['x'], row['y'], row['label'], fontdict=fontdict or text_fontdict, rotation=rotate_labels)

    style_axes(ax, show_axes, show_box, show_grid, axes_at_origin)

    if title:
        ax.set_title(title, fontdict=title_fontdict)

    if show_legend:
        ax.legend(title='Cluster')

    if filename:
        plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')
    
    if not return_fig:
        plt.close()

def normalize_dhat(d_hat, n_samples):
    """ Normalize dissimilarity predictions. """
    return d_hat * np.sqrt((n_samples * (n_samples - 1) / 2) / np.sum(d_hat**2))

def draw_shepard_diagram(X, D, ax=None, show_grid=False, show_rank_correlation=True):
    """
    Draw a Shepard diagram of input dissimilarities vs map distances.

    Parameters:
    X (np.ndarray): Configuration of objects on the map, shape (n_samples, n_dims).
    D (np.ndarray): Dissimilarity matrix, shape (n_samples, n_samples).
    ax (matplotlib.axes.Axes, optional): Axes object to draw the diagram on.
    show_grid (bool, optional): Whether to show grid lines on the plot.
    show_rank_correlation (bool, optional): Whether to display the rank correlation coefficient.
    """
    distances = cdist(X, X, metric='euclidean')
    distances_flat = distances[np.tril_indices(len(distances), -1)]
    disparities_flat = D[np.tril_indices(len(D), -1)]

    # Fit isotonic regression to the flattened arrays
    ir = IsotonicRegression()
    disp_hat = ir.fit_transform(X=disparities_flat, y=distances_flat)
    disp_hat = normalize_dhat(disp_hat, X.shape[0])

    # Prepare data for plotting
    df = pd.DataFrame({
        'Disparities': disparities_flat,
        'Distances': distances_flat,
        'Fitted Distances': disp_hat
    }).sort_values('Disparities')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    
    # Plotting the original and fitted distances
    ax.scatter(df['Disparities'], df['Distances'], color="darkblue", label="Original", alpha=0.6)
    ax.plot(df['Disparities'], df['Fitted Distances'], color="orange", label="Fitted", marker='o', linestyle='-')
    
    ax.set_xlabel('Input Dissimilarity', fontdict=axis_label_fontdict)
    ax.set_ylabel('Map Distance', fontdict=axis_label_fontdict)
    ax.legend()

    if show_grid:
        ax.grid(True)

    # Display Spearman rank correlation if requested
    if show_rank_correlation:
        rank_corr = spearmanr(df['Disparities'], df['Distances'])[0]
        ax.text(0.5, -0.15, f'Rank Correlation: {rank_corr:.2f}', transform=ax.transAxes, 
                ha='center', fontsize=14)

    ax.set_ylim(0, df['Distances'].max() * 1.15)
    ax.set_xlim(df['Disparities'].min() * 0.99, df['Disparities'].max() * 1.01)
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    return ax

def validate_inputs(X_t, color_t, incl_t, n_samples):
    """ Validate the shape and consistency of inputs. """
    if np.any([X.shape != X_t[0].shape for X in X_t]):
        raise ValueError("All input arrays need to be of similar shape.")
    if color_t is not None and any(color.shape[0] != n_samples for color in color_t):
        raise ValueError("Misshaped class arrays.")
    if incl_t is not None and any(incl.shape[0] != n_samples for incl in incl_t):
        raise ValueError("Misshaped inclusion arrays.")

def prepare_transparencies(n_periods, start, end, final):
    """ Prepare transparency values for the plotting periods. """
    return np.linspace(start, end, n_periods-1).tolist() + [final]


def draw_dynamic_map(X_t, color_t=None, size_t=None, incl_t=None, label = None, show_arrows=False,
                     show_last_positions_only=False, time_labels=None, show_axes = True, show_box = True, 
                     axes_at_origin = False, show_grid = False, cmap = None, transparency_start=0.1,
                     transparency_end=0.4, transparency_final=.75, scatter_kws={}, **kwargs):
    """
    Visualizes dynamic map data over multiple periods with options to show movement paths and adjust visual features.
    
    Parameters
    ----------
    X_t : list of ndarray
        List of arrays containing coordinates for each period, where each array is of shape (n_samples, n_features).
    color_t : list of ndarray, optional
        List of arrays containing color or group identifiers for each period.
    size_t : list of ndarray, optional
        List of arrays containing sizes for each data point in each period.
    incl_t : list of ndarray, optional
        List of arrays indicating if a point should be included in the plot for each period.
    show_arrows : bool, optional
        If True, display arrows showing movement between periods. Default is False.
    show_last_positions_only : bool, optional
        If True, only the last period's positions are shown with arrows indicating the movements from prior periods. Default is False.
    time_labels : list of str, optional
        Labels for each period, displayed in plot annotations or titles.
    transparency_start : float, optional
        Starting transparency level for the first period in the dynamic map.
    transparency_end : float, optional
        Ending transparency level just before the last period in the dynamic map.
    transparency_final : float, optional
        Transparency level for the last period in the dynamic map.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function or for configuring plot aspects.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the dynamic map, only if not plotted on an existing axis.
    """

    n_periods = len(X_t)
    n_samples = X_t[0].shape[0]

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        cmap = plt.get_cmap('tab10')

    incl_t = [np.ones(n_samples, dtype=bool) if incl is None else incl for incl in incl_t] if incl_t else [np.ones(n_samples, dtype=bool)] * n_periods

    ax = kwargs.get('ax')
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('fig_size', (6,6)))
        return_fig = True
    else:
        fig = ax.figure
        return_fig = False

    for t in range(n_periods):
        sizes = size_t[t] if size_t else np.full(n_samples, DEFAULT_BUBBLE_SIZE)
        if not color_t is None:
            unique_colors = np.unique(color_t[t])
            if len(unique_colors) > cmap.N:
                warnings.warn("Number of color labels exceeds the number of available colors in the colormap; colors will repeat.")
            color_indices = pd.Categorical(color_t[t], categories=unique_colors).codes
            colors = cmap(color_indices % cmap.N)
            edge_colors = np.array(['black'] * n_samples)  # Default to no edge color when using color mapping
        else:
            colors = np.array([cmap(0)] * n_samples)  # Default to the first color in the cmap
            edge_colors = np.array(['black'] * n_samples)  # Black edges when default coloring

        valid_indices = incl_t[t]
        if t == n_periods - 1:
            transparency = transparency_final  # Use final transparency for the last period
        else:
            transparency = np.linspace(transparency_start, transparency_end, n_periods-1)[t]

        scatter_args = {'edgecolors': edge_colors[valid_indices]}
        scatter_args.update(scatter_kws)

        # Only plot points for the last period if show_last_positions_only is True
        if not show_last_positions_only or t == n_periods - 1:
            ax.scatter(X_t[t][valid_indices, 0], X_t[t][valid_indices, 1], alpha=transparency, s=sizes[valid_indices], 
                    c=colors[valid_indices], **scatter_args)

        if show_arrows and t > 0:
            for i in range(n_samples):
                if incl_t[t][i] and incl_t[t-1][i]:
                    start_point = X_t[t-1][i]
                    end_point = X_t[t][i]
                    if not np.array_equal(start_point, end_point):
                        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                                color='gray', alpha=transparency, linewidth=1)

            if label is not None and t == n_periods - 1:  # Check explicitly if label is not None
                for i, txt in enumerate(label):
                    if valid_indices[i]:
                        ax.text(X_t[t][i, 0], X_t[t][i, 1], txt, fontsize=DEFAULT_FONT_SIZE)

    style_axes(ax, show_axes, show_box, show_grid, axes_at_origin)
    ax.axis('equal')

    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'], dpi=300)

    if not return_fig:
        plt.close(fig)

def plot_data(X, colors, sizes, inclusions, labels, transparency_start, transparency_end, transparency_final, period, n_periods, ax, show_last_positions_only):
    """
    Helper function to manage data plotting for each period.
    """
    transparency = np.linspace(transparency_start, transparency_end, n_periods)[period]
    if show_last_positions_only and period == n_periods - 1:
        transparency = transparency_final
    valid_indices = inclusions
    ax.scatter(X[valid_indices, 0], X[valid_indices, 1],
               alpha=transparency, s=sizes if sizes is not None else DEFAULT_BUBBLE_SIZE,
               color=colors[valid_indices])
    if period == n_periods - 1 and labels is not None:
        for i, txt in enumerate(labels):
            if valid_indices[i]:
                ax.text(X[i, 0], X[i, 1], txt, fontsize=DEFAULT_FONT_SIZE)

def draw_trajectories(Y_ts, labels, selected_labels = None, 
    show_axes = False, show_box = True, show_grid = False, axes_at_origin = False,
    annotate_periods = True, period_labels = None, fig_size = None, **kwargs):
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

    ax = kwargs.get('ax')
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('fig_size', (6,6)))
        return_fig = True
    else:
        fig = ax.figure
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

    if not return_fig:
        plt.close(fig)

def fit_attribute(coords, attribute_label, attribute_values, ax, coord_range):
    """Fit an attribute to the map and display the resultant vector."""
    # Store the current limits to restore them later
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    X = coords
    y = attribute_values
    est = sm.OLS(y, sm.add_constant(X))
    result = est.fit()
    vector = result.params[1:]  # Ignore intercept

    r_squared = result.rsquared
    length = ((x_lim[1] - x_lim[0]) / 2) * r_squared

    ax.arrow(0, 0, vector[0] * length, vector[1] * length, linestyle='--', lw=.25, alpha=.75, width=.001, color='grey', head_width=0.1)
    ax.text(vector[0] * 1.1 * length, vector[1] * 1.1 * length, attribute_label, fontdict={'size': 8, 'color': 'darkblue', 'weight': 'normal'})

    # Reset the limits to their original values
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def fit_attributes(map_coords, df_attributes, ax):
    """Fit multiple attributes and display their vectors on the map."""
    coord_range = (np.min(map_coords), np.max(map_coords))  # Get min and max for scaling
    for attribute in df_attributes.columns:
        fit_attribute(
            coords=map_coords, 
            attribute_label=attribute,
            attribute_values=df_attributes[attribute], 
            ax=ax,
            coord_range=coord_range)
        
def draw_map_sequence(X_t, color_t=None, size_t=None, incl_t=None, n_cols=4, time_labels=[], 
                      show_axes=True, show_box=True, axes_at_origin=False, show_grid=False,
                      cmap=None, fontdict=None, scatter_kws={}, **kwargs):
    """
    Draw a sequence of static maps next to each other.
    
    Parameters
    ----------
    X_t : list of ndarray
        List of arrays containing map coordinates for each period.
    color_t : list of ndarray, optional
        Color for each data point per period.
    size_t : list of ndarray, optional
        Size for each data point per period.
    incl_t : list of ndarray, optional
        Inclusion array to filter points for each period.
    n_cols : int, optional
        Maximum number of maps shown in one row.
    time_labels : list of str, optional
        Labels for each period, displayed as titles.
    show_axes : bool, optional
        Show axes of the plot.
    show_box : bool, optional
        Show a box around the plot.
    axes_at_origin : bool, optional
        Draw axes lines through the origin.
    show_grid : bool, optional
        Show grid lines on the plot.
    cmap : str or Colormap, optional
        Colormap to use for coloring the points.
    fontdict : dict, optional
        Font dictionary for the labels.
    scatter_kws : dict, optional
        Additional keyword arguments to pass to the scatter plot function.
    **kwargs : dict
        Additional keyword arguments for figure creation and styling.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the sequence of maps.
    """
    n_periods = len(X_t)
    n_rows = int(np.ceil(n_periods / n_cols))
    fig_size = kwargs.get('fig_size', (4 * n_cols, 4 * n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
    axs = np.atleast_2d(axs)  # Ensure axs is 2D array even if single row
    
    if len(time_labels) == 0:
        time_labels = ["Period " + str(t+1) for t in range(n_periods)]

    for t in range(n_periods):
        ax = axs[t // n_cols, t % n_cols]
        map_kws = {
            'X': X_t[t],
            'ax': ax,
            'label': None,  # Adjust based on actual parameters
            'color': color_t[t] if color_t else None,
            'size': size_t[t] if size_t else None,
            'inclusions': incl_t[t] if incl_t else None,
            'show_axes': show_axes,
            'show_box': show_box,
            'axes_at_origin': axes_at_origin,
            'show_grid': show_grid,
            'cmap': cmap,
            'fontdict': fontdict,
            'scatter_kws': scatter_kws
        }
        draw_map(**map_kws)
        if not time_labels is None:
            ax.set_title(time_labels[t])

        # Set all subplots to the same limits
        ax.set_xlim(np.min([ax.get_xlim()[0] for ax in axs.flat]), np.max([ax.get_xlim()[1] for ax in axs.flat]))
        ax.set_ylim(np.min([ax.get_ylim()[0] for ax in axs.flat]), np.max([ax.get_ylim()[1] for ax in axs.flat]))

    # Hide unused axes
    for idx in range(n_periods, n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].axis('off')

    return fig