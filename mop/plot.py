"""
Collection of functions used to generate plots
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2

"""

import loompy
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from . import general_utils
from . import loom_utils
from . import plot_helpers as ph

# Start log
plot_log = logging.getLogger(__name__)


def scatter_attr(loom_file,
                 x_axis,
                 y_axis,
                 plot_attr,
                 color_attr=None,
                 valid_attr=None,
                 highlight=None,
                 s=2,
                 downsample_number=None,
                 downsample_attr=None,
                 legend=False,
                 output=None,
                 xlim='auto',
                 ylim='auto',
                 x_label=None,
                 y_label=None,
                 title=None,
                 as_heatmap=False,
                 cbar_label=None,
                 low_p=1,
                 high_p=99,
                 figsize=(8, 6),
                 close=False,
                 **kwargs):
    """
    Plots scatter of cells in which each cluster is marked with a unique color
    
    Args:
        loom_file (str): Path to loom file
        x_axis (str): Attribute in loom_file specifying x-coordinates
        y_axis (str): Attribute in loom_file specifying y-coordinates
        plot_attr (str): Column attribute specifying basis of plotting
        color_attr (str): Optional, attribute specifying per cell colors
        valid_attr (str): Optional, attribute specifying cells to include
        highlight (str/list): Optional, only specified clusters will be colored
        s (int): Size of points on scatter plot
        downsample_number (int): Number to downsample to
        downsample_attr (str): Attribute to downsample by
        legend (bool): Includes legend with plot
        output (str): Optional, saves plot to a file
        xlim (tuple/str): Limits for x-axis
            auto to set based on data
        ylim (tuple/str): Limits for y-axis
            auto to set based on data
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        as_heatmap (bool): Plots attribute values as a heatmap
        cbar_label (str): Optional, adds and names colorbar
        low_p (int): Low end for percentile normalization (0-100)
        high_p (int): High end for percentile normalization (0-100)
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes figure
        **kwargs: keyword arguments for matplotlib's scatter
    
    Adapted from code by Fangming Xie
    """
    # Get indices
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    # Set-up dataframe 
    with loompy.connect(filename=loom_file, mode='r') as ds:
        df_plot = pd.DataFrame(
            {plot_attr: ds.ca[plot_attr][col_idx],
             'x_val': ds.ca[x_axis][col_idx].astype(float),
             'y_val': ds.ca[y_axis][col_idx].astype(float)})
        if color_attr is None:
            if as_heatmap:
                df_plot['color'] = ph.percentile_norm(
                    counts=df_plot[plot_attr],
                    low_p=low_p,
                    high_p=high_p)
            else:
                df_plot = ph.get_category_colors(df_plot=df_plot,
                                                 category_label=plot_attr,
                                                 color_label='color')
        else:
            df_plot['color'] = ds.ca[color_attr][col_idx]
        if downsample_attr is not None:
            df_plot[downsample_attr] = ds.ca[downsample_attr][col_idx]
    # Handle downsampling
    if downsample_number is not None:
        if isinstance(downsample_number, int):
            if downsample_attr is None:
                downsample_attr = plot_attr
            idx_to_use = []
            for item in df_plot[downsample_attr].unique():
                ds_idx = np.where(df_plot[downsample_attr] == item)[0]
                if ds_idx.shape[0] <= downsample_number:
                    idx_to_use.append(ds_idx)
                else:
                    subsample = np.random.choice(a=ds_idx,
                                                 size=downsample_number)
                    idx_to_use.append(subsample)
            idx_to_use = np.hstack(idx_to_use)
            df_plot = df_plot.iloc[idx_to_use, :]
        else:
            raise ValueError('downsample_number must be an integer')
    # Handle highlighting
    if highlight is not None:
        if isinstance(highlight, str):
            highlight = [highlight]
        elif isinstance(highlight, list) or isinstance(highlight, np.ndarray):
            pass
        else:
            raise ValueError('Unsupported type for highlight')
        hl_idx = pd.DataFrame(np.repeat([True], repeats=df_plot.shape[0]),
                              index=df_plot[plot_attr].values,
                              columns=['idx'])
        hl_idx['idx'].loc[highlight] = False
        tmp = df_plot['color'].copy()
        tmp.loc[hl_idx['idx'].values] = 'Null'
        df_plot['color'] = tmp
        highlight = True
    else:
        highlight = False
    # Make figure
    ph.plot_scatter(df_plot=df_plot,
                    x_axis='x_val',
                    y_axis='y_val',
                    col_opt='color',
                    s=s,
                    legend=legend,
                    legend_labels=plot_attr,
                    highlight=highlight,
                    output=output,
                    xlim=xlim,
                    ylim=ylim,
                    cbar_label=cbar_label,
                    x_label=x_label,
                    y_label=y_label,
                    title=title,
                    figsize=figsize,
                    close=close,
                    **kwargs)


def scatter_feature(loom_file,
                    x_axis,
                    y_axis,
                    feat_id,
                    layer,
                    feat_attr='Accession',
                    scale_attr=None,
                    clust_attr=None,
                    valid_attr=None,
                    highlight=None,
                    s=2,
                    downsample=None,
                    legend=False,
                    output=None,
                    xlim='auto',
                    ylim='auto',
                    x_label=None,
                    y_label=None,
                    title=None,
                    cbar_label=None,
                    low_p=1,
                    high_p=99,
                    gray_noncoverage=False,
                    coverage_layer=None,
                    figsize=(8, 6),
                    close=False,
                    **kwargs):
    """
    Plots scatter of cells in which each cluster is marked with a unique color
    
    Args:
        loom_file (str): Path to loom file
        x_axis (str): Attribute in loom_file specifying x-coordinates
        y_axis (str): Attribute in loom_file specifying y-coordinates
        layer (str): Layer for counts to be displayed
        feat_id (str): ID for feature of interest
        feat_attr (str): Row attribute containing feat_id
        scale_attr (str): Name of attribute to scale counts by
        clust_attr (str): Name of attribute containing cluster identities
            Used with downsample
        valid_attr (str): Optional, attribute specifying cells to include
        highlight (str/list): Optional, only specified clusters will be colored
        s (int): Size of points on scatter plot
        downsample (int): Number of cells to downsample to
        legend (bool): Includes legend with plot
        output (str): Optional, saves plot to a file
        xlim (tuple/str): Limits for x-axis
            auto to set based on data
        ylim (tuple/str): Limits for y-axis
            auto to set based on data
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        cbar_label (str): Optional, adds and names colorbar
        low_p (int): Low end for percentile normalization (0-100)
        high_p (int): High end for percentile normalization (0-100)
        gray_noncoverage (bool): Set non-covered features  to gray values
            Useful for methylation data
        coverage_layer (str): Layer used to identify covered features
        figsize (tuple): Size of scatter plot figure
        close (bool): Do not plot figure inline
        **kwargs: keyword arguments for matplotlib's scatter
    
    Adapted from code by Fangming Xie
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    # Set-up dataframe 
    with loompy.connect(filename=loom_file, mode='r') as ds:
        feat_idx = np.ravel(np.where(ds.ra[feat_attr] == feat_id))
        if feat_idx.shape[0] > 1:
            raise ValueError('Too many feature matches')
        if feat_idx.shape[0] == 0:
            raise ValueError('Feature was not found')
        counts = np.ravel(
            ds.layers[layer][feat_idx, :][:, col_idx].astype(float))
        if scale_attr is not None:
            scale_factor = ds.ca[scale_attr][col_idx]
            counts = np.divide(counts,
                               scale_factor,
                               out=np.zeros_like(counts),
                               where=scale_factor != 0)
        df_plot = pd.DataFrame({'x_val': ds.ca[x_axis][col_idx].astype(float),
                                'y_val': ds.ca[y_axis][col_idx].astype(float)})

        if clust_attr:
            df_plot[clust_attr] = ds.ca[clust_attr][col_idx].astype(str)
        if gray_noncoverage:
            if coverage_layer is None:
                raise ValueError('coverage_layer is not provided')
            else:
                add_gray = True
                cov_count = np.ravel(
                    ds.layers[layer][feat_idx, :][:, col_idx].astype(float))
                cov_idx = cov_count > 0
        else:
            add_gray = False
    # Add colors
    df_plot['color'] = ph.percentile_norm(counts=counts,
                                          low_p=low_p,
                                          high_p=high_p)
    # Downsample
    if downsample is not None:
        if isinstance(downsample, int):
            idx_to_use = []
            if clust_attr:

                for cluster in df_plot[clust_attr].unique():
                    clust_idx = np.where(df_plot[clust_attr] == cluster)[0]
                    if clust_idx.shape[0] <= downsample:
                        idx_to_use.append(clust_idx)
                    else:
                        subsample = np.random.choice(a=clust_idx,
                                                     size=downsample)
                        idx_to_use.append(subsample)
                idx_to_use = np.hstack(idx_to_use)
            else:
                idx_to_use = np.random.choice(a=np.arange(df_plot.shape[0]),
                                              size=downsample)
            df_plot = df_plot.iloc[idx_to_use, :]
        else:
            raise ValueError('downsample must be an integer')
    # Highlight
    if highlight is not None and clust_attr is not None:
        hl_idx = pd.DataFrame(np.arange(0, df_plot.shape[0]),
                              index=df_plot[clust_attr].values,
                              columns=['idx'])
        hl_idx = hl_idx.loc[highlight]
        df_plot = df_plot.iloc[hl_idx['idx'].values]
    # Make figure
    if add_gray:
        df_noncov = df_plot.copy()
        df_noncov['color'] = np.repeat('lightgray', df_noncov.shape[0])
        fig, ax = plt.subplots(figsize=figsize)
        ph.plot_scatter(df_plot=df_noncov,
                        x_axis='x_val',
                        y_axis='y_val',
                        col_opt='color',
                        s=s,
                        fig=fig,
                        ax=ax,
                        **kwargs)
        if np.sum(cov_idx) > 0:
            ph.plot_scatter(df_plot=df_plot.loc[cov_idx],
                            x_axis='x_val',
                            y_axis='y_val',
                            col_opt='color',
                            s=s,
                            legend=legend,
                            legend_labels=clust_attr,
                            output=output,
                            xlim=ax.get_xlim(),
                            ylim=ax.get_ylim(),
                            x_label=x_label,
                            y_label=y_label,
                            title=title,
                            figsize=figsize,
                            cbar_label=cbar_label,
                            close=close,
                            fig=fig,
                            ax=ax,
                            **kwargs)
    else:
        ph.plot_scatter(df_plot=df_plot,
                        x_axis='x_val',
                        y_axis='y_val',
                        col_opt='color',
                        s=s,
                        legend=legend,
                        legend_labels=clust_attr,
                        output=output,
                        xlim=xlim,
                        ylim=ylim,
                        x_label=x_label,
                        y_label=y_label,
                        title=title,
                        figsize=figsize,
                        cbar_label=cbar_label,
                        close=close,
                        **kwargs)


def sankey(loom_file,
           left_attr,
           right_attr,
           left_color=None,
           right_color=None,
           line_color=None,
           valid_attr=None,
           left_label=None,
           right_label=None,
           title=None,
           figsize=(8, 6),
           output=None,
           close=False):
    """
    Generates Sankey (river) plots between two attributes
        Typically two different types of clustering

    Args:
        loom_file (str): Path to loom file
        left_attr (str): Column attribute containing left side values
        right_attr (str): Column attribute containing right side values
        left_color (str): Optional, column attribute with left colors
        right_color (str): Optional, column attribute with right colors
        line_color (str): Optional, column attribute with line colors
        valid_attr (str): Optional, column attribute specifying cells to include
        left_label (str): Label for left hand side of the plot
            If None, is set to left_attr
        right_label (str): Label for right hand side of the plot
            If None, is set to right_attr
        title (str): Optional, title of plot
        figsize (tuple): Size of outputput figure
        output (str): Optional, output file name
        close (bool): Optional, close figure after generating
    Adapted from pysankey by:
        Anneya Golob
        marcomanz
        pierre-sassoulas
        jorwoods
        (https://github.com/anazalea/pySankey/)
    """
    # Handle labels
    if left_label is None:
        left_label = left_attr
    if right_label is None:
        right_label = right_attr
    # Make dataframe of data
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_attr,
                                          columns=True,
                                          as_bool=True,
                                          inverse=False)
    num_cells = np.sum(valid_idx)
    with loompy.connect(loom_file) as ds:
        df_plot = pd.DataFrame({'left_value': ds.ca[left_attr][valid_idx],
                                'right_value': ds.ca[right_attr][valid_idx],
                                'left_weight': np.ones(num_cells),
                                'right_weight': np.ones(num_cells)},
                               index=range(num_cells))
    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    l_labels = general_utils.nat_sort(df_plot['left_value'].unique())[::-1]
    r_labels = general_utils.nat_sort(df_plot['right_value'].unique())[::-1]
    for l_label in l_labels:
        left_dict = {}
        right_dict = {}
        left_df = df_plot[df_plot['left_value'] == l_label]
        for r_label in r_labels:
            right_df = left_df[left_df['right_value'] == r_label]
            left_dict[r_label] = right_df['left_weight'].sum()
            right_dict[r_label] = right_df['right_weight'].sum()
        ns_l[l_label] = left_dict
        ns_r[l_label] = right_dict
    # Determine positions of left label patches and total widths
    left_width = defaultdict()
    top_values = []
    for i, l_label in enumerate(l_labels):
        tmp_df = df_plot[df_plot['left_value'] == l_label]
        tmp_dict = {'left': tmp_df['left_weight'].sum()}
        if i == 0:
            tmp_dict['bottom'] = 0
            tmp_dict['top'] = tmp_dict['left']
        else:
            curr_label = l_labels[i - 1]
            curr_offset = df_plot['left_weight'].sum() * 0.02
            tmp_dict['bottom'] = left_width[curr_label]['top'] + curr_offset
            tmp_dict['top'] = tmp_dict['bottom'] + tmp_dict['left']
        top_values.append(tmp_dict['top'])
        left_width[l_label] = tmp_dict
    # Determine positions of right label patches and total widths
    right_width = defaultdict()
    for i, r_label in enumerate(r_labels):
        tmp_df = df_plot[df_plot['right_value'] == r_label]
        tmp_dict = {'right': tmp_df['right_weight'].sum()}
        if i == 0:
            tmp_dict['bottom'] = 0
            tmp_dict['top'] = tmp_dict['right']
        else:
            curr_label = r_labels[i - 1]
            curr_offset = df_plot['right_weight'].sum() * 0.02
            tmp_dict['bottom'] = right_width[curr_label]['top'] + curr_offset
            tmp_dict['top'] = tmp_dict['bottom'] + tmp_dict['right']
        top_values.append(tmp_dict['top'])
        right_width[r_label] = tmp_dict
    # Determine vertical aspect of plot
    x_max = np.max(top_values) / 30
    # Make color labels
    if left_color is None:
        left_col = pd.DataFrame(
            {'color': ph.get_random_colors(len(l_labels))},
            index=l_labels[::-1])
    else:
        with loompy.connect(loom_file) as ds:
            left_col = pd.DataFrame({'color': ds.ca[left_color][valid_idx]},
                                    index=ds.ca[left_attr][valid_idx])
            left_col = left_col[~left_col.index.duplicated(keep='first')]
            left_col = left_col.loc[l_labels[::-1]]
    if right_color is None:
        right_col = pd.DataFrame(
            {'color': ph.get_random_colors(len(r_labels))},
            index=r_labels[::-1])
    else:
        with loompy.connect(loom_file) as ds:
            right_col = pd.DataFrame({'color': ds.ca[right_color][valid_idx]},
                                     index=ds.ca[right_attr][valid_idx])
            right_col = right_col[~right_col.index.duplicated(keep='first')]
            right_col = right_col.loc[r_labels[::-1]]
    if line_color is None:
        line_col = left_col.copy()
    else:
        with loompy.connect(loom_file) as ds:
            line_col = pd.DataFrame({'color': ds.ca[line_color][valid_idx]},
                                    index=ds.ca[left_attr][valid_idx])
            line_col = line_col[~line_col.index.duplicated(keep='first')]
            line_col = line_col.loc[l_labels[::-1]]
    # Make plot
    fig, ax = plt.subplots(figsize=figsize)
    for l_label in l_labels:
        bottom_pos = left_width[l_label]['bottom']
        left_pos = left_width[l_label]['left']
        ax.fill_between([-0.02 * x_max, 0],
                        2 * [bottom_pos],
                        2 * [bottom_pos + left_pos],
                        color=left_col.loc[l_label][0],
                        alpha=0.99)
        ax.text(-0.05 * x_max,
                bottom_pos + 0.5 * left_pos,
                l_label,
                {'ha': 'right', 'va': 'center'})
    for r_label in r_labels:
        bottom_pos = right_width[r_label]['bottom']
        right_pos = right_width[r_label]['right']
        ax.fill_between([x_max, 1.02 * x_max], 2 * [bottom_pos],
                        2 * [bottom_pos + right_pos],
                        color=right_col.loc[r_label][0],
                        alpha=0.99)
        ax.text(1.05 * x_max,
                bottom_pos + 0.5 * right_pos,
                r_label,
                {'ha': 'left', 'va': 'center'})
    for l_label in l_labels:
        for r_label in r_labels:
            good_idx = np.logical_and(df_plot['left_value'] == l_label,
                                      df_plot['right_value'] == r_label)
            if np.sum(good_idx) > 0:
                left_bottom = left_width[l_label]['bottom']
                right_bottom = right_width[r_label]['bottom']
                left_ns = ns_l[l_label][r_label]
                right_ns = ns_r[l_label][r_label]
                left_strip = left_bottom + left_ns
                right_strip = right_bottom + right_ns
                # Determine lines
                ys_d = np.array(50 * [left_bottom] + 50 * [right_bottom])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [left_strip] + 50 * [right_strip])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                # Update bottom edges
                left_width[l_label]['bottom'] += left_ns
                right_width[r_label]['bottom'] += right_ns
                # Add lines
                ax.fill_between(np.linspace(0, x_max, len(ys_d)),
                                ys_d,
                                ys_u,
                                alpha=0.5,
                                color=line_col.loc[l_label][0])
    ax.set_ylabel(left_label)
    ax.yaxis.set_label_coords(-0.05,.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax_r = plt.gca().twinx()
    ax_r.set_ylabel(right_label)
    ax_r.yaxis.set_label_coords(1.05,.5)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)
    ax_r.spines['bottom'].set_visible(False)
    ax_r.spines['left'].set_visible(False)
    ax_r.get_xaxis().set_ticks([])
    ax_r.get_yaxis().set_ticks([])
    if title is not None:
        ax.set_title(title)
    if output is not None:
        fig.savefig(output, dpi=300)
    plt.show()
    if close:
        plt.close()


def confusion_matrix(loom_file,
                     row_attr,
                     column_attr,
                     normalize_by=None,
                     diagonalize=True,
                     valid_attr=None,
                     xlabel=None,
                     ylabel=None,
                     title=None,
                     cmap='Reds',
                     cbar_label=None,
                     figsize=(8, 6),
                     output=None,
                     close=False):
    """
    Plots a confusion matrix between two attributes
        Typically used to compare two differnet cluster assignments
    
    Args:
        loom_file (str): Path to loom file
        row_attr (str): Attribute specifying rows of plot
        column_attr (str): Attribute specifying columns of plot
        normalize_by (str/int): Optional, normalize by rows or columns
            Rows can be indicated by rows or 0
            Columns can be indicated by columns or 1
        diagonalize (bool): Organize confusion matrix along diagonal
        valid_attr (str): Attribute specifying cells to include
        xlabel (str): Optional, label for x axis
        ylabel (str): Optional, label for y axis
        title (str): Optional, title of plot
        cmap (str): Matplotlib cmap option
        cbar_label (str): Optional, label for colorbar
        figsize (tuple): Size of output figure
        output (str): Optional, name of output file
        close (bool): Close figure after plotting
    """
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_attr,
                                          columns=True,
                                          as_bool=True,
                                          inverse=False)
    # Get data
    with loompy.connect(loom_file) as ds:
        row_vals = ds.ca[row_attr][valid_idx]
        col_vals = ds.ca[column_attr][valid_idx]
    confusion = ph.gen_confusion_mat(row_vals=row_vals,
                                     col_vals=col_vals,
                                     normalize_by=normalize_by,
                                     diagonalize=diagonalize)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(confusion.values,
                   cmap=cmap)
    ax.set_xticks(np.arange(confusion.shape[1]))
    ax.set_xticklabels(confusion.columns.values)
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    ax.set_yticks(np.arange(confusion.shape[0]))
    ax.set_yticklabels(confusion.index.values)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    cbar = ax.figure.colorbar(im,
                              ax=ax)
    if cbar_label is not None:
        cbar.ax.set_ylabel(cbar_label,
                           rotation=90)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output,
                    dpi=300)
    if close:
        plt.close()
    plt.show()


def boxplot_feature(loom_file,
                    category_attr,
                    feat_id,
                    layer,
                    feat_attr='Accession',
                    scale_attr=None,
                    color_attr=None,
                    valid_attr=None,
                    highlight=None,
                    x_label=None,
                    y_label=None,
                    title=None,
                    legend=False,
                    output=None,
                    figsize=(8, 6),
                    close=False):
    """
    Makes a boxplot of a feature's counts

    Args:
        loom_file (str): Path to loom file
        category_attr (str): Name of column attribute for categories
        feat_id (str): Name of column attribute for values
        layer (str): Name of layer containing count data
        feat_attr (str): Row attribute containing feat_id
        scale_attr (str): Optional, attribute specifying scale for values
            Useful for methylation data
        color_attr (str): Optional, column attribute with color values
        valid_attr (str): Optional, column attribute specifying cells to include
        highlight (str/list): Optional, categories to plot
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        legend (bool): Includes legend with plot
        output (str): Optional, saves figure to this file path
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes matplotlib figure

    """
    # Get categorical dataframe
    df_plot = ph.prep_feature_dist(loom_file=loom_file,
                                   category_attr=category_attr,
                                   feat_id=feat_id,
                                   layer=layer,
                                   feat_attr=feat_attr,
                                   scale_attr=scale_attr,
                                   color_attr=color_attr,
                                   valid_attr=valid_attr,
                                   highlight=highlight)
    if color_attr is None:
        color_attr = 'color'
    # Plot data
    ph.plot_boxviolin(df_plot=df_plot,
                      category_label=category_attr,
                      value_label=layer,
                      color_label=color_attr,
                      plot_type='box',
                      cat_order=highlight,
                      title=title,
                      x_label=x_label,
                      y_label=y_label,
                      legend=legend,
                      output=output,
                      figsize=figsize,
                      close=close)


def violinplot_feature(loom_file,
                       category_attr,
                       feat_id,
                       layer,
                       feat_attr='Accession',
                       scale_attr=None,
                       color_attr=None,
                       valid_attr=None,
                       highlight=None,
                       x_label=None,
                       y_label=None,
                       title=None,
                       legend=False,
                       output=None,
                       figsize=(8, 6),
                       close=False):
    """
    Makes a violin plot of a feature's counts

    Args:
        loom_file (str): Path to loom file
        category_attr (str): Name of column attribute for categories
        feat_id (str): Name of column attribute for values
        layer (str): Name of layer containing count data
        feat_attr (str): Row attribute containing feat_id
        scale_attr (str): Optional, attribute specifying scale for values
            Useful for methylation data
        color_attr (str): Optional, column attribute with color values
        valid_attr (str): Optional, column attribute specifying cells to include
        highlight (str/list): Optional, categories to plot
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        legend (bool): Includes legend with plot
        output (str): Optional, saves figure to this file path
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes matplotlib figure

    """
    # Get categorical dataframe
    df_plot = ph.prep_feature_dist(loom_file=loom_file,
                                   category_attr=category_attr,
                                   feat_id=feat_id,
                                   layer=layer,
                                   feat_attr=feat_attr,
                                   scale_attr=scale_attr,
                                   color_attr=color_attr,
                                   valid_attr=valid_attr,
                                   highlight=highlight)
    if color_attr is None:
        color_attr = 'color'
    # Plot data
    ph.plot_boxviolin(df_plot=df_plot,
                      category_label=category_attr,
                      value_label=layer,
                      color_label=color_attr,
                      plot_type='violin',
                      cat_order=highlight,
                      title=title,
                      x_label=x_label,
                      y_label=y_label,
                      legend=legend,
                      output=output,
                      figsize=figsize,
                      close=close)


def boxplot_attr(loom_file,
                 category_attr,
                 value_attr,
                 color_attr=None,
                 valid_attr=None,
                 highlight=None,
                 x_label=None,
                 y_label=None,
                 title=None,
                 legend=False,
                 output=None,
                 figsize=(8, 6),
                 close=False):
    """
    Makes a boxplot of a column attribute

    Args:
        loom_file (str): Path to loom file
        category_attr (str): Name of column attribute for categories
        value_attr (str): Name of column attribute for values
        color_attr (str): Optional, column attribute with color values
        valid_attr (str): Optional, column attribute specifying cells to include
        highlight (str/list): Optional, categories to plot
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        legend (bool): Includes legend with plot
        output (str): Optional, saves figure to this file path
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes matplotlib figure

    """
    # Get categorical dataframe
    df_plot = ph.prep_categorical_dist(loom_file=loom_file,
                                       category_attr=category_attr,
                                       value_attr=value_attr,
                                       color_attr=color_attr,
                                       valid_attr=valid_attr,
                                       highlight=highlight)
    if color_attr is None:
        color_attr = 'color'
    # Plot data
    ph.plot_boxviolin(df_plot=df_plot,
                      category_label=category_attr,
                      value_label=value_attr,
                      color_label=color_attr,
                      plot_type='box',
                      cat_order=highlight,
                      title=title,
                      x_label=x_label,
                      y_label=y_label,
                      legend=legend,
                      output=output,
                      figsize=figsize,
                      close=close)


def violinplot_attr(loom_file,
                    category_attr,
                    value_attr,
                    color_attr=None,
                    valid_attr=None,
                    highlight=None,
                    x_label=None,
                    y_label=None,
                    title=None,
                    legend=False,
                    output=None,
                    figsize=(8, 6),
                    close=False):
    """
    Makes a violin plot of a column attribute

    Args:
        loom_file (str): Path to loom file
        category_attr (str): Name of column attribute for categories
        value_attr (str): Name of column attribute for values
        color_attr (str): Optional, column attribute with color values
        valid_attr (str): Optional, column attribute specifying cells to include
        highlight (str/list): Optional, categories to plot
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        legend (bool): Includes legend with plot
        output (str): Optional, saves figure to this file path
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes matplotlib figure

    """
    # Get categorical dataframe
    df_plot = ph.prep_categorical_dist(loom_file=loom_file,
                                       category_attr=category_attr,
                                       value_attr=value_attr,
                                       color_attr=color_attr,
                                       valid_attr=valid_attr,
                                       highlight=highlight)
    if color_attr is None:
        color_attr = 'color'
    # Plot data
    ph.plot_boxviolin(df_plot=df_plot,
                      category_label=category_attr,
                      value_label=value_attr,
                      color_label=color_attr,
                      plot_type='violin',
                      cat_order=highlight,
                      title=title,
                      x_label=x_label,
                      y_label=y_label,
                      legend=legend,
                      output=output,
                      figsize=figsize,
                      close=close)


def barplot_attr(loom_file,
                 category_attr,
                 value_attr,
                 color_attr=None,
                 valid_attr=None,
                 highlight=None,
                 legend=False,
                 output=None,
                 x_label=None,
                 y_label=None,
                 title=None,
                 figsize=(8, 6),
                 close=False):
    """
    Plots barplot of attribute data

    Args:
        loom_file (str): Path to loom file
        category_attr (str): Attribute containing values for x-axis
        value_attr (str): Attribute containing values for bar plot
        color_attr (str): Optional, attribute specifying per cell colors
        valid_attr (str): Optional, attribute specifying cells to include
        highlight (str/list): Optional, plot only specified category_attr values
        legend (bool): Includes legend with plot
        output (str): Optional, saves plot to a file
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes figure

    """
    # Get indices
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    # Set-up dataframe
    with loompy.connect(loom_file, mode='r') as ds:
        df_plot = pd.DataFrame({category_attr: ds.ca[category_attr][col_idx],
                                value_attr: ds.ca[value_attr][col_idx]})
        if color_attr is None:
            df_plot = ph.get_category_colors(df_plot=df_plot,
                                            category_label=value_attr,
                                            color_label='color')
        else:
            df_plot['color'] = ds.ca[color_attr][col_idx]
    # Handle highlighting
    df_plot, is_high = ph.process_highlight(df_plot=df_plot,
                                              highlight_attr=category_attr,
                                              highlight_values=highlight)
    if is_high:
        df_plot = df_plot.loc[df_plot.index[df_plot[category_attr].isin(highlight)]]
    # Prep for bar plot
    bar_info = df_plot.groupby(category_attr)[
        value_attr].value_counts().unstack().fillna(0)
    bar_info = bar_info.div(bar_info.sum(1), axis=0)
    bar_info = bar_info.loc[general_utils.nat_sort(bar_info.index.values)]
    color_df = df_plot[[value_attr, 'color']].drop_duplicates(keep='first')
    color_df = color_df.set_index([value_attr], drop=True)
    # Make plot
    fig = plt.figure(figsize=figsize)
    bottom_value = 0
    objs = []
    for value in color_df.index.values:
        obj = plt.bar(bar_info.index.values,
                      bar_info[value],
                      bottom=bottom_value,
                      color=color_df.loc[value][0])
        objs.append(obj)
        bottom_value += bar_info[value]
    # Modify figure
    if x_label is None:
        x_label = category_attr
    plt.xlabel(x_label)
    if y_label is None:
        y_label = 'Fraction'
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.xticks(rotation=45)
    if legend:
        l_h = plt.legend(objs,
                         color_df.index.values,
                         bbox_to_anchor=(1.04, 1),
                         loc='upper left')
    plt.show()
    if output:
        if legend:
            fig.savefig(output,
                        bbox_extra_artists=(l_h,),
                        bbox_inches='tight')
        else:
            fig.savefig(output,
                        dpi=300)
        plot_log.info('Saved figure to {}'.format(output))
    if close:
        plt.close()
