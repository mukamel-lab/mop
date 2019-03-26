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
import matplotlib.lines as mlines
import seaborn as sns
from . import general_utils
from . import loom_utils

# Start log
ph_log = logging.getLogger(__name__)


def get_random_colors(n):
    """
    Generates n random colors

    Args:
        n (int): Number of colors

    Returns:
        colors (list): List of colors

    Written by Chris Keown
    """
    # '#FFFF00', '#FF34FF','#FFDBE5', '#FEFFE6',
    colors = ['#1CE6FF', '#FF4A46', '#008941', '#006FA6', '#A30059',
              '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF',
              '#997D87',
              '#5A0007', '#809693', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53',
              '#FF2F80',
              '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9',
              '#B903AA',
              '#D16100', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8',
              '#013349',
              '#00846F', '#372101', '#FFB500', '#A079BF', '#CC0744', '#C0B9B2',
              '#001E09',
              '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68',
              '#7A87A1', '#788D66',
              '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648',
              '#0086ED', '#886F4C',
              '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9',
              '#FF913F', '#938A81',
              '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757',
              '#C8A1A1', '#1E6E00',
              '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600',
              '#D790FF', '#9B9700',
              '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0',
              '#3A2465', '#922329',
              '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804',
              '#324E72', '#6A3A4C',
              '#83AB58', '#001C1E', '#D1F7CE', '#004B28', '#C8D0F6', '#A3A489',
              '#806C66', '#222800',
              '#BF5650', '#E83000', '#66796D', '#DA007C', '#FF1A59', '#8ADBB4',
              '#1E0200', '#5B4E51',
              '#C895C5', '#320033', '#FF6832', '#66E1D3', '#CFCDAC', '#D0AC94',
              '#7ED379', '#012C58',
              '#7A7BFF', '#D68E01', '#353339', '#78AFA1', '#FEB2C6', '#75797C',
              '#837393', '#943A4D',
              '#B5F4FF', '#D2DCD5', '#9556BD', '#6A714A', '#001325', '#02525F',
              '#0AA3F7', '#E98176',
              '#DBD5DD', '#5EBCD1', '#3D4F44', '#7E6405', '#02684E', '#962B75',
              '#8D8546', '#9695C5',
              '#E773CE', '#D86A78', '#3E89BE', '#CA834E', '#518A87', '#5B113C',
              '#55813B', '#E704C4',
              '#00005F', '#A97399', '#4B8160', '#59738A', '#FF5DA7', '#F7C9BF',
              '#643127', '#513A01',
              '#6B94AA', '#51A058', '#A45B02', '#1D1702', '#E20027', '#E7AB63',
              '#4C6001', '#9C6966',
              '#64547B', '#97979E', '#006A66', '#391406', '#F4D749', '#0045D2',
              '#006C31', '#DDB6D0',
              '#7C6571', '#9FB2A4', '#00D891', '#15A08A', '#BC65E9', '#FFFFFE',
              '#C6DC99', '#203B3C',
              '#671190', '#6B3A64', '#F5E1FF', '#FFA0F2', '#CCAA35', '#374527',
              '#8BB400', '#797868',
              '#C6005A', '#3B000A', '#C86240', '#29607C', '#402334', '#7D5A44',
              '#CCB87C', '#B88183',
              '#AA5199', '#B5D6C3', '#A38469', '#9F94F0', '#A74571', '#B894A6',
              '#71BB8C', '#00B433',
              '#789EC9', '#6D80BA', '#953F00', '#5EFF03', '#E4FFFC', '#1BE177',
              '#BCB1E5', '#76912F',
              '#003109', '#0060CD', '#D20096', '#895563', '#29201D', '#5B3213',
              '#A76F42', '#89412E',
              '#1A3A2A', '#494B5A', '#A88C85', '#F4ABAA', '#A3F3AB', '#00C6C8',
              '#EA8B66', '#958A9F',
              '#BDC9D2', '#9FA064', '#BE4700', '#658188', '#83A485', '#453C23',
              '#47675D', '#3A3F00',
              '#061203', '#DFFB71', '#868E7E', '#98D058', '#6C8F7D', '#D7BFC2',
              '#3C3E6E', '#D83D66',
              '#2F5D9B', '#6C5E46', '#D25B88', '#5B656C', '#00B57F', '#545C46',
              '#866097', '#365D25',
              '#252F99', '#00CCFF', '#674E60', '#FC009C', '#92896B', '#1E2324',
              '#DEC9B2', '#9D4948',
              '#85ABB4', '#342142', '#D09685', '#A4ACAC', '#00FFFF', '#AE9C86',
              '#742A33', '#0E72C5',
              '#AFD8EC', '#C064B9', '#91028C', '#FEEDBF', '#FFB789', '#9CB8E4',
              '#AFFFD1', '#2A364C',
              '#4F4A43', '#647095', '#34BBFF', '#807781', '#920003', '#B3A5A7',
              '#018615', '#F1FFC8',
              '#976F5C', '#FF3BC1', '#FF5F6B', '#077D84', '#F56D93', '#5771DA',
              '#4E1E2A', '#830055',
              '#02D346', '#BE452D', '#00905E', '#BE0028', '#6E96E3', '#007699',
              '#FEC96D', '#9C6A7D',
              '#3FA1B8', '#893DE3', '#79B4D6', '#7FD4D9', '#6751BB', '#B28D2D',
              '#E27A05', '#DD9CB8',
              '#AABC7A', '#980034', '#561A02', '#8F7F00', '#635000', '#CD7DAE',
              '#8A5E2D', '#FFB3E1',
              '#6B6466', '#C6D300', '#0100E2', '#88EC69', '#8FCCBE', '#21001C',
              '#511F4D', '#E3F6E3',
              '#FF8EB1', '#6B4F29', '#A37F46', '#6A5950', '#1F2A1A', '#04784D',
              '#101835', '#E6E0D0',
              '#FF74FE', '#00A45F', '#8F5DF8', '#4B0059', '#412F23', '#D8939E',
              '#DB9D72', '#604143',
              '#B5BACE', '#989EB7', '#D2C4DB', '#A587AF', '#77D796', '#7F8C94',
              '#FF9B03', '#555196',
              '#31DDAE', '#74B671', '#802647', '#2A373F', '#014A68', '#696628',
              '#4C7B6D', '#002C27',
              '#7A4522', '#3B5859', '#E5D381', '#FFF3FF', '#679FA0', '#261300',
              '#2C5742', '#9131AF',
              '#AF5D88', '#C7706A', '#61AB1F', '#8CF2D4', '#C5D9B8', '#9FFFFB',
              '#BF45CC', '#493941',
              '#863B60', '#B90076', '#003177', '#C582D2', '#C1B394', '#602B70',
              '#887868', '#BABFB0',
              '#030012', '#D1ACFE', '#7FDEFE', '#4B5C71', '#A3A097', '#E66D53',
              '#637B5D', '#92BEA5',
              '#00F8B3', '#BEDDFF', '#3DB5A7', '#DD3248', '#B6E4DE', '#427745',
              '#598C5A', '#B94C59',
              '#8181D5', '#94888B', '#FED6BD', '#536D31', '#6EFF92', '#E4E8FF',
              '#20E200', '#FFD0F2',
              '#4C83A1', '#BD7322', '#915C4E', '#8C4787', '#025117', '#A2AA45',
              '#2D1B21', '#A9DDB0',
              '#FF4F78', '#528500', '#009A2E', '#17FCE4', '#71555A', '#525D82',
              '#00195A', '#967874',
              '#555558', '#0B212C', '#1E202B', '#EFBFC4', '#6F9755', '#6F7586',
              '#501D1D', '#372D00',
              '#741D16', '#5EB393', '#B5B400', '#DD4A38', '#363DFF', '#AD6552',
              '#6635AF', '#836BBA',
              '#98AA7F', '#464836', '#322C3E', '#7CB9BA', '#5B6965', '#707D3D',
              '#7A001D', '#6E4636',
              '#443A38', '#AE81FF', '#489079', '#897334', '#009087', '#DA713C',
              '#361618', '#FF6F01',
              '#006679', '#370E77', '#4B3A83', '#C9E2E6', '#C44170', '#FF4526',
              '#73BE54', '#C4DF72',
              '#ADFF60', '#00447D', '#DCCEC9', '#BD9479', '#656E5B', '#EC5200',
              '#FF6EC2', '#7A617E',
              '#DDAEA2', '#77837F', '#A53327', '#608EFF', '#B599D7', '#A50149',
              '#4E0025', '#C9B1A9',
              '#03919A', '#1B2A25', '#E500F1', '#982E0B', '#B67180', '#E05859',
              '#006039', '#578F9B',
              '#305230', '#CE934C', '#B3C2BE', '#C0BAC0', '#B506D3', '#170C10',
              '#4C534F', '#224451',
              '#3E4141', '#78726D', '#B6602B', '#200441', '#DDB588', '#497200',
              '#C5AAB6', '#033C61',
              '#71B2F5', '#A9E088', '#4979B0', '#A2C3DF', '#784149', '#2D2B17',
              '#3E0E2F', '#57344C',
              '#0091BE', '#E451D1', '#4B4B6A', '#5C011A', '#7C8060', '#FF9491',
              '#4C325D', '#005C8B',
              '#E5FDA4', '#68D1B6', '#032641', '#140023', '#8683A9', '#CFFF00',
              '#A72C3E', '#34475A',
              '#B1BB9A', '#B4A04F', '#8D918E', '#A168A6', '#813D3A', '#425218',
              '#DA8386', '#776133',
              '#563930', '#8498AE', '#90C1D3', '#B5666B', '#9B585E', '#856465',
              '#AD7C90', '#E2BC00',
              '#E3AAE0', '#B2C2FE', '#FD0039', '#009B75', '#FFF46D', '#E87EAC',
              '#DFE3E6', '#848590',
              '#AA9297', '#83A193', '#577977', '#3E7158', '#C64289', '#EA0072',
              '#C4A8CB', '#55C899',
              '#E78FCF', '#004547', '#F6E2E3', '#966716', '#378FDB', '#435E6A',
              '#DA0004', '#1B000F',
              '#5B9C8F', '#6E2B52', '#011115', '#E3E8C4', '#AE3B85', '#EA1CA9',
              '#FF9E6B', '#457D8B',
              '#92678B', '#00CDBB', '#9CCC04', '#002E38', '#96C57F', '#CFF6B4',
              '#492818', '#766E52',
              '#20370E', '#E3D19F', '#2E3C30', '#B2EACE', '#F3BDA4', '#A24E3D',
              '#976FD9', '#8C9FA8',
              '#7C2B73', '#4E5F37', '#5D5462', '#90956F', '#6AA776', '#DBCBF6',
              '#DA71FF', '#987C95',
              '#52323C', '#BB3C42', '#584D39', '#4FC15F', '#A2B9C1', '#79DB21',
              '#1D5958', '#BD744E',
              '#160B00', '#20221A', '#6B8295', '#00E0E4', '#102401', '#1B782A',
              '#DAA9B5', '#B0415D',
              '#859253', '#97A094', '#06E3C4', '#47688C', '#7C6755', '#075C00',
              '#7560D5', '#7D9F00',
              '#C36D96', '#4D913E', '#5F4276', '#FCE4C8', '#303052', '#4F381B',
              '#E5A532', '#706690',
              '#AA9A92', '#237363', '#73013E', '#FF9079', '#A79A74', '#029BDB',
              '#FF0169', '#C7D2E7',
              '#CA8869', '#80FFCD', '#BB1F69', '#90B0AB', '#7D74A9', '#FCC7DB',
              '#99375B', '#00AB4D',
              '#ABAED1', '#BE9D91', '#E6E5A7', '#332C22', '#DD587B', '#F5FFF7',
              '#5D3033', '#6D3800',
              '#FF0020', '#B57BB3', '#D7FFE6', '#C535A9', '#260009', '#6A8781',
              '#A8ABB4', '#D45262',
              '#794B61', '#4621B2', '#8DA4DB', '#C7C890', '#6FE9AD', '#A243A7',
              '#B2B081', '#181B00',
              '#286154', '#4CA43B', '#6A9573', '#A8441D', '#5C727B', '#738671',
              '#D0CFCB', '#897B77',
              '#1F3F22', '#4145A7', '#DA9894', '#A1757A', '#63243C', '#ADAAFF',
              '#00CDE2', '#DDBC62',
              '#698EB1', '#208462', '#00B7E0', '#614A44', '#9BBB57', '#7A5C54',
              '#857A50', '#766B7E',
              '#014833', '#FF8347', '#7A8EBA', '#274740', '#946444', '#EBD8E6',
              '#646241', '#373917',
              '#6AD450', '#81817B', '#D499E3', '#979440', '#011A12', '#526554',
              '#B5885C', '#A499A5',
              '#03AD89', '#B3008B', '#E3C4B5', '#96531F', '#867175', '#74569E',
              '#617D9F', '#E70452',
              '#067EAF', '#A697B6', '#B787A8', '#9CFF93', '#311D19', '#3A9459',
              '#6E746E', '#B0C5AE',
              '#84EDF7', '#ED3488', '#754C78', '#384644', '#C7847B', '#00B6C5',
              '#7FA670', '#C1AF9E',
              '#2A7FFF', '#72A58C', '#FFC07F', '#9DEBDD', '#D97C8E', '#7E7C93',
              '#62E674', '#B5639E',
              '#FFA861', '#C2A580', '#8D9C83', '#B70546', '#372B2E', '#0098FF',
              '#985975', '#20204C',
              '#FF6C60', '#445083', '#8502AA', '#72361F', '#9676A3', '#484449',
              '#CED6C2', '#3B164A',
              '#CCA763', '#2C7F77', '#02227B', '#A37E6F', '#CDE6DC', '#CDFFFB',
              '#BE811A', '#F77183',
              '#EDE6E2', '#CDC6B4', '#FFE09E', '#3A7271', '#FF7B59', '#4E4E01',
              '#4AC684', '#8BC891',
              '#BC8A96', '#CF6353', '#DCDE5C', '#5EAADD', '#F6A0AD', '#E269AA',
              '#A3DAE4', '#436E83',
              '#002E17', '#ECFBFF', '#A1C2B6', '#50003F', '#71695B', '#67C4BB',
              '#536EFF', '#5D5A48',
              '#890039', '#969381', '#371521', '#5E4665', '#AA62C3', '#8D6F81',
              '#2C6135', '#410601',
              '#564620', '#E69034', '#6DA6BD', '#E58E56', '#E3A68B', '#48B176',
              '#D27D67', '#B5B268',
              '#7F8427', '#FF84E6', '#435740', '#EAE408', '#F4F5FF', '#325800',
              '#4B6BA5', '#ADCEFF',
              '#9B8ACC', '#885138', '#5875C1', '#7E7311', '#FEA5CA', '#9F8B5B',
              '#A55B54', '#89006A',
              '#AF756F', '#2A2000', '#7499A1', '#FFB550', '#00011E', '#D1511C',
              '#688151', '#BC908A',
              '#78C8EB', '#8502FF', '#483D30', '#C42221', '#5EA7FF', '#785715',
              '#0CEA91', '#FFFAED',
              '#B3AF9D', '#3E3D52', '#5A9BC2', '#9C2F90', '#8D5700', '#ADD79C',
              '#00768B', '#337D00',
              '#C59700', '#3156DC', '#944575', '#ECFFDC', '#D24CB2', '#97703C',
              '#4C257F', '#9E0366',
              '#88FFEC', '#B56481', '#396D2B', '#56735F', '#988376', '#9BB195',
              '#A9795C', '#E4C5D3',
              '#9F4F67', '#1E2B39', '#664327', '#AFCE78', '#322EDF', '#86B487',
              '#C23000', '#ABE86B',
              '#96656D', '#250E35', '#A60019', '#0080CF', '#CAEFFF', '#323F61',
              '#A449DC', '#6A9D3B',
              '#FF5AE4', '#636A01', '#D16CDA', '#736060', '#FFBAAD', '#D369B4',
              '#FFDED6', '#6C6D74',
              '#927D5E', '#845D70', '#5B62C1', '#2F4A36', '#E45F35', '#FF3B53',
              '#AC84DD', '#762988',
              '#70EC98', '#408543', '#2C3533', '#2E182D', '#323925', '#19181B',
              '#2F2E2C', '#023C32',
              '#9B9EE2', '#58AFAD', '#5C424D', '#7AC5A6', '#685D75', '#B9BCBD',
              '#834357', '#1A7B42',
              '#2E57AA', '#E55199', '#316E47', '#CD00C5', '#6A004D', '#7FBBEC',
              '#F35691', '#D7C54A',
              '#62ACB7', '#CBA1BC', '#A28A9A', '#6C3F3B', '#FFE47D', '#DCBAE3',
              '#5F816D', '#3A404A',
              '#7DBF32', '#E6ECDC', '#852C19', '#285366', '#B8CB9C', '#0E0D00',
              '#4B5D56', '#6B543F',
              '#E27172', '#0568EC', '#2EB500', '#D21656', '#EFAFFF', '#682021',
              '#2D2011', '#DA4CFF',
              '#70968E', '#FF7B7D', '#4A1930', '#E8C282', '#E7DBBC', '#A68486',
              '#1F263C', '#36574E',
              '#52CE79', '#ADAAA9', '#8A9F45', '#6542D2', '#00FB8C', '#5D697B',
              '#CCD27F', '#94A5A1',
              '#790229', '#E383E6', '#7EA4C1', '#4E4452', '#4B2C00', '#620B70',
              '#314C1E', '#874AA6',
              '#E30091', '#66460A', '#EB9A8B', '#EAC3A3', '#98EAB3', '#AB9180',
              '#B8552F', '#1A2B2F',
              '#94DDC5', '#9D8C76', '#9C8333', '#94A9C9', '#392935', '#8C675E',
              '#CCE93A', '#917100',
              '#01400B', '#449896', '#1CA370', '#E08DA7', '#8B4A4E', '#667776',
              '#4692AD', '#67BDA8',
              '#69255C', '#D3BFFF', '#4A5132', '#7E9285', '#77733C', '#E7A0CC',
              '#51A288', '#2C656A',
              '#4D5C5E', '#C9403A', '#DDD7F3', '#005844', '#B4A200', '#488F69',
              '#858182', '#D4E9B9',
              '#3D7397', '#CAE8CE', '#D60034', '#AA6746', '#9E5585', '#BA6200']
    return colors[:n]


def find_limits(df_plot,
                axis):
    """
    Generates a tuple of limits for a given axis
    
    Args:
        df_plot (dataframe): Contains x/y-coordiantes for scatter plot
        axis (str): Column in df_plot containing axis coordinates
    
    Returns
        lims (tuple): Limits along given axis
    
    Adapted from code by Fangming Xie
    """
    lims = [np.nanpercentile(df_plot[axis].values, 0.1),
            np.nanpercentile(df_plot[axis].values, 99.9)]
    lims[0] = lims[0] - 0.1 * (lims[1] - lims[0])
    lims[1] = lims[1] + 0.1 * (lims[1] - lims[0])
    return tuple(lims)


def get_category_colors(df_plot,
                        category_label,
                        color_label='color'):
    """
    Generates unique colors for each member of a category

    Args:
        df_plot (dataframe): Contains categories
        category_label (str): Column in df_plot containing categories
        color_label (str): Output column containing color values

    Returns
        df_plot (dataframe): Same as input df_plot with added color column
    """
    if color_label is None:
        color_label = 'color'
    unq_cat = general_utils.nat_sort(df_plot[category_label].unique())
    col_opts = pd.DataFrame({category_label: unq_cat})
    col_opts[color_label] = get_random_colors(col_opts.shape[0])
    df_plot = pd.merge(df_plot,
                       col_opts,
                       left_on=category_label,
                       right_on=category_label)
    return df_plot


def plot_scatter(df_plot,
                 x_axis='x_val',
                 y_axis='y_val',
                 col_opt=None,
                 s=2,
                 legend=False,
                 legend_labels=None,
                 output=None,
                 xlim='auto',
                 ylim='auto',
                 highlight=False,
                 x_label=None,
                 y_label=None,
                 title=None,
                 figsize=(8, 6),
                 cbar_label=None,
                 close=False,
                 fig=None,
                 ax=None,
                 **kwargs):
    """
    Plots scatter of cells in which each cluster is marked with a unique color
    
    Args:
        df_plot (dataframe): Contains x/y-coordinates for scatter plot
        x_axis (str): Column in df_plot containing x-axis coordinates
        y_axis (str): Column in df_plot containing y-axis coordinates
        col_opt (str): Optional, column in df_plot containing color values
            If not provided, default is black
        s (int): Size of points on scatter plot
        legend (bool): Includes legend with plot
        legend_labels (str): Optional, column containing legend labels
        output (str): Optional, saves plot to a file
        xlim (tuple/str): Limits for x-axis
            auto to set based on data
        ylim (tuple/str): Limits for y-axis
            auto to set based on data
        highlight (bool): If true, highlights certain cells
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        title (str): Optional, title for plot
        figsize (tuple): Size of scatter plot figure
        cbar_label (str): Optional, if present adds colorbar and labels
        close (bool): If true, closes matplotlib figure
        fig (object): Optional, plot figure if already generated
        ax (object): Optional, axis for plots if already generated
        **kwargs: keyword arguments for matplotlib's scatter
    
    Adpated from code by Fangming Xie
    """
    if col_opt is None:
        col_opt = 'color'
        df_plot[col_opt] = 'k'
    # Make plot
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is None and fig is not None:
        raise ValueError('Both ax and fig must be provided')
    elif ax is not None and fig is None:
        raise ValueError('Both ax and fig must be provided')
    if highlight:
        ax.scatter(df_plot[x_axis].values,
                   df_plot[y_axis].values,
                   s=s,
                   c='lightgray',
                   alpha=0.1,
                   **kwargs)
        use_idx = np.where(df_plot[col_opt] != 'Null')[0]
        im = ax.scatter(df_plot[x_axis].iloc[use_idx].values,
                        df_plot[y_axis].iloc[use_idx].values,
                        s=s,
                        c=df_plot[col_opt].iloc[use_idx].values,
                        **kwargs)
    else:
        im = ax.scatter(df_plot[x_axis].values,
                        df_plot[y_axis].values,
                        s=s,
                        c=df_plot[col_opt].values,
                        **kwargs)
    # Modify figure
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if xlim is None and ylim is None:
        pass
    elif xlim == 'auto' or ylim == 'auto':
        ax.set_aspect('auto')
        xlim = find_limits(df_plot=df_plot,
                           axis=x_axis)
        ax.set_xlim(xlim)
        ylim = find_limits(df_plot=df_plot,
                           axis=y_axis)
        ax.set_ylim(ylim)
    elif xlim == 'equal' or ylim == 'equal':
        ax.set_aspect('equal')
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    # Add colorbar
    if cbar_label is not None:
        cbar = plt.colorbar(im,
                            ax=ax)
        cbar.set_label(cbar_label,
                       rotation=270,
                       labelpad=10)
    # Add legend
    if legend and legend_labels is not None:
        if highlight:
            df_plot = df_plot.iloc[use_idx]
        df_legend = df_plot[[legend_labels, col_opt]]
        df_legend = df_legend.drop_duplicates(keep='first')
        df_legend = df_legend.set_index(keys=legend_labels,
                                        drop=True)
        df_legend = df_legend.loc[
            general_utils.nat_sort(df_legend.index.values)]
        handles = []
        for row in df_legend.itertuples(index=True, name='legend'):
            tmp_hand = mlines.Line2D([],
                                     [],
                                     color=getattr(row, col_opt),
                                     marker='.',
                                     linestyle='',
                                     label=getattr(row, 'Index'))
            handles.append(tmp_hand)
        l_h = plt.legend(handles=handles,
                         bbox_to_anchor=(1.04, 1),
                         loc='upper left')
    else:
        l_h = None
    # Save figure
    if output:
        if l_h is None:
            fig.savefig(output,
                        dpi=300)
        else:
            fig.savefig(output,
                        bbox_extra_artists=(l_h,),
                        bbox_inches='tight')
        ph_log.info('Saved figure to {}'.format(output))
    if close:
        plt.close()


def set_value_by_percentile(count,
                            low_p,
                            high_p):
    """
    Sets a count below or above a percentile to the given percentile
    
    Args:
        count (float): Count value
        low_p (float): Lowest percentile value
        high_p (float): Highest percentile value
    
    Returns:
        normalized (float): Count normalized by percentile
        
    Adapted from code written by Fangming Xie
    """
    if count < low_p:
        return low_p
    elif count > high_p:
        return high_p
    else:
        return count


def percentile_norm(counts,
                    low_p,
                    high_p):
    """
    Sets the lowest/highest values for counts to be their percentiles
    
    Args:
        counts (1D array): Array of count values
        low_p (int): Lowest percentile value allowed (0-100)
        high_p (int): Highest percentile value allowed (0-100)
    
    Returns:
        normalized (1D array): Array of normalized count values
    
    Adapted from code by Fangming Xie
    """
    low_p = np.nanpercentile(counts, low_p)
    high_p = np.nanpercentile(counts, high_p)
    normalized = [set_value_by_percentile(i, low_p, high_p) for i in
                  list(counts)]
    normalized = np.array(normalized)
    return normalized


def gen_confusion_mat(row_vals,
                      col_vals,
                      normalize_by=None,
                      diagonalize=False):
    """
    Generates a diagonalized confusion matrix
    
    Args:
        row_vals (ndarray): Values for rows in confusion matrix
        col_vals (ndarray): Values for columns in confusion matrix
        normalize_by (str): Optional, normalize by rows or columns
        diagonalize (bool): Optional, set values along a diagonal
    
    Returns:
        mat (ndarray): Confusion matrix with diagonalized data
        
    Written by Fangming Xie with modifications by Wayne Doyle
    """
    # Cross-tabulate data
    mat = pd.crosstab(row_vals, col_vals)
    # Normalize
    if normalize_by is None:
        pass
    elif 'row' in str(normalize_by).lower() or normalize_by == 0:
        mat = mat.divide(mat.sum(axis=1), axis=0)
    elif 'col' in str(normalize_by).lower() or normalize_by == 1:
        mat = mat.divide(mat.sum(axis=0), axis=1)
    # Diagonalize matrix
    if diagonalize:
        transposed = False
        if mat.shape[0] > mat.shape[1]:
            mat = mat.T.copy()
            transposed = True
        orig_rows = mat.index.values
        orig_cols = mat.columns.values
        diag_mat = mat.values.copy()
        new_rows = orig_rows.copy()
        new_cols = orig_cols.copy()
        # Put largest values in corner
        for idx in range(min(diag_mat.shape)):
            tmp_mat = diag_mat[idx:, idx:]
            i, j = np.unravel_index(tmp_mat.argmax(),
                                    tmp_mat.shape)
            dm = idx + 1
            # update_rows
            new_vals = diag_mat[idx, :].copy()
            diag_mat[idx, :] = diag_mat[idx + i, :].copy()
            diag_mat[idx + i, :] = new_vals
            new_vals = new_rows[idx]
            new_rows[idx] = new_rows[idx + i]
            new_rows[idx + i] = new_vals
            # swap col idx, idx+j
            new_vals = diag_mat[:, idx].copy()
            diag_mat[:, idx] = diag_mat[:, idx + j].copy()
            diag_mat[:, idx + j] = new_vals
            new_vals = new_cols[idx]
            new_cols[idx] = new_cols[idx + j]
            new_cols[idx + j] = new_vals
        col_num = diag_mat.shape[1]
        if dm == col_num:
            pass
        elif dm < col_num:  # free columns
            col_dict = {}
            sorted_col_idx = np.arange(dm)
            free_col_idx = np.arange(dm, col_num)
            linked_rowcol_idx = diag_mat[:, dm:].argmax(axis=0)
            for col in sorted_col_idx:
                col_dict[col] = [col]
            for col, key in zip(free_col_idx, linked_rowcol_idx):
                col_dict[key] = col_dict[key] + [col]
            new_col_order = np.hstack(
                [col_dict[key] for key in sorted(col_dict.keys())])
            diag_mat = diag_mat[:, new_col_order].copy()
            new_cols = new_cols[new_col_order]
        else:
            raise ValueError('out of bounds indexing')
        mat = pd.DataFrame(diag_mat,
                           index=new_rows,
                           columns=new_cols)
        if transposed:
            mat = mat.T
    return mat


def plot_boxviolin(df_plot,
                   category_label,
                   value_label,
                   color_label,
                   plot_type,
                   cat_order=None,
                   title=None,
                   x_label=None,
                   y_label=None,
                   legend=False,
                   output=None,
                   figsize=(8, 6),
                   close=False):
    """
    Plots box plot data

    Args:
        df_plot (dataframe): Contains category and value data
        category_label (str): df_plot column containing categories
        value_label (str): df_plot column containing values
        color_label (str): df_plot column containing colors
        plot_type (str): Type of seaborn distribution plot
            box
            violin
        x_label (str): Optional, label for x-axis
        y_label (str): Optional, label for y-axis
        cat_order (str,list): Order of categorical variables
        title (str): Optional, title for plot
        legend (bool): Includes legend with plot
        output (str): Optional, saves figure to this file path
        figsize (tuple): Size of scatter plot figure
        close (bool): If true, closes matplotlib figure
    """
    # Handle cat_order
    if cat_order is None:
        plot_order = general_utils.nat_sort(df_plot[category_label].unique())
    elif isinstance(cat_order,str):
        plot_order = [cat_order]
    elif isinstance(cat_order,list):
        plot_order = cat_order
    else:
        raise ValueError('cat_order must be  a list or string')
    # Make plot
    df_legend = df_plot[[category_label, color_label]]
    df_legend = df_legend.drop_duplicates(keep='first')
    df_legend = df_legend.set_index(category_label, drop=True)
    df_legend = df_legend.loc[general_utils.nat_sort(df_legend.index.values)]
    fig, ax = plt.subplots(figsize=figsize)
    if 'box' in plot_type.lower():
        sns.boxplot(x=category_label,
                    y=value_label,
                    hue=category_label,
                    dodge=False,
                    palette=df_legend[color_label].to_dict(),
                    order=plot_order,
                    data=df_plot,
                    ax=ax)
    elif 'violin' in plot_type.lower():
        sns.violinplot(x=category_label,
                       y=value_label,
                       hue=category_label,
                       dodge=False,
                       palette=df_legend[color_label].to_dict(),
                       order=plot_order,
                       data=df_plot,
                       ax=ax)
    else:
        raise ValueError('Unsupported plot_type value, must be box or violin')
    plt.xticks(rotation=45)
    ax.get_legend().remove()
    # Edit plot
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if legend:
        handles = []
        for row in df_legend.itertuples(index=True, name='legend'):
            tmp_hand = mlines.Line2D([],
                                     [],
                                     color=getattr(row, color_label),
                                     marker='.',
                                     linestyle='',
                                     label=getattr(row, 'Index'))
            handles.append(tmp_hand)
        l_h = plt.legend(handles=handles,
                         bbox_to_anchor=(1.04, 1),
                         loc='upper left')
    else:
        l_h = None
    if output is not None:
        if l_h is None:
            fig.savefig(output,
                        dpi=300)
        else:
            fig.savefig(output,
                        bbox_extra_artists=(l_h,),
                        bbox_inches='tight')
        ph_log.info('Saved figure to {}'.format(output))
    if close:
        plt.close()
    plt.show()


def prep_feature_dist(loom_file,
                      category_attr,
                      feat_id,
                      layer,
                      feat_attr='Accession',
                      scale_attr=None,
                      color_attr=None,
                      valid_attr=None,
                      highlight=None):
    """
    Makes a dataframe for plotting feature count information

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

    Returns:
        df_plot (dataframe): Contains data for plotting
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
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
        df_plot = pd.DataFrame({category_attr: ds.ca[category_attr][col_idx],
                                layer: counts},
                               index=np.arange(col_idx.shape[0]))
        if color_attr is None:
            df_plot = get_category_colors(df_plot=df_plot,
                                          category_label=category_attr,
                                          color_label=color_attr)
        else:
            df_plot['color'] = ds.ca[color_attr][col_idx]
    if highlight is not None:
        if isinstance(highlight, str):
            highlight = [highlight]
        if isinstance(highlight, list) or isinstance(highlight, np.ndarray):
            pass
        else:
            raise ValueError('Unsupported type for highlight')
        hl_idx = pd.DataFrame(np.repeat([False], repeats=df_plot.shape[0]),
                              index=df_plot[category_attr],
                              columns=['idx'])
        hl_idx['idx'].loc[highlight] = True
        df_plot = df_plot.loc[hl_idx['idx'].values]
    return df_plot


def prep_categorical_dist(loom_file,
                          category_attr,
                          value_attr,
                          color_attr=None,
                          valid_attr=None,
                          highlight=None):
    """
    Makes a dataframe for plotting categorical data distributions

    Args:
        loom_file (str): Path to loom file
        category_attr (str): Name of column attribute for categories
        value_attr (str): Name of column attribute for values
        color_attr (str): Optional, column attribute with color values
        valid_attr (str): Optional, column attribute specifying cells to include
        highlight (str/list): Optional, categories to plot

    Returns:
        df_plot (dataframe): Contains categories and values for plotting
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        df_plot = pd.DataFrame({category_attr: ds.ca[category_attr][col_idx],
                                value_attr: ds.ca[value_attr][col_idx]},
                               index=np.arange(col_idx.shape[0]))
        if color_attr is None:
            df_plot = get_category_colors(df_plot=df_plot,
                                          category_label=category_attr,
                                          color_label='color')
        else:
            df_plot[color_attr] = ds.ca[color_attr][col_idx]
    if highlight is not None:
        if isinstance(highlight, str):
            highlight = [highlight]
        if isinstance(highlight, list) or isinstance(highlight, np.ndarray):
            pass
        else:
            raise ValueError('Unsupported type for highlight')
        hl_idx = pd.DataFrame(np.repeat([False], repeats=df_plot.shape[0]),
                              index=df_plot[category_attr],
                              columns=['idx'])
        hl_idx['idx'].loc[highlight] = True
        df_plot = df_plot.loc[hl_idx['idx'].values]
    return df_plot


def process_highlight(df_plot,
                      highlight_attr,
                      highlight_values):
    """
    Restricts df_plot to highlight_values

    Args:
        df_plot (df): Dataframe containing values for plotting
        highlight_attr (str): Column in df_plot containing highlight_values
        highlight_values (str/list): Values to restrict to

    Returns:
        df_plot (df): Possibly restricted dataframe
        highlight (bool): Reports if restricted or not
    """
    if highlight_values is not None:
        if isinstance(highlight_values, str):
            highlight_values = [highlight_values]
        elif isinstance(highlight_values, list) or isinstance(highlight_values,
                                                              np.ndarray):
            pass
        else:
            raise ValueError('Unsupported type for highlight_values')
        hl_idx = pd.DataFrame(np.ones(df_plot.shape[0], dtype=bool),
                              index=df_plot[highlight_attr].values,
                              columns=['idx'])
        hl_idx['idx'].loc[highlight_values] = False
        tmp = df_plot['color'].copy()
        tmp.loc[hl_idx['idx'].values] = 'Null'
        df_plot['color'] = tmp
        highlight = True
    else:
        highlight = False
    return df_plot, highlight
