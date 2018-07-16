#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_vis

Created on Fri Jan 19 13:53:57 2018

@author: Edward Rowland

This module contains code to create Bokeh visualisations in python from 
the traffic flow and GDP data. 


"""
import os
import sys
import math

import itertools as it
import pandas as pd
import numpy as np
import statsmodels.api as sm



#plotting
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,  Label, LinearAxis, Title, LinearColorMapper, ColorBar, BasicTicker
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import gridplot
from bokeh.palettes import Category10, Plasma256
from bokeh.transform import transform

#stats
from scipy.signal import correlate
from scipy.stats import linregress

#where our scripts are stored
file_dir = os.getcwd()
sys.path.append(file_dir) # assume other scripts are where we are


#%%
def multiline_plot(src,
                   title,
                   x_label,
                   y_label,
                   x_feature,
                   key_names = None,
                   data_cols = None,
                   colours = Category10,
                   bold_col = "red",
                   line_thick = 1.5,
                   bold_think = 1.5,
                   bold_feature = "gdp_growth",
                   alpha = 0.75,
                   tb_loc = "above",
                   plot_width = 1000,
                   plot_height = 750,
                   leg_loc = "bottom_right",
                   leg_ort = "vertical",
                   font_size = "8pt",
                   messages = True
                  ):
    """multiline_plot
    
    Creates a bokeh plot with multiple data series' plotted as a line. One 
    of these data series can be selected to be of a particular colour and 
    thickness, useful for highlighting one particular data series. It is also
    possible to show/hide particular series by clicking on its entry in the 
    legend
    
    Args:
        src (ColumnDataSource): contains data to plot
        data_cols (list): columns in src containing data to plot, if not 
            passed, it plots all columns 
        title (string): Title of the plot
        x_label (string): x axis label
        y_label (string): y axis label
        x_feature (string):column containing the year the data corrisponds to
        data_cols (list): the columns containing data to plot
        colours (Palette): Colour palette to use when plotting
        bold_col (string): Colour for the highligted data series
        line_thick (numeric): Line thickness for the highlighted data series
        bold_feature "string": name of the column in src containing data to 
            highlight
        alpha (numeric): alpha (transparency) of the plotted lines
        tb_loc (sting): tool bar location 
            see: https://bokeh.pydata.org/en/latest/docs/user_guide/tools.html
        plot_width (numeric): plot size
        plot_height (numeric): plot size
        leg_loc (string): location of the legend
            see: https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#legends
        leg_ort (string): orientation of the legend
            see: https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#legends
        font_size (string): size of font of the axis label text
        
    Returns:
        plot (bokeh figure): the figure created
        
    """

    # plot all the columns if data_cols not passed
    if data_cols is None: data_cols =  src.to_df().columns.tolist() 
    #as no key names have been specified
    if key_names is None: key_names = data_cols
    
    #create the bokeh plot
    plot = figure(title =  title,
                  x_axis_label = x_label,
                  y_axis_label = y_label,
                  toolbar_location = tb_loc,
                  plot_width = plot_width , 
                  plot_height = plot_height
                  )

    #create colour palette cycler
    colour_cycler = it.cycle(colours)

    #plot the data
    for item, key, col in zip(data_cols, 
                              key_names,
                              colour_cycler
                             ):
        #is this a feature we plot with a thicker red line?
        if item == bold_feature:
            plot.line(x_feature, 
                      item, 
                      line_color = bold_col,
                      line_width = line_thick * bold_think,
                      legend = key,
                      source = src
                      )
            if messages:
                print(key)
        else:
            plot.line(x_feature,
                      item,
                      line_color = col,
                      source = src,
                      legend = key,
                      line_alpha = alpha,
                      line_width = line_thick
                      )
            if messages:
                print(key)
            
    #make sure all the ticks are labeled
    n_ticks = len(src.to_df().index.unique().tolist())
    plot.xaxis[0].ticker.desired_num_ticks = n_ticks
    
    #legend setup
    plot.legend.location = leg_loc
    plot.legend.orientation = leg_ort
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = font_size 
        
    return plot
                
#%%
    
def cross_correlation_charts(df,
                             data_cols, 
                             key_names,
                             corr_name,
                             colour = Category10,
                             corr_col = "gdp_growth",
                             x_label = "Lag (years)",
                             y_label = "Signal overlap",
                             tb_loc = "right",
                             plot_x = 1000,
                             plot_y = 400,
                             lag_line_col = "black",
                             lag_line_alpha = 0.75,
                             lag_line_dash = "dashed",
                             messages = True
                            ):
    """cross_correlation_charts
    
    Creates a tabbed bokeh plot with multiple data series' cross correlation
    function with a single other series with a line and label on the chart
    that gives the lag with the highest signal overlap.
    
    Args:
        df (DataFrame): contains data to plot
        data_cols (list): columns in src containing data to plot, if not 
            passed, it plots all columns 
        key_names (List): more readable names for data_cols, if not passed
            data_cols is used
        x_label (string): x axis label
        y_label (string): y axis label
        cor_col (string): column in df containing data series to cross 
            correlate with data columns in data_col
        colours (Palette): Colour palette to use when plotting
        lag_line_col (string): Colour for the line indicating lag with max
            overlap
        lag_line_alpha (numeric): alpha (transparency) of the lag line
        tb_loc (sting): tool bar location 
        lag_line_dash (string): Dash pattern for the line indicating lag with 
            max overlap
            see: https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#line-properties
        plot_x (numeric): plot width in pixes on screen
        plot_y (numeric): plot height in pixes on screen

        
    Returns:
        (bokeh Tabs): Widget containing the plotted charts
        
    """
    
    # plot all the columns if data_cols not passed
    if data_cols is None: data_cols =  df.columns.tolist() 
    #as no key names have been specified
    if key_names is None: key_names = data_cols
    
    if corr_name is None: corr_name = corr_col
    
    #remove the one column we are correlating everything else with
    feat_cols = [col 
                 for col in data_cols
                 if col != corr_col
                ]
    lags = [i for i in range(-len(df.index),
                             (len(df.index)) - 1
                            )]
    
    plot_lst = list()
    if messages:
        print(data_cols)
        print(feat_cols)
        print(colour)
        print(key_names)
    for col, colour, name in zip(feat_cols, colour[len(feat_cols)], key_names):
        
        #compute cross_correlation
        y = correlate(df[corr_col], df[col])
    
        #create line indicating lag
        max_index = np.argmax(y)
        max_overlap = lags[max_index]
        max_line_y = [i for i in range(int(max(y)))]
        max_line_x = [max_overlap] * len(max_line_y)

        # #min value to for -ve correlations
        # min_index = np.argmin(y)
        # min_overlap = lags[min_index]
        # min_line_y = [i for i in range(int(min(y)))]
        # min_line_x = [min_overlap] * len(min_line_y)
        #
        #create figure
        plot_title = (corr_name + " and " 
                      + name 
                      + " mean flow cross-correlation"
                     )
                
        p = figure(title = plot_title,
                   x_axis_label = x_label,
                   y_axis_label = y_label,
                   toolbar_location = tb_loc,
                   plot_width = plot_x, 
                   plot_height = plot_y
                  )
        
        #plot crosscorelation function
        p.line(lags, 
               y,
               line_color = colour,
               line_width = 2
              )
        
        #indicate max and min points for lag
        p.line(max_line_x,
               max_line_y,
               line_color = lag_line_col,
               line_alpha = lag_line_alpha,
               line_dash = lag_line_dash
               )

        # p.line(min_line_x,
        #        min_line_y,
        #        line_color = lag_line_col,
        #        line_alpha = lag_line_alpha,
        #        line_dash = lag_line_dash
        #        )

        label_text = ("Max overlap at lag of {} years".format(max_overlap))

        label = Label(x = 5,
                      y = 30,
                      x_units='screen',
                      text =  label_text,
                      text_font_size = "8pt",
                      render_mode='css',
                      border_line_color='black',
                      border_line_alpha = 0.5,
                      background_fill_color = 'white'
                     )

        #create tabbed plots
        p.add_layout(label)
        tab = Panel(child = p, title = name)
        plot_lst.append(tab)   
        
    #put together and show the plots
    return Tabs(tabs = plot_lst)
    
#%%
def get_pdf(data):
    """get_pdf
    
    Compute the pdf of a data series
    
    Args:
        data (DataFrame)
    Returns:
        pdf (array): The probablity density function
        kde.support (array): for plotting 
    """
    #format data
    data = data.dropna().astype(float).values
    #get kde
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit()
    counts, bins = np.histogram(data)
    pdf = (kde.density/kde.density.max())*max(counts)
    
    return pdf, kde.support

#%%
def pdf_plot(src, 
             feature,
             plot = None,
             x_suffix = "_support",
             y_suffix = "_density",
             plot_size = (600,150),
             axis_swap = False,
             logo = None,
             toolbar = None
            ):
    """pdf_plot
    
    Creates a probablity density plot to be show with a scatter plot in
    the joint_plot function
    
    Args:
        src (ColumnDataSource): we create the plot from the data here
        feature (string): the feature in src to plot
        plot (Figure): bokeh figure to create the plot with, will create
            our own if this is None.
        x_suffix (string): suffix to append to feature to indicate the column 
            in src of the values to plot on the x axis
        x_suffix (string): suffix to append to feature to indicate the column 
            in src of the values to plot on the x axis
        plot_size (List like): width and height of the plot
        axis_swap (bool): When set to True, swap the axes, so x becomes y
            and y becomes x
        logo (bool): Display the Bokeh logo or not
        toolbar (bool): 
    Returns:
        plot (Figure): Bokeh Figure of the pdf for the feature indicated by 
            feature in the source data, src
    """
    
    #switch the axis
    if axis_swap:
        swap = y_suffix
        y_suffix = x_suffix
        x_suffix = swap
    
    #create our own plot if we don't have one    
    if plot is None:    
        plot = figure(plot_width = plot_size[0],
                      plot_height = plot_size[1])
    
    #plot the pdf    
    plot.line(x = feature + x_suffix, 
              y = feature + y_suffix,
              source = src
             )
    #set some plot options         
    plot.axis.visible = False
    plot.grid.visible = False
    plot.outline_line_color = None
    
    plot.toolbar.logo = logo
    plot.toolbar_location = toolbar
    
    return plot   
#%%
def plot_scatter(src,
                 x_feature,
                 y_feature,
                 x_label,
                 y_label,
                 plot = None,
                 plot_size = (600,450),
                 fit_line = False,
                 fit_line_col = "black",
                 logo = None,
                 toolbar = None
                ):
    """scatter_plot
    
    Creates a scatter plot to be show with probablity density function plots in
    the joint_plot function
    
    Args:
        src (ColumnDataSource): we create the plot from the data here
        y_feature (string): the feature in src to plot on the x axis
        y_feature (string): the feature in src to plot on the y axis
        x_label (string): Label for the x axis
        y_label (string): Label for the y axis
        plot (Figure): bokeh figure to create the plot with, will create
            our own if this is None.
        plot_size (List like): width and height of the plot
        fit_line (bool): Plot a linear least squares line of best fit
        fit_line_col: (string): colour of the line of best fit,
        logo (bool): Display the Bokeh logo or not
        toolbar (bool): 
    Returns:
        plot (Figure): Bokeh Figure showing a scatter plot for two features
            in the source data, src
    """
    
    #create our own plot if we don't have one   
    if plot is None:    
        plot = figure(x_axis_label = x_label,
                      y_axis_label = y_label,
                      plot_width = plot_size[0],
                      plot_height = plot_size[1]
                     )
    #get the best fit line and plot it
    if fit_line:    
        line_src, r, p = lin_reg_line(src, 
                                      x_feature,
                                      y_feature
                                    )
        plot.line(x = "x_vals",
                  y = "fit",
                  source = line_src,
                  line_color = fit_line_col,
                  line_alpha = 0.5
                 )
        
        """
        #for shading and error
        ##get the data, sort_values to make sure patch is drawn as rectangle
        line_df = line_src.to_df() 
        
        band_y = np.append(line_df.sort_values("x_vals")["+_err"], 
                           line_df.sort_values(by = "x_vals",
                                               ascending = False
                                              )["-_err"]
                          )
                          
        band_x = np.append(line_df.sort_values("x_vals")["x_vals"],
                           line_df.sort_values(by = "x_vals",
                                               ascending = False
                                              )["x_vals"]
                          )
        
        #and plot
        plot.patch(band_x,
                   band_y, 
                   color = "grey",
                   fill_alpha = 0.05,
                   line_alpha = 0.5,
                   line_dash = "dotted"
                  )
        """
           
        #add in the correlation coeff and alpha level as text
        ann_text = "Pearson's coef: {0} \np-value: {1}".format(round(r,2),
                                                               round(p,3)
                                                              )

        plot.add_layout(Title(text = ann_text, align = "center"),"below")

    #data    
    plot.circle(x = x_feature,
                y = y_feature,
                source = src
               )
    
    #graph formatting
    plot.toolbar.logo = logo
    plot.toolbar_location = toolbar
    
    plot.extra_x_ranges = {'top_joint_axis': plot.x_range}
    plot.add_layout(LinearAxis(x_range_name='top_joint_axis'), 'above')
    plot.extra_y_ranges = {'right_joint_axis': plot.y_range}   
    plot.add_layout(LinearAxis(y_range_name='right_joint_axis'), 'right')
    

    
    return plot
#%%
def frange(start,stop,step):
    
    """frange
    
    generator that starts at a number, increase by an increment until the 
    number it yields exceeds a limit
    
    Args:
        start (numeric): The number we start at
        stop (numeric): Stop yeilding when we exceed this
        step (numeric): The increment
    
    """
    i = start
    
    while i < stop:
        yield i
        i += step
#%%
def lin_reg_line(source,
                 x_feature,
                 y_feature):
    """lin_reg_line
    
    Creates a linear regression line for plot_scatter
    
    Args:
        source (ColumnDataSource): Contains the data to fit
        x_feature: feature on the x axis (IV)
        y_feature: feature on the y axis (DV)
        
    returns: line_src (ColumnDataSource) contains fit, the fitted line 
        (predicted y values), +_err and -_err the + and - one standard error 
        lines and x_vals, the x value for each fitted value in fit 
    
    """
    
    #dataframe makes this easier
    df = source.to_df()
    
    slope, intercept, r, p, std_err = linregress(df[x_feature], 
                                                 df[y_feature]
                                                )
    x_vals = [x for x in frange(min(df[x_feature]), max(df[x_feature]), 0.1)]
    
    #lines for plotting
    lse_line = [x*slope+intercept for x in df[x_feature].tolist()]
    pos_err_line = [y + std_err for y in lse_line]
    neg_err_line = [y - std_err for y in lse_line]
    
    #turn into source for plotting
    line_src = ColumnDataSource({"fit" : lse_line,
                                 "+_err" : pos_err_line,
                                 "-_err" : neg_err_line,
                                 "x_vals" : df[x_feature].tolist()
                               })
    return line_src, r, p
#%%
def joint_plot(source_lst, 
               x_feature, 
               y_feature,
               x_label,
               y_label,
               plot_size = (800,600),
               size_ratio = 0.85,
               fit_line = True,
               logo = None,
               toolbar = None
              ):
    """joint_plot
    
    Creates a scatter plot to be show with probablity density function plots in
    the joint_plot function
    
    Args:
        source_list (List): contains three ColumnDataSource object, the first
            two are for the x and y pdf chart respectively, the third is for
            the scatter plot.
        x_feature (string): the feature in src to plot on the x axis
        y_feature (string): the feature in src to plot on the y axis
        x_label (string): Label for the x axis
        y_label (string): Label for the y axis
        plot_size (List like): width and height of the plot
        plot_ratio (numeric): Relative size of scatter plot, larger the value
            the bigger the scatter plot relative to the pdf plots. Values 
            between 0 - 1.
        fit_line (bool): Plot a linear least squares line of best fit
        logo (bool): Display the Bokeh logo or not
        toolbar (bool): 
    Returns:
        joint_plot (Gridplot): Bokeh GridPlot object with the scatter and 
            pdf plots within it.
    
    
    """
    #compute the sizes of the pdf and scatter plot
    pdf_size_ratio = 1 - size_ratio
    scatter_size = tuple(int(round(dim * size_ratio))
                         for dim in plot_size
                        )

    x_pdf_size = (scatter_size[0],
                  int(plot_size[1] * pdf_size_ratio)
                 )

    y_pdf_size = (int(plot_size[0] * pdf_size_ratio),
                  scatter_size[1]
                 )

    #create the pdf plots
    y_pdf = pdf_plot(source_lst[0], 
                     y_feature,
                     plot_size = y_pdf_size,
                     axis_swap = True
                    )
    x_pdf = pdf_plot(source_lst[1],
                     x_feature,
                     plot_size = x_pdf_size
                    )
    
    #create the scatter plot
    scatter = plot_scatter(source_lst[2],
                           x_feature = x_feature,
                           y_feature = y_feature,
                           x_label = x_label,
                           y_label = y_label,
                           plot_size = scatter_size,
                           fit_line = fit_line
                          )
    
    #remove/add logos and toolbars
    x_pdf.toolbar.logo = logo
    y_pdf.toolbar.logo = logo
    scatter.toolbar.logo = logo
    x_pdf.toolbar_location = toolbar
    y_pdf.toolbar_location = toolbar
    scatter.toolbar_location = toolbar
    
    #combine the plots together
    joint_plot = gridplot([x_pdf],
                          [scatter, y_pdf]
                         )
    
    return joint_plot
#%%
def lag_gdp(df,
            lag,
            year_col = "year_label",
            gdp_cols = ["GDP Growth", "gdp"],
            area_cols = ["area_name", "area_level"]
           ):
    """lag_gdp
    
    Lags the gdp data year
    
    Args:
        df (DataFrame): Contains the gdp and traffic data
        lag (numeric): how many years to lag the gdp data
        year_col (string): Column in df containing the year
        gdp_cols (List): Columns in df containing gdp data
        area_cols (List): Columns in df contain geographic data
    
    returns (DataFrame): contains data with traffic values matched to lagged
        gdp data
    
    """
    
    #splice dataframes
    gdp_df = df[gdp_cols + area_cols].reset_index()
    traffic_df = df.drop(gdp_cols, axis = 1).reset_index()

    #lag year
    gdp_df[year_col] = gdp_df[year_col] + lag

    #merge, dropping years we don't have both sets of data for
    return pd.merge(left = traffic_df,
                    right = gdp_df,
                    on = ["year_label"] + area_cols,
                    how = "inner"
                   )


#%%

def create_sources(df, 
                   traffic_feature,
                   nuts_level,
                   region_name,
                   gdp_feature = ["GDP Growth","gdp"],
                   nuts_col = "area_level",
                   region_col = "area_name"

                  ):
    """create_sources
    
    Creates sources for the joint_plot
    
    Args:
        df (DataFrame): data for the joint plot
        nuts_level (string): heirarcical level for the region to create the
            jointplot for
        region_name (string): name of the region to create the jointplot for
        gdp_feature (List): columns in df containing gdp data
        nuts_col (string): column in df containing the nuts level for the 
            region
        region_col (string): column in df containing the region name for the
            data to plot
            
    returns source_list (List): data for joint_plot. Contains three 
        ColumnDataSource objects. The first two contain the data for the pdf 
        plots on the x and y axis. The third contains the data for the scatter
        plot.           
    
    """
    
    features = [gdp_feature, traffic_feature]
    #make sure we sort by the index
    df.sort_index(inplace = True)
    mask = (df[nuts_col] == nuts_level) & (df[region_col] == region_name)
    
    df = df[mask]
    
    source_lst = list()
    all_feat_lst = list()
    
    pdf_df = pd.DataFrame()
    
    #seperate column sources for gdp and traffic feature
    for feature in features:
    
        pdf_feat_lst = [feature + "_density",
                        feature + "_support"
                       ]
        
        all_feat_lst = all_feat_lst + pdf_feat_lst
        
        pdf_df[pdf_feat_lst[0]], pdf_df[pdf_feat_lst[1]] = get_pdf(df[feature])
    
        source_lst.append(ColumnDataSource(pdf_df))
    
    #combined column source
    source_lst.append(ColumnDataSource(df[features]))
    
    return source_lst


def create_corr_df(df,
                   x_features,
                   x_name,
                   y_features = None,
                   y_name = None,
                   coef_name = "R"
                   ):
    """
    Takes a Dataframe and correalates two sets of features in that DataFrame and
        returns the coefficients in a tall DataFrame
    Args:
        df (DataFrame): contains x and y features to correlate
        x_features (list): first set of features to correlate
        x_name (String): name of the column containing x_features in the returned DataFrame
        y_features (list): second set of features to correlate - optional; will correlate
            x_features with itself if not given
        y_name (String): name of the column containing y_features in the returned DataFrame
            - optional; if not passed is [x_name]_2
        coef_name: name of the coefficents column in returned DataFrame

    Returns:
        corr_df (DataFrame): Tall format dataframe containing columns for both sets of
            correlated features and the correlation coefficent

    """
    # do we only have one set of features?
    if y_features is None: y_features = x_features
    if y_name is None: y_name = "".format("{}_2", x_name)

    # initialise array
    corr_arr = np.zeros((len(y_features), len(x_features)))

    # populate the array with correlation coefs
    for x, y in it.product(x_features, y_features):
        y_pos = y_features.index(y)
        x_pos = x_features.index(x)
        corr_arr[y_pos, x_pos] = np.corrcoef(df[y],
                                             df[x]
                                             )[1, 0]

    # turn into pandas DataFrame and reshape into tall
    corr_df = pd.DataFrame(corr_arr, index = y_features, columns = x_features)
    corr_df.index.name = y_name
    corr_df.columns.name = x_name
    corr_df = corr_df.reset_index().melt(id_vars = y_name, value_name = coef_name)

    return corr_df


def create_heatmap(df,
                   x_col = "eco_ind",
                   y_col = "vehicle_type",
                   val_col = "R",
                   colour_palette = Plasma256[::1],
                   title_text = "traffic flow vs economic indicators",
                   plot_shape = [1000, 1000],
                   rect_shape = [1, 1],
                   n_ticks = 9,
                   min_val = -1,
                   max_val = 1
                   ):

    """
    creates a bokeh plot of a heatmap between two columns in a pandas dataframe
    Args:
        df (DataFrame): contains values to plot
        x_col(String): name of col containing features to plot on the x axis
        y_col(String): name of col containing features to plot on the y axis
        val_col(String): name of column containing values to colur
        colour_palette: bokeh colour palette to use for the graph
        title_text(String): name of the graph
        plot_shape (iterable): dimensions of the plot
        rect_shape (iterable): dimensions of the rectangles making up the heatmap
        n_ticks (int): number of ticks to display on the colour bar
        min_val (numeric): min val for the colour bar
        max_val (numeric): max val for the colour bar

    Returns: bokeh plot object

    """
    # make things we need

    src = ColumnDataSource(df)

    colour_mapper = LinearColorMapper(palette = colour_palette,
                                      low = min_val,
                                      high = max_val
                                      )
    colours = transform(val_col, colour_mapper)

    # create the figure
    tool_tips = [("coef:", "${}".format(val_col))]

    p = figure(plot_width = plot_shape[0],
               plot_height = plot_shape[1],
               x_range = list(df[x_col].drop_duplicates()),
               y_range = list(df[y_col].drop_duplicates()),
               toolbar_location = 'above',
               title = title_text,
               tooltips = tool_tips
               )

    # make the heatmap
    p.rect(x = x_col,
           y = y_col,
           width = rect_shape[0],
           height = rect_shape[1],
           source = src,
           fill_color = colours,
           line_color = None
           )

    p.xaxis.major_label_orientation = math.pi / 2

    # colour key
    color_bar = ColorBar(color_mapper = colour_mapper,
                         location = (0, 0),
                         ticker = BasicTicker(desired_num_ticks = n_ticks)
                        )

    # stick them together
    p.add_layout(color_bar, 'right')

    return p

def auto_corr_plot(src,
                   title,
                   plot_shape = [450,300],
                   x_label = "lag",
                   y_label = "autocorrelation",
                   lag_col = "lag",
                   corr_col = "auto_corr",
                   symbol_colour = "mediumseagreen",
                   symbol_size = 8,
                   line_colour = "cornflowerblue",
                   line_width = 2,
                   err_colour = "darkseagreen",
                   err_alpha = 0.2,
                   plot_errors = True,
                   err_cols = ["upper_err", "lower_err"]
                   ):

    # create the plot from the autocorrelation results
    plot = figure(title = title,
                  x_axis_label = x_label,
                  y_axis_label = y_label,
                  plot_width=plot_shape[0],
                  plot_height=plot_shape[1],
                  )

    if plot_errors:

        #set up the variables for the patch to plot errors
        x_err = np.delete(src.data[lag_col], 0)
        up_err = np.delete(src.data[err_cols[0]] - src.data[corr_col], 0)
        low_err = np.delete(src.data[err_cols[1]] - src.data[corr_col], 0)

        band_x = np.append(x_err, x_err[::-1])
        band_y = np.append(low_err, up_err[::-1])

        # and plot the patch for the errors
        plot.patch(band_x,
                   band_y,
                   color = err_colour,
                   fill_alpha = err_alpha
                   )

    # plot the "stems"
    plot.segment(x0 = lag_col,
                 y0 = 0,
                 x1 = lag_col,
                 y1 = corr_col,
                 source = src,
                 color = line_colour,
                 line_width = line_width
                 )

    # plot the leaves
    plot.circle(x = lag_col,
                y = corr_col,
                size = symbol_size,
                color = symbol_colour,
                line_color = line_colour,
                line_width = line_width,
                source = src
                )

    return plot


