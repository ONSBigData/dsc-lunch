#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exploratory_analysis

A module that contains functions for exploratory analysis of traffic and 
gdp

This contains functions

Created on Mon Nov 27 11:48:26 2017

@author: Edward Rowland
"""

#%%
import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import holoviews as hv
import itertools as it

import matplotlib.pylab as plt
import scipy.stats as stats


from holoviews.operation import gridmatrix

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer 

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

#custom modules
file_dir = os.getcwd()
sys.path.append(file_dir) # assume other scripts are where we are
import dataset_processing as dat
#%%
   
def feature_hist(df, 
                 features, 
                 messages = False, 
                 sample_ratio = 0.1,
                 kde_colour = "r",
                 kde_line_width = 3,
                 hist_colour = "b",
                 hist_line_width= 3,
                 hist_alpha = 0.5
                 ):
    """feature_hist
    
    plots a histogram with kernel density estimation for each feature in a df
    
    Arguments:
        df (DataFrame): the data to plot
        features(DataFrame): columns in df containing data to plot
        messages (bool): set to true to print out progress messages
        sample_ratio (float): the fraction of observations in df to plot 
            to reduce overplotting and speed up runtime
        kde_colour (char): matplotlib colour for kde plot
        kde_line_width (float): kde plot line width
        hist_colour (char): matplotlib colour for histogram plot
        hist_line_width (float): hist plot line width
        hist_alpha (float): transparency setting for histogram
    
    Returns: 
        plots_lst (list): list of matplotlib plots for each feature
        
    """
    plots_lst = list()
    #sample for speed with large DataFrames
    working_df = df.sample(frac = sample_ratio)
    #plot histgram for each feature column
    for feature in features:
        if messages: print("plotting: " + feature)
        fig = sns.distplot(working_df[feature].dropna(),
                           rug = True,
                           kde_kws = {"color" : kde_colour,
                                      "lw" : kde_line_width,
                                      "label": "KDE"
                                     },
                           hist_kws = {"histtype": "step",
                                       "linewidth": hist_line_width,
                                       "color" : hist_colour,
                                       "alpha" : hist_alpha
                                      }
                          )
        
        fig.set_title(feature + " histogram")
        fig.set_xlabel(feature + " annual average daily flow")
        plt.show()
        plots_lst.append(fig)
        if messages: print ("plotting complete")
    return plots_lst

#%%
    
def create_grid_scatter(df, 
                        group_col,
                        bg_colour = "#efe8e2",
                        alpha_level = 0.1,
                        point_size = 1,
                        sample_ratio = 0.1,
                        messages = False
                       ):
    
    """create_grid_scatter
    
    builds a grid of scatterplots, useful for checking multicolinearity.
    
    Arguments:
        df (DataFrame) the data to plot
        group_col (string): to colour the points by group
        bg_colour (string): colour hex code of the background
        alpha_level (float): tranparency of plots
        point_size (float): size of points on plot
        sample_ratio (float): ratio of observations in df to use for speed and 
            to prevent overplotting
        messages (bool): print messages
        
        Returns:
            fig (holoviews plot): the grid scatter
    
    """
    #create dataset
    if messages: print("creating dataset for plotting...")
    feats_hv = hv.Dataset(df.sample(frac = sample_ratio))
    grouped_feats_hv = feats_hv.groupby(group_col, 
                                        container_type=hv.NdOverlay
                                        )
    #plot options
    if messages: print("formatting plot...")
    plot_opts = dict(bgcolor = bg_colour)
    style = dict(alpha = alpha_level, size = point_size)
    #plot
    if messages: print("plotting")
    grid_corr = gridmatrix(grouped_feats_hv, diagonal_type=hv.Scatter)
    fig = grid_corr({'Scatter': {'plot': plot_opts, 'style': style}})
    
    return fig


#%%

def plot_pearsons(df):
    
    """plot_pearsons
    
    compute persons correlation coefficents and plot in heatmap
    
    Arguments:
        df (DataFrame): numerical values to plot
        
    Returns:
        corr_arr (array): array of correlation coefficients
        fig (matplotlib axes): heatmap of correlations
    """
    corr_arr = df.corr()
    fig = sns.heatmap(corr_arr,
                      xticklabels = corr_arr.columns.values,
                      yticklabels = corr_arr.columns.values)
    
    return corr_arr, fig
#%%
def qq_plot(df, features, distribution = "norm"):
    
    """qq_plot
    
    qq plots for each column in features stored in df
    
    Arguments: 
        df (dataframe): contains data to be plotted
        features (list): names of columns in df to plot
        distribution (string) distribution to check data against

    
    """
    for feature in features: 
        stats.probplot(df[feature], dist = distribution, plot=plt)
        plt.title(feature)    
        plt.show()
        
#%%     
def linearity_plots(df, features, nuts_val):
    
    """linearity_plots 
    
    Creates plots to check for non-linearities in features vs the DV.
    
    Arguments:
        df (DataFrame): contains data to plot
        features (list): column names of the IVs to check against the DV
        nuts_val (string): column name of the DV
    
    Returns:
        fig_lst (list): list of seaborne (matplotlib) plot objects for
            the plot of each IV vs DV
    
    """
   
    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    nuts_mask = df[nuts_val].notnull()
    fig_lst = list()
    for feature in features:
        feat_mask = df[feature].notnull()
        #only plot data we have values for
        null_mask = feat_mask & nuts_mask
        
        fig = sns.jointplot(x = feature, y = nuts_val, data = df[null_mask])
        fig_lst.append(fig)
        
    return fig_lst
        
#%%

def loadings_plot(PCA_data, 
                  coeff, 
                  labels = None,
                  x_comp = 0,
                  y_comp = 1,
                  alpha_level = 0.5,
                  arrow_colour = "r",
                  text_colour = "g",
                  dots_per_inch = 500,
                  size = 1,
                  text_size = "small"
                 ):
    
    """loadings_plot
    
    Create a loadings plot to show the influence of each feature on a component
    
    Arguments:
        PCA_data 
    """
    
    #get the PC scores to plot
    x_score = PCA_data[:, x_comp]
    y_score = PCA_data[:, y_comp]
    
    scale_x = 1.0/(x_score.max() - x_score.min())
    scale_y = 1.0/(y_score.max() - y_score.min())
    n_features = coeff.shape[0]
    plt.subplots(dpi =dots_per_inch)
    plt.scatter(x_score * scale_x, 
                y_score * scale_y,
                s = size)
    
    #set the text for the arrow if none is given
    if labels is None: 
        labels = ["Var" + str(feat + 1) for feat in range(n_features)]
        
    #plot each feature arrow with label
    for feature in range(n_features):
        #plot the arrow
        plt.arrow(0,
                  0, 
                  coeff[feature, x_comp],
                  coeff[feature, y_comp],
                  alpha = alpha_level,
                  color = arrow_colour
                 )
        
        #plot the text for the arrow
        plt.text(coeff[feature, x_comp]* 1.15, 
                 coeff[feature, y_comp] * 1.15,
                 labels[feature],
                 color = text_colour, 
                 ha = 'center', 
                 va = 'center',
                 size = text_size
                )    

#%%

def all_descriptive_stats(loc, 
                          stats_for,
                          file_year_col = "aadfyear",
                          file_regex = ".*traffic_gdp_[0-9]",
                          ext = "csv",
                          messages = False,
                          rcat_col = "rcat",
                          nuts2_col = "nuts2_name",
                          nuts3_col = "nuts3_name",
                          area_col_suffix = "_name",
                          veh_col = "vehicle_measure",
                          area_name_col = "area_name",
                          area_level_col = "area_level",
                          year_col = "year",
                          gdp_col = "gdp",
                          gdp_unit_col = "gdp_unit",
                          road_all_indicator = "all",
                          test_range = None
                         ):
    
    """compute_descriptive_stats
    
    Computes descriptive stats with pandas describe function for NUTS2 & 3 
    areas to compare against that area's gdp.
    
    Arguments:
        loc (string): folder location with the datafiles inside
        file_regex (string): regex expression to select the files to load
        ext (string): extension of the files to load
        messages(bool): indicate to print progress messages
        stats_for (list): List of columns that we want to aggreagate.
        file_year_col (string): column in file where the year of the
            observations are stored.
        rcat_col (string): Column in df that stores the road category.
        nuts2_col (string): Column in df that stores the nuts2 category.
        nuts3_col (string): Column in df that stores the nuts3 category.
        area_col_suffix (string): Suffix to add to the nuts level that gives
            the column in df that stores the area name.
        veh_col (string): Name of the column in describe_df where the traffic
            flow measure is stored
        area_name_col (string): Name of the column in describe_df where the 
            area name is stored.
        area_level_col (string): Name of the column in describe_df where the 
            nuts level is stored.
        year_level_col (string): Name of the column in describe_df where the 
            year is stored.
        gdp_col (string): Name of the column in describe_df where the 
            gdp value is stored.
        gdp_unit_col (string): Name of the column in describe_df where the 
            gdp unit is stored.
        road_all_indicator (string): Value in rcat_col in describe_df that
            corrisponds to stats for all road categories
            
    Returns: 
        describe_df (DataFrame): Contains descriptive stats by road type 
        and NUTS area for observations in df

    """
    
    # these are the files that store the data we want to iterate over
    files = dat.files_looks_like(loc, looks_like = file_regex, ext = ext)
    df_lst = list()
    
    #for testing a limited range of the files in loc to see if this works
    #as it takes a while with all of them
    if test_range is not None: 
        files = files[0:test_range]
        print("Testing with: %s " %files)
    for file in files:
        if messages: print("loading file %s" %file)
        df = pd.read_csv(file)
        #get the year, assuming all data corrisponds to the same year
        year = df[file_year_col].unique()[0]
        if messages: print("year: %s" %year)
        if messages: print("computing stats...")
        
        #get a list of dataframes with stats
        stats_df = compute_descriptive_stats(df, 
                                    year, 
                                    stats_for,
                                    rcat_col = rcat_col,
                                    nuts2_col = nuts2_col,
                                    nuts3_col = nuts3_col,
                                    area_col_suffix = area_col_suffix ,
                                    veh_col = veh_col,
                                    area_name_col = area_name_col,
                                    area_level_col = area_level_col,
                                    year_col = year_col,
                                    gdp_col = gdp_col,
                                    gdp_unit_col = gdp_unit_col,
                                    road_all_indicator = road_all_indicator
                                            )
        stats_df.reset_index(inplace=True, drop=True)                     
        df_lst.append(stats_df)
        if messages: print("...done")
        if messages: print("file year: %s" %df_lst[-1].year)
    if messages: print("concatenating")
    descriptive_df = pd.concat(df_lst)
    if messages: print("done")
    return descriptive_df

#%%
        
def compute_descriptive_stats(df, 
                              year,                              
                              stats_for,
                              rcat_col = "rcat",
                              nuts2_col = "nuts2_name",
                              nuts3_col = "nuts3_name",
                              area_col_suffix = "_name",
                              growth_col_suffix = "_growth",
                              ratio_col_suffix = "_gdp_ratio",
                              veh_col = "vehicle_measure",
                              area_name_col = "area_name",
                              area_level_col = "area_level",
                              year_col = "year",
                              gdp_col = "gdp",
                              gdp_unit_col = "gdp_unit",
                              growth_col = "gdp_growth",
                              ratio_col = "gdp_ratio",
                              road_all_indicator = "all",
                              national = "nat",
                              nat_level = "nuts1",
                              nat_gdp_col = "gdp",
                              nat_gdp_unit_col = "gdp_unit",
                              nat_gdp_growth_col = "gdp_growth"
                              
                              ):
    
    """compute_descriptive_stats
    
    Computes descriptive stats with pandas describe function for NUTS2 & 3 
    areas to compare against that area's gdp.
    
    Arguments:
        df (DataFrame): Contains the traffic data by observation point that
            we wish to aggregate
        year (string): Year the df corresponds to.
        stats_for (list): List of columns that we want to aggreagate.
        rcat_col (string): Column in df that stores the road category.
        nuts2_col (string): Column in df that stores the nuts2 category.
        nuts3_col (string): Column in df that stores the nuts3 category.
        area_col_suffix (string): Suffix to add to the nuts level that gives
            the column in df that stores the area name.
        growth_col_suffix (string): Suffix to add to the nuts level that gives
            the column in df that stores the gdp growth.
        ratio_col_suffix (string): Suffix to add to the nuts level that gives
            the column in df that stores the gdp ratio on previous year.
        veh_col (string): Name of the column in describe_df where the traffic
            flow measure is stored
        area_name_col (string): Name of the column in describe_df where the 
            area name is stored.
        area_level_col (string): Name of the column in describe_df where the 
            nuts level is stored.
        year_level_col (string): Name of the column in describe_df where the 
            year is stored.
        gdp_col (string): Name of the column in describe_df where the 
            gdp value is stored.
        gdp_unit_col (string): Name of the column in describe_df where the 
            gdp unit is stored.
        growth_col (string): Name of the column in describe_df where the 
            gdp growth value is stored.
        ratio_col (string): Name of the column in describe_df where the 
            gdp ratio on previous year value is stored.
        road_all_indicator (string): Value in rcat_col in describe_df that
            corrisponds to stats for all road categories,
        national (string): label to go in df[area_col] indicating this is
            stats for the nation
        nat_level (string): label to go in df[area_level_col] indicating this 
            is stats for the nation    
            
    Returns: 
        describe_df (DataFrame): Contains descriptive stats by road type 
        and NUTS area for observations in df
        
    Todo:
        refactor with yearly descriptive stats, see comments

    """
    stats_lst = list()
    
    nuts2_areas = df[nuts2_col].dropna().unique().tolist()
    nuts3_areas = df[nuts3_col].dropna().unique().tolist()

    road_cats = df[rcat_col].unique().tolist()
    road_cats.append(road_all_indicator)

    #combine nuts2 and nuts3 areas
    nuts_areas = [(area, "nuts2") for area in nuts2_areas]
    for area in nuts3_areas: nuts_areas.append((area, "nuts3"))
    nuts_areas.append((national, nat_level)) #all areas
    
    #flat is better than nested
    for product in it.product(nuts_areas, road_cats):
      
        #unpack
        area_name = product[0][0]
        area_col = product[0][1]
        road_cat = product[1]
        
        #subselect by nuts area and road category

        if (area_name is national) & (road_cat is road_all_indicator) :
            # include everything
            mask = [True] * df.shape[0]
        elif area_name is national:
            # all of a particular road in the nation
            mask = df[rcat_col] ==  road_cat 
        elif road_cat is road_all_indicator:
            # all roads in a nuts area
            mask =  df[area_col + area_col_suffix] == area_name  
        else:
            #only certain road types in a nuts area
            mask = ((df[area_col + area_col_suffix] == area_name)
                    & (df[rcat_col] == road_cat)
                   )
   
        #to store data in tall format wrt measure type
        stats = df[mask][stats_for].describe().T
        stats.reset_index(inplace = True)
        stats.rename(columns = {"index" : veh_col}, inplace = True)
        
        #we know what this responds to
        stats[area_name_col] = area_name
        stats[area_level_col] = area_col
        stats[year_col] = year
        stats[rcat_col] = road_cat
        
        #get the gdp...
        
        if area_name is national: 
            #...of the nation
            #take first value as they are all the same
            stats[gdp_col] = df[nat_gdp_col][0]
            stats[gdp_unit_col] = df[nat_gdp_unit_col][0]
            stats[growth_col] = df[nat_gdp_growth_col][0]
            
            stats[ratio_col] = 1 +(df[nat_gdp_col][0]/100)
        else:
            #...of each nuts area
            area_mask =  df[area_col + area_col_suffix] == area_name
            stats[gdp_col] = df[area_mask][area_col+"_value"].unique()[0]
            stats[gdp_unit_col] = df[area_mask][area_col+"_unit"].unique()[0]
            stats[growth_col] = df[area_mask][area_col 
                                              + growth_col_suffix
                                             ].unique()[0]
            stats[ratio_col] = df[area_mask][area_col 
                                             + ratio_col_suffix
                                            ].unique()[0]
        stats_lst.append(stats)  
    
    describe_df = pd.concat(stats_lst)
    
    return describe_df


    
#%%
def compute_PCs(df, features,
                strat = "median",
                n_comp = 4):
    
    """compute_PCs
    
    standardises and imputes data then computes principle components
    
    Arguments: 
        df (DataFrame): the data to find the PCs of
        features (list): columns in df with the independent variables
        strategy (string): imputation strategy
        n_comp (int): number of principle components 
    """
    
    #create arrays
    X = df[features].values
    
    #create pipeline for data processing
    
    pipe_steps = []
    
    pipe_steps.append(("impute", Imputer(strategy = strat)))
    pipe_steps.append(("scale", StandardScaler()))
    pca_pipe = Pipeline(pipe_steps)    
    
    pca = PCA()
    pca.fit(pca_pipe(X))
    
