#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""feature_creation

Contains functions for creating features from the AADF traffic flow set
Created on Mon Nov 27 13:32:29 2017

@author: Edwardb Rowland
"""
#%%
import os
import sys
import re

import pandas as pd
import numpy as np


file_dir = os.getcwd()
sys.path.append(file_dir) # assume other scripts are where we are
import dataset_processing as dat

#%%
def create_ratios(df, 
                  denom, 
                  numerators, 
                  suffix = "_ratio", 
                  inf_to_na = True
                  ):
    """create_ratios
    
    computes ratios of traffic flow for the different vehcile types for each
    observation
    
    Arguments:
        df (Dataframe): contains the traffic data
        denom (string): column containing denominator value to compute ratios
        numerators (array): column names containing numerator values 
            to compute ratios
        suffix (string): column name suffix to store the ratio of each traffic
            flow ratio computed here
        inf_to_na (bool): change inf values to numpy.na values
    
    Returns: 
        df (Dataframe): as passed, with additional columns for the computed 
            ratios
        
    
    """
    for numer in numerators:
        col_name = numer + suffix
        df[col_name] = df[numer].divide(df[denom])
    
        if inf_to_na: df[col_name].replace([np.inf, -np.inf], np.nan)
    return df
        
        
#%%
def compute_ratios(loc, 
                   file_regex = ".*traffic_gdp_[0-9]",
                   flow_column_text = "fd",
                   all_veh_flow_col = "fdall_mv",
                   messages = False
                  ):

    #files with data to compute ratios for
    traffic_data_files = dat.files_looks_like(loc,
                                              looks_like = file_regex
                                              )
    for file in traffic_data_files:
        
        if messages: print("loading %s" %file)
        
        traffic_gdp_df = pd.read_csv(file)
        
        # get the columns to compute ratios with 
        mask =  [flow_column_text in col 
                 for col in  traffic_gdp_df.columns.tolist()
                ]
        numerators = traffic_gdp_df.columns.values[mask]

        if messages: print("computing ratios")
        # remove the overall traffic flow as 
        #this will only give values of 1
        index = np.argwhere(numerators == all_veh_flow_col)
        numerators = np.delete(numerators, index)

        #..and compute the ratios
        traffic_gdp_df = create_ratios(traffic_gdp_df,
                                       all_veh_flow_col,
                                       numerators
                                       ) 
        #overwrite that file
        if messages: print("saving")
        traffic_gdp_df.to_csv(file)
#%%
def log_transform(df, features =  None, prefix = "log_", feature_str = "fd"):
    
    """log_transform
    
    creates log values for data in specified columns
    
    Arguments: 
        df (Dataframe): data with values to transform
        features (list): list of column names with values to covert to log
        prefix (string): prefix to put on new columns with log values
    
    Returns:
        df (Dataframe): the dataframe with new columns containing the log
            transform data
    """
    #get features if we don't have any
    if features is None:
        mask =  [(feature_str in col) for col in df.columns.tolist()]
        features = df.columns.values[mask]
        
    for feature in features:
        df[prefix + feature] = np.log(df[feature])
        #change the infinite values so they ar easier to deal with/drop
        df["log_" + feature].replace([np.inf, -np.inf],
                                     np.nan,
                                     inplace = True)
    return df

#%%
def compute_log_transform(loc, 
                          messages = False, 
                          file_regex = ".*traffic_gdp_[0-9]",
                          feature_str = "fd",
                          prefix = "log"
                          ):
    """ compute_log_trasform

        loads data stored in csvs in a folder and applies a log transform to
        specified columns, then saves them as the same filename
        
        Arguments:
            loc (string): folder location with the datafiles inside
            messages (bool): indicates whether to return printed messages
            file_regex (string): regex expression to select the files to load
            feature_str (string): to identify columns for log_transform
            prefix (string): to add to column names with transformed data
            
        
    """    
    
    files = dat.files_looks_like(loc, looks_like = file_regex)
    
    for file in files:
        if messages: print("loading: %s" %file)
        df = pd.read_csv(file)
        
        if messages: print("applying log transform")
        df = log_transform(df, feature_str = feature_str, prefix = prefix)
        
        if messages: print("saving")
        df.to_csv(file)

#%%
def get_road_cats(df, road_col, junc_col, road_cat_col):
    
    """get_road_cats
    
    finds the DFT road category for intersecting roads at the junction the 
    observation was recorded at.
    
    Arguments:
        df (DataFrame): road data
        road_col (string): column in df that has road names with the DFT 
            category
        junc_col (string): column in df containing intersecting roads we want
            to find the road for
        road_cat_col (string): column containting the category of road in 
            road_col
            
    Returns: 
        junc_type (list): list of road category for roads in junc_col
        
    """
    junc_type = list()
    #get the road name
    road_lst = df[road_col].tolist()
    #sometimes two road names are given
    road_lst = [road.rstrip(" ") for road in road_lst]
    road_lst = [road.rstrip("/") for road in road_lst]
    
    
    for junc in df[junc_col]:
        #incidate Local authority boundary
        if junc is "LA Boundary": 
            junc_type.append("LA")
        else:
            #find matching roads...
            road_mask = [str(junc) in road for road in road_lst]
            if any(road_mask):
                #...get their category
                junc_type.append(df[road_cat_col][road_mask].tolist()[0])
            else:
                #...if we cant find the road
                junc_type.append(np.nan)
            
    return junc_type

#%%
    
def classify_road(road_names):
    
    """Classify_road
    
    the collection points list the junctions that the data is recorded from,
    a-junction and b-junction. we create a new feature based upon this data
    using the following rules

    |data rule            | example                    |  classifiction |
    |--------------------:|:--------------------------:|:---------------|
    |first character is m | m79                        | Motorway       |
    |first character is a | a38                        | a road         |
    |first character is b | b3135                      | b road         |
    |is 'la boundary'     | 'la boundary'              | boundary       |
    |other                | church st, caerau, maesteg | minor road     |
    |no alphanumeic text  | "    "                     | unclassified   | 

    
    works out a type of junction for each junction
    in a list
    
    Arguments:
        road_names (Series): the road names 
    Returns
        road_type(list) : road types
    
    """
    #get rid of NaNs etc
    null_mask = road_names.isnull()
    road_names[null_mask] = "" # so they are entered as unclassified
    
    # classifications of junctions 
    road_type = ["motorway" if str(road).startswith("m")
                 else "a road" if str(road).startswith("a") 
                 else "b road" if str(road).startswith("b") 
                 else "boundary" if "la boundary" in str(road)
                 else "minor road" if (re.search("[a-zA-Z]", str(road)) is not None)                         
                 else "unclassified" 
                 for road in road_names.str.lower().tolist() 
                ]    
    
    return road_type

#%%
    
def classify_junctions(loc,  
                       file_regex = ".*traffic_gdp_[0-9]",
                       ext = "csv",
                       messages = False,
                       a_col = "a-junction",
                       b_col = "b-junction",
                       cat_col = "rcat",
                       road_col = "road",
                       cat_suffix = "_cats",
                       class_suffix = "_class",
                       col_dict = {"level_0": "road_type",
                                   "level_1" : "observation_no"
                                   }
                      ):

    """classify_junctions
    
    creates features to indicate road type of the roads that intersect at
    the junction where the observation was made.
    
    loc (string): folder location with the datafiles inside
    file_regex (string): regex expression to select the files to load
    ext (string): extension of the files to load
    messages(bool): indicate to print progress messages
    a_col (string): column in where a junction is stored
    b_col (string): column in where b junction is stored
    cat_col (string): column in where road category is stored
    road_col (string): column in where road name where the observation was made
        is stored
    cat_suffix (string): suffix to add to a_col and b_col to give the column
        name for the categories for each of those
    class_suffix (string): suffix to add to a_col and b_col to give the column
        name for the categories for each of those
    col_dict (dict): dictionary for renaming columns
    

    """        
    #list of files to load
    
    files = dat.files_looks_like(loc, looks_like = file_regex, ext = ext)
    
    for file in files:
        if messages: print("loading file %s" %file)
        df = pd.read_csv(file)
        
        if messages: print("tweaking column names")
        df.rename(columns = col_dict, inplace = True)
        #categories
        if messages: print("finding junction categories")
        df[a_col + cat_suffix] = get_road_cats(df, road_col, a_col, cat_col)
        df[b_col + cat_suffix] = get_road_cats(df, road_col, b_col, cat_col)
        #classes
        
        if messages: print("finding junction classes")
        df[a_col + class_suffix] = classify_road(df[a_col])
        df[b_col + class_suffix] = classify_road(df[b_col])
        
        if messages: print("saving")
        df.to_csv(file)
        if messages: print("finished") 
        
  