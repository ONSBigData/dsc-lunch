# -*- coding: utf-8 -*-
"""dataset_processing

Created on Tue Aug  8 11:13:20 2017

@author: Edward Rowland

A module containing functions for processing data for the
GDP_nowcasting project

"""
#%%
import os
import gc
import warnings
import logging
import re
import glob

import pandas as pd
import numpy as np
import geopandas as gpd
import shapely as sh
import itertools as it

from dateutil.relativedelta import relativedelta
#%%
def traffic_munging(df,
           val_str = "fd",
            val_name = "aadf",
            melt = False):

    """traffic_munging
    
    Data muning for the traffic data, converts column names to lower case
    set id and value columns of the dataframe and melts the dataframe to tall
    data format
    
    Arguments: 
        df (dataframe): the traffic data
        val_str (string): this is the column in df that contains values for
                          traffic flow
        val_name (string): the name to give the traffic flow values column in
                           the melted dataset
        melt (bool): indicate whether or not to melt the dataset
        
    Returns: 
        df (dataframe): the processesd traffic data
    """

    #this frustrates me
    df.columns = [str.lower(name) for name in df.columns]
    #these are the value columns
    val_mask = [str.lower(val_str) in col for col in df.columns]
    val_cols = df.columns[val_mask]

    #cols that are not the value columns are the id cols
    id_cols = df.columns[np.invert(val_mask)]

    #to tall
    if melt:
        df  = pd.melt(df,
                       id_vars = id_cols,
                       value_vars = val_cols,
                       var_name = "vehicle_type",
                       value_name = val_name)


    return df

    #%%
def load_traffic_data(
        proj_dir = "/home/eddr/Documents/Projects/GDP_nowcasting",
        traffic_loc = "/data/traffic/",
        major_roads = "AADF-data-major-roads.csv",
        minor_roads = "AADF-data-minor-roads.csv",
        year = 2015,
        key = ["major", "minor"],#year to get data for
        e_n_col = "east_north",
        e_col = "s ref e",
        n_col = "s ref n",
        lat_long_col = "lat_long",
        lat_col = "s ref latitude",
        long_col = "s ref longitude"
                      ):
    
    """load_traffic_data
    
    function that loads the aadf data from the dft and sticks it all together.
    calls traffic_munging
    
    Arguments:
        proj_dir (str): directory of the project
        traffic_loc (str): folder with the traffic data
        major_roads (str): name of the major roads aadf data file
        minor_roads (str): name of the minor roads aadf data file
        year (int): year of the data we want
        key (list): a list of length 2 with a reference to the two data files
        e_n_ocl (str): name of the column in traffic_df that will contain 
                       the easting and northing geospatail point data
        e_col (str): column in traffic_df containing the easting co-ord
        n_col (str): column in traffic_df containing the northing co-ord
        lat_long_col (str):  name of the column in traffic_df that will contain 
                             the latitude and longitude geospatial point data
        lat_col (str): column in traffic_df containing the latitude co-ord
        long_col (str): column in traffic_df containing the longitude co-ord
        
    Returns:
        traffic_df (dataframe): contains traffic data from both the major and
                                minor roads datasets, processed from the year
                                specified.
    """
    
    #load data
    os.chdir(proj_dir)
    major_roads_df = pd.read_csv(os.getcwd() + traffic_loc + major_roads)
    minor_roads_df = pd.read_csv(os.getcwd() + traffic_loc + minor_roads)

    # combine files
    traffic_df = pd.concat([major_roads_df, minor_roads_df],
                           keys = key)

    # mung
    traffic_df = traffic_munging(traffic_df)
    mask = traffic_df.aadfyear == year
    # we only take the most recent year as dataset is large, like will
    # comfortably bork 16Gb of ram large
    traffic_df = traffic_df[mask]
    traffic_df.head()

    #create gis spatial point fields
    traffic_df[lat_long_col] = [sh.geometry.Point(lat, long)
                                for lat, long in zip(traffic_df[lat_col],
                                                     traffic_df[long_col])
                               ]
    #easting and northing
    traffic_df[e_n_col] = [sh.geometry.Point(east, north)
                           for east, north in zip(traffic_df[e_col],
                                                  traffic_df[n_col])
                          ]
    #tidy mem
    del major_roads_df, minor_roads_df
    gc.collect

    return traffic_df

#%%   
def load_shapefile(shp_lookup,
                   shp_col, 
                   nuts_dir = "/data/geographic/shapefiles/NUTS/",
                   year = 2015,
                   year_col = "years",
                   logger = logging.getLogger(__name__)
                  ):
    """load_shapefile
    Load a shapefile with name stored in a dataframe column for a particular 
    year, also in a column within a directory.
    
    Arguments:
        shp_lookup (dataframe): contains the names of shapefiles by year
        shp_col (str): name of the column containing the name of the datafile
        nuts_dir (str): directory where the shapefile is kept
    Returns:
        shp (GeoDataFrame): contains the shapefile, is empty if no shapefile
                            is found
    Warnings: 
        OSError : will return shp as an empty GeoDataFrame
        TypeError: as OSError
    """
    #get the filename
    year_mask = shp_lookup[year_col] == year
    shp_filename  = shp_lookup.loc[year_mask][shp_col].values[0]
    
    #try and load it
    try:
        print(os.getcwd() + nuts_dir + str(shp_filename))
        shp = gpd.read_file(os.getcwd() + nuts_dir + str(shp_filename))
    #returns a black shapefile an
    except (OSError, TypeError) as e:
        shp = gpd.GeoDataFrame()
        logger.warning("Shapefile name or directory for: " +
                       os.getcwd() +
                       nuts_dir +
                       str(shp_filename) + 
                       " not found, empty GeoDataFrame returned")
        
    return shp
#%%
def mung_shapefiles(gdf,
                    column_replace = {"nuts2_name" : ["nuts218nm", 
                                      "nuts215nm"
                                     ],
                      "nuts3_name" : ["nuts318nm",
                                      "nuts315nm"
                                     ],
                      "nuts_code" : ["nuts218cd", 
                                     "nuts215cd",
                                     "nuts318cd",
                                     "nuts315cd",
                                     "eurostat_n"
                                    ],
                      "shape_length" : ["SHAPE_Leng", 
                                        "st_lengths"
                                       ],
                      "shape_area" : ["st_areasha"]
                     }
                    ):

    """mung_shapefiles
    
    processing the shapefiles so they merge/append etc. better by homogenising 
    column names 
    
    Arguments:
        gdf (geodataframe): data loaded from a shapefile
        column_replace (dict): key/value pairs where the value is a list of
        strings that if they are a name of a column in gdf are replaced by 
        the key
    Return:
        gdf (geodataframe) : shapefile data with homogenised column names
    
    """
    
    #lowercase column names
    gdf.columns = [str.lower(name) for name in gdf.columns]
    
    #lowercase dict
    column_replace = dict((key.lower(), 
                           [value.lower() for value in values] #we expect values in dict are lists of strings
                          )
                          for key, values in column_replace.items())
    
    # search the through the dict of column names to replace 
    for new_name, old_names in column_replace.items():
        
        #rename columns that need replacing
        gdf.columns = [str.lower(new_name) 
                       if column_name in old_names 
                       else column_name 
                       for column_name in gdf.columns.tolist()]
        
    return gdf
 
#%%    
def merge_traffic_with_nuts(shapefile_lookup,
                     proj_dir = "/home/eddr/Documents/Projects/GDP_nowcasting",
                     traffic_loc = "/data/traffic/",
                     working_dir = "/working/data/",
                     major_roads = "AADF-data-major-roads.csv",
                     minor_roads = "AADF-data-minor-roads.csv",
                     columns = ["nuts2_ew", 
                                "nuts2_sco",
                                "nuts3_ew",
                                "nuts3_sco"
                               ],
                     nuts_levels= ["nuts2",
                                   "nuts3"
                                  ],
                     df_suffix = "traffic",
                     geo_col = "east_north",
                     lookup_years_col = "years",
                     e_col = "s ref e",
                     n_col = "s ref n",
                     lat_long_col = "lat_long",
                     lat_col = "s ref latitude",
                     long_col = "s ref longitude",
                     crs_df = 0,
                     logger = logging.getLogger(__name__)
                     ):
    
    """merge_traffic_with_nuts
       
    This takes the files for each year with aadf traffic dataset and then finds
    the NUTS regions that the observation point sits in, then merges the 
    details of the NUTS region with that observation using a spatial join.
    Outputs the resulting merged df as a .csv in the working directory    
    
    Arguments:
        shapefile_lookup (Dataframe): contains the names of the shapefiles
            to merge onto the traffic data
        proj_dir (string): the project directory where the folder containing
            traffic is stored
        traffic_loc (string): folder within proj_dir where the traffic data is
        working_dir (string): folder where the working data files are stored
        major_roads (string): name of the file containing aadf observations for
            major roads
        minor_roads (string): name of the file containing aadf observations for
            minor roads
        columns (list): columns in shape_lookup that contain shapefile names
        nuts_levels (list): names of the different NUTS levels
        df_suffix (string): to append to column names from the traffic data 
            after the merge
        geo_col (string): column in traffic data to put the spatial point 
            easobject created from easting and northing co-ords
        e_col (string): column that contains the easting co-ord in traffic data
        n_col (string): column that contains the northing co-ord in traffic 
            data
        lat_long_col(string) column in traffic data to put the lattitude and
            longitude spatial point bject created from co-ords
        lat_col (string): column that contains the latitude co-ord in traffic 
            data
        long_col (string): column that contains the longitude co-ord in traffic 
            data
        crs_df (int): location of the crs data to copy from the shapefile over
                        to the merged traffic/nuts merged geodataframe,
        logger (logging.logger): logger for logging warnings, errors etc.
        
    Warnings:
        IOError : if the merged dataframe cannot be created and saved into
            a csv file
    """
    #get min and max years...
    min_year = min(shapefile_lookup[lookup_years_col])
    max_year = max(shapefile_lookup[lookup_years_col])+1
    
    #...and iterate through them#
    for year in range(min_year, max_year):
        #load the traffic dataset
        traffic_df = load_traffic_data(year = year,
                                       proj_dir = proj_dir, 
                                       traffic_loc = traffic_loc,
                                       major_roads = major_roads,
                                       minor_roads = minor_roads,
                                       e_n_col = geo_col,
                                       e_col = e_col,
                                       n_col = n_col,
                                       lat_long_col = lat_long_col,
                                       long_col = long_col
                                      )
    
        #load the relevant shapefiles
        shp_lst = [load_shapefile(shp_lookup = shapefile_lookup, 
                                  shp_col = col, 
                                  year = year
                                 )
                   for col in columns
                  ]
        
        #mung column names before merge
        shp_lst = [mung_shapefiles(shp_file) for shp_file in shp_lst]
        #we need a geographic dataframe
        gdf = gpd.GeoDataFrame(traffic_df,
                               crs = shp_lst[crs_df].crs,
                               geometry = geo_col
                              )
        gdf.reset_index(inplace = True)
        
        # each nuts level
        for level in nuts_levels:
            #get the relevant shape files for that nuts level
            nuts_mask = [level in col for col in columns]
            nuts_list = list(it.compress(shp_lst, nuts_mask))
            #combine shapefiles into one
            nuts_shp = pd.concat(nuts_list)
            #spatial merge these with traffic dataset 
            gdf = gpd.sjoin(gdf, 
                             nuts_shp, 
                             how = "left", 
                             op = "within", 
                             rsuffix = level)

        #once merged with nuts, output the dataset
        try:
            file = os.getcwd() + working_dir + df_suffix + str(year)
            gdf.to_csv(file + ".csv")
        except IOError:
            logger.warning("Could not create " + file, ", skipping")

            
#%%

def merge_regions(df, 
                  nuts2_shp, 
                  nuts3_shp, 
                  geo_col = "east_north", 
                  df_suffix = "traffic"
                 ):
    
    """merge_regions
    Function that merges a pandas dataframe containing geospatail points
    with the descriptive data (e.g region name) from the nuts2 and nuts3
    shapefiles.
    
    Arguments:
        df (dataframe): containing data (e.g from load_traffic_data) to merge
        nuts2_shp (geodataframe): nuts2 region shapefile
        nuts3_shp (geodataframe): nuts3 region shapefile
        geo_col (str): column name in df that contains the spatial point data
        df_suffix (str): to add to columnnames in df in returned geodataframe
        
    Returns:
        gdf (geodataframe): made up of df merged with nuts2 and nuts3 
                                shapefiles data
                                
    Warnings:
    """
    if (nuts2_shp is not None): 
        #create a geopandas dataframe and remove index as merge doesn't like
        #multiindex dataframes
        
        gdf = gpd.GeoDataFrame(df, crs = nuts2_shp.crs, geometry = geo_col)
        gdf.reset_index(inplace = True)
        #return the geodataframe merged with the shapefile region data by spatial
        #point
        gdf =  gpd.sjoin(gdf, 
                         nuts2_shp, 
                         how = "left", 
                         op = "within", 
                         lsuffix = df_suffix, 
                         rsuffix = "nuts2")
    else:
        warnings.warn("nuts2_shp is empty, unable to merge NUTS2 regions")
    
    if (nuts3_shp is not None):    
        
        #if gdf hasn't been created, i.e nuts2_shp has not been passed
        try:
            gdf
        except NameError:
            gdf = gpd.GeoDataFrame(df, crs = nuts2_shp.crs, geometry = geo_col)
        
            
        gdf =  gpd.sjoin(gdf, 
                             nuts3_shp, 
                             how = "left", 
                             op = "within", 
                             rsuffix = "nuts3")
    else:
        warnings.warn("nuts3_shp is empty, unable to merge NUTS3 regions")

    try :
        gdf
    except NameError:
        warnings.warn("Did not merge any NUTS level regions")
        return None
    
    return gdf

#%%
def load_gdp(file, 
             area, 
             loc = "/data/GDP/", 
             geo_col = "geo",
             time_col = "time",
             format_cols = True):
    
    """load_gdp
    loads and does a bit of cleaning of gsp_data
    
    Arguments:
        loc (string): location of the file to be loaded.
        file (string): name of the file to be loaded.
        area (string): NUTS level the dataset is set at
        geo_col (string): column in file containing geograpical area name    
        time_col (string): column containing data on the time the data refers
            to
        format_cols (bool): indicates if we need to format the column names
    Returns:
        gdp_df (dataframe): gdp data for nuts regions 
    """
    
    gdp_df  = (pd.read_csv(os.getcwd() + loc + file, 
                                 encoding = "latin-1"))
    
    # lowercase for consitency/easy referencing
    gdp_df.columns = [str.lower(name) for name in gdp_df.columns]

    
    if format_cols:
        #set location area type
        gdp_df[geo_col] = gdp_df[geo_col].str.lower()
   
        #more sensible column names
        gdp_df.rename(columns = {"geo" : area,
                                 "value" : area + "_value",
                                 "unit" : area + "_unit",
                                 "flag and footnotes" : area 
                                                        + "_flag and footnotes"
                                 },
                              inplace = True
                     )   
        
    gdp_col = area + "_value"
    #formatting
    gdp_df = string_to_num(gdp_df, [gdp_col])
    gdp_df[time_col] = pd.to_datetime(gdp_df[time_col], format = "%Y")
    
    #get gdp change measures
    gdp_df = get_growth(df = gdp_df,
                        area_col = area,
                        gdp_col = area + "_value".capitalize(),               
                        time_col = "time" ,
                        ratio_col = area + "_gdp_ratio",
                        growth_col = area + "_growth"
                       ) 
    
    return gdp_df    
#%%
def merge_traffic_with_gdp(traffic_df, 
                           nuts2_gdp_df,
                           nuts3_gdp_df,
                           nat_gdp_df,
                           file_loc = "/working/data/",
                           traffic_year_col = "aadfyear",
                           traffic_nuts2_col = "nuts2_name",
                           traffic_nuts3_col = "nuts3_name",
                           gdp_year_col = "time",
                           nat_year_col = "year",
                           gdp_nuts2_col = "nuts2",
                           gdp_nuts3_col = "nuts3",
                           gdp_nuts2_val_col = "nuts2_value",
                           gdp_nuts3_val_col = "nuts3_value"
                          ):
    
    """merge_traffic_with_gdp
    
    merges traffic data that has been labelled with the correct data nuts 
    regions with the gdp for those regions
    
    Arguments:
        traffic_df (DataFrame): contains the traffic data
        nuts2_gdp_df (DataFrame): nuts2 regions gdp data
        nuts3_gdp_df (DataFrame): nuts2 regions gdp data
        national_gdp_df (DataFrame): national gdp data
        traffic_year_col (string): column in traffic_df that indicates what
                                   year the data corresponds to
        nat_year_col (string): column in nat_gdp_df that indicates what
                                   year the data corresponds to
        traffic_nuts2_col (string): column indicating nut2 region observation
                                    is in, in traffic_df
        traffic_nuts3_col (string): column indicating nut3 region observation
                                    is in, in traffic_df
        gdp_year_col (string): column in nuts2_gdp_df and nuts3_gdp_df where
                                year is stored
        gdp_nuts2_col (string): column for nuts2 region in nuts2_gdp_df
        gdp_nuts3_col (string): column for nuts3 region in nuts3_gdp_df
    """
    
    #lower case for consitency
    traffic_df[traffic_nuts2_col] = [str(name).lower()
                            for name in traffic_df[traffic_nuts2_col].tolist()
                                    ]
    traffic_df[traffic_nuts3_col] = [str(name).lower()
                            for name in traffic_df[traffic_nuts3_col].tolist()
                                    ]
    
    #one merge for each nuts region
    if nuts2_gdp_df is not None:
        traffic_gdp_df = pd.merge(left = traffic_df, 
                                  right = nuts2_gdp_df,
                                  left_on = [traffic_year_col, 
                                             traffic_nuts2_col
                                            ],
                                  right_on = [gdp_year_col ,
                                              gdp_nuts2_col
                                             ],
                                  how = "left"
                                 )
        
    if nuts3_gdp_df is not None:   
        traffic_gdp_df = pd.merge(left = traffic_gdp_df, 
                                  right = nuts3_gdp_df,
                                  left_on = [traffic_year_col,
                                             traffic_nuts3_col
                                            ],
                                  right_on = [gdp_year_col, 
                                              gdp_nuts3_col
                                             ],
                                  how = "left"
                                 )
        
    if nat_gdp_df is not None:   
        traffic_gdp_df = pd.merge(left = traffic_gdp_df, 
                                  right = nat_gdp_df,
                                  left_on = [traffic_year_col
                                            ],
                                  right_on = [nat_year_col
                                             ],
                                  how = "left"
                                 )
    
    #assume first in this unique col is the year
    years = pd.DatetimeIndex(traffic_gdp_df[ traffic_year_col])
    
    year = years.year.unique()[0]
    
    

    filename = "traffic_gdp_" + str(year) + ".csv"
        
    #gdp values are acutally given as strings, so change
    traffic_gdp_df = string_to_num(traffic_gdp_df, 
                                   [gdp_nuts2_val_col, gdp_nuts3_val_col]
                                   )
    
    #save as a working data file
    traffic_gdp_df.to_csv(os.getcwd() + file_loc + filename)
    
    return traffic_gdp_df
    
#%%

def merge_nuts_regions(df,
                       df_suffix = "traffic", 
                       proj_dir = "/home/eddr/Documents/Projects/GDP_nowcasting",
                       shape_loc = "/data/geographic/shapefiles/",
                       nuts2_loc = "nuts2_2018/",
                       nuts3_loc = "nuts3_2018/",
                       nuts2_file = "NUTS_Level_2_January_2018_Full_Clipped_Boundaries_in_the_United_Kingdom.shp",
                       nuts3_file = "NUTS_Level_3_January_2018_Full_Extent_Boundaries_in_the_United_Kingdom.shp",
                       geo_col = "east_north"
                      ):
    

    """merge_nuts_regions
    
    Basically a wrapper for merge_regions that loads in the nuts shapefiles
    
    arguments:
        df (dataframe): containing data (e.g from load_traffic_data) to merge
        df_suffix (str): to add to the column names that were originally in df
        shape_loc (str): directory of the shapefiles
        nuts2_loc (str): folder for the nuts2 shapefile
        nuts3_loc (str): folder for the nuts3 shapefile
        nuts2_file (str): nuts2 shapefile name
        nuts3_file (str): nuts3 shapefile name
        
    returns:
        merge_regions output (geodataframe): made up of df merged with nuts2 
                                             and nuts3 shapefiles data
    """
    
    nuts2_shp = gpd.read_file(proj_dir + shape_loc + nuts2_loc + nuts2_file)
    nuts3_shp = gpd.read_file(proj_dir + shape_loc + nuts3_loc + nuts3_file)

    return merge_regions(df, nuts2_shp, nuts3_shp, geo_col, df_suffix)

#%%
    
def get_map(maps_df,
            countries_col = "countries", 
            country_dict = {"uk" : ["united_kingdom"],
                            "eng_wal" : ["england_wales"],
                            "sco" : ["scotland"]},
            shapefile_col = "shapefiles",
            filename_col = "filename",
            nuts_col = "nuts_level",
            year_col = "year"
           ):


    #is there a shapefile with the whole of the uk?
    if any(name in maps_df[countries_col] for name in country_dict["uk"]):
        
        #ignore all that aren't a nuts2 uk shape file predating the dataset
        uk_mask = [country in country_dict["uk"]
                   for country in maps_df[countries_col]
                  ]
        
        #get the map
        uk_map = get_most_recent_map(maps_df[uk_mask])
    
    #there isn't a shape file with the whole of the uk
    else:
        #find the england and wales shapefile
        eng_wal_mask = [country in country_dict["eng_wal"]
                        for country in maps_df[countries_col]
                       ]
        eng_wal_map = get_recent_map(maps_df[eng_wal_mask])
        
        #find scotland shapefile
        sco_mask = [country in country_dict["sco"]
                    for country in maps_df[countries_col]
                   ] 
        sco_map = get_recent_map(maps_df[sco_mask])
        
        #merge the two maps, just like in 1707
        uk_shp = eng_wal_map[shapefile_col].append(sco_map)
        
        uk_map = pd.Dataframe(
            {filename_col : {"eng_wal_map" : eng_wal_map[filename_col],
                             "sco_map" : sco_map[filename_col]
                            },
             nuts_col : eng_wal_map[nuts_col],
             year_col : {"eng_wal_map_year" : eng_wal_map[year_col],
                         "sco_map_year" : sco_map[year_col]},
             shapefile_col : uk_shp            
            }
        )
    return uk_map
#%%
    
def merge_all_traffic_gdp(nuts2_gdp_df,
                          nuts3_gdp_df,
                          nat_gdp_df,
                          traffic_loc,
                          file_loc = "/working/data/",
                          traffic_year_col = "aadfyear",
                          traffic_nuts2_col = "nuts2_name",
                          traffic_nuts3_col = "nuts3_name",
                          gdp_year_col = "time",
                          nat_year_col = "year",
                          gdp_nuts2_col = "nuts2",
                          gdp_nuts3_col = "nuts3",
                          gdp_nuts2_val_col = "nuts2_value",
                          gdp_nuts3_val_col = "nuts3_value"
                         ):
    """merge_all_traffic_gdp
    
    merges all traffic data files in the given folder, that have been labelled 
    with the correct nuts regions, with the gdp for those regions
    
    Arguments:
        nuts2_gdp_df (DataFrame): nuts2 regions gdp data
        nuts3_gdp_df (DataFrame): nuts2 regions gdp data
        national_gdp_df (DataFrame): national gdp data
        traffic_loc (string): folder path for where all the traffic data is 
                              stored
        traffic_year_col (string): column in traffic_df that indicates what
                                   year the data corresponds to
        traffic_nuts2_col (string): column indicating nut2 region observation
                                    is in, in traffic_df
        traffic_nuts3_col (string): column indicating nut3 region observation
                                    is in, in traffic_df
        gdp_year_col (string): column in nuts2_gdp_df and nuts3_gdp_df where
                                year is stored
        nat_year_col (string): column in nat_gdp_df that indicates what
                           year the data corresponds to
        gdp_nuts2_col (string): column for nuts2 region in nuts2_gdp_df
        gdp_nuts3_col (string): column for nuts3 region in nuts3_gdp_df
    """

    #find files that look like traffic data
    traffic_data_files = files_looks_like(traffic_loc)

    #load each file and merge with gdp
    for file in traffic_data_files:    
        traffic_df = pd.read_csv(file)
        #format time
        traffic_df[traffic_year_col] = pd.to_datetime(
                                                traffic_df[traffic_year_col], 
                                                format = "%Y"
                                                     )
    
        merge_traffic_with_gdp(traffic_df, 
                               nuts2_gdp_df = nuts2_gdp_df,
                               nuts3_gdp_df = nuts3_gdp_df,
                               nat_gdp_df = nat_gdp_df,
                               file_loc = file_loc,
                               traffic_year_col = traffic_year_col ,
                               traffic_nuts2_col = traffic_nuts2_col,
                               traffic_nuts3_col = traffic_nuts3_col,
                               gdp_year_col = gdp_year_col,
                               nat_year_col = nat_year_col,
                               gdp_nuts2_col = gdp_nuts2_col,
                               gdp_nuts3_col = gdp_nuts3_col
                              )

#%% 

def files_looks_like(loc, looks_like = ".*traffic[0-9]", ext = "csv"):
    
    """file_looks_like
    returns files in a folder of a specified extenstion that has a filename 
    like a partiular reg expression
    
    Arguments:
        loc (string): path of the folder where the files are stored
        looks_like (string): regex to match the filename with
        ext (string): extention of the files to match e.g csv
    
    Returns:
        matches (list): the files that match the regex with the extension.
        
    
    """
    files = glob.glob(loc + "/*" + ext)
    mask = [bool(re.match(looks_like, file)) for file in files]
    matches = list(it.compress(files,mask))
    
    return matches

#%% 

def drop_duplicate_rows(df,
                        col_to_del = "Unnamed"):
    
    """drop_duplicate_rows
    
    Sometimes in data munging the rows get repeated with some extra 
    unwanted columns with names like "Unnamed: 1". This solves that issue
    
    Arguments:
        df (DataFrame): dataframe with the repeating rows
        col_to_del (string): name of column or word in column names we need 
            to get rid of
    
    Returns: 
        df (Dataframe): dataframe without repeating rows in fixed, incremental
            index
    """
    
    #drop these unneeded cols and rows
    drop_mask = [col_to_del in col for col in df.columns.tolist()]
    drop_col = df.columns[drop_mask].tolist()
    df.drop(drop_col, inplace = True, axis = 1)
    df.drop_duplicates(inplace = True)
    
    #our index is broken, so we fix..
    df.reset_index(drop = True, inplace = True)
    
    return df
    
#%%
def string_to_num(df, cols):
    
    """string_to_num - GDP and traffic flow correlations over time
    
    converts all values in a column from string to numerical values
    
    Arguments: 
        df (DataFrame): data to convert to string
        cols (list): list of columns in df for conversion
    
    Return:
        df (DataFrame): now with specified columns as a string
    """
    #each of the chosen columns in df...
    for col in cols:
        #...format and change their type to numeric
        if pd.api.types.is_string_dtype(df[col]):
            df[col]  = df[col].str.replace("\D+","")
            df[col] = pd.to_numeric(df[col])
    
    return df
#%%
def load_all_traffic_data(loc, 
                          load_columns = ["road", "rcat"],
                          regex = ".*traffic_gdp_[0-9]",
                          ext = "csv"
                         ):
    
    files = files_looks_like(loc = loc, looks_like = regex, ext = ext)

    df_lst = [pd.read_csv(file, usecols = load_columns) 
              for file in files
             ]
    df = pd.concat(df_lst, axis = 0)
    
    return df
    
#%%
   
def get_growth(df,
               area_col,
               gdp_col,
               messages = False,
               time_col = "time",
               ratio_col = "gdp_ratio",
               growth_col = "growth"
               ):
    
    """get_growth
    
    Calculates gdp growth for this year's figure based on the previous year.
    
    Arguments:
        df (Dataframe): GDP data by nuts region.
        area_col (string): column in df where the nuts area name is stored
        gdp_col (string): Column in df where the gdp value is stored
        messages (bool): Print out messages?
        time_col (string): Column in df giving the year of the gdp figure.
        ratio_col (string): Column in df  to store the gdp ratio from the 
            previous year.
        growth_col (string): Column in df to store the gdp growth from the 
            previous year.
        
    Returns: 
        df (DataFrame): As passed, except with two new columns named as string
            in ratio_col and growth col with gdp ratio and gdp growth for that
            region based on last year
    """
    gdp_ratio = list()
    growth = list()

    for index,row in df.iterrows():
        
        if messages: print(row)
        
        prev_year = row[time_col] - relativedelta(years = 1)
        mask = (df[area_col] == row[area_col]) & (df[time_col] == prev_year)
        
        #is there any previous year's data for this region we can look at?
        if not df[mask][gdp_col].unique().any():
            #if there isn't any, we can't calc growth figures
            gdp_ratio.append(np.nan)
            growth.append(np.nan)
        else:
            #calc growth figures
            prev_gdp = df[mask][gdp_col].unique()[0]
            ratio = row[gdp_col]/prev_gdp
            gdp_ratio.append(ratio)
            growth.append((ratio - 1)*100) # % growth rate
            
        if messages: 
            print("prev_gdp: %s" %prev_gdp)
            print("gdp ratio: %s" %gdp_ratio)
            print("growth rate: %s" %growth)

    df[ratio_col] = gdp_ratio
    df[growth_col] = growth
    
    return df
    
#%%
def load_stats(loc = os.getcwd() + "/working/data/",
               name = "traffic_stats.csv",
               year_col = "year",
               year_label_col = "year_label",
               format_year = True
               ):
    """load_stats
    
    loads the .csv file with descriptive statistics by nuts region 
    and nationally
    
    Arguments:
        loc (string): path to where the file is stored        
        file (string): name of the file
        year_col (string): column in df that contains year the data was 
            collected
        year_label_col (string); Column in df where the year is shown as an
            integer, useful for plotting.
        format_year (bool): format year or not
    Returns: 
        df (Dataframe): Contains the descriptive statistics 
    """
    
    df = pd.read_csv(loc + name)

    if format_year:
        #format year
        df[year_col] = pd.to_datetime(df[year_col], format = "%Y-%m-%d")
        #int for plotting
        df[year_label_col] = [int(date.year) for date in df[year_col]]
    
    return df

#%%
def unmelt_by_col(df,
                  key_col,
                  value_cols,
                  validation = "one_to_one",
                  messages = False 
                 ):
    
    """unmelt_by_col
    
    Takes a dataframe in tall format with a key column and multiple value 
    columns and turns it into wide format.
    All other columns not in key_col or value_col are assumed to be index
    columns
    
    Arguments:
        df (DataFrame): data in tall format
        key_col (string); column in df that stores the keys
        value_cols (list): columns in df that contain the values
        validation (string): validation to perfom on the unmelted data; check
            pandas documentation for mergeing for details or set this as None
            to avoid. 
        messages (bool): print messages, useful for debugging
            
    Returns: 
        unmelted_df (DataFrame): df in wide format
    """
    
    #the unique keyd we want to unmelt by
    items = df[key_col].unique()

    #get the first data frame containing only the first key value
    unmelted_df = df[df[key_col] == items[0]]

    #now get each other dataframe containing only each key value to merge   
    for index, item in enumerate(items[1:]):
        if messages:
            print("item: %s" %item)
            print("item by index : %s" %items[index])
            print("index: %s" %index)
        #subset the data
        mask = df[key_col] == item
        #merge this subset with the rest
        unmelted_df = pd.merge(left = unmelted_df, 
                               right = df[mask],
                               how = "outer",
                               on = value_cols,
                               suffixes = ["_" + items[index],
                                           "_" + item
                                           ],
                               validate = validation
                              )
    return unmelted_df
