#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:26:38 2025

@author: ghielmin
"""


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from datetime import datetime
os.chdir('/home/ghielmin/Desktop/snowpat-main/')
import snowpat.pysmet as smet
os.chdir('/home/ghielmin/Desktop/data_thesis')


def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def calculate_elevation_difference(alt1, alt2):
    return abs(alt1 - alt2)


# List of file names to process
files_to_process = [
    "snowpack/WFJ2/output/WFJ2_WFJ2_MS_SNOW.smet",
    "snowpack/WFJ2/output/WFJ2_WFJ2_MS_SNOW_excludePSUM.smet",
    "snowpack/WFJ2/output/WFJ2_WFJ2_MS_SNOW_unheatedgauges.smet"
]

file_names = [os.path.basename(file_path) for file_path in files_to_process]
name = 0
# Loop through each file and process it
for file in files_to_process:
    # Import IMIS data and read smet file
    imis_stn = "WFJ2"
    imis_station = smet.read(file)
    imis_data_pandas = imis_station.data
    imis_data_numpy = imis_station.toNumpy()
    imis_station_id = imis_station.meta_data.station_id
    imis_lon = imis_station.meta_data.location.longitude
    imis_lat = imis_station.meta_data.location.latitude
    imis_alt = imis_station.meta_data.location.altitude
    
    folder_path = 'mch/SMET/'
    all_files = glob.glob(os.path.join(folder_path, "*.smet"))

    radius = 35
    selected_stations = []
    swissmetnet_data = []

    # Define the stations within a radius of 35 km around the IMIS station we want to analyze
    for file in all_files:
        file = smet.read(file)
        data_pandas = file.data
        data_numpy = file.toNumpy()
        station_id = file.meta_data.station_id
        lon = file.meta_data.location.longitude
        lat = file.meta_data.location.latitude
        alt = file.meta_data.location.altitude

        dist = calculate_distance(imis_lat, imis_lon, lat, lon)
        elev_diff = calculate_elevation_difference(imis_alt, alt)

        if dist <= radius:
            selected_stations.append(station_id)

            station_info = {
                'station_id': station_id,
                'longitude': lon,
                'latitude': lat,
                'altitude': alt,
                'distance_to_imis': dist,
                'elevation_diff': elev_diff,
                'precipitation': data_pandas['PSUM'].values
            }

            swissmetnet_data.append(station_info)

    swissmetnet_df = pd.DataFrame(swissmetnet_data)
    swissmetnet_dataset_sorted = swissmetnet_df.sort_values(by='elevation_diff', ascending=True)

    # Now select the top 5 stations based on smallest elevation difference
    top_stations = swissmetnet_dataset_sorted.head(5)

    list_stations = top_stations.station_id

    ##########################################
    r = 1
    list_r = []

    imis_data_pandas['timestamp'] = pd.to_datetime(imis_data_pandas['timestamp'])

    imis_data_pandas['Precip'] = imis_data_pandas['MS_Snow'] + imis_data_pandas['MS_Rain']
    
    # Step 1: Create a column for 6-hour intervals
    imis_data_pandas.set_index('timestamp', inplace=True)
    start_date = imis_data_pandas.index.min()  # Start date of IMIS data
    end_date = imis_data_pandas.index.max()
    
    imis_data_pandas_hourly = imis_data_pandas.copy()

    # Resample the data to 6-hour periods and calculate the cumulative sum
    imis_data_pandas['Cumulative_Precip'] = imis_data_pandas['Precip'].resample('6H').sum()
    
    imis_data_pandas = imis_data_pandas[imis_data_pandas['Cumulative_Precip'] > 0.1]

    for i in list_stations:
        # Read SwissMetNet data for the current station
        mch = pd.read_csv(
            f"mch/SMET/{i}.smet", skiprows=14, sep='\s+')

        if i == 'ZER':
            mch.columns = ['timestamp', 'PSUM']
        else:
            mch.columns = ['timestamp', 'PSUM']

        # Replace -999 with 0 (missing data handling)
        mch.loc[mch['PSUM'] == -999, 'PSUM'] = 0

        # Convert timestamp to datetime
        mch['timestamp'] = pd.to_datetime(mch['timestamp'])
        
        mch.set_index('timestamp', inplace=True)
        mch = mch.loc[start_date:end_date]
        mch_hourly = mch.copy()

        # Resample the data to 6-hour periods and calculate the cumulative sum
        mch['Cumulative_PSUM'] = mch['PSUM'].resample('6H').sum()
        
        #mch = mch[mch['Cumulative_PSUM'] > 0.1]

        mch['Cumulative_PSUM' + str(r)] = mch['Cumulative_PSUM']
        
        mch = mch.reset_index()

        # Merge the IMIS data with the precipitation data from the current station
        imis_data_pandas = pd.merge(
            imis_data_pandas, mch[['timestamp', 'Cumulative_PSUM' + str(r)]], on='timestamp', how='inner')
        #imis_data_pandas.fillna(0, inplace=True)

        # Prepare data for linear regression
        X = imis_data_pandas[['Cumulative_PSUM' + str(r)]].values.reshape(-1, 1)
        y = imis_data_pandas['Cumulative_Precip']  # Target variable (IMIS precipitation)

    
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions for the IMIS station
        predictions = model.predict(X)

        # Calculate R2 score for the model
        r2 = r2_score(y, predictions)

        # Plotting the results
        plt.figure(figsize=(8, 6))
        plt.plot(imis_data_pandas.index, y,
                 color='blue', label='Actual data')
        plt.plot(imis_data_pandas.index, predictions,
                 color='red', label='Regression Line')
        plt.xlabel('Timestamp')
        plt.ylabel('Precipitation (mm)')
        plt.title(f'Linear Regression: Station {r} to IMIS')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Store the R2 value for this station's regression model
        list_r.append((i, r2, r))

        r += 1
        
    list_r_sorted = sorted(list_r, key=lambda x: x[1], reverse=True)  # x[1] is the R2 value
    
    # Select the top 3 stations
    top_stations = list_r_sorted[:2]
    
    # Extract the station IDs and their corresponding column names
    top_station_ids = [x[0] for x in top_stations]
    top_station_columns = [f'Cumulative_PSUM{x[2]}' for x in top_stations]
    
    # Prepare data for multiple linear regression
    X_multiple = imis_data_pandas[top_station_columns].values  # Use top station columns as predictors
    y = imis_data_pandas['Cumulative_Precip'].values  # Target variable
    
    # Initialize and fit the multiple linear regression model
    model_multiple = LinearRegression(fit_intercept=False)
    model_multiple.fit(X_multiple, y)
    
    # Make predictions using the multiple regression model
    predictions_multiple = model_multiple.predict(X_multiple)
    
    # Calculate R2 score for the multiple regression model
    r2_multiple = r2_score(y, predictions_multiple)
    print(f"R² for multiple regression with top stations: {r2_multiple:.3f}")
    
    
    # Plotting the multiple regression results
    plt.figure(figsize=(8, 6))
    plt.plot(imis_data_pandas.index, y, color='blue', label='Actual data')
    plt.plot(imis_data_pandas.index, predictions_multiple, color='green', label='Multiple Regression Line')
    plt.xlabel('Timestamp')
    plt.ylabel('Precipitation (mm)')
    plt.title('Multiple Regression: Top Stations to IMIS')
    plt.legend()
    plt.grid(True)
    plt.show()  
    
        # print coefficient
    print(list_r)
    file_name = file_names[name]
    print(f"The correlation coefficient for the file {file_name} is R²: {r2_multiple}")
    name = name + 1
    
    
    ################################
    imis_data_pandas.set_index('timestamp', inplace=True)
    predictions_hourly_resampled_series = pd.Series(predictions_multiple, 
                                                index=imis_data_pandas.index)

# Interpolating between the 6-hourly predictions
    predictions_hourly = predictions_hourly_resampled_series.resample('H').interpolate(method='linear')

    #####################################################
    # now we apply this multiple linear regression on hourly data
    ##############################
    imis_data_pandas_hourly = imis_data_pandas_hourly[imis_data_pandas_hourly['Precip'] > 0.1]
    r = 0
    for i in top_station_ids:
        # Read SwissMetNet data for the current station
        mch = pd.read_csv(
            f"mch/SMET/{i}.smet", skiprows=14, sep='\s+')

        if i == 'ZER':
            mch.columns = ['timestamp', 'PSUM']
        else:
            mch.columns = ['timestamp', 'PSUM']

        # Replace -999 with 0 (missing data handling)
        mch.loc[mch['PSUM'] == -999, 'PSUM'] = 0

        # Convert timestamp to datetime
        mch['timestamp'] = pd.to_datetime(mch['timestamp'])
        
        mch.set_index('timestamp', inplace=True)
        mch = mch.loc[start_date:end_date]

        #mch = mch[mch['Cumulative_PSUM'] > 0.1]

        mch['PSUM' + str(r)] = mch['PSUM']
        
        mch = mch.reset_index()

        # Merge the IMIS data with the precipitation data from the current station
        imis_data_pandas_hourly = pd.merge(
            imis_data_pandas_hourly, mch[['timestamp', 'PSUM' + str(r)]], on='timestamp', how='inner')
        #imis_data_pandas.fillna(0, inplace=True)
        r += 1
        
        # Prepare data for linear regression
    top_station_ids = [x[0] for x in top_stations]
    top_station_columns = [f'PSUM{x[2]}' for x in top_stations]
    
    psum_columns = [col for col in imis_data_pandas_hourly.columns if col.startswith('PSUM')]

    # Store the selected columns
    psum_data = imis_data_pandas_hourly[psum_columns]
    # Drop the column 'PSUM24' from the DataFrame
    psum_data = psum_data.drop(columns=['PSUM24'])
    
    
    # If you want to store the data for use in the regression model, you can assign it to X_hourly
    X_hourly = psum_data.values
            
        # Prepare data for multiple linear regression
    # X_multiple = imis_data_pandas_hourly[top_station_columns].values
    y = imis_data_pandas_hourly['Precip']  # Target variable (IMIS precipitation)

        # Make predictions for the IMIS station
    predictions = model_multiple.predict(X_hourly)

        # Calculate R2 score for the model
    r2_hourly = r2_score(y, predictions)
        