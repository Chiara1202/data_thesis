#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:24:08 2025

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
from sklearn.metrics import r2_score
from datetime import datetime

import snowpat.pysmet as smet



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

    imis_data_pandas = imis_data_pandas[imis_data_pandas['Precip'] > 0.1]

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

        mch['PSUM' + str(r)] = mch['PSUM']
        mch.PSUM.plot()
        mch = mch[mch['PSUM'] > 0.1]

        # Merge the IMIS data with the precipitation data from the current station
        imis_data_pandas = pd.merge(
            imis_data_pandas, mch[['timestamp', 'PSUM' + str(r)]], on='timestamp', how='inner')

        # Prepare data for linear regression
        X = imis_data_pandas[['PSUM' + str(r)]].values.reshape(-1, 1)
        y = imis_data_pandas['Precip']  # Target variable (IMIS precipitation)

        # Initialize and fit the LinearRegression model
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
        
        
        # print coefficient
        file_name = file_names[name]
        print(f"The correlation coefficient for the file {file_name} is R²: {r2}")
        name = name + 1


    
# ############################################   
# from scipy.stats import pearsonr
# X = imis_data_pandas[['PSUM1']]
# y = imis_data_pandas['New'].values  # Dataset 2

# # Fit the linear regression model
# #model = LinearRegression()
# model = LinearRegression()
# model.fit(X, y)

# # Regression parameters
# slope = model.coef_[0]
# intercept = model.intercept_
# r2 = model.score(X, y)


# print(f"Regression equation: y = {slope:.2f}x + {intercept:.2f}")
# print(f"R² value: {r2:.2f}")

# # Predict values
# y_pred = model.predict(X)
# #y_pred[y_pred == y_pred[0]] = 0

# r, _ = pearsonr(y, y_pred)
# r2 = r2_score(y, y_pred)
# plt.plot(imis_data_pandas.timestamp, y,
#           color='blue', label='Actual data')
# plt.plot(imis_data_pandas.timestamp, y_pred,
#           color='k', label='Regression Line')

# #plt.axhline(y=1.12951, color='r', linestyle='-', label="Horizontal Line at y=0")
# plt.xlabel('Precipitation Dataset 1')
# plt.ylabel('Precipitation Dataset 2')
# plt.title('Linear Regression of Precipitation Data')
# plt.legend()
# plt.grid()
# plt.show()

# #############################
# #y_pred[y_pred < 1.16] = 0


# #####################################
# imis_data_pandas['regression'] = y_pred

# df = pd.merge(complete_pandas, imis_data_pandas[['timestamp', 'regression']], on='timestamp', how='left')

# # Fill the missing 'Predicted_Precipitation' (where precipitation < 0.1) with zeros
# df['regression'] = df['regression'].fillna(0)

# df.loc[df['New'] == 0, 'regression'] = 0

# ###########################################################
# plt.figure(figsize=(8, 6))

# plt.plot(df.timestamp, df.New,
#          color='g', label='Predictor')
# plt.plot(df.timestamp, df.regression,
#          color='red', label='Regression Line')


# # plt.plot(imis_data_pandas.index, y,
# #           color='blue', label='Actual data')
# # plt.plot(imis_data_pandas.index, y_pred_new,
# #          color='b', label='Regression Line')

# #plt.axhline(y=1.12951, color='r', linestyle='-', label="Horizontal Line at y=0")
# plt.xlabel('Precipitation Dataset 1')
# plt.ylabel('Precipitation Dataset 2')
# plt.title('Linear Regression of Precipitation Data')
# plt.legend()
# plt.grid()
# plt.show()
