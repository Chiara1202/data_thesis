#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:11:00 2025

@author: ghielmin
"""
import os 
os.chdir("/home/ghielmin/Desktop/data_thesis/")
import subprocess

result = subprocess.run('snowpack -c snowpack/WFJ2/WFJ2_WFJ2_MS_SNOW.ini -b 2013-09-01T00:00 -e 2018-12-01T00:00 > log.txt 2>&1', shell=True, text=True, capture_output=True)

# Print the output (optional)
print("Output:", result.stdout)
print("Error (if any):", result.stderr)
