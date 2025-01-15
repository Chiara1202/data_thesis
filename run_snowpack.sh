#!/bin/bash
# Specify the output directory
output_dir="snowpack/WFJ2/output"

# Check if the output directory exists, and create it if it doesn't
if [ ! -d "$output_dir" ]; then
    echo "Output directory doesn't exist. Creating it..."
    mkdir "$output_dir"
else
    echo "Output directory already exists."
fi

# Run the snowpack command for the first file and log output
snowpack -c snowpack/WFJ2/WFJ2_WFJ2_MS_SNOW.ini -b 2013-09-02T00:00 -e 2018-12-01T00:00 > log1.txt 2>&1
echo "Finished processing WFJ2_WFJ2_MS_SNOW.ini"

# Run the snowpack command for the second file and log output
snowpack -c snowpack/WFJ2/WFJ2_WFJ2_MS_SNOW_excludePSUM.ini -b 2013-09-02T00:00 -e 2018-12-01T00:00 > log2.txt 2>&1
echo "Finished processing WFJ2_WFJ2_MS_SNOW_excludePSUM.ini"

# Run the snowpack command for the third file and log output
snowpack -c snowpack/WFJ2/WFJ2_WFJ2_MS_SNOW_unheatedgauges.ini -b 2013-09-02T00:00 -e 2018-12-01T00:00 > log3.txt 2>&1
echo "Finished processing WFJ2_WFJ2_MS_SNOW_unheatedgauges.ini"

echo "All files processed. Check log1.txt, log2.txt, and log3.txt for details."

