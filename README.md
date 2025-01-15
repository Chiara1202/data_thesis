# To reproduce:

- run `run_snowpack.sh`, this will run the SNOWPACK model on three different ini files which have the following characteristics:
   - WFJ2_WFJ2_MS_SNOW.ini: is the default experiment
   - WFJ2_WFJ2_MS_SNOW_excludePSUM.ini: PSUM is not taken into account
   - WFJ2_WFJ2_MS_SNOW_unheatedgauges.ini: the filter Unheated Rain Gauge is applied

  The output are saved in /snowpack/WFJ2/output
