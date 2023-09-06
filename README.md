# orbital_coverage

For an exoplanet, this tool provides the orbital phase coverage of existing NenuFAR/LOFAR observations, while predicting future observing windows to cover the entire orbital phase.

## Installation

Before running the code, ensure you have Python3 installed.

Install the required libraries using pip:

```bash
pip install numpy matplotlib astropy h5py astroquery
```

## Usage

This tool can be used by simply download (git clone) and execute the orbital_coverage_interactive.py script. To view the help message and understand the available arguments, run: 

```bash
./orbital_coverage_interactive.py -h
```

A list of available arguments is given below:

```bash
usage: orbital_coverage_interactive.py [-h] -t TARGET -f FIELD [-i INSTRUMENT] [-o OBSERVATION] [-b BADDATA]
                                       [-fmin FREQMIN] [-m MODE] [-p PERIOD] [-pe PERIODERR] [-j JD] [-pre PREDICT]
                                       [-e ELEVATION] [-s SUN] [-l] [-a]

For an exoplanet, this code provides the orbital phase coverage of existing NenuFAR/LOFAR observations, while
predicting future observing windows to cover the entire orbital phase.

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        Target name. Must be the same format as used in the NASA exoplanet archive, with spaces
                        replaced with underscores (e.g. tau_boo_b).
  -f FIELD, --field FIELD
                        Field name. Must be the same one used in NenuFAR observations. For example, if field name is
                        TAU_BOO, the code will look for existing observing data directories named with TAU_BOO,
                        while exclude directories with names containing CALIBRATOR.
  -i INSTRUMENT, --instrument INSTRUMENT
                        Telescope used for observation. Can be LOFAR or NENUFAR. For LOFAR, only BEAMFORM
                        observations for TAU BOO are supported for now. Default NENUFAR.
  -o OBSERVATION, --observation OBSERVATION
                        Type of observations to be found. Can be IMAGING or BEAMFORM. Used to search for existing
                        data in nenufar-nri or nenufar-tf directories. Default BEAMFORM.
  -b BADDATA, --baddata BADDATA
                        Bad data list. If True, observations given in the bad data list will be excluded. Default
                        True.
  -fmin FREQMIN, --freqmin FREQMIN
                        Filter previous observations with minimum observing frequency below XX MHz. Only used in
                        BEAMFORM mode. Default 1000.
  -m MODE, --mode MODE  Code driving mode, can be AUTO or MANUAL. If AUTO, the code grabs orbital parameters from
                        NASA exoplanet archive. If MANUAL, the user needs to input orbital parameters. Default AUTO.
  -p PERIOD, --period PERIOD
                        Period of the exoplanet in days. Only used in manual mode. Default 0.
  -pe PERIODERR, --perioderr PERIODERR
                        Period Error of the exoplanet in days. Only used in manual mode. Default 0.
  -j JD, --jd JD        Time of Conjuction (Transit Midpoint) in JD. Only used in manual mode. Default 0.
  -pre PREDICT, --predict PREDICT
                        Predict length in days. Giving observing window for the target in next XX days. Default 30.
  -e ELEVATION, --elevation ELEVATION
                        Target elevation limit in degrees. Only give observing window with target elevation > XX
                        degress. Default 40.
  -s SUN, --sun SUN     Sun elevation limit in degrees. Only give observing window with Sun elevation < XX degress.
                        Default -18.
  -l, --limit           Orbital coverage limit. If set, only give observing window with oribital phase uncovered by
                        previous observations. Default False.
  -a, --avoid           Avoiding observing windows already booked by others. If set, an observing schedule file is
                        required as input. Default False.
```

## Examples

1. *Student A wishes to investigate the exoplanet Kepler-42 c*.

Although one could attempt to use the tool just with the exoplanet's name, it's essential to provide at least two input arguments: the target name and the field name.

- **Target Name**: This is the name of the exoplanet in the NASA exoplanet archive format. Replace spaces with underscores for compatibility. For instance, the target name for Kepler-42 c becomes **kepler_42_c**. While we suggest using lowercase, it seems that the NASA exoplanet archive is case-insensitive.

- **Field Name**: For NenuFAR observations, this is the name utilized during scheduling, which typically corresponds to the star's name. For Kepler-42 c, the field name is **KEPLER_42**. This must be in uppercase due to NenuFAR's naming conventions.

Given this, Student A can execute the command:

```bash
./orbital_coverage_interactive.py -t kepler_42_c -f KEPLER_42
```

**WARNING**: Exoplanet systems often have alternate names (e.g., Kepler-42 c is also known as KOI-961 c). We strongly advise users to consult the NenuFAR schedule to ensure they're using the correct field names.

2. *Postdoc B would like to investigate the exoplanet ups And e. However, they prefer to use orbital parameters from a specific reference.*