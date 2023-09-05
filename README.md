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

## Quick examples

1. Student A wishes to investigate the exoplanet Kepler-42 c.

Although one could attempt to use the tool just with the exoplanet's name, it's essential to provide at least two input arguments: the target name and the field name.

- **Target Name**: This is the name of the exoplanet in the NASA exoplanet archive format. Replace spaces with underscores for compatibility. For instance, the target name for Kepler-42 c becomes **kepler_42_c**. While we suggest using lowercase, it seems that the NASA exoplanet archive is case-insensitive.

- **Field Name**: For NenuFAR observations, this is the name utilized during scheduling, which typically corresponds to the star's name. For Kepler-42 c, the field name is **KEPLER_42**. This must be in uppercase due to NenuFAR's naming conventions.

Given this, Student A can execute the command:

```bash
./orbital_coverage_interactive.py -t kepler_42_c -f KEPLER_42
```

**WARNING**: Exoplanet systems often have alternate names (e.g., Kepler-42 c is also known as KOI-961 c). We strongly advise users to consult the NenuFAR schedule to ensure they're using the correct field names.
