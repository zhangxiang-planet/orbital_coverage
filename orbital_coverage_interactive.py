#!/usr/bin/env python3

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy import units as u
import matplotlib.dates as mdates
import urllib
import json
import argparse
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import re
import h5py

def extract_reference_name(column_value):
    # Convert the MaskedColumn value to string
    html_str = str(column_value)
    
    # The regex pattern captures text between ">" and "</a>"
    match = re.search(r'>(.*?)<\/a>', html_str)
    if match:
        # Replaces &amp; with &
        return match.group(1).replace('&amp;', '&')
    return "Unknown Reference"  # Default in case regex doesn't find a match

def parse_directory_name(dirname):
    start_date, start_time, end_date, end_time = dirname.split("_")[:4]
    return start_date, start_time, end_date, end_time

def convert_to_julian_date(date, time):
    date_obj = datetime.strptime(date + time, '%Y%m%d%H%M%S')
    julian_date = Time(date_obj, scale='utc')
    return julian_date.jd

# Convert phases to bin indices
def phase_to_bin(phase, bin_edges):
    return np.digitize(phase, bin_edges) - 1

# Function to split observation times into 1-minute segments, for plotting
def split_observation_into_segments(start, end):
    segment_duration = 10 / (24 * 60)  # 1 minutes in days
    num_segments = int((end - start) / segment_duration)
    segments = np.linspace(start, end, num_segments, endpoint=False)
    return segments

def format_func(x):
    return f"{x:.6f}"

def extract_subband_values(filename):
    # Regular expression pattern to match the desired line and extract the numbers
    pattern = r'Beam\[0\]\.subbandList=\[(\d+)\.\.(\d+)\]'

    with open(filename, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Return the matched values as integers
                return int(match.group(1)) * 0.1953125, int(match.group(2)) * 0.1953125

    return None, None

def extract_freq_values(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "Frequency (MHz) min,max" in line:
            parts = line.split(":")
            freq_min, freq_max = map(float, parts[1].strip().split())
            return freq_min, freq_max
        
    return None, None


##################################

# Parameters for target and telescope
parser = argparse.ArgumentParser(description='For an exoplanet, this code provides the orbital phase coverage of existing NenuFAR/LOFAR observations, while predicting future observing windows to cover the entire orbital phase.')
parser.add_argument('-t', '--target', type=str, required=True, 
    help='Target name. Must be the same format as used in the NASA exoplanet archive, with spaces replaced with underscores (e.g. tau_boo_b).')
parser.add_argument('-f', '--field', type=str, required=True, 
    help='Field name. Must be the same one used in NenuFAR observations. For example, if field name is TAU_BOO, the code will look for existing observing data directories named with TAU_BOO, while exclude directories with names containing CALIBRATOR.')
parser.add_argument('-i', '--instrument', type=str, default='NENUFAR', 
    help='Telescope used for observation. Can be LOFAR or NENUFAR. For LOFAR, only BEAMFORM observations for TAU BOO are supported for now. Default NENUFAR.')
parser.add_argument('-o', '--observation', type=str, default='BEAMFORM', 
    help='Type of observations to be found. Can be IMAGING or BEAMFORM. Used to search for existing data in nenufar-nri or nenufar-tf directories. Default BEAMFORM.')
parser.add_argument('-b', '--baddata', type=bool, default=True, 
    help='Bad data list. If True, observations given in the bad data list will be excluded. Default True.')
parser.add_argument('-fmin', '--freqmin', type=float, default=1000, 
    help='Filter previous observations with minimum observing frequency below XX MHz. Only used in BEAMFORM mode. Default 1000.')
parser.add_argument('-m', '--mode', type=str, default='AUTO', 
    help='Code driving mode, can be AUTO or MANUAL. If AUTO, the code grabs orbital parameters from NASA exoplanet archive. If MANUAL, the user needs to input orbital parameters. Default AUTO.')
parser.add_argument('-p', '--period', type=float, default=0, 
    help='Period of the exoplanet in days. Only used in manual mode. Default 0.')
parser.add_argument('-pe', '--perioderr', type=float, default=0, 
    help='Period Error of the exoplanet in days. Only used in manual mode. Default 0.')
parser.add_argument('-j', '--jd', type=float, default=0, 
    help='Time of Conjuction (Transit Midpoint) in JD. Only used in manual mode. Default 0.')
parser.add_argument('-pre', '--predict', type=float, default=30, 
    help='Predict length in days. Giving observing window for the target in next XX days. Default 30.')
parser.add_argument('-d', '--date', type=str, default='NOW', 
    help='Start date of prediction. Format YYYY-MM-DD. Default the current date.')
parser.add_argument('-e', '--elevation', type=float, default=40, 
    help='Target elevation limit in degrees. Only give observing window with target elevation > XX degress. Default 40.')
parser.add_argument('-s', '--sun', type=float, default=-18, 
    help='Sun elevation limit in degrees. Only give observing window with Sun elevation < XX degress. Default -18.')
parser.add_argument('-l', '--limit', action='store_true', default=False, 
    help='Orbital coverage limit. If set, only give observing window with oribital phase uncovered by previous observations. Default False.')
parser.add_argument('-a', '--avoid', action='store_true', default=False, 
    help='Avoiding observing windows already booked by others. If set, an observing schedule file is required as input. Default False.')
parser.add_argument('-w', '--window', type=float, default=2, 
    help='Only give observing windows longer than X hours. Default 2.')

args = parser.parse_args()
target_name = args.target
field_name = args.field
obs_mode = args.observation
drive_mode = args.mode
predict_length = args.predict
target_elevation = args.elevation
sun_elevation = args.sun

if args.instrument == "NENUFAR":
    if obs_mode == "BEAMFORM":
        base_paths_LT02 = glob.glob('/databf/nenufar-tf/LT02/????/??/')
        base_paths_ES02 = glob.glob('/databf/nenufar-tf/ES02/????/??/')
        base_paths = base_paths_LT02 + base_paths_ES02
        base_paths.sort()
    elif obs_mode == "IMAGING":
        base_paths_LT02 = glob.glob('/databf/nenufar-nri/LT02/????/??/')
        base_paths_ES02 = glob.glob('/databf/nenufar-nri/ES02/????/??/')
        base_paths = base_paths_LT02 + base_paths_ES02
        base_paths.sort()
    else:
        print('Error: Observing mode can only be IMAGING or BEAMFORM.')
        exit(1)
elif args.instrument == "LOFAR":
    if obs_mode == "BEAMFORM":
        base_paths_1 = glob.glob('/databf2/lofar/EXO/LOFAR/LC13_027/L??????/raw/')
        base_paths_2 = glob.glob('/databf2/lofar/EXO/LC7_013/L??????/raw/')
        base_paths = base_paths_1 + base_paths_2
        base_paths.sort()
    elif obs_mode == "IMAGING":
        print("Sorry! IMAGING mode is not supported with LOFAR. Please contact the developers if you need.")
        exit(1)
    else:
        print('Error: Observing mode can only be IMAGING or BEAMFORM.')
        exit(1)
else:
    print('Error: Observing instrument can only be NENUFAR or LOFAR.')
    exit(1) 

if len(base_paths) == 0:
    print("No detection is found with the field name. Maybe try another?")

if args.baddata == True:
    if os.path.exists(field_name+'_exclude.dat'):
        bad_data =  np.genfromtxt(field_name+'_exclude.dat', dtype=str)
    elif os.path.exists('/cep/nenufar/nenufar/pro/exoplanets/exclude-from-analysis/'+field_name+'_exclude.dat'):
        bad_data =  np.genfromtxt('/cep/nenufar/nenufar/pro/exoplanets/exclude-from-analysis/'+field_name+'_exclude.dat', dtype=str)
    else:
        print('Exclude data list not provided.')
        bad_data = []

# exoclock_planets = json.loads(urllib.request.urlopen('https://www.exoclock.space/database/planets_json').read())

table = NasaExoplanetArchive.query_object(target_name.replace("_", " "))

# Find the reference with smallest error in orbital period
mask = np.logical_and(table['soltype'] == 'Published Confirmed', table['pl_tranmid']>0)
small_err = np.nanmin(table[mask]['pl_orbpererr1'] - table[mask]['pl_orbpererr2'])
best_reference = table[(table['pl_orbpererr1'] - table['pl_orbpererr2']) == small_err]

reference = best_reference

target = SkyCoord(reference['ra'], reference['dec'])

if args.instrument == "NENUFAR":
    location = EarthLocation(lat=47.3821*u.deg, lon=2.1948*u.deg, height=136.0*u.m)  
elif args.instrument == "LOFAR":
    location = EarthLocation(lat=52.9060*u.deg, lon=6.8688*u.deg, height=20.0*u.m)
else:
    print('Error: Observing instrument can only be NENUFAR or LOFAR.')
    exit(1)
    
p_e = float(reference['pl_orbper'].value[0])
p_e_error = float(small_err.value / 2)
jd0 = float(reference['pl_tranmid'].value[0])

# summary file
f = open(target_name + '_' + args.instrument + '_' + obs_mode + '_' + drive_mode + '_summary.txt', 'w')

print("----------")
f.write("----------\n")

if drive_mode == 'MANUAL':
    p_e = args.period
    p_e_error = args.perioderr
    jd0 = args.jd
    print("Code is running in MANUAL mode.")
    print("Reference Name:", extract_reference_name(reference['pl_refname']))  # Replace with actual column name if different
    # print("Publication Date:", reference['pl_pubdate'])
    print("Orbital Period:", p_e)  # Replace with actual column name if different
    print("Error in Orbital Period:", p_e_error)  # Replace with actual column name if different
    print("Time of Conjuction (Transit Midpoint):", jd0)

    f.write("Code is running in MANUAL mode.\n")
    f.write("Reference Name: " + str(extract_reference_name(reference['pl_refname'])) + '\n')
    f.write("Orbital Period: " + str(p_e) + '\n')
    f.write("Error in Orbital Period: " + str(p_e_error) + '\n')
    f.write("Time of Conjuction (Transit Midpoint): " + str(jd0) + '\n')
else:
    print("Code is running in AUTO mode.")
    print("Reference Name:", extract_reference_name(reference['pl_refname']))  # Replace with actual column name if different
    # print("Publication Date:", reference['pl_pubdate'])
    print("Orbital Period:", p_e)  # Replace with actual column name if different
    print("Error in Orbital Period:", p_e_error)  # Replace with actual column name if different
    print("Time of Conjuction (Transit Midpoint):", jd0)

    f.write("Code is running in AUTO mode.\n")
    f.write("Reference Name: " + str(extract_reference_name(reference['pl_refname'])) + '\n')
    f.write("Orbital Period: " + str(p_e) + '\n')
    f.write("Error in Orbital Period: " + str(p_e_error) + '\n')
    f.write("Time of Conjuction (Transit Midpoint): " + str(jd0) + '\n')

# Will lose phase coherence (10 percent of one orbit) by what time?
esti_cut = float((p_e/p_e_error)*(0.1*p_e) + jd0)
# print(esti_cut)
esti_cut_utc = Time(esti_cut, format='jd', scale='utc').iso

print('Will lose phase coherence (10 percent of one orbit) by:', esti_cut_utc)
print("----------")

f.write('Will lose phase coherence (10 percent of one orbit) by: ' + str(esti_cut_utc) + '\n')
f.write("----------\n")


##################################

# Initialize lists for observation start and end times
t_starts = []
t_ends = []

if args.instrument == "NENUFAR":
    for base_path in base_paths:
        for dirname in os.listdir(base_path):
            if field_name in dirname and "CALIBRATOR" not in dirname and not any(bad_dir in dirname for bad_dir in bad_data):
                if obs_mode == 'IMAGING':
                    start_date, start_time, end_date, end_time = parse_directory_name(dirname)
                    t_starts.append(convert_to_julian_date(start_date, start_time))
                    t_ends.append(convert_to_julian_date(end_date, end_time))
                elif obs_mode == 'BEAMFORM':
                    file_paths = glob.glob(base_path + '/' + dirname + "/*.parset")
                    if len(file_paths) == 0:
                        file_paths = glob.glob(base_path + '/' + dirname + "/L1/*.parset")
                    file_paths.sort()

                    # Sometimes the parset file doesn't exist...
                    if len(file_paths) > 0:
                        f_min, f_max = extract_subband_values(file_paths[0])
                    else:
                        file_paths = glob.glob(base_path + '/' + dirname + "/L1/*.spectra.txt")
                        if len(file_paths) == 0:
                            file_paths = glob.glob(base_path + '/' + dirname + "/*.spectra.txt")
                        if len(file_paths) == 0:
                            print("No frequency information was found in directory " + base_path + '/' + dirname + ". Please double check.")
                            exit(1)
                        file_paths.sort()
                        f_min, f_max = extract_freq_values(file_paths[0])                   

                    if f_min == None:
                        print("No frequency information was found in file " + file_paths[0] + ". Please double check.")
                        exit(1)

                    if f_min < args.freqmin:
                        start_date, start_time, end_date, end_time = parse_directory_name(dirname)
                        t_starts.append(convert_to_julian_date(start_date, start_time))
                        t_ends.append(convert_to_julian_date(end_date, end_time))
                else:
                    print("Error: Observing mode can only be IMAGING or BEAMFORM.")
                    exit(1)

                    
elif args.instrument == "LOFAR":
    lofar_ids = []
    for base_path in base_paths:
        lofar_ids.append(base_path.split('/')[-3])
        files = glob.glob(base_path + '/*.h5')
        files.sort()
        with h5py.File(files[0], 'r') as file:
            if 'Tau Boo' in file.attrs['TARGETS'] or 'Tau Bootis' in file.attrs['TARGETS']:
                t_start = Time(file.attrs['OBSERVATION_START_MJD'], format='mjd')
                t_end = Time(file.attrs['OBSERVATION_END_MJD'], format='mjd')
                t_starts.append(t_start.jd)
                t_ends.append(t_end.jd)
else:
    print('Error: Observing instrument can only be NENUFAR or LOFAR.')
    exit(1) 
    
t_starts = np.array(t_starts)
t_ends = np.array(t_ends)

# Compute phases of existing observations
phases_observed_starts = ((t_starts - jd0) / p_e) % 1
phases_observed_ends = ((t_ends - jd0) / p_e) % 1

t_starts_utc = Time(t_starts, format='jd', scale='utc').iso
t_ends_utc = Time(t_ends, format='jd', scale='utc').iso

f.write("\nThis section gives summary of observations already taken.\nColumns: Observation start time (JD), Observation end time (JD), Orbital Phase at start time, Orbital Phase at end time, Observation start time (UTC), Observation end time (UTC)\n\n")

if len(t_starts) == 0:
    print('No previous observation was found.')
else:
    vfunc = np.vectorize(format_func)
    t_phase_observed = np.vstack((vfunc(t_starts).astype(str), vfunc(t_ends).astype(str), vfunc(phases_observed_starts).astype(str), vfunc(phases_observed_ends).astype(str), t_starts_utc.astype(str), t_ends_utc.astype(str)))
    np.savetxt(f, t_phase_observed.T, fmt='%s', delimiter=', ')


    # Splitting observations into segments
    all_segments = []
    for start, end in zip(t_starts, t_ends):
        segments = split_observation_into_segments(start, end)
        all_segments.extend(segments)

    all_segments = np.array(all_segments)
    phases_segments = ((all_segments - jd0) / p_e) % 1
    datetime_segments = Time(all_segments, format='jd', scale='utc')

    if args.instrument == "LOFAR":
        # We need to plot 2017 and 2020 data seperately
        mask_2017 = datetime_segments.iso[:].astype(str) < '2018'
        mask_2020 = datetime_segments.iso[:].astype(str) > '2019'

    datetime_segments = np.array(datetime_segments.to_value('unix', subfmt='float') / 86400)

    # Calculate elevations for these segments
    target_altaz_segments = target.transform_to(AltAz(obstime=Time(all_segments, format='jd'), location=location))
    elevations_segments = np.array(target_altaz_segments.alt.deg)

    # Plot 1: Orbital Phase Distribution of Observations
    if args.instrument == "LOFAR":    
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # vmin, vmax = 30, 55

        # sc1 = axs[0].scatter(phases_segments[mask_2020], datetime_segments[mask_2020], c=elevations_segments[mask_2020], marker='o', cmap='viridis', vmin=vmin, vmax=vmax)
        sc1 = axs[0].scatter(phases_segments[mask_2020], datetime_segments[mask_2020], marker='o')
        axs[0].text(phases_segments[mask_2020], datetime_segments[mask_2020], lofar_ids[mask_2020], fontsize=8)
        
        axs[0].set_xlim(0,1)
        # axs[0].set_title(target_name + ": Orbital Phase of Past Observations (2017)")
        # axs[0].set_xlabel("Orbital Phase")
        # axs[0].set_ylabel("Time (UT)")
        axs[0].yaxis.set_major_locator(mdates.AutoDateLocator())
        axs[0].yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        # sc2 = axs[1].scatter(phases_segments[mask_2017], datetime_segments[mask_2017], c=elevations_segments[mask_2017], marker='o', cmap='viridis', vmin=vmin, vmax=vmax)
        sc2 = axs[1].scatter(phases_segments[mask_2017], datetime_segments[mask_2017], marker='o')
        axs[1].text(phases_segments[mask_2017], datetime_segments[mask_2017], lofar_ids[mask_2017], fontsize=8)
        
        axs[1].set_xlim(0,1)
        # axs[1].set_title(target_name + ": Orbital Phase of Past Observations (2020)")
        # axs[1].set_xlabel("Orbital Phase")
        # axs[1].set_ylabel("Time (UT)")
        axs[1].yaxis.set_major_locator(mdates.AutoDateLocator())
        axs[1].yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        for ax in axs:
            ax.set_xlabel("Orbital Phase")
            ax.set_ylabel("Time (UT)")
        fig.suptitle(target_name + ": Orbital Phase of Past Observations")

        # fig.subplots_adjust(right=0.85)
        # cbar_ax = fig.add_axes([0.87, 0.05, 0.03, 0.9])  # Adjust the dimensions as needed

        # # Add colorbar
        # cbar = fig.colorbar(sc1, cax=cbar_ax)
        # cbar.set_label('Target Elevation (deg)')

# Adjust layout
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    
        plt.savefig(target_name + '_' + args.instrument + '_' + obs_mode + '_' + drive_mode + '_orbital_phase.png', dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

    else:
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(phases_segments, datetime_segments, c=elevations_segments, marker='o', cmap='viridis')
        plt.colorbar(sc, label='Target Elevation (deg)')
        plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xlim(0,1)
        plt.title(target_name + ": Orbital Phase of Past Observations")
        plt.xlabel("Orbital Phase")
        plt.ylabel("Time (UT)")
        plt.tight_layout()
        plt.savefig(target_name + '_' + args.instrument + '_' + obs_mode + '_' + drive_mode + '_orbital_phase_collected.png', dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

    #####################################

    # Determine covered phases using histogram
    num_bins = 360
    bin_edges = np.linspace(0, 1, num_bins+1)
    phase_coverage = np.zeros(num_bins)

    for phase_start, phase_end in zip(phases_observed_starts, phases_observed_ends):
        start_bin = int(phase_start * num_bins)
        end_bin = int(phase_end * num_bins)
        
        if end_bin < start_bin:
            phase_coverage[start_bin:] += 1
            phase_coverage[:end_bin+1] += 1
        else:
            phase_coverage[start_bin:end_bin+1] += 1

    covered_bins = bin_edges[:-1][phase_coverage > 0]

    # Plot 2: Histogram of Phase Coverage
    plt.figure(figsize=(12, 6))
    plt.bar(bin_edges[:-1], phase_coverage, width=1/num_bins, align='edge')
    plt.xticks(np.arange(0, 1.05, 0.05))
    plt.title(target_name + " Histogram of Phase Coverage")
    plt.xlabel("Orbital Phase")
    plt.ylabel("Number of Observations")
    plt.tight_layout()
    plt.savefig(target_name + '_' + args.instrument + '_' + obs_mode + '_' + drive_mode + '_phase_coverage_hist.png', dpi=300, bbox_inches='tight', facecolor='w')
    plt.close()

####################################

# Compute the time range for the next month
if args.date == 'NOW':
    start_time = Time(datetime.now())
else:
    start_time = Time(args.date)

end_time = start_time + timedelta(days=predict_length)
delta_t = timedelta(minutes=10)  
times = Time([start_time + i*delta_t for i in range(int((end_time - start_time).sec/delta_t.seconds))])

phases = ((times.jd - jd0) / p_e) % 1

times_all = times 
phases_all = phases

# Filter by available observing window (not already occupied)
if args.avoid == True:
    schedules = glob.glob('*_booking.csv')
    schedules.sort()

    if len(schedules) == 0:
        print("Error: No schedule files found.")
        exit(1)

    new_schedule = schedules[-1]
    print("Using future schedule file: " + new_schedule)
    schedule_data = np.genfromtxt(new_schedule, delimiter=',', skip_header=1, usecols=(0, 1, 2), dtype=str)
    schedule_start = Time(schedule_data[:,0], format='iso')
    schedule_end = Time(schedule_data[:,1], format='iso')

    mask_schedule = np.zeros(len(times), dtype=bool)

    # Loop over the time blocks
    for start, end in zip(schedule_start, schedule_end):
        # Update the mask for times within the current block
        mask_schedule = np.logical_or(mask_schedule, (times >= start) & (times <= end))

    good_schedule = ~mask_schedule

    times = times[good_schedule]
    phases = phases[good_schedule]

# Compute the AltAz of the target and Sun over the time range
altaz_target = target.transform_to(AltAz(obstime=times, location=location))
altaz_sun = get_sun(times).transform_to(AltAz(obstime=times, location=location))

altitudes_target = altaz_target.alt.deg
altitudes_sun = altaz_sun.alt.deg

num_bins = 360
bin_edges = np.linspace(0, 1, num_bins+1)

phases_bin_indices = phase_to_bin(phases, bin_edges)

if len(t_starts) == 0:
    covered_bin_indices = []
else:
    covered_bin_indices = np.unique(phase_to_bin(covered_bins, bin_edges))

# Filter by altitude > 40°, nighttime, and not-covered phases
if args.limit == False:
    good_slots = (altitudes_target > target_elevation) & (altitudes_sun < sun_elevation)
else:
    good_slots = (altitudes_target > target_elevation) & (altitudes_sun < sun_elevation) & (~np.isin(phases_bin_indices, covered_bin_indices))

if np.sum(good_slots) == 0:
    print("No good observing window in next 30 days. Sorry!")
    exit(0)

times = times[good_slots]
phases = phases[good_slots]
altitudes_target = altitudes_target[good_slots]


# Find observing slots longer than 1 hour
time_diffs = (times[1:].jd - times[:-1].jd) * 24 * 60
gaps = np.where(time_diffs > 15)[0]

cluster_start = [times[0]]  # the first timestamp is always the start of the first cluster
cluster_end = []

for g in gaps:
    cluster_end.append(times[g])
    cluster_start.append(times[g + 1])
cluster_end.append(times[-1])  # the last timestamp is always the end of the last cluster

cluster_start = Time(cluster_start)
cluster_end = Time(cluster_end)

# Compute the duration of each cluster in hours
cluster_durations = (cluster_end.jd - cluster_start.jd) * 24  # converting days to hours

# Identify clusters with durations > X hour
long_clusters_indices = np.where(cluster_durations > args.window)[0]
long_clusters_mask = cluster_durations > args.window

long_clusters_start = cluster_start[long_clusters_mask]
long_clusters_end = cluster_end[long_clusters_mask]
long_clusters_start_phase = ((long_clusters_start.jd - jd0) / p_e) % 1
long_clusters_end_phase = ((long_clusters_end.jd - jd0) / p_e) % 1

f.write('\n-----------\n')
f.write("This section gives summary of observation windows in next 30 days.\nColumns: Observation start time (JD), Observation end time (JD), Orbital Phase at start time, Orbital Phase at end time, Observation start time (UTC), Observation end time (UTC)\n\n")

vfunc = np.vectorize(format_func)
t_phase_predicted = np.vstack((vfunc(long_clusters_start.jd).astype(str), vfunc(long_clusters_end.jd).astype(str), vfunc(long_clusters_start_phase).astype(str), vfunc(long_clusters_end_phase).astype(str), long_clusters_start.iso.astype(str), long_clusters_end.iso.astype(str)))
np.savetxt(f, t_phase_predicted.T, fmt='%s', delimiter=', ')

long_clusters_times = []

for idx in long_clusters_indices:
    # Identify the start and end indices in the original "times" array for each long cluster
    start_idx = np.where(times == cluster_start[idx])[0][0]
    end_idx = np.where(times == cluster_end[idx])[0][0]

    # Append the relevant timestamps to the long_clusters_times list
    long_clusters_times.extend(times[start_idx:end_idx + 1])

# Convert the list back to a Time array
long_clusters_times = Time(long_clusters_times)
bool_mask = np.isin(times, long_clusters_times)

# print(altitudes_target[bool_mask])

# Plotting
plt.figure(figsize=(10, 8))
sc = plt.scatter(phases_all, times_all.to_value('unix', subfmt='float') / 86400, c='gray', marker='.', s=3, alpha=0.1)
sc = plt.scatter(phases[bool_mask], times[bool_mask].to_value('unix', subfmt='float') / 86400, c=np.array(altitudes_target[bool_mask]), marker='o', cmap='viridis')

plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xlim(-0.15,1.15)
plt.colorbar(sc, label='Target Elevation (deg)')
plt.title(target_name + " Observing Slots in Next 30 days")
plt.xlabel("Orbital Phase")
plt.ylabel("Time (UT)")
# Annotate the start and end times of each long cluster
for start, end in zip(cluster_start[long_clusters_indices], cluster_end[long_clusters_indices]):
    start_phase = phases[times == start][0]
    end_phase = phases[times == end][0]
    
    # Formatting time to only display month, day, hour, and minute
    start_str = start.strftime("%m-%d %H:%M")
    end_str = end.strftime("%m-%d %H:%M")
    
    plt.annotate(start_str,
                 (start_phase, start.to_value('unix', subfmt='float') / 86400), 
                 xytext=(-40, -10), textcoords='offset points', 
                 fontsize=8, color='tab:orange')
    
    plt.annotate(end_str,
                 (end_phase, end.to_value('unix', subfmt='float') / 86400), 
                 xytext=(-5, 5), textcoords='offset points', 
                 fontsize=8, color='tab:orange')

plt.tight_layout()
plt.savefig(target_name + '_' + args.instrument + '_' + obs_mode + '_' + drive_mode + '_observing_slot_predict.png', dpi=300, bbox_inches='tight', facecolor='w')
plt.close()

f.write('\n-----------\n')
f.close()

