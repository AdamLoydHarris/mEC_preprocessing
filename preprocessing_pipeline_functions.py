import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy.signal import resample
from collections import namedtuple
from datetime import datetime, date
import re

def load_cluster_labels(cluster_group_tsv, cluster_kslabel_tsv):
    """
    Load cluster labels from cluster_group.tsv and cluster_KSlabel.tsv,
    returning a dictionary: { cluster_id: label }.
    
    The priority is:
      - cluster_group.tsv (if present)
      - fallback to cluster_KSlabel.tsv (if the cluster is missing a label in the first)
    """
    # cluster_group file
    if os.path.exists(cluster_group_tsv):
        df_group = pd.read_csv(cluster_group_tsv, sep='\t')
        # If needed, rename columns if they don't match exactly:
        if 'cluster_id' not in df_group.columns or 'group' not in df_group.columns:
            df_group.columns = ['cluster_id', 'group']
    else:
        df_group = pd.DataFrame(columns=['cluster_id', 'group'])
    
    # cluster_KSlabel file
    if os.path.exists(cluster_kslabel_tsv):
        df_ks = pd.read_csv(cluster_kslabel_tsv, sep='\t')
        if 'cluster_id' not in df_ks.columns or 'KSlabel' not in df_ks.columns:
            df_ks.columns = ['cluster_id', 'KSlabel']
    else:
        df_ks = pd.DataFrame(columns=['cluster_id', 'KSlabel'])
    
    # Convert cluster_id to int in both for easier merges
    df_group['cluster_id'] = df_group['cluster_id'].astype(int, errors='ignore')
    df_ks['cluster_id']    = df_ks['cluster_id'].astype(int, errors='ignore')
    
    # Merge them into a single DataFrame, outer join so we keep all
    df_merge = pd.merge(df_group, df_ks, on='cluster_id', how='outer')
    
    # Create a dictionary mapping cluster_id -> final label
    cluster_label_dict = {}
    for idx, row in df_merge.iterrows():
        clust_id = int(row['cluster_id'])
        
        group_label = None
        ks_label    = None
        
        if 'group' in row and pd.notnull(row['group']):
            group_label = row['group']
        if 'KSlabel' in row and pd.notnull(row['KSlabel']):
            ks_label = row['KSlabel']
        
        # Priority: if group_label is available, use it; otherwise fallback to ks_label
        final_label = group_label if group_label is not None else ks_label
        cluster_label_dict[clust_id] = final_label
    
    return cluster_label_dict

def get_good_clusters(cluster_label_dict):
    """
    Return a sorted list of cluster IDs considered 'good'.
    Adjust the condition below for your labeling convention:
    e.g. 'good', 'Good', 'su', 'SU'
    """
    good = []
    for clust_id, label in cluster_label_dict.items():
        if label in ['good', 'Good', 'su', 'SU']:
            good.append(clust_id)
    return sorted(good)

def bin_spikes(spike_times_ms, spike_clusters, good_clusters,
               bin_size_ms=25,
               session_offset=0,
               session_duration_ms=None):
    """
    Bin 'good' clusters' spikes into 25 ms bins for one session.

    - spike_times_ms: 1D array of spike times (ms, in concatenated timeline).
    - spike_clusters: 1D array of cluster IDs for each spike_time.
    - good_clusters: list of cluster IDs that are 'good'.
    - bin_size_ms: size of each bin in ms (default = 25 ms).
    - session_offset: the starting time (ms) of this session in the *concatenated* timeline.
    - session_duration_ms: total duration of this session from that offset.
    
    Returns a 2D array of shape (n_good_clusters, n_time_bins).
    """
    if session_duration_ms <= 0:
        # If there's no valid time range, return an empty matrix
        return np.zeros((len(good_clusters), 0), dtype=np.int32)
    
    t_start = session_offset
    t_end   = session_offset + session_duration_ms
    
    spikes_remaining = sum(spike_times_ms>t_end)
    print(f"Spikes remaining after this session: {spikes_remaining}")
    # Boolean mask for spikes in [t_start, t_end)
    in_session_mask = (spike_times_ms >= t_start) & (spike_times_ms < t_end)
    
    # Subset spike times & clusters
    # (Make sure these arrays are 1D with .squeeze())
    sess_spike_times    = spike_times_ms[in_session_mask].squeeze() - t_start
    sess_spike_clusters = spike_clusters[in_session_mask].squeeze()
    
    # Figure out how many bins we need
    n_bins = int(np.ceil(session_duration_ms / bin_size_ms))
    
    # We'll create a 2D array: shape = (len(good_clusters), n_bins)
    spike_matrix = np.zeros((len(good_clusters), n_bins), dtype=np.int32)
    
    # Make a quick index from cluster_id -> row index in spike_matrix
    cluster_index_map = {clust_id: i for i, clust_id in enumerate(good_clusters)}
    
    # Digitize times to bin indices
    bin_indices = (sess_spike_times // bin_size_ms).astype(int)
    
    # Accumulate counts in each bin
    for t_bin, clust in zip(bin_indices, sess_spike_clusters):
        if 0 <= t_bin < n_bins:  # just a safety check
            if clust in cluster_index_map:
                spike_matrix[cluster_index_map[clust], t_bin] += 1
    
    return spike_matrix


def extract_cohort_and_mouse_id(filepath):
    """
    Extracts the cohort number and mouse ID from a given file path.
    Args:
        filepath (str): The file path from which to extract the cohort number and mouse ID.

    Returns:
        tuple: A tuple containing the cohort number (str) and mouse ID (str) if the pattern
               is found, otherwise (None, None).
    """
    match = re.search(r"cohort(\d+)/.*?/([^/]+)/", filepath)
    if match:
        cohort = match.group(1)  # Extract the cohort number
        mouse_id = match.group(2)  # Extract the mouse ID
        return cohort, mouse_id
    else:
        print('cannot retrieve mouse and cohort from filepath')
        return None, None


def load_json(json_path):
    
    with open(json_path, 'r') as f:
        session_dict = json.load(f)
    
    session_items = session_dict.items()

    return session_items
    

def load_metadata(cohort, mouse, data_folder):

    """
    Load metadata for a given mouse in a given cohort.
    Returns two DataFrames: one for awake, one for sleep.
    """
    data_directory = f"{data_folder}/cohort{cohort}/"   
    metadata_path= f"{data_directory}/MetaData.xlsx - {mouse}.csv"

    metadata_awake = pd.read_csv(metadata_path, delimiter=',',dtype=str)
    filtered_metadata_awake=metadata_awake[metadata_awake['Include']==1]
    
    try:
        metadata_path_sleep= f"{data_directory}/MetaData.xlsx - {mouse}_sleep.csv"
        metadata_sleep = pd.read_csv(metadata_path_sleep, delimiter=',',dtype=str)
        filtered_metadata_sleep=metadata_sleep[metadata_sleep['Include']==1]
        return filtered_metadata_awake, filtered_metadata_sleep

    except FileNotFoundError:
        print('No sleep metadata found for this mouse')
        return filtered_metadata_awake, None
    

def extract_ephys_date_time(filepath):
    """
    Extracts the date and time from a given file path using a regular expression.

    The function searches for a date-time pattern in the format 'YYYY-MM-DD_HH-MM-SS'
    within the provided file path. If a match is found, it returns the date and time
    as separate strings. If no match is found, it returns (None, None).

    Args:
        filepath (str): The file path containing the date-time information.

    Returns:
        tuple: A tuple containing the date and time as strings (date, time).
               If no match is found, returns (None, None).
    """
    # Regular expression to capture date-time format
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filepath)
    if match:
        date = match.group(1)
        time = match.group(2)
        return date, time

    return None, None

def wake_or_sleep(ephys_filepath, metadata_awake, metadata_sleep):
    """
    Determines whether an electrophysiology (ephys) session is an awake or sleep session based on metadata.
    Args:
        ephys_filepath (str): The file path to the ephys data file.
        metadata_awake (pd.DataFrame): A DataFrame containing metadata for awake sessions.
        metadata_sleep (pd.DataFrame): A DataFrame containing metadata for sleep sessions.
    Returns:
        str: 'awake' if the session is an awake session, 'sleep' if the session is a sleep session, or None if the session is not found in either metadata.
    """

    sleep_status = None

    date, time = extract_ephys_date_time(ephys_filepath)
    
    # first try awake metadata

    filtered_awake =  metadata_awake[(metadata_awake["Date"] == date) & (metadata_awake["Ephys"] == time)]
    
    if len(filtered_awake>1):
        print('More than one awake entry found for this date and time')
    if len(filtered_awake==0):
        print('Not an awake session')
    if len(filtered_awake==1):
        print('awake session')
        sleep_status = 'awake'

    if sleep_status =='awake':
        return sleep_status
    
    else:
        # then try sleep metadata
        filtered_sleep = metadata_sleep[(metadata_sleep["Date"] == date) & (metadata_sleep["Ephys"] == time)]

        if len(filtered_sleep>1):
            print('More than one sleep entry found for this date and time')
        if len(filtered_sleep==0):
            print('Not an sleep session')
        if len(filtered_sleep==1):
            print('sleep session')
            sleep_status = 'sleep'

        return sleep_status


def get_awake_sessions_info(ephys_filepath, metadata_awake):

    """
    uses ephys file path from json and metadata dataframe
    to retrieve information about the session and timestamps
    to find tracking data and pycontrol data
    """

    date, time = extract_ephys_date_time(ephys_filepath)

    filtered_awake =  metadata_awake[(metadata_awake["Date"] == date) & (metadata_awake["Ephys"] == time)]
    
    Structure = filtered_awake["Structure"]
    Structure_abstract = filtered_awake["Structure_abstract"]
    Structure_no = filtered_awake["Structure_no"]
    Tracking_timestamp = filtered_awake["Tracking"]
    Behaviour_timestamp = filtered_awake["Behaviour"]

    return date, Structure, Structure_abstract, Structure_no, Tracking_timestamp, Behaviour_timestamp


def get_behaviour_txt(data_path, mouse, cohort, date, Behaviour_timestamp, Structure_abstract, int_subject_IDs=True):
    """
    Retrieves the pycontrol output file for awake sessions and produces a dictionary 
    containing the raw pycontrol data.
    Parameters:
    data_path (str): The base directory path where the data is stored.
    mouse (str): The identifier for the mouse.
    cohort (str): The cohort number or identifier.
    date (str): The date of the session in 'YYYYMMDD' format.
    Behaviour_timestamp (str): The timestamp of the behaviour session.
    Returns:
    dict: A dictionary where keys are event names and values are numpy arrays of event times.
    The function performs the following steps:
    1. Constructs the file path for the behaviour data file.
    2. Reads the file and extracts relevant information.
    3. Parses session information including experiment name, task name, subject ID, and start date.
    4. Extracts state and event IDs, and session data.
    5. Converts subject ID to integer if `int_subject_IDs` is True.
    6. Returns a dictionary with event names as keys and numpy arrays of event times as values.
    Note:
    - The function assumes that the file format and structure are consistent with the expected format.
    """

    
    Behaviourfile_path = f"{data_path}/cohort{cohort}/{mouse}/behaviour/{mouse}-{date}-{Behaviour_timestamp}.txt"

    print('Importing data file: '+os.path.split(Behaviourfile_path)[1])

    with open(Behaviourfile_path, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]


    # Extract and store session information.
    file_name = os.path.split(Behaviourfile_path)[1]
    Event = namedtuple('Event', ['time','name'])

    info_lines = [line[2:] for line in all_lines if line[0]=='I']

    experiment_name = next(line for line in info_lines if 'Experiment name' in line).split(' : ')[1]
    task_name       = next(line for line in info_lines if 'Task name'       in line).split(' : ')[1]
    subject_ID_string    = next(line for line in info_lines if 'Subject ID'      in line).split(' : ')[1]
    datetime_string      = next(line for line in info_lines if 'Start date'      in line).split(' : ')[1]


    if int_subject_IDs: # Convert subject ID string to integer.
        subject_ID = int(''.join([i for i in subject_ID_string if i.isdigit()]))
    else:
        subject_ID = subject_ID_string

    datetime = datetime.strptime(datetime_string, '%Y/%m/%d %H:%M:%S')
    datetime_string = datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Extract and store session data.

    state_IDs = eval(next(line for line in all_lines if line[0]=='S')[2:])
    event_IDs = eval(next(line for line in all_lines if line[0]=='E')[2:])
    variable_lines = [line[2:] for line in all_lines if line[0]=='V']

    if Structure_abstract not in ['ABCD','AB','ABCDA2','ABCDE','ABCAD']:
        pass
    else:
        structurexx = next(line for line in variable_lines if 'active_poke' in line).split(' active_poke ')[1]
        if 'ot' in structurexx:
            structurex=structurexx[:8]+']'
        else:
            structurex=structurexx

        if Structure_abstract in ['ABCD','AB','ABCDE']:
            structure=np.asarray((structurex[1:-1]).split(',')).astype(int)
        else:
            structure=structurex

        ID2name = {v: k for k, v in {**state_IDs, **event_IDs}.items()}
        data_lines = [line[2:].split(' ') for line in all_lines if line[0]=='D']
        events = [Event(int(dl[0]), ID2name[int(dl[1])]) for dl in data_lines]
        times = {event_name: np.array([ev.time for ev in events if ev.name == event_name])  
                    for event_name in ID2name.values()}
  
    return times



def make_trial_times_array(times_dic, target_binning=25, Structure_abstract='ABCD'):
    """
    Generate an array of trial onset times from a dictionary of event timestamps.
    Parameters:
    times_dic (dict): Dictionary containing pycontrol trial events and their timestamps. 
                        The keys are event names, and the values are lists of timestamps.
    target_binning (int, optional): The bin size to convert timestamps into. Default is 25 ms.
    Structure_abstract (str, optional): The structure of the session. Common types are 'ABCD', 'ABCDE', 'AB', 'ABCAD'. 
                                        Default is 'ABCD'.
    Returns:
    np.ndarray: A 2D array where each row corresponds to a trial and each column corresponds to a state onset time.
                The times are aligned such that the first state onset (e.g., 'A_on') is set to 0, and all times are 
                converted to the specified bin size.
    Raises:
    ValueError: If an unknown Structure_abstract is provided or if a required state is not found in times_dic.
    Notes:
    - The function aligns the trial times so that the first state onset (e.g., 'A_on') is set to 0.
    - The timestamps are converted from milliseconds to the specified bin size.
    - For 'ot' sessions, the first 'A_on' event is registered as 'A_on_first', and subsequent 'A_on' events are standard.

    """
   
    trial_times = []

    if Structure_abstract == 'ABCD':
        states = ['A_on', 'B_on', 'C_on', 'D_on']
    elif Structure_abstract == 'ABCDE':
        states = ['A_on', 'B_on', 'C_on', 'D_on', 'E_on']
    elif Structure_abstract == 'AB':
        states = ['A_on', 'B_on']
    elif Structure_abstract == 'ABCAD':
        states = ['A_on', 'B_on', 'C_on', 'A2_on', 'D_on']
    else:
        raise ValueError(f"Unknown Structure_abstract: {Structure_abstract}")

    for state in states:
        if state not in times_dic:
            raise ValueError(f"State {state} not found in times_dic")

    first_A_on = times_dic['A_on'][0]

    for i in range(len(times_dic[states[0]])):
        trial = []
        for state in states:
            if state == 'A_on' and 'A_on_first' in times_dic:
                trial.append(times_dic['A_on_first'][i] - first_A_on)
            else:
                trial.append(times_dic[state][i] - first_A_on)
        trial_times.append(trial)

    trial_times = np.array(trial_times)
    trial_times = (trial_times / target_binning).astype(int)

    return trial_times

def get_sync_to_trial_offset(times_dic, target_binning=25):
    """
    Calculate the offset between the first 'rsync' event and the first 'A_on' event in bins.
    
    Args:
        times_dic (dict): Dictionary containing event times.
        target_binning (int): The bin size to convert timestamps into. Default is 25 ms.
    
    Returns:
        float: The offset between the first 'rsync' event and the first 'A_on' event in bins.
    
    Raises:
        ValueError: If 'rsync' or 'A_on' events are not found in times_dic.
    """
    rsync_times = times_dic['rsync']
    a_on_times = times_dic.get('A_on_first', times_dic['A_on'])

    first_rsync = rsync_times[0]
    first_a_on = a_on_times[0]

    offset_ms = first_a_on - first_rsync
    offset_bins = (offset_ms / target_binning).astype(int)

    return offset_bins


def make_pokes_dic(times_dic, target_binning = 25):

    """
    Aligns poke times to the first 'A_on' event and bins them.
    Parameters:
    times_dic (dict): Dictionary containing event times.
    target_binning (int, optional): The binning interval. Default is 25.
    Returns:
    dict: Dictionary with aligned and binned poke times.
    """

    first_a_on = times_dic.get('A_on_first', times_dic['A_on'])[0]
    pokes_dic = {}

    for key in times_dic.keys():
        if 'poke' in key and 'out' not in key:
            aligned_pokes = times_dic[key] - first_a_on
            aligned_pokes = aligned_pokes[aligned_pokes >= 0]
            aligned_pokes = (aligned_pokes / target_binning).astype(int)
            pokes_dic[key] = aligned_pokes

    return pokes_dic


def get_behaviour_tsv(data_path, mouse, cohort, Behaviour_timestamp):

    """
    retrieves the pycontrol output file for bigmaze tsv sessions and produces a dictionary 
    containing the events in file. 
    for our bigmaze files this will just be RSYNC as we are just doing openfield or objectvector 
    sessions
    IMPORTANT: the later version of pycontrol has timestamps in seconds instead of milliseconds
    so we will here be multiplying by 1000 to keep units the same
    """
    Behaviourfile_path =   f"{data_path}/cohort{cohort}/{mouse}/behaviour/{mouse}-{date}-{Behaviour_timestamp}.tsv"
    behaviour_df = pd.read_csv(Behaviourfile_path, sep='\t')
    behaviour_df["time"] = behaviour_df["time"]*1000
    df_rsync = of_behaviour.query("type == 'event' & subtype == 'sync' & content == 'rsync'")


    return behaviour_df, df_rsync["time"]



def get_tracking(data_path, mouse, cohort, date, Tracking_timestamp):
    """
    Retrieves tracking data for a given mouse from a specified cohort and date.
    Parameters:
    data_path (str): The base directory path where the data is stored.
    mouse (str): The identifier for the mouse.
    cohort (int): The cohort number.
    date (str): The date of the tracking data in 'YYYY-MM-DD' format.
    Tracking_timestamp (str): The timestamp associated with the tracking data.
    Returns:
    tuple: A tuple containing the following elements:
        - xy (DataFrame or None): DataFrame containing the xy coordinates from the sleap file, or None if not found.
        - pinstate (DataFrame or None): DataFrame containing the pinstate data, or None if not found.
        - ROIs (DataFrame or None): DataFrame containing the ROIs data, or None if not found.
        - hd (DataFrame or None): DataFrame containing the head direction data, or None if not found.
    Raises:
    FileNotFoundError: If any of the required files are not found, a message is printed and the corresponding return value is set to None.
    """

    if cohort == 7:
        sleap_tag = 'cleaned_coordinates'
        head_direction_tag = 'head_direction'
        ROIs_tag = 'ROIs'

    behaviour_folder = f"{data_path}/cohort{cohort}/{mouse}/behaviour"
    files_in_behaviour = os.listdir(behaviour_folder)

    # pinstate
    try:
        print(f"finding pinstate file for {mouse}_pinstate_{date}-{Tracking_timestamp}")
        pinstate_path = f"{behaviour_folder}/{mouse}_pinstate_{date}-{Tracking_timestamp}.csv"
        pinstate = pd.read_csv(pinstate_path)
    except FileNotFoundError:
        pinstate = None
        print('Cannot find pinstate file')

    # xy coords
    try:
        coords_search = [i for i in behaviour_folder if date in i and Tracking_timestamp in i and sleap_tag in i]
        if len(coords_search==1):
            xy = pd.read_csv(os.path.join(behaviour_folder, coords_search[0]))
        else:
            print(f'check for duplicates or missing file for {date}_{Tracking_timestamp}')
            xy=None
    except FileNotFoundError:
        print('Cannot find sleap file')
        xy = None

    # ROIs:
    try:
        ROIs_search = [i for i in behaviour_folder if date in i and Tracking_timestamp in i and ROIs_tag in i]
        if len(ROIs_search==1):
            ROIs = pd.read_csv(os.path.join(behaviour_folder, ROIs_search[0]))
        else:
            print(f'check for duplicates or missing file for {date}_{Tracking_timestamp} ROIs')
            ROIs=None
    except FileNotFoundError:
        print('Cannot find ROIs file')
        ROIs=None
    
    # head_direction 

    try:
        hd_search = [i for i in behaviour_folder if date in i and Tracking_timestamp in i and head_direction_tag in i]
        if len(ROIs_search==1):
            hd = pd.read_csv(os.path.join(behaviour_folder, hd_search[0]))
        else:
            print(f'check for duplicates or missing file for {date}_{Tracking_timestamp} head directions')
            hd=None
    except FileNotFoundError:
        print('Cannot find head direction file')
        hd=None
 

    return  xy, pinstate, ROIs, hd


def resample_tracking(sleap_df, pinstate, camera_fps=60, target_fps=40, body_part = 'head_back'):

    """
    takes in sleap dataframe and pinstate and returna
    a resampled and first sync pulse truncated
    list of xy coordinates from a select stable bodypart
    """

    sync_indices = np.where(pinstate > np.median(pinstate))[0]

    first_sync_idx = sync_indices[0]
    print(f"First sync pulse detected at frame index: {first_sync_idx}")

    # Trim the SLEAP data to remove rows before the first sync pulse
    print("Trimming data to start after the first sync pulse...")
    sleap_trimmed = sleap_df[first_sync_idx:]

    print(f"Resampling data from {camera_fps} FPS to {target} FPS...")
    resample_factor = target_fps / camera_fps

    coords = []

    for column in [f'{body_part}.x', f'{body_part}.y']:
        resampled_column = resample(sleap_trimmed[column].values, resampled_length)
        coords.append(resampled_column)
    
    return coords


def resample_ROIs(ROIs_df, pinstate, camera_fps=60, target_fps=40, body_part='head_back'):
    """
    Resamples the ROIs dataframe to match the target FPS.
    Args:
        ROIs_df (pd.DataFrame): DataFrame containing the ROIs data.
        pinstate (np.array): Array containing the pinstate data.
        camera_fps (int): Original frames per second of the camera.
        target_fps (int): Target frames per second for resampling.
        body_part (str): The column name in the ROIs_df to be resampled.
    Returns:
        pd.Series: Resampled series of the specified body part.
    """
    sync_indices = np.where(pinstate > np.median(pinstate))[0]
    first_sync_idx = sync_indices[0]
    print(f"First sync pulse detected at frame index: {first_sync_idx}")

    # Trim the ROIs data to remove rows before the first sync pulse
    print("Trimming data to start after the first sync pulse...")
    ROIs_trimmed = ROIs_df.iloc[first_sync_idx:]

    # Calculate the resampling factor
    resample_factor = camera_fps / target_fps

    # Resample the specified body part column
    body_part_column = ROIs_trimmed[body_part]
    resampled_length = int(len(body_part_column) / resample_factor)
    resampled_body_part = body_part_column.iloc[::int(resample_factor)].reset_index(drop=True)

    return resampled_body_part

def resample_headDirections(HD_df, pinstate, camera_fps=60, target_fps=40, body_part = 'head_back'):
    
    """
    takes in HD dataframe and pinstate and returns
    a resampled and first sync pulse truncated
    list of xy coordinates from a select stable bodypart
    """

    sync_indices = np.where(pinstate > np.median(pinstate))[0]

    first_sync_idx = sync_indices[0]
    print(f"First sync pulse detected at frame index: {first_sync_idx}")

    # Trim the SLEAP data to remove rows before the first sync pulse
    print("Trimming data to start after the first sync pulse...")
    HD_df_trimmed = HD_df[first_sync_idx:]

    print(f"Resampling data from {camera_fps} FPS to {target} FPS...")
    resample_factor = target_fps / camera_fps

    HD_resampled = pd.DataFrame()

    for column in HD_df_trimmed.columns:
        resampled_column = resample(HD_df_trimmed[column].values, resampled_length)
        HD_resampled[column] = resampled_column
    
    return HD_resampled


def align_to_first_A(offset_bin, xy, ROIs, head_direction, ephys_mat):
    """
    Aligns the data to the first 'A' state by cutting everything before the offset bin.
    
    Args:
        offset_bin (int): The offset bin to align to the first 'A' state.
        xy (np.ndarray): Resampled xy coordinates.
        ROIs (pd.DataFrame): Resampled ROIs data.
        head_direction (pd.DataFrame): Resampled head direction data.
        ephys_mat (np.ndarray): Ephys firing rate matrix.
    
    Returns:
        tuple: Aligned xy coordinates, ROIs data, head direction data, and ephys firing rate matrix.
    """
    aligned_xy = xy[offset_bin:]
    aligned_ROIs = ROIs.iloc[offset_bin:].reset_index(drop=True)
    aligned_head_direction = head_direction.iloc[offset_bin:].reset_index(drop=True)
    aligned_ephys_mat = ephys_mat[:, offset_bin:]

    return aligned_xy, aligned_ROIs, aligned_head_direction, aligned_ephys_mat


