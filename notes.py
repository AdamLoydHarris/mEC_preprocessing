

kilosort_folder = "/Volumes/behrens/adam_harris/Taskspace_abstraction_mEC/Data/cohort7/Sorted/bp01/28032024_31032024_combined_all"
json_path =os.path.join(kilosort_folder, [i for i in os.listdir(kilosort_folder) if i.endswith('.json')][0])
print(json_path)

with open(json_path, 'r') as f:
    session_dict = json.load(f)

print(session_dict)
# session_dict: { file_path: file_size_in_bytes, ... }
# e.g. "/path/to/2024-03-21_12-59-35/continuous.dat": 28106707968

# Sort the sessions by file_path or by date/time—adjust to your preference
session_items = sorted(session_dict.items(), key=lambda x: x[0])

# -------------------------------------------------------------------------
# 2) Compute absolute session boundaries in the concatenated timeline (ms)
# -------------------------------------------------------------------------
bytes_per_sample = 768
sampling_rate_hz = 30000  # 30 kHz
# => each sample = 768 bytes
# => 1 ms corresponds to 30 samples at 30 kHz

session_info = []  # will hold (file_path, start_ms, end_ms) in the *concatenated* timeline
cumulative_ms = 0.0

for (file_path, file_size) in session_items:
    # number of raw samples in this session
    n_samples = file_size / bytes_per_sample
    
    # convert to ms:  n_samples / 30  (since 30 samples = 1 ms)
    duration_ms = n_samples / 30.0
    start_ms = cumulative_ms
    end_ms   = cumulative_ms + duration_ms
    
    session_info.append((file_path, start_ms, end_ms))
    cumulative_ms = end_ms

# -------------------------------------------------------------------------
# 3) Load spike_times and spike_clusters from Kilosort output
#    Convert from sample indices (30 kHz) to ms
# -------------------------------------------------------------------------


spike_times_samples = np.load(os.path.join(kilosort_folder, 'spike_times.npy'))
spike_clusters      = np.load(os.path.join(kilosort_folder, 'spike_clusters.npy'))

# Make sure they are both squeezed to 1D
spike_times_samples = spike_times_samples.squeeze()
spike_clusters      = spike_clusters.squeeze()

# Convert from samples to ms
spike_times_ms = spike_times_samples / 30.0

# -------------------------------------------------------------------------
# 4) Load cluster labels from TSVs & choose 'good' clusters
# -------------------------------------------------------------------------
cluster_group_tsv   = os.path.join(kilosort_folder, 'cluster_group.tsv')
cluster_kslabel_tsv = os.path.join(kilosort_folder, 'cluster_KSlabel.tsv')
cluster_labels = load_cluster_labels(cluster_group_tsv, cluster_kslabel_tsv)

good_clusters = get_good_clusters(cluster_labels)
print(f"Found {len(good_clusters)} 'good' clusters.")

# -------------------------------------------------------------------------
# 5) For each session, load sample_numbers.npy to find the first sync pulse,
#    then truncate everything prior to that pulse for binning.
# -------------------------------------------------------------------------
bin_size_ms = 25
session_spike_counts = []  # will hold one array per session

for i, (file_path, session_abs_start, session_abs_end) in enumerate(session_info):
    # The absolute session boundaries in the concatenated timeline are
    #  [session_abs_start, session_abs_end).
    file_path_volumes = file_path.replace('/ceph/', '/Volumes/')
    session_dir = os.path.dirname(file_path_volumes)
    sync_file   = os.path.join(session_dir, 'sample_numbers.npy')
    print(sync_file)
    
    if not os.path.exists(sync_file):
        print(f"Warning: sync_file not found for {file_path}. Skipping first-sync truncation.")
        # We'll treat the entire session from session_abs_start to session_abs_end
        # i.e. no extra offset.
        first_sync_ms = 0.0
    else:
        # Load the sample_numbers array
        sample_numbers = np.load(sync_file).squeeze()
        # Typically it's 1D. We'll assume the first sync pulse is sample_numbers[0].
        first_sync_sample = sample_numbers[0]
        # Convert that to ms
        first_sync_ms = first_sync_sample / 30.0
    
    # Now we want to shift the session's start forward by first_sync_ms
    # So that time=0 in our local bins corresponds to the first sync pulse
    new_session_start = session_abs_start + first_sync_ms
    new_session_duration = session_abs_end - new_session_start
    
    # Bin the spikes for this truncated session
    spike_count_mat = bin_spikes(
        spike_times_ms=spike_times_ms,
        spike_clusters=spike_clusters,
        good_clusters=good_clusters,
        bin_size_ms=bin_size_ms,
        session_offset=new_session_start,
        session_duration_ms=new_session_duration
    )
    
    session_spike_counts.append(spike_count_mat)
    
    print(f"Session {i}: {file_path}")
    print(f"   Original: start_ms={session_abs_start:.2f}, end_ms={session_abs_end:.2f}")
    print(f"   First sync pulse at {first_sync_sample} samples => {first_sync_ms:.2f} ms")
    print(f"   Final binning range: [{new_session_start:.2f}, {new_session_start + new_session_duration:.2f}) ms")
    print(f"   Output shape: {spike_count_mat.shape}\n")

# -------------------------------------------------------------------------
# 6) Save each session’s 2D spike count matrix
# -------------------------------------------------------------------------
output_dir = '/Users/AdamHarris/Desktop/test_output_mats'
os.makedirs(output_dir, exist_ok=True)

for i, (file_path, _, _) in enumerate(session_info):
    out_name = f"binnedSpikes_session{i}.npy"
    out_path = os.path.join(output_dir, out_name)
    np.save(out_path, session_spike_counts[i])
    print(f"Saved binned spikes for session {i} -> {out_path}")

print("All sessions processed and binned spike arrays saved.")

if __name__ == '__main__':
main()







def filter_low_fr(session_spike_counts, good_clusters)
    n_sessions = len(session_spike_counts)
    n_good     = len(good_clusters)
    FR_means   = np.zeros((n_sessions, n_good), dtype=np.float32)
    
    for s in range(n_sessions):
        spike_count_mat = session_spike_counts[s]  # shape: (n_good, n_time_bins)
        total_spikes = np.sum(spike_count_mat, axis=1)  # sum across bins => shape: (n_good,)
        
        # each bin is 25 ms => 0.025 s. So total session time in seconds:
        total_time_s = spike_count_mat.shape[1] * (bin_size_ms / 1000.0)
        
        # Firing rate (spikes/sec) for each cluster
        # if total_time_s=0, we'd get a divide-by-zero; we can handle that
        if total_time_s > 0:
            cluster_fr = total_spikes / total_time_s
        else:
            cluster_fr = np.zeros(n_good, dtype=np.float32)
        
        FR_means[s, :] = cluster_fr
    
    # session_means: overall FR across all clusters for each session
    # if session_means == 0 => that session is considered invalid
    session_means = np.mean(FR_means, axis=1)  # shape: (n_sessions,)
    session_mask  = (session_means != 0.0)     # ignore sessions w/ zero overall FR
    
    # -------------------------------------------------------------------------
    # 7) Filter out clusters whose FR dips below threshold in ANY valid session
    # -------------------------------------------------------------------------
    FR_THRESHOLD = 0.002  # e.g. 0.002 spikes/s
    # For each cluster, we want to see if FR < 0.002 in any session where session_mask==True
    # => cluster fails if it is < 0.002 in ANY valid session.
    
    # FR_means[session_mask, :] => shape: (#valid_sessions, n_good)
    # (FR_means[session_mask, :] < FR_THRESHOLD) => bool array of same shape
    # .any(axis=0) => True for any cluster that fails
    clusters_below_threshold = (FR_means[session_mask, :] < FR_THRESHOLD).any(axis=0)
    # The final mask is True for clusters that pass
    FR_mask = ~clusters_below_threshold  # shape: (n_good,)
    
    # We'll create a new list of truly-good clusters
    filtered_good_clusters = [c for c, keep in zip(good_clusters, FR_mask) if keep]
    print(f"Filtering: {np.sum(~FR_mask)} clusters fail FR< {FR_THRESHOLD} criterion.") 
    print(f"Remaining: {np.sum(FR_mask)} clusters.")
    
    # Now, also filter the session_spike_counts arrays
    filtered_spike_counts = []
    for s in range(n_sessions):
        mat = session_spike_counts[s]  # shape: (n_good, n_time_bins)
        mat_filtered = mat[FR_mask, :] # keep only clusters that pass
        filtered_spike_counts.append(mat_filtered)
    
    # -------------------------------------------------------------------------
    # 8) Save final, filtered results
    # -------------------------------------------------------------------------
    # We can either overwrite the old output or store in new files
    filtered_output_dir = '/Users/AdamHarris/Desktop/test_output_mats_filtered_2123'
    os.makedirs(filtered_output_dir, exist_ok=True)
    
    for s, (file_path, _, _) in enumerate(session_info):
        out_path = os.path.join(filtered_output_dir, f"binnedSpikes_session{s}.npy")
        np.save(out_path, filtered_spike_counts[s])
    
    # We might also want to save the new list of clusters
    cluster_out_path = os.path.join(filtered_output_dir, "good_clusters_filtered.npy")
    np.save(cluster_out_path, np.array(filtered_good_clusters))
    
    print("All sessions binned and firing-rate-filtered. Final # of clusters:", len(filtered_good_clusters))

if __name__ == '__main__':
    main()