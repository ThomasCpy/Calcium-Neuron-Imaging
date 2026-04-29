import os
import re
import glob
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import distance
from scipy.cluster import hierarchy
import scipy.signal as signal
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from skimage import io, filters, feature, exposure, morphology, segmentation, measure
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict

# ==========================================
# IO AND METADATA SETUP
# ==========================================

def parse_filename_metadata(filename):
    """
    extracts time, well, field, and channel from the filename string.
    expected format pattern: ...t001G02f32d1...
    """
    pattern = re.compile(r't(\d{3})([A-H]\d{2})f(\d{2})d(\d)')
    match = pattern.search(filename)
    
    if match:
        t_point, well, field, channel = match.groups()
        return {
            'time': int(t_point), # cast to int for easier sorting later
            'well': well,
            'field': field,
            'channel': channel
        }
    return None

def setup_dynamic_output(base_path, file_list, folder_name="Output Calcium"):
    """
    scans files to dynamically create the nested well > field folder structure.
    """
    output_dir = os.path.join(base_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    structure_map = {}
    
    for f_path in file_list:
        meta = parse_filename_metadata(os.path.basename(f_path))
        if not meta: continue
            
        well = meta['well']
        field = meta['field']
        
        well_path = os.path.join(output_dir, well)
        os.makedirs(well_path, exist_ok=True)
        
        field_path = os.path.join(well_path, f"Field_{field}")
        os.makedirs(field_path, exist_ok=True)
        
        structure_map[(well, field)] = field_path
        
    print(f"generated dynamic calcium plate structure in: {output_dir}")
    return output_dir, structure_map

# ==========================================
# IMAGE PROCESSING FUNCTIONS
# ==========================================

def load_as_float(image_path):
    """loads image and dynamically scales bit-depth ceiling to 0.0 - 1.0"""
    if not os.path.exists(image_path):
        return None
        
    img = io.imread(image_path, as_gray=True)
    if np.max(img) == 0:
        return img.astype(np.float32)
        
    img_max = np.max(img)
    
    if img.dtype == np.uint8 or img_max <= 255:
        scale_factor = 255.0
    elif img_max <= 4095:       
        scale_factor = 4095.0
    elif img_max <= 16383:      
        scale_factor = 16383.0
    else:                       
        scale_factor = 65535.0
        
    return img.astype(np.float32) / scale_factor

def preprocess_image(img, tophat_radius=15):
    """cleans up noise, equalizes lighting locally, and flattens background"""
    img_denoised = filters.median(img, morphology.disk(1))
    
    img_clahe = exposure.equalize_adapthist(img_denoised, kernel_size=32, clip_limit=0.02)
    
    # tophat on the clahe image isolates the cells from any residual background gradients
    selem = morphology.disk(tophat_radius)
    img_tophat = morphology.white_tophat(img_clahe, footprint=selem)
    
    return img_denoised, img_clahe, img_tophat

def segment_calcium_cells(img_tophat, original_img, method='triangle', min_cell_area=80, roi_radius=6, min_distance=10):
    """
    uses clahe-enhanced inputs to segment highly heterogeneous calcium fields.
    """
    img_smoothed = filters.gaussian(img_tophat, sigma=1.0)
    
    if method == 'triangle':
        thresh = filters.threshold_triangle(img_smoothed)
    elif method == 'li':
        thresh = filters.threshold_li(img_smoothed + 1e-8)
    else:
        thresh = filters.threshold_otsu(img_smoothed)
        
    mask = img_smoothed > (thresh * 0.8)
    mask_cleaned = morphology.remove_small_objects(mask, min_size=min_cell_area)
    
    # smoothing the distance to prevent false positives
    distance_img = ndi.distance_transform_edt(mask_cleaned)
    distance_smoothed = filters.gaussian(distance_img, sigma=2.0)
    
    # topography balancing shape (distance) and signal (intensity)
    weighted_topography = distance_smoothed * img_smoothed
    
    # find regional maxima. min_distance dictates how close two somas can physically be before fusing.
    coords = feature.peak_local_max(weighted_topography, min_distance=min_distance, labels=mask_cleaned)
    marker_mask = np.zeros(distance_img.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)
    
    # watershed to cut touching dendrites
    cell_labels = segmentation.watershed(-weighted_topography, markers, mask=mask_cleaned)
    
    # locate the brightest central core for the roi
    soma_rois = []
    props = measure.regionprops(cell_labels, intensity_image=original_img)
    
    for prop in props:
        y0, x0 = prop.weighted_centroid
        soma_rois.append({
            'y': int(y0),
            'x': int(x0),
            'radius': roi_radius
        })
        
    return cell_labels, soma_rois

def extract_roi_intensities(img, rois):
    """
    extracts the mean pixel intensity for each defined roi from the provided image.
    we use the raw denoised image here to preserve linear fluorescence dynamics.
    """
    intensities = {}
    h, w = img.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    
    for roi in rois:
        # create circular mask for the roi
        mask = ((y_grid - roi['y'])**2 + (x_grid - roi['x'])**2) <= roi['radius']**2
        if np.any(mask):
            intensities[roi['id']] = np.mean(img[mask])
        else:
            intensities[roi['id']] = 0.0
            
    return intensities

def save_side_by_side_qc(img_raw, cell_labels, stable_rois, out_path, base_name):
    """plots the raw preprocessed image against the segmented masks and consensus stable rois."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    p2, p98 = np.percentile(img_raw, (2, 98))
    img_vis = exposure.rescale_intensity(img_raw, in_range=(p2, p98))
    
    axes[0].imshow(img_vis, cmap='inferno')
    axes[0].set_title(f'{base_name} - raw input', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_vis, cmap='inferno')
    
    # plot the temporal boundaries of the current frame if they exist
    if cell_labels is not None and np.max(cell_labels) > 0:
        boundaries = segmentation.find_boundaries(cell_labels, mode='inner')
        bound_rgba = np.zeros((*img_vis.shape, 4), dtype=np.float32)
        bound_rgba[boundaries] = [0, 1, 0, 0.4]  # faint green borders for current activity
        axes[1].imshow(bound_rgba)
        
    # plot the stable, cross-temporal rois
    for roi in stable_rois:
        circ = mpatches.Circle(
            (roi['x'], roi['y']), 
            radius=roi['radius'], 
            fill=False, 
            edgecolor='red', 
            linewidth=1.5
        )
        axes[1].add_patch(circ)
        # keeping labels small to prevent clutter in dense networks
        axes[1].text(roi['x'] + roi['radius'], roi['y'] - roi['radius'], str(roi['id']), color='white', fontsize=7, fontweight='bold')
            
    axes[1].set_title(f'stable consensus rois ({len(stable_rois)} cells)', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='black')
    fig.clf()
    plt.close('all')

# ==========================================
# SIGNAL PROCESSING & TRACE ANALYSIS
# ==========================================

def calculate_df_f0(trace, window=20, method='percentile'):
    """
    calculates delta f / f0. 
    method 'percentile': uses a rolling percentile (handles slow drift/photobleaching).
    method 'global_decay': fits a 2nd degree polynomial to the resting state to cancel initial slope.
    """
    trace_array = np.array(trace)
    
    if method == 'percentile':
        # use 20th percentile as baseline (f0)
        f0 = ndi.percentile_filter(trace_array, percentile=20, size=window, mode='nearest')
    elif method == 'global_decay':
        # fit a quadratic decay curve to capture the global slope/photobleaching.
        x = np.arange(len(trace_array))
        # to prevent spikes from pulling the baseline up, we fit only to the lower 50% of the trace
        median_val = np.median(trace_array)
        mask = trace_array <= median_val
        
        if np.sum(mask) < 3: # fallback if trace is exceptionally flat/short
            mask = np.ones_like(trace_array, dtype=bool)
            
        coeffs = np.polyfit(x[mask], trace_array[mask], deg=2)
        f0 = np.polyval(coeffs, x)
    else:
        raise ValueError(f"unknown baseline method: {method}")
        
    # prevent division by zero
    f0[f0 == 0] = np.nan 
    df_f0 = (trace_array - f0) / f0
    return df_f0

def analyze_traces_and_plot(well_field_data, stable_rois, dest_dir, well, field, fps=1.0, peak_prominence=0.1, peak_distance=3, baseline_method='percentile'):
    """
    sorts temporal data, extracts raw and delta f / f0, finds peaks, 
    and plots all cells overlapped side-by-side (raw vs normalized) to check compensation.
    returns a list of dictionaries with stats per cell.
    """
    timepoints = sorted(list(well_field_data.keys()))
    if not timepoints: return []
    
    cell_stats = []
    n_cells = len(stable_rois)
    
    # setup trace plot: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    
    t_sec = np.array(timepoints) / fps
    # generate a color palette for the cells so overlapping lines are distinguishable
    colors = plt.cm.viridis(np.linspace(0, 1, n_cells))
    
    for i, roi in enumerate(stable_rois):
        roi_id = roi['id']
        raw_trace = [well_field_data[t].get(roi_id, np.nan) for t in timepoints]
        
        # interpolate missing data if a frame extraction failed
        s = pd.Series(raw_trace)
        s = s.interpolate(method='linear').bfill().ffill()
        raw_trace = s.values
        
        df_f0 = calculate_df_f0(raw_trace, method=baseline_method)
        
        # find peaks (prominence threshold filters out noise ripples)
        peaks, properties = signal.find_peaks(df_f0, prominence=peak_prominence, distance=peak_distance)
        
        freq = len(peaks) / (len(timepoints) / fps) # peaks per second
        # amplitude: average peak height relative to baseline, or max-min if no peaks
        if len(peaks) > 0:
            amp = np.mean(df_f0[peaks])
        else:
            amp = np.nanmax(df_f0) - np.nanmin(df_f0)
            
        cell_stats.append({
            'well': well,
            'field': field,
            'roi_id': roi_id,
            'frequency_hz': freq,
            'amplitude_df_f0': amp
        })
        
        # plot raw trace overlapped
        axes[0].plot(t_sec, raw_trace, color=colors[i], lw=1.2, alpha=0.7)
        
        # plot normalized trace overlapped
        axes[1].plot(t_sec, df_f0, color=colors[i], lw=1.2, alpha=0.7)
        
    axes[0].set_title(f"{well} Field {field} - raw fluorescence", fontweight='bold')
    axes[0].set_ylabel("raw intensity (a.u.)")
    axes[0].set_xlabel("time (s)")
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    axes[1].set_title(f"{well} Field {field} - $\Delta F/F_0$ (normalized)", fontweight='bold')
    axes[1].set_ylabel(r"$\Delta F/F_0$")
    axes[1].set_xlabel("time (s)")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
        
    plt.tight_layout()
    out_path = os.path.join(dest_dir, f"{well}_Field{field}_traces.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    return cell_stats

# ==========================================
# CONSENSUS ROI TRACKING
# ==========================================

def compute_stable_rois_for_field(field_data, total_timepoints, presence_threshold=0.75, clustering_dist=10):
    """
    clusters all detected rois across timepoints to find consistent cells.
    if a cell appears in at least (presence_threshold * 100)% of frames, it becomes a stable roi.
    """
    all_points = []
    all_times = []
    
    # flatten the temporal data
    for t_point, rois in field_data.items():
        for r in rois:
            all_points.append([r['x'], r['y']])
            all_times.append(t_point)
            
    if not all_points:
        return []
        
    all_points = np.array(all_points)
    all_times = np.array(all_times)
    
    # hierarchical clustering based on spatial distance
    if len(all_points) > 1:
        dist_matrix = distance.pdist(all_points)
        z = hierarchy.linkage(dist_matrix, method='complete')
        labels = hierarchy.fcluster(z, t=clustering_dist, criterion='distance')
    else:
        labels = np.array([1])
        
    stable_rois = []
    roi_counter = 1
    
    unique_clusters = set(labels)
    for cluster_id in unique_clusters:
        idx = np.where(labels == cluster_id)[0]
        cluster_times = all_times[idx]
        
        unique_appearances = len(set(cluster_times))
        
        if (unique_appearances / total_timepoints) >= presence_threshold:
            cluster_pts = all_points[idx]
            mean_x, mean_y = np.mean(cluster_pts, axis=0)
            
            stable_rois.append({
                'id': roi_counter,
                'x': int(mean_x),
                'y': int(mean_y),
                'radius': 6 
            })
            roi_counter += 1
            
    return stable_rois

# ==========================================
# PARALLEL WORKER ENGINE
# ==========================================

def _worker_extract_rois(args):
    """pass 1: isolates rois for a single timepoint without plotting."""
    (file_path, meta, threshold_method, tophat_radius, roi_radius, min_soma_dist) = args
    
    img_raw = load_as_float(file_path)
    if img_raw is None or np.max(img_raw) == 0: 
        return (meta, [])
        
    img_denoised, img_clahe, img_tophat = preprocess_image(img_raw, tophat_radius=tophat_radius)
    
    _, soma_rois = segment_calcium_cells(
        img_tophat, 
        original_img=img_denoised, 
        method=threshold_method, 
        min_cell_area=80, 
        roi_radius=roi_radius,
        min_distance=min_soma_dist
    )
    
    return (meta, soma_rois)

def _worker_process_frame(args):
    """pass 2: plots stable rois and extracts raw intensity for this frame."""
    (file_path, meta, stable_rois, output_paths_dict, threshold_method, tophat_radius, roi_radius, min_soma_dist) = args
    
    well = meta['well']
    field = meta['field']
    time_point = meta['time']
    
    dest_dir = output_paths_dict.get((well, field))
    if not dest_dir: return None
    
    base_filename = os.path.basename(file_path).split('.')[0]
    out_img_path = os.path.join(dest_dir, f"t{time_point:03d}_seg.png")
    
    img_raw = load_as_float(file_path)
    if img_raw is None or np.max(img_raw) == 0: return None
        
    img_denoised, img_clahe, img_tophat = preprocess_image(img_raw, tophat_radius=tophat_radius)
    
    # extract intensity from raw denoised image
    intensities = extract_roi_intensities(img_denoised, stable_rois)
    
    if not os.path.exists(out_img_path):
        cell_labels, _ = segment_calcium_cells(
            img_tophat, 
            original_img=img_denoised, 
            method=threshold_method, 
            min_cell_area=80, 
            roi_radius=roi_radius,
            min_distance=min_soma_dist
        )
        save_side_by_side_qc(img_denoised, cell_labels, stable_rois, out_img_path, base_filename)
        
    return (well, field, time_point, intensities)

def process_calcium_directory(
    data_dir, 
    condition_map,
    target_channel="1", 
    threshold_method="triangle",
    tophat_radius=20,
    roi_radius=6,
    min_soma_dist=10,
    presence_threshold=0.75,
    acquisition_fps=1.0,
    peak_prominence=0.1,
    peak_distance=3,
    baseline_method='percentile'
):
    """master pipeline orchestrator."""
    search_pattern = os.path.join(data_dir, "*.[tT][iI][fF]*")
    all_files = glob.glob(search_pattern)
    
    target_files = []
    for f in all_files:
        meta = parse_filename_metadata(os.path.basename(f))
        if meta and meta['channel'] == target_channel:
            target_files.append((f, meta))
            
    if not target_files:
        print("no valid calcium files found matching the regex pattern.")
        return

    base_out_dir = os.path.dirname(data_dir)
    stats_out_dir, structure_dict = setup_dynamic_output(base_out_dir, [f[0] for f in target_files], "Output Calcium Analysis")
    
    if roifile is None:
        print("\n[!] warning: 'roifile' package not found. Fiji .roi files will not be exported.")
        print("    to enable this feature, run: pip install roifile")
    
    max_workers = os.cpu_count() if os.cpu_count() else 1
    
    # ---------------------------------------------------------
    # PASS 1: segment every frame to find physical cell locations
    # ---------------------------------------------------------
    print(f"\n[pass 1/3] extracting dynamic rois across {max_workers} cpu cores...")
    field_temporal_rois = defaultdict(lambda: defaultdict(list))
    total_timepoints_per_field = defaultdict(set)
    
    extract_tasks = [(f_path, meta, threshold_method, tophat_radius, roi_radius, min_soma_dist) for f_path, meta in target_files]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_extract_rois, t): t for t in extract_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(extract_tasks), desc="segmenting"):
            meta, soma_rois = future.result()
            well_field_key = (meta['well'], meta['field'])
            field_temporal_rois[well_field_key][meta['time']] = soma_rois
            total_timepoints_per_field[well_field_key].add(meta['time'])
            
    # ---------------------------------------------------------
    # CONSENSUS: define stable cells
    # ---------------------------------------------------------
    print("\ncomputing consensus stable rois for each field...")
    stable_rois_dict = {}
    for wf_key, time_data in field_temporal_rois.items():
        total_t = len(total_timepoints_per_field[wf_key])
        stable_rois = compute_stable_rois_for_field(time_data, total_timepoints=total_t, presence_threshold=presence_threshold, clustering_dist=min_soma_dist)
        stable_rois_dict[wf_key] = stable_rois
        print(f"  > {wf_key[0]} field {wf_key[1]}: found {len(stable_rois)} stable cells.")
        
        # export to fiji
        export_fiji_rois(stable_rois, stats_out_dir, wf_key[0], wf_key[1])

    # ---------------------------------------------------------
    # PASS 2: extract trace intensities and save qc images
    # ---------------------------------------------------------
    print(f"\n[pass 2/3] extracting raw intensities and plotting qc masks...")
    plot_tasks = []
    for f_path, meta in target_files:
        wf_key = (meta['well'], meta['field'])
        stable_rois = stable_rois_dict.get(wf_key, [])
        plot_tasks.append((f_path, meta, stable_rois, structure_dict, threshold_method, tophat_radius, roi_radius, min_soma_dist))

    # nested dict to hold time series data: traces_data[well][field][time][roi_id] = intensity
    traces_data = defaultdict(lambda: defaultdict(dict))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_process_frame, t): t for t in plot_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(plot_tasks), desc="processing signals"):
            result = future.result()
            if result:
                well, field, t_point, intensities = result
                traces_data[well][field][t_point] = intensities

    # ---------------------------------------------------------
    # PASS 3: process signals, plot traces, and aggregate stats
    # ---------------------------------------------------------
    print(f"\n[pass 3/3] computing df/f0, finding peaks, and generating statistics...")
    all_cell_stats = []
    
    for well, fields in traces_data.items():
        for field, time_data in fields.items():
            dest_dir = structure_dict[(well, field)]
            stable_rois = stable_rois_dict[(well, field)]
            
            stats = analyze_traces_and_plot(
                time_data, stable_rois, dest_dir, well, field, 
                fps=acquisition_fps, 
                peak_prominence=peak_prominence, 
                peak_distance=peak_distance,
                baseline_method=baseline_method
            )
            all_cell_stats.extend(stats)
            
    # compile into dataframe and generate violin plots
    if all_cell_stats:
        df_stats = pd.DataFrame(all_cell_stats)
        # map well to experimental condition (fallback to well name if not in map)
        df_stats['condition'] = df_stats['well'].map(lambda w: condition_map.get(w, w))
        
        csv_path = os.path.join(stats_out_dir, "calcium_cell_statistics.csv")
        df_stats.to_csv(csv_path, index=False)
        
        # violin plots for frequency and amplitude
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.violinplot(data=df_stats, x='condition', y='frequency_hz', ax=axes[0], palette='pastel', inner="quartile")
        sns.stripplot(data=df_stats, x='condition', y='frequency_hz', ax=axes[0], color='black', alpha=0.5, size=3)
        axes[0].set_title('Oscillation Frequency by Condition')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.violinplot(data=df_stats, x='condition', y='amplitude_df_f0', ax=axes[1], palette='pastel', inner="quartile")
        sns.stripplot(data=df_stats, x='condition', y='amplitude_df_f0', ax=axes[1], color='black', alpha=0.5, size=3)
        axes[1].set_title('Oscillation Amplitude by Condition')
        axes[1].set_ylabel(r'Amplitude ($\Delta F/F_0$)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_out_dir, "population_violin_plots.png"), dpi=200)
        plt.close()
        
    print(f"\ncalcium pipeline complete. output saved to: {stats_out_dir}")

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    data_path = r"C:/Users/lab/OneDrive - Vandria/Bureau/Calcium imaging/Pilot experiment"
    
    # user maps wells to conditions here for the violin plots
    exp_condition_map = {
        'G02': 'Control',
        'G03': 'Control',
        'G04': 'Control',
        'G05': 'Control',
        # any well not listed will just show up as its well name (e.g., 'C01')
    }
    
    process_calcium_directory(
        data_dir=data_path,
        condition_map=exp_condition_map,
        target_channel="1",       
        threshold_method="li",  
        tophat_radius=25,         
        roi_radius=6,             
        min_soma_dist=10,
        presence_threshold=0.75, 
        acquisition_fps=1.0,      # change to your actual camera frame rate (hz)
        peak_prominence=0.1,      # min height above baseline to count as a true calcium peak
        peak_distance=3,          # min frames between spikes to prevent double counting noise
        baseline_method='global_decay' # toggle between 'percentile' and 'global_decay'
    )