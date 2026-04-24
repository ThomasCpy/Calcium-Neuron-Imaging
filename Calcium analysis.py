import os
import re
import glob
import numpy as np
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, filters, feature, exposure, morphology, segmentation, measure
from tqdm import tqdm
import concurrent.futures

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
            'time': t_point,
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
    distance = ndi.distance_transform_edt(mask_cleaned)
    distance_smoothed = filters.gaussian(distance, sigma=2.0)
    
    # topography balancing shape (distance) and signal (intensity)
    weighted_topography = distance_smoothed * img_smoothed
    
    # find regional maxima. min_distance dictates how close two somas can physically be before fusing.
    coords = feature.peak_local_max(weighted_topography, min_distance=min_distance, labels=mask_cleaned)
    marker_mask = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)
    
    # watershed to cut touching dendrites
    cell_labels = segmentation.watershed(-weighted_topography, markers, mask=mask_cleaned)
    
    # locate the brightest central core for the roi
    # we use the non-clahe original_img here so the centroid reflects true calcium peaks
    soma_rois = []
    props = measure.regionprops(cell_labels, intensity_image=original_img)
    
    for prop in props:
        y0, x0 = prop.weighted_centroid
        soma_rois.append({
            'id': prop.label,
            'y': int(y0),
            'x': int(x0),
            'radius': roi_radius
        })
        
    return cell_labels, soma_rois

def save_side_by_side_qc(img_raw, cell_labels, soma_rois, out_path, base_name):
    """plots the raw preprocessed image against the segmented masks and soma rois."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    p2, p98 = np.percentile(img_raw, (2, 98))
    img_vis = exposure.rescale_intensity(img_raw, in_range=(p2, p98))
    
    axes[0].imshow(img_vis, cmap='inferno')
    axes[0].set_title(f'{base_name} - raw input', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_vis, cmap='inferno')
    
    if cell_labels is not None and np.max(cell_labels) > 0:
        boundaries = segmentation.find_boundaries(cell_labels, mode='inner')
        bound_rgba = np.zeros((*img_vis.shape, 4), dtype=np.float32)
        bound_rgba[boundaries] = [0, 1, 0, 0.5]  # green borders
        axes[1].imshow(bound_rgba)
        
        for roi in soma_rois:
            circ = mpatches.Circle(
                (roi['x'], roi['y']), 
                radius=roi['radius'], 
                fill=False, 
                edgecolor='red', 
                linewidth=1.2
            )
            axes[1].add_patch(circ)
            # keeping labels small to prevent clutter in dense networks
            axes[1].text(roi['x'] + roi['radius'], roi['y'] - roi['radius'], str(roi['id']), color='white', fontsize=6)
            
    axes[1].set_title(f'segmented boundaries & soma rois ({len(soma_rois)} cells)', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='black')
    fig.clf()
    plt.close('all')

# ==========================================
# PARALLEL WORKER ENGINE
# ==========================================

def _worker_process_timepoint(args):
    """isolated worker for handling a single calcium imaging timepoint."""
    (file_path, meta, output_paths_dict, threshold_method, tophat_radius, roi_radius, min_soma_dist) = args
    
    well = meta['well']
    field = meta['field']
    time_point = meta['time']
    
    dest_dir = output_paths_dict.get((well, field))
    if not dest_dir: return None
    
    base_filename = os.path.basename(file_path).split('.')[0]
    out_img_path = os.path.join(dest_dir, f"t{time_point}_seg.png")
    
    if os.path.exists(out_img_path): return
    
    img_raw = load_as_float(file_path)
    if img_raw is None or np.max(img_raw) == 0: return None
        
    # preprocess with clahe
    img_denoised, img_clahe, img_tophat = preprocess_image(img_raw, tophat_radius=tophat_radius)
    
    # segment using clahe features but extract true peaks from the raw denoised image
    labels, soma_rois = segment_calcium_cells(
        img_tophat, 
        original_img=img_denoised, 
        method=threshold_method, 
        min_cell_area=80, 
        roi_radius=roi_radius,
        min_distance=min_soma_dist
    )
    
    save_side_by_side_qc(img_denoised, labels, soma_rois, out_img_path, base_filename)
    return True

def process_calcium_directory(
    data_dir, 
    target_channel="1", 
    threshold_method="triangle",
    tophat_radius=20,
    roi_radius=6,
    min_soma_dist=10
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
    _, structure_dict = setup_dynamic_output(base_out_dir, [f[0] for f in target_files], "Output Calcium Segmentation")
    
    tasks = []
    for f_path, meta in target_files:
        tasks.append((f_path, meta, structure_dict, threshold_method, tophat_radius, roi_radius, min_soma_dist))

    max_workers = os.cpu_count() if os.cpu_count() else 1
    print(f"\nlaunching parallel extraction on {max_workers} cpu cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_process_timepoint, t): t for t in tasks}
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing Timepoints"):
            pass
            
    print("\ncalcium pipeline complete.")

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    data_path = r"C:/Users/lab/OneDrive - Vandria/Bureau/Calcium imaging/Pilot experiment"
    
    process_calcium_directory(
        data_dir=data_path,
        target_channel="1",       
        threshold_method="li",  
        tophat_radius=25,         
        roi_radius=6,             
        min_soma_dist=10          
    )