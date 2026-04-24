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
    # regex to capture t(3 digits), well(letter+2 digits), f(2 digits), d(1 digit)
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
    returns a dictionary mapping (well, field) -> output_directory
    """
    output_dir = os.path.join(base_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    structure_map = {}
    
    for f_path in file_list:
        meta = parse_filename_metadata(os.path.basename(f_path))
        if not meta: continue
            
        well = meta['well']
        field = meta['field']
        
        # create well folder
        well_path = os.path.join(output_dir, well)
        os.makedirs(well_path, exist_ok=True)
        
        # create field subfolder
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
    elif img_max <= 4095:       # 12-bit camera
        scale_factor = 4095.0
    elif img_max <= 16383:      # 14-bit camera
        scale_factor = 16383.0
    else:                       # 16-bit camera
        scale_factor = 65535.0
        
    return img.astype(np.float32) / scale_factor

def preprocess_image(img, tophat_radius=15):
    """cleans up noise and equalizes uneven illumination common in fluo imaging"""
    # median filter for salt and pepper noise
    img_denoised = filters.median(img, morphology.disk(1))
    
    # tophat subtracts background gradients
    selem = morphology.disk(tophat_radius)
    img_tophat = morphology.white_tophat(img_denoised, footprint=selem)
    
    return img_denoised, img_tophat

def segment_calcium_cells(img_tophat, original_img, method='triangle', min_cell_area=100, roi_radius=6):
    """
    uses watershed to separate close cells, weighting the distance by intensity.
    extracts the high intensity soma center for roi placement.
    """
    img_smoothed = filters.gaussian(img_tophat, sigma=1.5)
    
    if method == 'triangle':
        thresh = filters.threshold_triangle(img_smoothed)
    elif method == 'li':
        thresh = filters.threshold_li(img_smoothed + 1e-8)
    else:
        thresh = filters.threshold_otsu(img_smoothed)
        
    mask = img_smoothed > thresh
    mask_cleaned = morphology.remove_small_objects(mask, min_size=min_cell_area)
    
    # distance transform multiplied by intensity to strongly bias peak detection towards the soma
    distance = ndi.distance_transform_edt(mask_cleaned)
    weighted_topography = distance * img_smoothed
    
    # find regional maxima (soma centers)
    coords = feature.peak_local_max(weighted_topography, min_distance=15, labels=mask_cleaned)
    marker_mask = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)
    
    # watershed to cut touching dendrites/cells
    cell_labels = segmentation.watershed(-weighted_topography, markers, mask=mask_cleaned)
    
    # locate the brightest central core of each label to set the elliptic/circular roi
    soma_rois = []
    props = measure.regionprops(cell_labels, intensity_image=original_img)
    
    for prop in props:
        # weighted_centroid pulls the coordinate onto the brightest pixel of the body
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
        # overlay green borders for the entire cellular region (dendrites + body)
        boundaries = segmentation.find_boundaries(cell_labels, mode='inner')
        bound_rgba = np.zeros((*img_vis.shape, 4), dtype=np.float32)
        bound_rgba[boundaries] = [0, 1, 0, 0.7]  # green, alpha 0.7
        axes[1].imshow(bound_rgba)
        
        # overlay red circular rois on the bright somas
        for roi in soma_rois:
            circ = mpatches.Circle(
                (roi['x'], roi['y']), 
                radius=roi['radius'], 
                fill=False, 
                edgecolor='red', 
                linewidth=1.5
            )
            axes[1].add_patch(circ)
            axes[1].text(roi['x'] + roi['radius'], roi['y'] - roi['radius'], str(roi['id']), color='red', fontsize=8)
            
    axes[1].set_title(f'segmented boundaries & soma rois ({len(soma_rois)} cells)', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='black')
    fig.clf()
    plt.close('all')


def _worker_process_timepoint(args):
    """isolated worker for handling a single calcium imaging timepoint."""
    (file_path, meta, output_paths_dict, threshold_method, tophat_radius, roi_radius) = args
    
    well = meta['well']
    field = meta['field']
    time_point = meta['time']
    
    dest_dir = output_paths_dict.get((well, field))
    if not dest_dir: return None
    
    base_filename = os.path.basename(file_path).split('.')[0]
    out_img_path = os.path.join(dest_dir, f"t{time_point}_seg.png")
    
    # bypass if already processed to allow resuming
    if os.path.exists(out_img_path): return
    
    img_raw = load_as_float(file_path)
    if img_raw is None or np.max(img_raw) == 0: return None
        
    # preprocess and isolate background
    img_denoised, img_tophat = preprocess_image(img_raw, tophat_radius=tophat_radius)
    
    # extract cells and bodies
    labels, soma_rois = segment_calcium_cells(
        img_tophat, 
        original_img=img_denoised, 
        method=threshold_method, 
        min_cell_area=150, 
        roi_radius=roi_radius
    )
    
    # generate qc visual
    save_side_by_side_qc(img_denoised, labels, soma_rois, out_img_path, base_filename)
    
    return True

def process_calcium_directory(
    data_dir, 
    target_channel="1", 
    threshold_method="triangle",
    tophat_radius=20,
    roi_radius=6
):
    """master pipeline orchestrator."""
    search_pattern = os.path.join(data_dir, "*.[tT][iI][fF]*")
    all_files = glob.glob(search_pattern)
    
    # filter files by target calcium channel using regex
    target_files = []
    for f in all_files:
        meta = parse_filename_metadata(os.path.basename(f))
        if meta and meta['channel'] == target_channel:
            target_files.append((f, meta))
            
    if not target_files:
        print("no valid calcium files found matching the regex pattern.")
        return

    # prepare  output structure based on discovered wells/fields
    base_out_dir = os.path.dirname(data_dir)
    _, structure_dict = setup_dynamic_output(base_out_dir, [f[0] for f in target_files], "Output Calcium Segmentation")
    
    tasks = []
    for f_path, meta in target_files:
        tasks.append((f_path, meta, structure_dict, threshold_method, tophat_radius, roi_radius))

    # deploying multiprocessing 
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

    data_path = r"C:/Users/lab/OneDrive - Vandria/Bureau/Calcium/Raw_Images"
    
    process_calcium_directory(
        data_dir=data_path,
        target_channel="1",      
        threshold_method="otsu",  # triangle, li, or otsu
        tophat_radius=25,         # size of background rolling ball subtraction
        roi_radius=7              # size of the soma bounding circle for future trace extraction
    )