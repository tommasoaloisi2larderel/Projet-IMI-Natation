import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import os

def rectify_pool(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image Shape: {img.shape}")
    
    # Create output directory for this image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("lane_outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Preprocessing and Lane Line Detection
    # Convert to HSV to isolate colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for "lane line" colors (often red/white/blue, but let's start with a general heuristic or edge detection)
    # The pool lines are usually distinct against the blue water.
    # Let's try edge detection first, maybe with a specific color mask if needed.
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Edge detection on enhanced image
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Save edges for inspection
    cv2.imwrite(os.path.join(output_dir, "debug_edges.png"), edges)
    
    # Hough Line Transform (Probabilistic)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Identify pool lanes: filter lines based on angle
    # Horizontal lines (lanes)
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Horizontal: close to 0 or 180
            if abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines.append(line)
                
            # Vertical: close to 90 or -90
            elif abs(abs(angle) - 90) < 30: # Allow some slant
                vertical_lines.append(line[0])
            
    print(f"Divided into {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines.")

    # Visualize detected vertical lines
    debug_v = img.copy()
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(debug_v, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "debug_vertical_lines.png"), debug_v)
    
    # Identify pool lanes: filter lines based on angle/length and merge them
    merged_lines = merge_lines(horizontal_lines, img.shape)
    
    line_image = img.copy()
    if merged_lines:
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Identify markings along the line
            # Sample pixel intensities (or specific color channels) along the line
            profile = get_line_profile(img, line)
            
            # Detect transitions (peaks/troughs in derivative)
            transitions = detect_transitions(profile)
            
            # Visualize detected points on the line
            for i in transitions:
                # Interpolate back to image coordinates
                t = i / len(profile)
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                cv2.circle(line_image, (px, py), 5, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(output_dir, "debug_lines_merged.png"), line_image)
    print(f"Detected markings saved to {output_dir}/debug_lines_merged.png")
    
    # Perspective Transform Logic using Line Distances (User Request)
    if len(merged_lines) >= 2:
        print(f"Using {len(merged_lines)} lines for perspective correction.")
        
        # Sort lines by vertical position (Y)
        # We use the midpoint Y to sort
        merged_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        
        src_pts = []
        dst_pts = []
        
        # Define target dimensions
        target_w = img.shape[1]
        target_h = img.shape[0]
        
        # Define target spacing for lines
        # We want them equidistant.
        # Let's map the top line to a margin and bottom line to a margin, distributing others in between.
        margin = 50
        total_spacing_h = target_h - 2 * margin
        if len(merged_lines) > 1:
            step = total_spacing_h / (len(merged_lines) - 1)
        else:
            step = 0 # Should not happen with >=2 check
            
            step = 0 
            
        # Debug: Print the Y-coordinates of the lines at the center
        center_x = img.shape[1] // 2
        y_coords = []
        for l in merged_lines:
            y_coords.append( (l[1] + l[3]) // 2 )
        print(f"Detected Y-coordinates (approx): {y_coords}")
        
        
        # Assemble optimization data
        # Horizontal Points
        h_pts = []
        h_targets = []
        
        for i, line in enumerate(merged_lines):
            x1, y1, x2, y2 = line
            target_y = margin + i * step
            
            h_pts.append([x1, y1])
            h_targets.append(target_y)
            h_pts.append([x2, y2])
            h_targets.append(target_y)
            
        h_pts = np.array(h_pts)
        h_targets = np.array(h_targets)
        
        # Vertical Points
        # We want x1' == x2' for each line. We don't care *where* in X they are, just that dx=0.
        v_pairs = []
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            v_pairs.append([x1, y1, x2, y2])
        v_pairs = np.array(v_pairs)
        
        print(f"Optimization with {len(h_pts)} horizontal points and {len(v_pairs)} vertical constraints.")

        # Optimization Function
        def homography_residuals(params, h_pts, h_targets, v_pairs):
            # params: h11, h12, h13, h21, h22, h23, h31, h32 (h33=1)
            h = np.append(params, 1.0).reshape(3, 3)
            
            residuals = []
            
            # 1. Horizontal Lines -> Y target
            if len(h_pts) > 0:
                ones = np.ones((len(h_pts), 1))
                pts_h = np.hstack((h_pts, ones))
                proj = pts_h @ h.T
                w = proj[:, 2]
                w[abs(w) < 1e-5] = 1e-5
                y_prime = proj[:, 1] / w
                
                # We also want to weakly constrain X to original X to avoid flipping?
                # Actually, main constraint is Y.
                err_y = (y_prime - h_targets) * 10.0 # Weight 10
                residuals.extend(err_y)
                
                # Weak X regularization for horizontal lines to keep image roughly centered/stable
                x_prime = proj[:, 0] / w
                err_x_reg = (x_prime - h_pts[:, 0]) * 0.01 
                residuals.extend(err_x_reg)

            # 2. Vertical Lines -> Vertical (dx = 0)
            if len(v_pairs) > 0:
                # p1
                p1 = v_pairs[:, :2]
                ones1 = np.ones((len(p1), 1))
                ph1 = np.hstack((p1, ones1))
                proj1 = ph1 @ h.T
                w1 = proj1[:, 2]
                w1[abs(w1) < 1e-5] = 1e-5
                x1_prime = proj1[:, 0] / w1
                
                # p2
                p2 = v_pairs[:, 2:]
                ones2 = np.ones((len(p2), 1))
                ph2 = np.hstack((p2, ones2))
                proj2 = ph2 @ h.T
                w2 = proj2[:, 2]
                w2[abs(w2) < 1e-5] = 1e-5
                x2_prime = proj2[:, 0] / w2
                
                # Error: x1' - x2' should be 0
                err_v = (x1_prime - x2_prime) * 5.0 # Weight 5
                residuals.extend(err_v)
            
            return np.array(residuals)

        # Initial Guess
        initial_h = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        
        print("Optimizing Homography with Vertical Constraints...")
        res = least_squares(homography_residuals, initial_h, args=(h_pts, h_targets, v_pairs), method='lm')
        
        h_opt = np.append(res.x, 1.0).reshape(3, 3)
        print(f"Optimized H:\n{h_opt}")
        print(f"Cost: {res.cost}")
        
        M = h_opt
        
        if M is not None:
             warped = cv2.warpPerspective(img, M, (target_w, target_h))
             cv2.imwrite(os.path.join(output_dir, "Piscine_top_down.png"), warped)
             
             comparison = np.hstack((img, warped))
             cv2.imwrite(os.path.join(output_dir, "comparison.png"), comparison)
             print("Saved Piscine_top_down.png and comparison.png")
             
             # PLOT DISTANCES
             # Measure actual distances in the warped image at center column
             center_col = target_w // 2
             
             # Sample the warped image to find lines? 
             # Or transform the original measurements?
             # Let's transform the original line points to see where they ended up
             
             ones = np.ones((len(h_pts), 1))
             src_h = np.hstack((h_pts, ones))
             dst_h = src_h @ M.T
             dst_pts_mapped = dst_h[:, :2] / dst_h[:, 2:3]
             
             # Extract Ys
             final_ys = dst_pts_mapped[:, 1]
             # Average Y per line (every 2 points)
             line_final_ys = []
             for i in range(0, len(final_ys), 2):
                 y_avg = (final_ys[i] + final_ys[i+1]) / 2
                 line_final_ys.append(y_avg)
                 
             line_final_ys = np.sort(line_final_ys)
             distances = np.diff(line_final_ys)
             
             print(f"Final Line Ys: {line_final_ys}")
             print(f"Final Distances: {distances}")
             
             # Plot
             plt.figure(figsize=(6, 4))
             plt.bar(range(len(distances)), distances)
             plt.xlabel("Gap Index")
             plt.ylabel("Distance (pixels)")
             plt.title("Distances between consecutive lines")
             plt.savefig(os.path.join(output_dir, "distances_plot.png"))
             print(f"Saved {output_dir}/distances_plot.png")
             
             # Step 4: Scale Measurement
             print("\n--- Scale Measurement ---")
             base_name = os.path.splitext(os.path.basename(image_path))[0]
             measure_pixel_scale(warped, len(merged_lines), margin, step, base_name)
             
        else:
             print("Could not compute homography.")
             
    else:
        print("Not enough lines detected to compute transform.")

def merge_lines(lines, shape):
    # Simplified merging: cluster lines by slope and intercept
    # Only keep long vertical-ish lines for now (assuming lane lines recede)
    filtered_lines = []
    if lines is None: return []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        
        # Filter for horizontal-ish lines (angle close to 0)
        # The pool lanes appear to be horizontal in the image
        # INCREASED LENGTH THRESHOLD to 200 to remove short noise
        if abs(angle) < 10 and length > 200:
             filtered_lines.append([x1, y1, x2, y2])
             
    # Sort by length
    filtered_lines.sort(key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2), reverse=True)
    
    # Improved Clustering
    # 1. Calculate the Y-value of each line at the center of the image.
    #    This gives a single robust metric for "height" of the line.
    center_x = shape[1] / 2
    
    line_metrics = []
    for line in filtered_lines:
        x1, y1, x2, y2 = line
        if x2 == x1: continue # vertical line, ignore
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y_center = slope * center_x + intercept
        line_metrics.append({'line': line, 'y': y_center, 'slope': slope, 'intercept': intercept})
        
    # 2. Sort by Y-center
    line_metrics.sort(key=lambda m: m['y'])
    
    merged = []
    if not line_metrics:
        return []
        
    # 3. Group lines within a threshold
    current_group = [line_metrics[0]]
    threshold = 20 # pixels at center of image
    
    for i in range(1, len(line_metrics)):
        metric = line_metrics[i]
        if abs(metric['y'] - current_group[-1]['y']) < threshold:
            current_group.append(metric)
        else:
            # Process current group
            merged.append(merge_group(current_group, shape))
            current_group = [metric]
    
    # Process last group
    if current_group:
        merged.append(merge_group(current_group, shape))
        
    return merged

def merge_group(group, shape):
    # Fit a single line to all points in the group
    all_x = []
    all_y = []
    
    for item in group:
        l = item['line']
        all_x.extend([l[0], l[2]])
        all_y.extend([l[1], l[3]])
        
    if not all_x: return [0,0,0,0]
    
    # Polyfit
    z = np.polyfit(all_x, all_y, 1)
    p = np.poly1d(z)
    
    # Determine extent (min/max x)
    # Reverting to use actual detected extent instead of forcing full width
    
    x_min = np.min(all_x)
    x_max = np.max(all_x)
    
    y_min = int(p(x_min))
    y_max = int(p(x_max))
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def get_line_profile(img, line):
    x1, y1, x2, y2 = line
    length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
    xs = np.linspace(x1, x2, length)
    ys = np.linspace(y1, y2, length)
    
    # Sample along the line
    profile = []
    for x, y in zip(xs, ys):
        if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
             pixel = img[int(y), int(x)]
             # Try simple Grayscale intensity or V channel
             # Or Saturation if markers are colorful
             # Let's use Red channel for now, or just sum (brightness)
             val = int(pixel[2])  # Red channel
             profile.append(val)
    

    return np.array(profile)

def detect_transitions(profile):
    # Find peaks in the profile
    if len(profile) == 0: return []
    
    # Robust peak detection
    # Standard deviation based threshold?
    mean = np.mean(profile)
    std = np.std(profile)
    threshold = mean + 1.5 * std  # Detect bright spots
    
    peaks = []
    for i in range(1, len(profile)-1):
        if profile[i] > threshold and profile[i] > profile[i-1] and profile[i] > profile[i+1]:
            peaks.append(i)
    
    # If no peaks found, try matching troughs (dark spots)
    if not peaks:
         threshold_dark = mean - 1.5 * std
         for i in range(1, len(profile)-1):
            if profile[i] < threshold_dark and profile[i] < profile[i-1] and profile[i] < profile[i+1]:
                peaks.append(i)
                
    return peaks

def measure_pixel_scale(warped_img, num_lanes, margin, step, base_name):
    # Strategy Change: Map R component along the lanes

    # Get Red Channel for profiling
    r_channel = warped_img[:, :, 2]
    width = warped_img.shape[1]
    
    # Re-calculate red_mask for Global Projection strategy
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped_img, cv2.COLOR_BGR2LAB)
    
    # HSV Thresholds
    lower_red1 = np.array([0, 40, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 40, 30])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    hsv_mask = mask1 | mask2
    
    # LAB Threshold
    l, a, b = cv2.split(lab)
    lab_mask = cv2.threshold(a, 140, 255, cv2.THRESH_BINARY)[1]
    
    # Combined Red Mask
    red_mask = cv2.bitwise_or(hsv_mask, lab_mask)
    cv2.imwrite("debug_red_mask.png", red_mask)

    
    plt.figure(figsize=(12, 10))
    
    profiles = []
    
    for i in range(num_lanes):
        y = int(margin + i * step)
        
        # Extract row
        # Averaging 3 pixels height for robustness?
        if y >= 1 and y < r_channel.shape[0] - 1:
            row = np.mean(r_channel[y-1:y+2, :], axis=0)
        else:
            row = r_channel[y, :]
            
        profiles.append(row)
        
        # Plot
        plt.subplot(num_lanes, 1, i+1)
        plt.plot(row, color='r', linewidth=1)
        plt.ylabel(f"Line {i+1}")
        plt.ylim(0, 255)
        # Turn off x labels for all but last
        if i < num_lanes - 1:
            plt.xticks([])
            
    plt.xlabel("X Pixel Position")
    plt.suptitle("Red Channel Profiles along Rectified Lanes")
    plt.tight_layout()
    plt.savefig("red_profiles.png")
    print("Saved red_profiles.png")
    
    # Save the data for potential analysis?
    # For now just visualization as requested.
    
    # Previous strategies were per-lane.
    # NEW STRATEGY: Global Vertical Projection (Sum Red Mask along Y)
    
    print("\n--- Global Vertical Projection Analysis ---")
    
    # Use the robust red_mask we generated earlier (Union of HSV and LAB)
    # Filter noise first?
    kernel = np.ones((5,5), np.uint8)
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel) # Remove small dots
    red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_CLOSE, kernel) # Close gaps
    
    # Sum along Y-axis -> Shape (width,)
    # If a column has red, the sum will be > 0.
    # Actually, let's count *how many* red pixels in each column.
    # red_mask is 0 or 255.
    vertical_projection = np.sum(red_mask_clean > 0, axis=0)
    
    # Normalize or just threshold?
    # We want columns that have a "significant" amount of red.
    # e.g., if there are 9 lanes, maybe at least 3-4 lanes should have red in that column?
    # But lines are thin.
    # Maybe just any red? Or > 5 pixels?
    projection_threshold = 5 
    
    active_cols = vertical_projection > projection_threshold
    
    # Find continuous runs of active columns
    diff = np.diff(np.concatenate(([0], active_cols.astype(np.uint8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Filter segments
    segments = []
    for s, e in zip(starts, ends):
        length = e - s
        if length > 50: # Minimum width to be a valid 5m chunk?
            segments.append((s, e, length))
            
    print(f"Detected vertical segments (Start, End, Width): {segments}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(vertical_projection, label='Red Pixel Count per Column')
    plt.axhline(projection_threshold, color='g', linestyle='--', label='Threshold')
    
    if segments:
        # Sort by width? The 5m segment should be the widest one usually?
        # Or usually the first one (closest to wall)?
        # Let's assume the longest one is the 5m.
        longest_segment = max(segments, key=lambda x: x[2])
        s_best, e_best, width_best = longest_segment
        
        print(f"Selected Best Segment: Start={s_best}, End={e_best}, Width={width_best}")
        
        scale = width_best / 5.0
        print(f"Estimated Scale (Global Projection): {scale:.2f} px/m")
        print(f"Estimated Visible Pool Width: {width / scale:.2f} m")
        
        # Visualize on the output global image
        debug_projection = warped_img.copy()
        cv2.line(debug_projection, (s_best, 0), (s_best, warped_img.shape[0]), (0, 255, 0), 2)
        cv2.line(debug_projection, (e_best, 0), (e_best, warped_img.shape[0]), (0, 255, 0), 2)
        cv2.putText(debug_projection, f"5m Width: {width_best}px", (s_best, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ANNOTATE FINAL OUTPUT with Scale Bar
        # Work on a copy so we don't pollute the warped_img used for lane extraction
        annotated_img = warped_img.copy()
        
        h, w = annotated_img.shape[:2]
        
        # Draw a bar representing 1 meter at bottom center
        bar_len_px = int(scale) # 1 meter
        start_x = (w - bar_len_px) // 2
        end_x = start_x + bar_len_px
        y_pos = h - 50
        
        # White background for text/bar
        cv2.rectangle(annotated_img, (start_x - 10, y_pos - 40), (end_x + 10, y_pos + 20), (0, 0, 0), -1)
        
        # Green Bar
        cv2.line(annotated_img, (start_x, y_pos), (end_x, y_pos), (0, 255, 0), 10)
        
        # Text
        text = f"1 Meter ({scale:.1f} px)"
        cv2.putText(annotated_img, text, (start_x, y_pos - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save overwritten output
        cv2.imwrite("Piscine_top_down.png", annotated_img)
        print(f"Saved annotated Piscine_top_down.png with scale bar.")
        
        cv2.imwrite("debug_red_vertical_projection_img.png", debug_projection)
        
    plt.title("Global Vertical Projection of Red Pixels")
    plt.xlabel("X Pixel Position")
    plt.ylabel("Count of Red Pixels")
    # EXPORT INDIVIDUAL LANES
    print("\n--- Exporting Individual Lanes ---")
    
    # Create a specific directory for this image's outputs
    import os
    # base_name is passed as argument
    output_dir = os.path.join("lane_outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Move/Save global debug images to this folder
    # We should have saved them here in the first place, but since we didn't,
    # let's just save the plot again here if possible, or move them.
    # Actually, let's just save the specific ones we have in memory now.
    
    # Save profiles plot
    plt.figure(figsize=(12, 10))
    # ... previous plotting logic was done above and closed ...
    # To avoid re-plotting, we should have passed output_dir to the plotting functions.
    # Refactoring slightly to just move the files or save them with path.
    
    # Let's just save the critical ones we have handles to:
    if 'debug_projection' in locals():
        cv2.imwrite(os.path.join(output_dir, "debug_red_vertical_projection.png"), debug_projection)
    
    # Also save the red mask if it exists
    if 'red_mask' in locals():
        cv2.imwrite(os.path.join(output_dir, "debug_red_mask.png"), red_mask)
        
    # Also save red profiles if we can re-save (or we just move the file)
    if os.path.exists("red_profiles.png"):
        os.rename("red_profiles.png", os.path.join(output_dir, "red_profiles.png"))
        
    if os.path.exists("red_vertical_projection.png"):
        os.rename("red_vertical_projection.png", os.path.join(output_dir, "red_vertical_projection.png"))
        
    # Ensure annotated_img exists even if no segments found (for Full_Rectified.png)
    if 'annotated_img' not in locals():
        annotated_img = warped_img.copy()
        
    if 'scale' not in locals():
        scale = 0

    # Save global rectified
    cv2.imwrite(os.path.join(output_dir, "Full_Rectified.png"), annotated_img)
    
    # Generate images for each lane (WATER BETWEEN LINES)
    num_swimming_lanes = num_lanes - 1
    
    for i in range(num_swimming_lanes):
        # Line i y-coord
        y_top = int(margin + i * step)
        # Line i+1 y-coord
        y_bottom = int(margin + (i+1) * step)
        
        # Crop
        lane_img = warped_img[y_top:y_bottom, :].copy()
        
        # Lane Numbering
        lane_id = num_swimming_lanes - i
        
        filename = f"Lane_{lane_id}_{int(scale)}px_per_m.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, lane_img)
        print(f"Saved {filepath}")


if __name__ == "__main__":
    import glob
    import os
    
    # Input directory
    input_folder = "Input"
    image_files = glob.glob(os.path.join(input_folder, "*.png"))
    
    if not image_files:
        print(f"No .png files found in {input_folder}/. Checking current directory...")
        image_files = glob.glob("*.png")
        # Filter out output images (start with debug_, comparison, Piscine_top_down, etc)
        image_files = [f for f in image_files if "debug_" not in f and "comparison" not in f and "Piscine_top_down" not in f and "red_vertical" not in f and "red_profiles" not in f]
    
    print(f"Leafs to process: {len(image_files)}")
    
    for img_path in image_files:
        print(f"\nProcessing {img_path}...")
        rectify_pool(img_path)
