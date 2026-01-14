import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Configuration
TEMPLATE_SIZE = (128, 128) 
MIN_CONFIDENCE = 0.3

# Initialize ORB detector
ORB_DETECTOR = cv2.ORB_create(nfeatures=500)


def load_image(path):
    """Load image with alpha channel support."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def to_grayscale(img):
    """Convert image to grayscale, handling alpha channel."""
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        # Use alpha as mask
        bgr = img[:, :, :3]
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = (gray.astype(np.float32) * alpha).astype(np.uint8)
        return gray
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_warning_light(img):
    """
    Extract warning light from image.
    For dark images (dashboards), find the largest bright connected component.
    """
    if img is None:
        return None
    
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if this is a dark image
    mean_brightness = gray.mean()
    
    if mean_brightness < 80:
        # Dark image - find bright regions
        # Use higher threshold to isolate the warning light
        _, bright = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Find the largest connected component (the warning light)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
        
        if num_labels > 1:
            # Find largest non-background component
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            
            # Create mask with just the largest component
            mask = np.zeros_like(gray)
            mask[labels == largest_idx] = 255
            return mask
        
        return bright
    else:
        # Normal/light image - use grayscale
        return gray


def normalize_image(gray):
    """
    Normalize image to white icon on black background.
    Steps:
    1. Apply threshold to get binary image
    2. Ensure white content on black background
    3. Crop to content with padding
    4. Resize to standard size
    """
    if gray is None or gray.size == 0:
        return None
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0) 
    
    # Use Otsu's method
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Determine polarity - we want white icon on black background
    white_ratio = np.mean(binary) / 255.0
    if white_ratio > 0.5:
        # Mostly white background, invert
        binary = 255 - binary
    
    # Clean up noise but preserve shape details
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small connected components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Find the largest non-background component
        areas = stats[1:, cv2.CC_STAT_AREA] 
        if len(areas) > 0:
            # Keep components that are at least 3% of the largest
            max_area = np.max(areas)
            min_area = max_area * 0.03
            
            clean = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels == i] = 255
            binary = clean
    
    # Find content and crop with padding
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add 15% padding
    pad = int(max(w, h) * 0.15)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(binary.shape[1] - x, w + 2 * pad)
    h = min(binary.shape[0] - y, h + 2 * pad)
    
    cropped = binary[y:y+h, x:x+w]
    
    # Make square by padding shorter dimension
    if h != w:
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = cropped
        cropped = square
    
    # Resize to standard size
    normalized = cv2.resize(cropped, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Create edge map for edge-based matching
    edges = cv2.Canny(normalized, 50, 150)
    
    return normalized, edges


def load_templates(template_dir):
    """Load and normalize all templates, keeping original images."""
    templates = {}
    
    if not template_dir.exists():
        print(f"Template directory not found: {template_dir}")
        return templates
    
    for path in template_dir.glob("*.png"):
        name = path.stem
        img = load_image(path)
        if img is None:
            continue
        
        gray = to_grayscale(img)
        result = normalize_image(gray)
        
        if result is not None:
            normalized, edges = result
            templates[name] = {
                'normalized': normalized,
                'edges': edges,
                'original': img
            }
            print(f"  Loaded: {name}")
    
    return templates


def rotate_image(img, angle):
    """Rotate image by angle (in degrees) around center."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated


def scale_image(img, scale):
    """Scale image by factor, then pad/crop back to original size."""
    h, w = img.shape[:2]
    
    # Scale the image
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad or crop back to original size
    result = np.zeros((h, w), dtype=img.dtype)
    
    if scale > 1.0:
        # Crop center
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        result = scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        # Pad center
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        result[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
    
    return result


def compute_similarity(img1, img2, edges1=None, edges2=None):
    """Compute multiple similarity metrics between two images."""
    if img1 is None or img2 is None:
        return {"ncc": 0, "ssim": 0, "hamming": 0, "shape": 0, "edge": 0, "orb": 0, "center": 0}
    
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 1. Normalized Cross-Correlation
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    ncc = float(result[0, 0]) if result.size > 0 else 0
    ncc = max(0, ncc)
    
    # 2. Structural Similarity (SSIM)
    try:
        ssim_score = ssim(img1, img2)
        ssim_score = max(0, ssim_score)
    except:
        ssim_score = 0
    
    # 3. Hamming distance on binary images
    xor = cv2.bitwise_xor(img1, img2)
    hamming = 1.0 - (np.sum(xor > 127) / xor.size)
    
    # 4. Shape descriptor similarity using Hu Moments
    moments1 = cv2.moments(img1)
    moments2 = cv2.moments(img2)
    
    try:
        hu1 = cv2.HuMoments(moments1).flatten()
        hu2 = cv2.HuMoments(moments2).flatten()
        # Use log scale for Hu moments
        hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
        hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
        # Compute similarity (inverse of distance)
        hu_distance = np.linalg.norm(hu1 - hu2)
        shape_score = max(0, 1.0 - hu_distance / 10.0)  # Normalize
    except:
        shape_score = 0
    
    # 5. Edge-based matching
    edge_score = 0
    if edges1 is not None and edges2 is not None:
        # Ensure same size
        if edges1.shape != edges2.shape:
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        # Compare edge maps using Hamming distance
        xor_edges = cv2.bitwise_xor(edges1, edges2)
        edge_score = 1.0 - (np.sum(xor_edges > 127) / xor_edges.size)
    
    # 6. ORB feature matching (rotation/scale invariant)
    orb_score = 0
    try:
        kp1, des1 = ORB_DETECTOR.detectAndCompute(img1, None)
        kp2, des2 = ORB_DETECTOR.detectAndCompute(img2, None)
        
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) > 0:
                # Count good matches with distance < 50
                good_matches = [m for m in matches if m.distance < 50]
                # Normalize by the smaller descriptor count
                max_possible = min(len(des1), len(des2))
                orb_score = len(good_matches) / max(max_possible, 1)
                orb_score = min(1.0, orb_score)  # Cap at 1.0
    except:
        orb_score = 0
    
    # 7. Center region pattern (helps distinguish ABS text from ! symbol)
    center_score = 0
    try:
        h, w = img1.shape[:2]
        # Extract center 40% region
        c_size = int(min(h, w) * 0.4)
        cy, cx = h // 2, w // 2
        y1, y2 = cy - c_size // 2, cy + c_size // 2
        x1, x2 = cx - c_size // 2, cx + c_size // 2
        
        center1 = img1[y1:y2, x1:x2]
        center2 = img2[y1:y2, x1:x2]
        
        # Compare center patterns using correlation
        if center1.size > 0 and center2.size > 0:
            result = cv2.matchTemplate(center1, center2, cv2.TM_CCOEFF_NORMED)
            center_score = max(0, float(result[0, 0]) if result.size > 0 else 0)
    except:
        center_score = 0
    
    return {
        "ncc": ncc,
        "ssim": ssim_score,
        "hamming": hamming,
        "shape": shape_score,
        "edge": edge_score,
        "orb": orb_score,
        "center": center_score
    }


def match_image(candidate, candidate_edges, templates):
    """
    Match candidate against all templates with multi-scale and rotation testing.
    Returns list of (name, score, metrics) sorted by score descending.
    """
    results = []
    rotation_angles = [0, -5, 5, -10, 10]  # Re-enable rotation
    scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # Test multiple scales
    
    for name, template_data in templates.items():
        template = template_data['normalized']
        template_edges = template_data['edges']
        
        best_score = 0
        best_metrics = None
        
        # Test multiple scales
        for scale in scale_factors:
            # Scale the candidate
            if scale == 1.0:
                scaled = candidate
                scaled_edges = candidate_edges
            else:
                scaled = scale_image(candidate, scale)
                scaled_edges = scale_image(candidate_edges, scale)
            
            # Test multiple rotations at this scale
            for angle in rotation_angles:
                if angle == 0:
                    transformed = scaled
                    transformed_edges = scaled_edges
                else:
                    transformed = rotate_image(scaled, angle)
                    transformed_edges = rotate_image(scaled_edges, angle)
                
                metrics = compute_similarity(transformed, template, transformed_edges, template_edges)
                
                # Weighted combination of 7 metrics
                # ORB and center pattern help distinguish similar icons (ABS vs TPMS)
                score = (
                    0.25 * metrics["ncc"] +
                    0.25 * metrics["ssim"] +
                    0.20 * metrics["hamming"] +
                    0.03 * metrics["shape"] +
                    0.08 * metrics["edge"] +
                    0.12 * metrics["orb"] +
                    0.07 * metrics["center"]
                )
                
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
        
        results.append((name, best_score, best_metrics))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def load_labels(label_path):
    """Load indicator labels and meanings."""
    if not label_path.exists():
        return {}
    
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_base_name(template_name):
    """Extract base indicator name (remove variant numbers)."""
    # Remove trailing digits
    name = template_name.rstrip("0123456789")
    # Handle underscore before number
    if name.endswith("_"):
        name = name[:-1]
    return name


def select_file():
    """Open file dialog to select an image."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select Warning Light Image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    
    if file_path:
        return Path(file_path)
    return None


def save_debug_images(candidate, templates, results, debug_dir):
    """Save debug images for inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save normalized candidate
    cv2.imwrite(str(debug_dir / "candidate.png"), candidate)
    
    # Save top 5 matches with comparison
    for i, (name, score, metrics) in enumerate(results[:5]):
        template = templates[name]['normalized']
        
        # Create side-by-side comparison
        comparison = np.hstack([candidate, template])
        cv2.imwrite(str(debug_dir / f"match_{i+1}_{name}_{score:.2f}.png"), comparison)


def show_results_window(candidate_img, template_img, name, confidence, meaning, action):
    """Display detection results in a GUI window."""
    window = tk.Tk()  # Use Tk instead of Toplevel
    window.title("Warning Light Detection Result")
    window.geometry("600x500")
    window.configure(bg='#2b2b2b')
    
    # Bring window to front
    window.lift()
    window.attributes('-topmost', True)
    window.after_idle(window.attributes, '-topmost', False)
    
    # Title
    title_label = tk.Label(
        window, 
        text="Warning Light Detected",
        font=("Arial", 18, "bold"),
        bg='#2b2b2b',
        fg='#ffffff'
    )
    title_label.pack(pady=10)
    
    # Image comparison frame
    image_frame = tk.Frame(window, bg='#2b2b2b')
    image_frame.pack(pady=10)
    
    # Convert images for tkinter (resize for display)
    display_size = 150
    
    # Handle different image formats
    if candidate_img.ndim == 2:
        candidate_rgb = cv2.cvtColor(candidate_img, cv2.COLOR_GRAY2RGB)
    elif candidate_img.shape[2] == 3:
        candidate_rgb = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
    elif candidate_img.shape[2] == 4:
        candidate_rgb = cv2.cvtColor(candidate_img, cv2.COLOR_BGRA2RGB)
    else:
        candidate_rgb = candidate_img
    
    if template_img.ndim == 2:
        template_rgb = cv2.cvtColor(template_img, cv2.COLOR_GRAY2RGB)
    elif template_img.shape[2] == 3:
        template_rgb = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
    elif template_img.shape[2] == 4:
        template_rgb = cv2.cvtColor(template_img, cv2.COLOR_BGRA2RGB)
    else:
        template_rgb = template_img
    
    candidate_resized = cv2.resize(candidate_rgb, (display_size, display_size))
    template_resized = cv2.resize(template_rgb, (display_size, display_size))
    
    # Convert to PIL
    candidate_pil = Image.fromarray(candidate_resized)
    template_pil = Image.fromarray(template_resized)
    
    candidate_tk = ImageTk.PhotoImage(candidate_pil)
    template_tk = ImageTk.PhotoImage(template_pil)
    
    # Candidate image
    candidate_label = tk.Label(image_frame, text="Your Image", font=("Arial", 10), bg='#2b2b2b', fg='#aaaaaa')
    candidate_label.grid(row=0, column=0, padx=20)
    candidate_img_label = tk.Label(image_frame, image=candidate_tk, bg='#2b2b2b')
    candidate_img_label.image = candidate_tk  # Keep reference
    candidate_img_label.grid(row=1, column=0, padx=20)
    
    # Arrow
    arrow_label = tk.Label(image_frame, text="→", font=("Arial", 24), bg='#2b2b2b', fg='#4CAF50')
    arrow_label.grid(row=1, column=1)
    
    # Template image
    template_label = tk.Label(image_frame, text="Matched Template", font=("Arial", 10), bg='#2b2b2b', fg='#aaaaaa')
    template_label.grid(row=0, column=2, padx=20)
    template_img_label = tk.Label(image_frame, image=template_tk, bg='#2b2b2b')
    template_img_label.image = template_tk  # Keep reference
    template_img_label.grid(row=1, column=2, padx=20)
    
    # Results frame
    results_frame = tk.Frame(window, bg='#3b3b3b', relief=tk.RAISED, bd=2)
    results_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    
    # Indicator name
    name_label = tk.Label(
        results_frame,
        text=f"Indicator: {name.upper().replace('_', ' ')}",
        font=("Arial", 14, "bold"),
        bg='#3b3b3b',
        fg='#FFA500'
    )
    name_label.pack(pady=5)
    
    # Confidence
    confidence_color = '#4CAF50' if confidence >= 0.6 else '#FFA500' if confidence >= 0.4 else '#FF5252'
    confidence_label = tk.Label(
        results_frame,
        text=f"Confidence: {confidence:.1%}",
        font=("Arial", 12),
        bg='#3b3b3b',
        fg=confidence_color
    )
    confidence_label.pack(pady=5)
    
    # Separator
    separator = tk.Frame(results_frame, height=2, bg='#555555')
    separator.pack(fill=tk.X, padx=10, pady=10)
    
    # Meaning
    if meaning:
        meaning_label = tk.Label(
            results_frame,
            text=f"Meaning:\n{meaning}",
            font=("Arial", 11),
            bg='#3b3b3b',
            fg='#ffffff',
            wraplength=500,
            justify=tk.LEFT
        )
        meaning_label.pack(pady=5, padx=10)
    
    # Action
    if action:
        action_label = tk.Label(
            results_frame,
            text=f"Action Required:\n{action}",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#FF5252',
            wraplength=500,
            justify=tk.LEFT
        )
        action_label.pack(pady=10, padx=10)
    
    # Close button
    close_btn = tk.Button(
        window,
        text="Close",
        font=("Arial", 11),
        bg='#4CAF50',
        fg='#ffffff',
        activebackground='#45a049',
        command=window.destroy,
        width=15,
        height=2
    )
    close_btn.pack(pady=10)
    
    # Center window on screen
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')
    
    window.mainloop()


class DetectorApp:
    """Main GUI application for warning light detection."""
    
    def __init__(self, templates, labels, debug_dir):
        self.templates = templates
        self.labels = labels
        self.debug_dir = debug_dir
        self.current_image = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Car Warning Light Detector v2")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Create UI
        self.create_menu_ui()
        
    def create_menu_ui(self):
        """Create the main menu interface."""
        # Title Section
        title_frame = tk.Frame(self.root, bg='#1e1e1e', padx=20, pady=20)
        title_frame.pack(fill='x')
        
        title_label = tk.Label(
            title_frame,
            text="CAR WARNING LIGHT DETECTOR v2",
            font=('Arial', 24, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Computer Vision Detection System • 19 CV Techniques • 7 Similarity Metrics",
            font=('Arial', 10),
            bg='#1e1e1e',
            fg='#888888'
        )
        subtitle_label.pack()
        
        # Instructions Section
        info_frame = tk.Frame(self.root, bg='#2b2b2b', padx=20, pady=10)
        info_frame.pack(fill='x')
        
        instructions = """
HOW TO USE:
   1. Click 'Load Image' button below
   2. Select a warning light image from your computer
   3. Wait for analysis (takes a few seconds)
   4. View results below
   5. Load another image to test again

TIPS: Works best with clear, well-lit warning lights • Supports dashboard photos and icon images
        """
        
        info_label = tk.Label(
            info_frame,
            text=instructions,
            font=('Courier', 10),
            bg='#2b2b2b',
            fg='#cccccc',
            justify='left'
        )
        info_label.pack()
        
        # Load Button
        button_frame = tk.Frame(self.root, bg='#2b2b2b', pady=10)
        button_frame.pack()
        
        self.load_btn = tk.Button(
            button_frame,
            text="LOAD IMAGE",
            font=('Arial', 14, 'bold'),
            bg='#0078d4',
            fg='white',
            padx=40,
            pady=15,
            cursor='hand2',
            command=self.load_and_process_image
        )
        self.load_btn.pack()
        
        # Results Section
        self.results_frame = tk.Frame(self.root, bg='#2b2b2b', padx=20, pady=20)
        self.results_frame.pack(fill='both', expand=True)
        
        # Initially show placeholder
        self.show_placeholder()
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text=f"{len(self.templates)} templates loaded - Ready to analyze",
            font=('Arial', 9),
            bg='#1e1e1e',
            fg='#4CAF50',
            pady=5
        )
        self.status_label.pack(fill='x', side='bottom')
        
    def show_placeholder(self):
        """Show placeholder when no image is loaded."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        placeholder_label = tk.Label(
            self.results_frame,
            text="No image loaded yet.\nClick 'LOAD IMAGE' to get started.",
            font=('Arial', 14),
            bg='#2b2b2b',
            fg='#666666'
        )
        placeholder_label.pack(expand=True)
        
    def load_and_process_image(self):
        """Load and process an image."""
        # Update status
        self.status_label.config(text="Selecting image...", fg='#FFC107')
        self.root.update()
        
        # Select file
        image_path = select_file()
        
        if not image_path:
            self.status_label.config(text="No file selected", fg='#f44336')
            return
        
        print(f"Selected: {image_path}")
        self.status_label.config(text=f"Analyzing: {Path(image_path).name}...", fg='#FFC107')
        self.root.update()
        
        # Load image
        img = load_image(image_path)
        if img is None:
            self.status_label.config(text="Failed to load image", fg='#f44336')
            messagebox.showerror("Error", "Failed to load the selected image.")
            return
        
        self.current_image = img
        
        # Process image
        gray = extract_warning_light(img)
        result = normalize_image(gray)
        
        if result is None:
            self.status_label.config(text="Failed to process image", fg='#f44336')
            messagebox.showerror("Error", "Failed to process the image.")
            return
        
        candidate, candidate_edges = result
        
        # Match against templates
        results = match_image(candidate, candidate_edges, self.templates)
        
        # Save debug images
        save_debug_images(candidate, self.templates, results, self.debug_dir)
        
        # Show results
        self.display_results(img, results)
        
        # Print to console
        print("\nTop 5 matches:")
        print("-" * 70)
        for i, (name, score, metrics) in enumerate(results[:5]):
            print(f"  {i+1}. {name:25} Score: {score:.1%}")
        
        best_name, best_score, _ = results[0]
        display_name = get_base_name(best_name)
        print(f"\nDETECTED: {display_name} ({best_score:.1%})")
        
    def display_results(self, img, results):
        """Display detection results in the GUI."""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        best_name, best_score, best_metrics = results[0]
        
        # Create two-column layout
        images_frame = tk.Frame(self.results_frame, bg='#2b2b2b')
        images_frame.pack(fill='both', expand=True)
        
        # Left: Input image
        left_frame = tk.Frame(images_frame, bg='#2b2b2b')
        left_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        tk.Label(
            left_frame,
            text="Input Image",
            font=('Arial', 12, 'bold'),
            bg='#2b2b2b',
            fg='#ffffff'
        ).pack(pady=5)
        
        # Convert and display input image
        img_display = self.prepare_display_image(img, max_size=300)
        img_tk = ImageTk.PhotoImage(img_display)
        img_label = tk.Label(left_frame, image=img_tk, bg='#2b2b2b')
        img_label.image = img_tk  # Keep reference
        img_label.pack()
        
        # Right: Matched template
        right_frame = tk.Frame(images_frame, bg='#2b2b2b')
        right_frame.pack(side='right', fill='both', expand=True, padx=10)
        
        tk.Label(
            right_frame,
            text="Matched Template",
            font=('Arial', 12, 'bold'),
            bg='#2b2b2b',
            fg='#ffffff'
        ).pack(pady=5)
        
        # Display matched template
        template_img = self.templates[best_name]['original']
        template_display = self.prepare_display_image(template_img, max_size=300)
        template_tk = ImageTk.PhotoImage(template_display)
        template_label = tk.Label(right_frame, image=template_tk, bg='#2b2b2b')
        template_label.image = template_tk  # Keep reference
        template_label.pack()
        
        # Results info
        info_frame = tk.Frame(self.results_frame, bg='#1e1e1e', padx=20, pady=15)
        info_frame.pack(fill='x', pady=10)
        
        # Detection result
        if best_score >= MIN_CONFIDENCE:
            confidence_color = '#4CAF50' if best_score >= 0.7 else '#FFC107'
            display_name = get_base_name(best_name)
            
            tk.Label(
                info_frame,
                text=f"DETECTED: {display_name}",
                font=('Arial', 14, 'bold'),
                bg='#1e1e1e',
                fg=confidence_color
            ).pack()
            
            tk.Label(
                info_frame,
                text=f"Confidence: {best_score:.1%}",
                font=('Arial', 12),
                bg='#1e1e1e',
                fg='#ffffff'
            ).pack(pady=5)
            
            # Get meaning and action
            base_name = get_base_name(best_name)
            meaning = ""
            action = ""
            
            if base_name in self.labels:
                info = self.labels[base_name]
                meaning = info.get('meaning', 'Unknown')
                action = info.get('action', 'Consult manual')
            elif best_name in self.labels:
                info = self.labels[best_name]
                meaning = info.get('meaning', 'Unknown')
                action = info.get('action', 'Consult manual')
            
            if meaning:
                tk.Label(
                    info_frame,
                    text=f"Meaning: {meaning}",
                    font=('Arial', 11),
                    bg='#1e1e1e',
                    fg='#cccccc',
                    wraplength=800
                ).pack(pady=5)
            
            if action:
                tk.Label(
                    info_frame,
                    text=f"Recommended Action: {action}",
                    font=('Arial', 11, 'bold'),
                    bg='#1e1e1e',
                    fg='#FFC107',
                    wraplength=800
                ).pack(pady=5)
            
            self.status_label.config(text=f"Detection complete: {display_name} ({best_score:.1%})", fg='#4CAF50')
        else:
            tk.Label(
                info_frame,
                text=f"Low Confidence Match",
                font=('Arial', 14, 'bold'),
                bg='#1e1e1e',
                fg='#FFC107'
            ).pack()
            
            tk.Label(
                info_frame,
                text=f"Best match: {best_name} ({best_score:.1%})\nImage quality may be poor or template not in database.",
                font=('Arial', 11),
                bg='#1e1e1e',
                fg='#cccccc'
            ).pack(pady=5)
            
            self.status_label.config(text=f"Low confidence: {best_name} ({best_score:.1%})", fg='#FFC107')
    
    def prepare_display_image(self, img, max_size=300):
        """Prepare image for display in GUI."""
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize to fit
        h, w = img_rgb.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        
        return Image.fromarray(img_rgb)
    
    def run(self):
        """Start the GUI application."""
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        print("\n" + "=" * 70)
        print(" " * 15 + "CAR WARNING LIGHT DETECTOR v2")
        print("=" * 70)
        print("\nGUI application started")
        print(f"{len(self.templates)} templates loaded")
        print("\nUse the GUI window to load and analyze warning light images")
        print("Close the window to exit the application\n")
        
        self.root.mainloop()


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    template_dir = data_dir / "templates"
    label_path = data_dir / "labels.json"
    debug_dir = data_dir / "debug_v2"
    
    # Load templates
    print("Loading templates...")
    templates = load_templates(template_dir)
    if not templates:
        print("No templates loaded!")
        return
    
    # Load labels
    labels = load_labels(label_path)
    
    # Create and run GUI application
    app = DetectorApp(templates, labels, debug_dir)
    app.run()


if __name__ == "__main__":
    main()
