import requests
from PIL import Image, ImageChops
from io import BytesIO
import time
from datetime import datetime
import os
import configparser
import glob
import numpy as np

# Tile URL template
URL_TEMPLATE = "https://backend.wplace.live/files/s0/tiles/{x}/{y}.png"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = ['snapshots', 'scaled', 'delta']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    return directories

def read_config(filename="config.cfg"):
    """
    Reads configuration from a file using the configparser module.
    
    Args:
        filename (str): The name of the configuration file.
        
    Returns:
        dict: A dictionary containing all configuration settings.
    """
    config = configparser.ConfigParser()
    
    # Check if the config file exists
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found. Using default settings.")
        return {
            "frequency": 900,
            "x_min": 471,
            "x_max": 478,
            "y_min": 843,
            "y_max": 847,
            "save_scaled": False,
            "scale_percent": 50,
            "scale_algorithm": "nearest",
            "save_deltas": False,
            "delta_threshold": 0
        }

    try:
        config.read(filename)
        # Read settings from the config file, or use defaults if they are missing
        settings = {
            "frequency": config.getint('settings', 'frequency', fallback=900),
            "x_min": config.getint('settings', 'x_min', fallback=471),
            "x_max": config.getint('settings', 'x_max', fallback=478),
            "y_min": config.getint('settings', 'y_min', fallback=843),
            "y_max": config.getint('settings', 'y_max', fallback=847),
            "save_scaled": config.getboolean('settings', 'save_scaled', fallback=False),
            "scale_percent": config.getint('settings', 'scale_percent', fallback=50),
            "scale_algorithm": config.get('settings', 'scale_algorithm', fallback='nearest').lower(),
            "save_deltas": config.getboolean('settings', 'save_deltas', fallback=False),
            "delta_threshold": config.getint('settings', 'delta_threshold', fallback=0)
        }
        
        # Validate settings
        if settings["save_deltas"] and settings["save_scaled"]:
            print("Warning: Delta comparison disabled because scaling is enabled.")
            settings["save_deltas"] = False
            
        if settings["scale_percent"] < 1 or settings["scale_percent"] > 100:
            print("Warning: Scale percent must be between 1-100. Using default of 50.")
            settings["scale_percent"] = 50
            
        # Validate scale algorithm
        valid_algorithms = ['nearest', 'bilinear', 'bicubic', 'lanczos']
        if settings["scale_algorithm"] not in valid_algorithms:
            print(f"Warning: Invalid scale algorithm '{settings['scale_algorithm']}'. Using 'nearest'.")
            settings["scale_algorithm"] = 'nearest'
            
        return settings
    except configparser.Error as e:
        print(f"Error reading config file: {e}. Using default settings.")
        return {
            "frequency": 900,
            "x_min": 471,
            "x_max": 478,
            "y_min": 843,
            "y_max": 847,
            "save_scaled": False,
            "scale_percent": 50,
            "scale_algorithm": "nearest",
            "save_deltas": False,
            "delta_threshold": 0
        }

def get_pil_resampling_filter(algorithm):
    """
    Convert algorithm string to PIL resampling filter.
    
    Args:
        algorithm (str): Algorithm name ('nearest', 'bilinear', 'bicubic', 'lanczos')
        
    Returns:
        PIL resampling filter constant
    """
    algorithm_map = {
        'nearest': Image.NEAREST,      # Sharp, pixelated - best for pixel art
        'bilinear': Image.BILINEAR,    # Smooth, good for most images  
        'bicubic': Image.BICUBIC,      # Smoother, good quality
        'lanczos': Image.LANCZOS       # Highest quality, best for photography
    }
    return algorithm_map.get(algorithm, Image.NEAREST)

def download_tile(x, y):
    """Downloads a single tile from the web."""
    url = URL_TEMPLATE.format(x=x, y=y)
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

def get_latest_snapshot():
    """
    Find the most recent snapshot file in the snapshots directory.
    
    Returns:
        str or None: Path to the latest snapshot file, or None if no snapshots found.
    """
    png_files = glob.glob("snapshots/stitched_*.png")
    
    if not png_files:
        return None
    
    # Sort by filename (which includes timestamp)
    png_files.sort()
    return png_files[-1]

def calculate_image_difference(img1, img2, threshold=0):
    """
    Calculate the difference between two images with custom visualization:
    - Purple pixels for deletions (pixels that were present in img1 but not in img2)
    - Green pixels for additions (pixels that were not in img1 but are in img2)
    - White pixels for modifications (pixels that changed color)
    
    Args:
        img1 (PIL.Image): First image (older)
        img2 (PIL.Image): Second image (newer)  
        threshold (int): Minimum difference threshold (0-255)
        
    Returns:
        PIL.Image or None: Custom difference image, or None if images are identical
    """
    try:
        # Ensure both images are the same size and mode
        if img1.size != img2.size:
            print("Warning: Images have different sizes, cannot compare")
            return None
            
        # Convert both to RGBA for comparison
        if img1.mode != 'RGBA':
            img1 = img1.convert('RGBA')
        if img2.mode != 'RGBA':
            img2 = img2.convert('RGBA')
            
        # Convert to numpy arrays for pixel-level analysis
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Create output array (RGBA)
        height, width = arr1.shape[:2]
        diff_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Track statistics
        deleted_pixels = 0
        added_pixels = 0
        modified_pixels = 0
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                pixel1 = arr1[y, x]  # [R, G, B, A]
                pixel2 = arr2[y, x]  # [R, G, B, A]
                
                # Check if pixels are transparent (alpha = 0)
                transparent1 = pixel1[3] == 0
                transparent2 = pixel2[3] == 0
                
                if not transparent1 and transparent2:
                    # Pixel was deleted (was solid, now transparent)
                    diff_array[y, x] = [128, 0, 128, 255]  # Purple
                    deleted_pixels += 1
                    
                elif transparent1 and not transparent2:
                    # Pixel was added (was transparent, now solid)
                    diff_array[y, x] = [0, 255, 0, 255]  # Green
                    added_pixels += 1
                    
                elif not transparent1 and not transparent2:
                    # Both pixels are solid, check if they're different
                    # Calculate color difference (ignoring alpha for color comparison)
                    color_diff = np.sqrt(np.sum((pixel1[:3].astype(int) - pixel2[:3].astype(int)) ** 2))
                    
                    if color_diff > threshold:
                        # Pixel was modified (color changed)
                        diff_array[y, x] = [255, 255, 255, 255]  # White
                        modified_pixels += 1
                    # If color difference is below threshold, leave as transparent (no change)
                    
                # If both are transparent or identical, leave as transparent (no change)
        
        # Check if there are any changes
        total_changes = deleted_pixels + added_pixels + modified_pixels
        
        if total_changes == 0:
            print("No differences detected between images")
            return None
            
        # Print change statistics
        print(f"Changes detected: {total_changes} pixels")
        if deleted_pixels > 0:
            print(f"  Deleted pixels (purple): {deleted_pixels}")
        if added_pixels > 0:
            print(f"  Added pixels (green): {added_pixels}")
        if modified_pixels > 0:
            print(f"  Modified pixels (white): {modified_pixels}")
        if threshold > 0:
            print(f"  Applied difference threshold: {threshold}")
            
        # Convert back to PIL Image
        diff_image = Image.fromarray(diff_array, 'RGBA')
        return diff_image
        
    except Exception as e:
        print(f"Error calculating image difference: {e}")
        return None

def stitch_tiles(x_min, x_max, y_min, y_max, save_scaled=False, scale_percent=50, 
                 scale_algorithm="nearest", save_deltas=False, delta_threshold=0):
    """
    Downloads and stitches tiles to create a single image.
    
    Args:
        x_min (int): The starting x coordinate.
        x_max (int): The ending x coordinate.
        y_min (int): The starting y coordinate.
        y_max (int): The ending y coordinate.
        save_scaled (bool): Whether to save a scaled version.
        scale_percent (int): Scale percentage (1-100).
        scale_algorithm (str): Scaling algorithm ('nearest', 'bilinear', 'bicubic', 'lanczos').
        save_deltas (bool): Whether to save difference images.
        delta_threshold (int): Minimum difference threshold for delta images.
    """
    # Ensure directories exist
    ensure_directories()
    
    try:
        # Download first tile to get size
        first_tile = download_tile(x_min, y_min)
        tile_w, tile_h = first_tile.size
    except Exception as e:
        print(f"Failed to fetch initial tile: {e}")
        return

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    
    # Create empty canvas
    stitched = Image.new("RGBA", (cols * tile_w, rows * tile_h))
    
    # Paste tiles
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            try:
                tile = download_tile(x, y)
                x_offset = (x - x_min) * tile_w
                y_offset = (y - y_min) * tile_h
                stitched.paste(tile, (x_offset, y_offset))
                print(f"Placed tile {x},{y}")
            except Exception as e:
                print(f"Failed to fetch tile {x},{y}: {e}")
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Always save the full-resolution version first
    full_filename = os.path.join("snapshots", f"stitched_{timestamp}.png")
    stitched.save(full_filename, 'PNG', optimize=True, compress_level=9)
    full_absolute_path = os.path.abspath(full_filename)
    full_file_size = os.path.getsize(full_filename)
    print(f"Saved full resolution: {full_absolute_path}")
    print(f"Full size: {full_file_size:,} bytes ({full_file_size / 1024 / 1024:.2f} MB)")
    
    # Handle delta comparison (only if not using scaling)
    if save_deltas and not save_scaled:
        # Get previous full-resolution snapshot for comparison
        png_files = glob.glob("snapshots/stitched_*.png")
        if len(png_files) > 1:  # We need at least 2 files to compare
            png_files.sort()
            latest_snapshot = png_files[-2]  # Second to last (previous one)
            try:
                print(f"Comparing with previous snapshot: {latest_snapshot}")
                previous_image = Image.open(latest_snapshot)
                
                # Calculate difference with custom visualization
                diff_image = calculate_image_difference(previous_image, stitched, delta_threshold)
                
                if diff_image:
                    # Save difference image in delta folder
                    delta_filename = os.path.join("delta", f"delta_{timestamp}.png")
                    diff_image.save(delta_filename, 'PNG', optimize=True)
                    delta_absolute_path = os.path.abspath(delta_filename)
                    print(f"Saved delta image: {delta_absolute_path}")
                else:
                    print("No significant changes detected, skipping delta image")
                    
            except Exception as e:
                print(f"Error creating delta image: {e}")
    
    # Handle scaled version if requested
    if save_scaled:
        try:
            # Calculate new size
            original_width, original_height = stitched.size
            new_width = int(original_width * scale_percent / 100)
            new_height = int(original_height * scale_percent / 100)
            
            # Get the appropriate resampling filter
            resampling_filter = get_pil_resampling_filter(scale_algorithm)
            
            # Create scaled version
            scaled_image = stitched.resize((new_width, new_height), resampling_filter)
            
            # Save scaled version
            scaled_filename = os.path.join("scaled", f"stitched_{timestamp}_scaled_{scale_percent}p_{scale_algorithm}.png")
            scaled_image.save(scaled_filename, 'PNG', optimize=True, compress_level=9)
            
            # Print scaled version info
            scaled_absolute_path = os.path.abspath(scaled_filename)
            scaled_file_size = os.path.getsize(scaled_filename)
            compression_ratio = (1 - scaled_file_size / full_file_size) * 100
            
            print(f"Saved scaled version: {scaled_absolute_path}")
            print(f"Scaled size: {scaled_file_size:,} bytes ({scaled_file_size / 1024 / 1024:.2f} MB)")
            print(f"Size reduction: {compression_ratio:.1f}%")
            print(f"Dimensions: {original_width}x{original_height} → {new_width}x{new_height}")
            print(f"Scale algorithm: {scale_algorithm}")
            
        except Exception as e:
            print(f"Error creating scaled image: {e}")

def create_sample_config():
    """Create a sample configuration file with all options."""
    config_content = """[settings]
# Capture frequency in seconds (900 = 15 minutes)
frequency = 900

# Tile coordinates for the area to capture
x_min = 471
x_max = 478
y_min = 843
y_max = 847

# Save a scaled-down version to reduce file size (true/false)
save_scaled = false

# Scale percentage for scaled version (1-100)
# 50 = half size, 25 = quarter size, etc.
scale_percent = 50

# Scaling algorithm for resizing:
# nearest   - Sharp, pixelated (best for pixel art, smallest files)
# bilinear  - Smooth, good for most images
# bicubic   - Smoother, good quality 
# lanczos   - Highest quality (best for photographs)
scale_algorithm = nearest

# Save delta/difference images comparing consecutive snapshots (true/false)
# Note: This is automatically disabled when save_scaled is true
# Delta images now use custom colors:
# - Purple: Deleted pixels (were present, now gone)
# - Green: Added pixels (were absent, now present)  
# - White: Modified pixels (changed color)
save_deltas = false

# Minimum difference threshold for delta images (0-255)
# 0 = show all differences, higher values = only show larger differences
delta_threshold = 0
"""
    
    with open('config_sample.cfg', 'w') as f:
        f.write(config_content)
    print("Created sample configuration file: config_sample.cfg")

if __name__ == "__main__":
    # Create sample config if it doesn't exist
    if not os.path.exists("config.cfg") and not os.path.exists("config_sample.cfg"):
        create_sample_config()
        print("Edit config_sample.cfg and rename it to config.cfg to customize settings.")
    
    # Ensure directories exist
    ensure_directories()
    
    settings = read_config("config.cfg")
    
    # Unpack settings for use
    frequency = settings.get("frequency")
    x_min = settings.get("x_min")
    x_max = settings.get("x_max")
    y_min = settings.get("y_min")
    y_max = settings.get("y_max")
    save_scaled = settings.get("save_scaled")
    scale_percent = settings.get("scale_percent")
    scale_algorithm = settings.get("scale_algorithm")
    save_deltas = settings.get("save_deltas")
    delta_threshold = settings.get("delta_threshold")
    
    # Print current settings and directory structure
    print("=== Current Settings ===")
    print(f"Frequency: {frequency}s ({frequency / 60:.1f} minutes)")
    print(f"Area: X={x_min}-{x_max}, Y={y_min}-{y_max}")
    print("Save format: PNG (lossless)")
    print("Output directory: ./snapshots/")
    
    if save_scaled:
        print(f"Save scaled: Yes ({scale_percent}% using {scale_algorithm})")
        print("Scaled directory: ./scaled/")
    else:
        print("Save scaled: No")
        
    print(f"Save deltas: {'Yes' if save_deltas else 'No'}")
    if save_deltas:
        print(f"Delta threshold: {delta_threshold}")
        print("Delta directory: ./delta/")
        print("Delta colors: Purple=Deleted, Green=Added, White=Modified")
    print("========================\n")

    while True:
        print(f"\nStarting new screenshot task...")
        print(f"Stitching area from X={x_min} to {x_max} and Y={y_min} to {y_max}")
        try:
            stitch_tiles(x_min, x_max, y_min, y_max, save_scaled, scale_percent, 
                        scale_algorithm, save_deltas, delta_threshold)
        except Exception as e:
            print(f"Error during stitching: {e}")
        
        print(f"Sleeping for {frequency / 60:.1f} minutes...\n")
        time.sleep(frequency)
