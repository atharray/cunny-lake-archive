import requests
from PIL import Image
from io import BytesIO
import time
from datetime import datetime

# Ranges, update if needed
X_MIN, X_MAX = 471, 478
Y_MIN, Y_MAX = 843, 847
frequency = 60 * 15 # Value in seconds, how often to take snapshot   # 60 seconds * 15 minutes = 900 seconds

# Tile URL template
URL_TEMPLATE = "https://backend.wplace.live/files/s0/tiles/{x}/{y}.png"

# Download one tile
def download_tile(x, y):
    url = URL_TEMPLATE.format(x=x, y=y)
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

def stitch_tiles():
    # Download first tile to get size
    first_tile = download_tile(X_MIN, Y_MIN)
    tile_w, tile_h = first_tile.size
    
    cols = X_MAX - X_MIN + 1
    rows = Y_MAX - Y_MIN + 1
    
    # Create empty canvas
    stitched = Image.new("RGBA", (cols * tile_w, rows * tile_h))
    
    # Paste tiles
    for x in range(X_MIN, X_MAX + 1):
        for y in range(Y_MIN, Y_MAX + 1):
            try:
                tile = download_tile(x, y)
                x_offset = (x - X_MIN) * tile_w
                y_offset = (y - Y_MIN) * tile_h
                stitched.paste(tile, (x_offset, y_offset))
                print(f"Placed tile {x},{y}")
            except Exception as e:
                print(f"Failed to fetch tile {x},{y}: {e}")
    
    # Timestamped filename
    filename = f"stitched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    stitched.save(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    while True:
        print("\nStarting new screenshot task...")
        try:
            stitch_tiles()
        except Exception as e:
            print(f"Error during stitching: {e}")
        
        print("Sleeping for 5 minutes...\n")
        time.sleep(frequency)
