# Cunny Lake Archive Tile Stitcher

Quickly generated script that will save cunny lake snapshots every 5 minutes.

This script downloads map tiles from **wplace.live** and stitches them together into a single combined image.  
It runs the screenshot task automatically every **15 minutes** (Default, adjustable), saving the output with a timestamped filename.

App is a quick WIP MVP for this emergency and might have bugs.

---

## Requirements
- Python **3.8+** (tested on Python 3.12)

#### Windows

Double clicks start.bat. screenshots should start appearing

If you have any errors it could be because your python version was installed incorrectly, download latest at https://python.org

ENSURE you select "Add to Path" option in installer.

#### MacOS

run:

```bash
chmod +x start.sh
./start.sh
```

Same disclaimers as above.


#### Alternative

If the above doesn't work try running the app directly:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate.bat
pip install -r requirements.txt
python cunny.py
```

```bash
# MacOS/Linux
python -m venv venv
source venv/scripts/activate
pip install -r requirements.txt
python cunny.py
```
