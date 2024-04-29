
# Import libraries
import os
import requests
from zipfile import ZipFile

# Import parameters
from utils.config import DATA_LINK, DATA_DIR, DOWNLOAD_RAW, DIRS

# Create directories if they do not exist
for d in DIRS:
    if not os.path.exists(d):
        os.makedirs(d)

# Download raw data if specified
if DOWNLOAD_RAW:
    zipfile = DATA_DIR + '/raw.zip'
    r = requests.get(DATA_LINK)
    with open(zipfile, 'wb') as f:
        f.write(r.content)
    with ZipFile(zipfile, 'r') as z:
        for file in z.namelist():
            if file.startswith('raw/'):
                z.extract(file, DATA_DIR)
    os.remove(zipfile)