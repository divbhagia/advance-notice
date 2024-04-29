
# Some parameters
DOWNLOAD_RAW = True
EXTRACT_IPUMS = False

# Download link for the raw data
DATA_LINK = 'https://www.dropbox.com/scl/fi/l45oehwvjucfz9xpafdj1/raw.zip?rlkey=g4zdee32tgwcox4boc9950zxo&st=5ojpr02p&dl=1'

# Specify directories
DATA_DIR = 'data'
RAW_DATA_DIR = DATA_DIR + '/raw'
IPUMS_DIR = RAW_DATA_DIR + '/ipums-extract'
OUTPUT_DIR = 'output'
QUANTS_DIR = 'scripts/quants'
TEST_QUANTS_DIR = 'tests/quants'
DIRS = [DATA_DIR, OUTPUT_DIR, QUANTS_DIR, TEST_QUANTS_DIR]

# Colors
class Colors:
    RED = '#d40404'
    BLACK = '#000000'
    BLUE = '#0303B6'

# Variables to keep
cpsvars = ['serial', 'cpsid', 'mish', 'pernum', 'hwtfinl', 'wtfinl', 
           'year', 'month', 'statefip', 'metro', 'metarea', 'county', 'faminc', 'sex', 'marst', 'empstat', 'labforce', 'occ1990', 'ind1990', 'classwkr', 'numjob', 'educ', 'race', 'durunem2', 
           'age', 'uhrsworkt', 'uhrswork1', 'durunemp']
dwsvars = ['dwstat', 'dwresp', 'dwrecall', 'dwfulltime', 'dwclass',
           'dwsuppwt', 'dws', 'dwreas', 'dwnotice', 'dwlastwrk', 'dwunion', 'dwben', 'dwexben', 'dwhi', 'dwind1990', 'dwocc1990', 'dwmove', 'dwhinow', 'dwyears', 'dwweekl', 'dwweekc', 'dwjobsince', 'dwhrswkc', 'dwwksun']
SELECTED_VARS = cpsvars + dwsvars

