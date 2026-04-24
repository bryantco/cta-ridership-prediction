import os
from pathlib import Path
import requests
import pandas as pd
import argparse

from utils import fetch_socrata_dataset

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cta_ridership", type=str, required=True)
parser.add_argument("--cta_stations", type=str, required=True)
args = parser.parse_args()

# If the CTA ridership data exists, read it in
if Path(args.cta_ridership).is_file():
    cta_df = pd.read_parquet(args.cta_ridership)
else:
    # If the CTA data does not exist, download it from Socrata
    cta_df = fetch_socrata_dataset('5neh-572f')

    # And save as a flat file
    os.makedirs(args.output_dir, exist_ok=True)
    cta_df.to_parquet(path=args.cta_ridership, engine='fastparquet', index=False)

cta_df.tail(10)

# Check that the file is up-to-date
# If the data exists and the last row number is smaller than the last row number on Socrata, re-download
nrow_in_data = cta_df.shape[0]
print(f'Number of rows in the data: {nrow_in_data}')

# Check against Socrata
url = "https://data.cityofchicago.org/resource/5neh-572f.json"
params = {
    "$select": "count(*)"
}

data_socrata_json = requests.get(url, params=params).json()
nrow_in_socrata = int(data_socrata_json[0]['count'])

if nrow_in_data == nrow_in_socrata:
    print('The local data is up-to-date.')
else:
    print('Downloading new data...')
    params_download = {
        "$offset": nrow_in_data
    }

    data_new = requests.get(url, params=params_download).json()
    df_new = pd.DataFrame(data_new)
    cta_df = pd.concat([cta_df, df_new], ignore_index=True)

    # Save the flat file
    os.makedirs(args.output_dir, exist_ok=True)
    cta_df.to_parquet(path=args.cta_ridership, engine='fastparquet', index=False)


if Path(args.cta_stations).is_file():
    cta_stations_df = pd.read_parquet(args.cta_stations)
else:
    cta_stations_df = fetch_socrata_dataset('8pix-ypme')

    # And save as a flat file
    os.makedirs(args.output_dir, exist_ok=True)
    cta_stations_df.to_parquet(path=args.cta_stations, engine='fastparquet', index=False)