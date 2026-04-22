import os
from pathlib import Path
import requests
import pandas as pd

from utils import fetch_socrata_dataset

# If the CTA ridership data exists, read it in
if Path("output/cta_ridership.parquet").is_file():
    cta_df = pd.read_parquet("output/cta_ridership.parquet")
else:
    # If the CTA data does not exist, download it from Socrata
    cta_df = fetch_socrata_dataset('5neh-572f')

    # And save as a flat file
    os.makedirs("output", exist_ok=True)
    cta_df.to_parquet(path='output/cta_ridership.parquet', engine='fastparquet', index=False)

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
    os.makedirs("output", exist_ok=True)
    cta_df.to_parquet(path='output/cta_ridership.parquet', engine='fastparquet', index=False)


if Path("output/cta_stations.parquet").is_file():
    cta_stations_df = pd.read_parquet("output/cta_stations.parquet")
else:
    cta_stations_df = fetch_socrata_dataset('8pix-ypme')

    # And save as a flat file
    os.makedirs("output", exist_ok=True)
    cta_stations_df.to_parquet(path='output/cta_stations.parquet', engine='fastparquet', index=False)