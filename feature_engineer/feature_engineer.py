import pandas as pd
import numpy as np
import json
import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cta_ridership", type=str, required=True)
parser.add_argument("--cta_stations", type=str, required=True)
parser.add_argument("--cta_ridership_with_features", type=str, required=True)
args = parser.parse_args()

cta_df = pd.read_parquet(args.cta_ridership)

cta_df.head(10)

cta_stations_df = pd.read_parquet(args.cta_stations)

cta_stations_df.head(10)

stop_id_unique = set(cta_stations_df["map_id"])

mask_station_id = cta_df["station_id"].isin(stop_id_unique)
# True if everything is valid
all_stations_in_stations_df = mask_station_id.all()
print(all_stations_in_stations_df)

invalid_rows = cta_df[~mask_station_id]
print(invalid_rows['stationname'].value_counts())

# First, deduplicate the stations df since it seems to have one row per cardinal direction.
cta_stations_df_merge = cta_stations_df.groupby('map_id').first().reset_index()
cta_stations_df_merge = cta_stations_df_merge[['map_id', 'red', 'blue', 'g', 'brn', 'p', 'y', 'pnk', 'o', 'location']]

# Then, add features for longitude and latitude
cta_stations_df_merge['location'] = cta_stations_df_merge['location'].apply(json.loads)
cta_stations_df_merge['lat'] = cta_stations_df_merge['location'].apply(lambda d: d.get('latitude')).astype(float)
cta_stations_df_merge['lon'] = cta_stations_df_merge['location'].apply(lambda d: d.get('longitude')).astype(float)

cta_df = cta_df.merge(
    cta_stations_df_merge,
    how='left',
    left_on='station_id',
    right_on='map_id'
).dropna(
    subset=['map_id']
)

# Create categorical line feature
conditions_cta_line = [
    cta_df['red'] == 1,
    cta_df['blue'] == 1,
    cta_df['g'] == 1,
    cta_df['brn'] == 1,
    cta_df['p'] == 1,
    cta_df['y'] == 1,
    cta_df['pnk'] == 1,
    cta_df['o'] == 1
]

choices_cta_line = [
    'red',
    'blue',
    'green',
    'brown',
    'purple',
    'yellow',
    'pink',
    'orange'
]

cta_df['line'] = np.select(conditions_cta_line, choices_cta_line, default='NA')

cta_df.head(10)

cta_df['line'].value_counts()

cta_df['date'] = pd.to_datetime(cta_df['date'])

cta_df['year'] = cta_df['date'].dt.year
cta_df['month'] = cta_df['date'].dt.month
cta_df['day'] = cta_df['date'].dt.day

cta_df['day_of_week_num'] = cta_df['date'].dt.weekday
cta_df['day_of_week_name'] = cta_df['date'].dt.day_name()

cta_df.head(10)

cta_df.head(10)

cta_df.tail(10)

cta_df.to_parquet(args.cta_ridership_with_features, engine='fastparquet', index=False)