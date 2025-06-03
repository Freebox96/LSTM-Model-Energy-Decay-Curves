import pandas as pd

# Load the two CSV files
csv_a = pd.read_csv('dataset/mim_full_source_receiver_data.csv')
csv_b = pd.read_csv('dataset/mhe_room_material_average.csv')

# Repeat each row of csv_a m times (where m = number of rows in csv_b)
csv_a_repeated = pd.DataFrame(csv_a.values.repeat(len(csv_b), axis=0), columns=csv_a.columns)

# Tile (repeat) csv_b to align with the repeated csv_a
csv_b_tiled = pd.concat([csv_b]*len(csv_a), ignore_index=True)

# Concatenate them side-by-side
merged = pd.concat([csv_a_repeated.reset_index(drop=True), csv_b_tiled.reset_index(drop=True)], axis=1)


# Drop specific columns from the merged result
columns_to_omit = ['Surface_Area', 'Volume']  
merged = merged.drop(columns=columns_to_omit, errors='ignore')

merged['ID'] = merged['ID'].astype(int)

# Save the result
merged.to_csv('dataset/full_dataset.csv', index=False)
