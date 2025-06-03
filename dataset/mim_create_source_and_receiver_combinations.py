import itertools
import csv
import random
import math
import numpy as np

# Define dimensions
lengths = [3, 3.5, 4, 4.5, 5, 5.5, 6]
widths = [3, 3.5, 4, 4.5, 5, 5.5, 6]
heights = [2.5, 3, 3.5, 4]

distance_from_edge = 1.0
min_distance_sr = 1.0
default_step = 0.4
fallback_step = 0.2

# --- Step 1: Generate unique (L, W) → coordinate sets ---
unique_LW = list(itertools.product(lengths, widths))
geometry_data = {}

def valid_pairwise_distance(sources, receivers, min_dist):
    for s, r in zip(sources, receivers):
        if math.hypot(s[0] - r[0], s[1] - r[1]) < min_dist:
            return False
    return True

def try_generate_positions(L, W, step, max_attempts=1000):
    valid_x = np.arange(distance_from_edge, L - distance_from_edge + 0.001, step)
    valid_y = np.arange(distance_from_edge, W - distance_from_edge + 0.001, step)

    for _ in range(max_attempts):
        sources = [(round(random.choice(valid_x), 2), round(random.choice(valid_y), 2)) for _ in range(3)]
        receivers = [(round(random.choice(valid_x), 2), round(random.choice(valid_y), 2)) for _ in range(3)]
        if valid_pairwise_distance(sources, receivers, min_distance_sr):
            return sources, receivers
    return None, None  # Signal failure

for L, W in unique_LW:
    sources, receivers = try_generate_positions(L, W, default_step)
    if sources is None:
        print(f"Fallback: Trying smaller step size for L={L}, W={W}")
        sources, receivers = try_generate_positions(L, W, fallback_step)

    if sources is None:
        print(f"Could not find valid config for L={L}, W={W}, using zero placeholders.")
        sources = receivers = [(0, 0)] * 3

    geometry_data[(L, W)] = (sources, receivers)

# --- Generate full data with heights ---
full_combinations = list(itertools.product(lengths, widths, heights))
full_data = []

id_counter = 1  # Initialize an ID counter

for L, W, H in full_combinations:
    sources, receivers = geometry_data[(L, W)]
    for s, r in zip(sources, receivers):
        row = [id_counter, L, W, H, 2*(L * W + L * H + W * H), L*W*H, s[0], s[1], 1.5, r[0], r[1], 1.5]
        full_data.append(row)
        id_counter += 1  # Increment the ID counter

# --- Write full data CSV ---
with open('dataset/mim_full_source_receiver_data.csv', 'w', newline='') as full_file:
    writer = csv.writer(full_file)
    header = ['ID', 'Length', 'Width', 'Height', 'Surface_Area', 'Volume', 'Source_X', 'Source_Y', 'Source_Z', 'Receiver_X', 'Receiver_Y', 'Receiver_Z']
    writer.writerow(header)
    writer.writerows(full_data)

print("File created:")
print(" mim_full_source_receiver_data.csv (one row per source-receiver combination)")
