import json
import sys


if len(sys.argv) != 3:
    print("Usage: python transfer_factor_calc.py <regionA> <regionB>")
    print("Example: python transfer_factor_calc.py regionA regionB")
    sys.exit(1)

region_1 = sys.argv[1]
region_2 = sys.argv[2]

file_1 = f"bin_by_bin_{region_1}_diff_values.json"
file_2 = f"bin_by_bin_{region_2}_diff_values.json"
# Load the data from the JSON files
try:
    with open(file_1, 'r') as f:
        data_1 = json.load(f)

    with open(file_2, 'r') as f:
        data_2 = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both JSON files are in the same directory as the script.")
    exit()

# Initialize the dictionary to store the results
transfer_factors = {}

# Get the set of common keys (histogram names) between the two files
common_keys = set(data_1.keys()) & set(data_2.keys())

# Iterate over each common histogram
for key in common_keys:
    values_1 = data_1[key]
    values_2 = data_2[key]
    
    # Initialize a list for the calculated ratios for the current histogram
    ratios = []
    
    # Determine the number of bins to process (the minimum of the two lists)
    num_bins = min(len(values_1), len(values_2))
    
    # Calculate the B/A ratio for each bin
    for i in range(num_bins):
        val_1 = values_1[i]
        val_2 = values_2[i]
        
        # Check for division by zero
        if val_1 == 0 or val_2 == 0:
            # Handle division by zero, setting to 1.0 as a neutral factor
            ratio = 1.0
        else:
            ratio = val_2 / val_1
        
        # Cap the ratio to 1.0 if it's negative
        if ratio < 0:
            final_ratio = 1.0
        else:
            final_ratio = ratio
            
        ratios.append(final_ratio)
        print(ratios)
    # Add the list of calculated ratios to our results dictionary
    transfer_factors[key] = ratios

# Save the final results to a new JSON file
output_filename = f'bin_by_bin_{region_2}_by_{region_1}.json'
with open(output_filename, 'w') as f:
    json.dump(transfer_factors, f, indent=4)

print(f"Successfully created '{output_filename}' with the calculated transfer factors.")
