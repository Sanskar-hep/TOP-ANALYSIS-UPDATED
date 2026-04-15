from coffea.util import load,save
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import hist
import json
import sys


def print_banner():

    HEADER = "\033[95m"
    INFO = "\033[92m"
    LINE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    banner = f"""
{LINE}{BOLD}====================================================================={RESET}
{HEADER}{BOLD}  ABCD Background Estimation Framework{RESET}
{LINE}---------------------------------------------------------------------{RESET}
{INFO}  Channel   : Electron + Jets{RESET}
{INFO}  Method    : Data-Driven QCD Extraction{RESET}
{INFO}  Strategy  : ABCD with Non-QCD Subtraction{RESET}
{INFO}  Era       : Run2_UL
{INFO}  Admin     : SANSKAR NANDA
{LINE}{BOLD}====================================================================={RESET}
"""

    print(banner)

print_banner()
# Check if the region argument is given
if len(sys.argv) != 2:
    print("Usage: python script.py <region> ")
    sys.exit(1)

region = sys.argv[1]


# Set HEP style
hep.style.use("CMS")

# Add the xsec and the ngen for all the eras of Run2
UL2016postVFP = {
    "ttbar_SemiLeptonic": [366.3, 144722000],
    "Tchannel": [134.2, 63073000],
    "Schannel": [2.215836, 5471000],
    "ttbar_FullyLeptonic": [88.5, 43546000],
    "tw_top": [39.65, 3368375],
    "tw_antitop": [39.65, 3654510],
    "DYJetsToLL": [6424.0, 82448544],
    "Tbarchannel": [80.0, 30609000],
    "WJetsToLNu_0J": [52780.0, 159756701],
    "WJetsToLNu_1J": [8832.0, 167292982],
    "WJetsToLNu_2J": [3276.0, 26790000],
    "WWTo2L2Nu": [11.09, 2900000],
    "WWTolnulnu": [10.48, 4932000],
    "WZTo2Q2L": [6.565, 13526954],
    "ZZTo2L2Nu": [0.974, 16826232],
    "ZZTo2Q2L": [3.676, 13740600],
}

UL2016preVFP = {
    "ttbar_SemiLeptonic": [366.3, 131106831],
    "ttbar_FullyLeptonic": [88.5, 37202073],
    "Tchannel": [134.2, 52437432],
    "Tbarchannel": [80.0, 29205915],
    "Schannel": [2.215836, 3592772],
    "tw_top": [39.65, 3294485],
    "tw_antitop": [39.65, 3176335],
    "DYJetsToLL": [6424.0, 95170552],
    "WJetsToLNu_0J": [52780.0, 121208493],
    "WJetsToLNu_1J": [8832.0, 84198168],
    "WJetsToLNu_2J": [3276.0, 27463756],
    "WWTo2L2Nu": [11.09, 3006596],
    "WWTolnulnu": [10.48, 4932000],
    "WZTo2Q2L": [6.565, 9780392],
    "ZZTo2L2Nu": [0.974, 16826232],
    "ZZTo2Q2L": [3.676, 10406942],
}

UL2017 = {
    "ttbar_SemiLeptonic":[366.3, 343257745],
    "ttbar_FullyLeptonic":[88.5, 105860011],
    "Tchannel":[134.2, 121728258],
    "Tbarchannel":[80.0, 65701149],
    "Schannel":[2.215836, 8866570],
    "tw_top":[39.65, 8506765],
    "tw_antitop":[39.65, 8433562],
    "DYJetsToLL":[6424.0, 131552392],
    "WJetsToLNu_0J":[52780.0, 135263983],
    "WJetsToLNu_1J":[8832.0,85950236],
    "WJetsToLNu_2J":[3276.0, 29987306],
    "WWTo2L2Nu":[11.09, 7071358],
    "WWTolnulnu":[10.79, 2000000],
    "WZTo2Q2L":[5.595, 18136497],
    "ZZTo2L2Nu": [0.974, 16826232],
    "ZZTo2Q2L":[3.676, 19134840],
    
}

UL2018 = {
    "ttbar_SemiLeptonic":[366.3, 472977862],
    "ttbar_FullyLeptonic":[88.5, 143830836],
    "Tchannel":[134.2, 166637158],
    "Tbarchannel":[80.0, 89985007],
    "Schannel":[2.215836, 12444591],
    "tw_top":[39.65, 11270430],
    "tw_antitop":[39.65, 10949620],
    "DYJetsToLL":[6424.0, 129037134],
    "WJetsToLNu_0J":[52780.0, 137259710],
    "WJetsToLNu_1J":[8832.0, 87594835],
    "WJetsToLNu_2J":[3276.0, 29028341],
    "WWTo2L2Nu":[11.09, 9962019],
    "WZTo2Q2L":[6.565, 17952068],
    "ZZTo2L2Nu":[0.974, 16826232],
    "ZZTo2Q2L":[3.676, 19082659]

}

eras = {
    "UL2016postVFP":UL2016postVFP,
    "UL2016preVFP": UL2016preVFP,
    "UL2017": UL2017,
    "UL2018":UL2018
}

lumi_config = {
    "UL2016postVFP": 16812.1,
    "UL2016preVFP": 19521.2,
    "UL2017": 41479.6,
    "UL2018":59222.7416,
}

file_tags = {
    "UL2016postVFP":"2016postVFP",
    "UL2016preVFP":"2016preVFP",
    "UL2017":"2017",
    "UL2018":"2018"
}

print("\n Available Eras")
era_list = list(eras.keys())

for i, name in enumerate(era_list, start=1):
    print(f"{i}. {name}")

while True:
    choice = input("\n Select the era for which you want to do the Data-Non QCD MC :  ")
    if choice.isdigit() and 1 <= int(choice) <= len(era_list):
        era = era_list[int(choice) - 1]
        break
    else:
        print("\n Please give a correct input, refer to the keys printed above")

selected_samples = eras[era]
lumi = lumi_config[era]
file_tag = file_tags[era]

factor = {k: (xsec * lumi / ngen) for k, (xsec, ngen) in selected_samples.items()}

print("===The details for the current era which is analysed===")
print(f"The luminosity value is :{lumi}")
print(f"The corresponding era is : {file_tag}")

print(f"Please check if the era being analysed, is the era you requested. If you see any mismatch contact the ADMIN for the above issue !!")

# Load coffea output
output = load(f"region{region}_ABCD_for_{file_tag}_with_nbtags_vs_id.coffea")

# Define the histograms you want to process
histogram_names = ["pTSum", "nJet", "FW1", "p2in", "AL", "delta_R", "planarity", "Sxz", "Szz"]  # Add more as needed

# Dictionary to store all QCD estimates
qcd_estimates = {}
bin_by_bin_diff = {}
# Loop through each histogram
for hist_name in histogram_names:
    print(f"\n{'='*60}")
    print(f"PROCESSING HISTOGRAM: {hist_name}")
    print(f"{'='*60}")
    
    mc_hist = []
    mc_labels = []
    data_hist = []
    
    print(f"Loading and scaling MC samples for {hist_name}...")
    
    # Check if histogram exists in the data
    hist_exists = False
    for dataset, nested in output.items():
        if hist_name in nested[dataset]:
            hist_exists = True
            break
    
    if not hist_exists:
        print(f"WARNING: {hist_name} not found in any dataset. Skipping...")
        continue
    
    # Loop over datasets for this histogram
    for dataset, nested in output.items():
        if hist_name not in nested[dataset]:
            continue
        hist_obj = nested[dataset][hist_name]

        if dataset == "DATA":
            data_hist.append(hist_obj)
        else:
            if dataset in factor:
                scale = factor[dataset]
                h_scaled = hist_obj * scale
                mc_hist.append(h_scaled)
                mc_labels.append(dataset)
            else:
                print(f"Skipping {dataset}: no normalization factor found")

    # Check if we have both data and MC for this histogram
    if len(data_hist) == 0:
        print(f"WARNING: No data found for {hist_name}. Skipping...")
        continue
    if len(mc_hist) == 0:
        print(f"WARNING: No MC samples found for {hist_name}. Skipping...")
        continue

    # Calculate total MC histogram by summing all MC samples
    print(f"Combining all MC samples for {hist_name}...")
    total_mc_hist = None
    for i, mc_h in enumerate(mc_hist):
        if total_mc_hist is None:
            total_mc_hist = mc_h.copy()
        else:
            total_mc_hist += mc_h
        print(f"  Added {mc_labels[i]}: {mc_h.sum().value:.1f} events")

    # Combine data histograms
    print(f"Combining data samples for {hist_name}...")
    combined_data = None
    for data_h in data_hist:
        if combined_data is None:
            combined_data = data_h.copy()
        else:
            combined_data += data_h

    print(f"Total Data events: {combined_data.sum().value:.0f}")
    print(f"Total MC events: {total_mc_hist.sum().value:.1f}")

    # Get bin information from the original histogram
    bin_edges = combined_data.axes[0].edges
    n_bins = len(combined_data.axes[0])
    bin_start = bin_edges[0]
    bin_stop = bin_edges[-1]

    print(f"Histogram binning info for {hist_name}:")
    print(f"  Number of bins: {n_bins}")
    print(f"  Start: {bin_start}")
    print(f"  Stop: {bin_stop}")

    # Create a new hist.Hist object for the difference
    difference_hist = combined_data.copy()
    difference_hist.reset()

    # Calculate the difference: Data - MC for each bin
    data_values = combined_data.values()
    mc_values = total_mc_hist.values()
    difference_values = data_values - mc_values
    difference_variances = combined_data.variances() + total_mc_hist.variances()
    
    #----Handle the negative events in the bins--(New addition) 
    #difference_values = np.maximum(difference_values ,0)
    
    # store bin by bin difference values:
    bin_by_bin_diff[hist_name] = difference_values.tolist()

    # Set the histogram values directly from the difference array
    difference_hist.view(flow=False)['value'] = difference_values
    difference_hist.view(flow=False)['variance'] = difference_variances

    print(f"Total difference (Data - MC): {np.sum(difference_values):.1f} events")
    
    qcd_fraction = np.sum(difference_values)/combined_data.sum().value
    print(f"qcd_fraction : {qcd_fraction * 100}")

    # Store the QCD estimate for this histogram
    qcd_estimates[hist_name] = difference_hist

    # Create the difference plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot the difference histogram
    difference_hist.plot1d(ax=ax, histtype='fill', alpha=0.7, color='steelblue',
                           edgecolor='black', linewidth=0.5, label='Data - Total MC')

    # Styling
    # Set appropriate axis labels based on histogram name
    if "pt" in hist_name.lower():
        xlabel = f'{hist_name.replace("_", " ").title()} [GeV]'
    elif "eta" in hist_name.lower():
        xlabel = f'{hist_name.replace("_", " ").title()}'
    elif "phi" in hist_name.lower():
        xlabel = f'{hist_name.replace("_", " ").title()}'
    else:
        xlabel = f'{hist_name.replace("_", " ").title()}'
    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Data - MC [Events]', fontsize=16)
    ax.set_title(f'Data - MC Difference for {hist_name.replace("_", " ").title()}', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=16, ncol=1, bbox_to_anchor=(0.98, 0.98),
              frameon=True, fancybox=True, shadow=True, framealpha=0.9,
              edgecolor='black', facecolor='white')

    # Set x-axis limits
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    # Auto-adjust y-axis with some padding
    diff_max = np.max(difference_values)
    diff_min = np.min(difference_values)
    ax.set_ylim(-diff_max, diff_max*1.1)

    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.savefig(f'QCD_hist_{hist_name}_{region}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print bin-by-bin results
    print(f"\nBIN-BY-BIN DIFFERENCE ANALYSIS for {hist_name}")
    print("="*60)
    print(f"{'Bin':>3} {'Range':>15} {'Data':>10} {'MC':>10} {'Difference':>12}")
    print("-" * 60)

    bin_centers = combined_data.axes[0].centers
    for i in range(len(bin_centers)):
        range_low = bin_edges[i]
        range_high = bin_edges[i+1]
        print(f"{i+1:>3} {range_low:>6.2f}-{range_high:<6.2f} {data_values[i]:>10.1f} "
              f"{mc_values[i]:>10.1f} {difference_values[i]:>12.1f}")

    print(f"\nSummary for {hist_name}:")
    print(f"  Positive differences (Data > MC): {np.sum(difference_values > 0)} bins")
    print(f"  Negative differences (Data < MC): {np.sum(difference_values < 0)} bins")
    print(f"  Zero differences: {np.sum(difference_values == 0)} bins")
    print(f"  Largest positive difference: {np.max(difference_values):.1f}")
    print(f"  Largest negative difference: {np.min(difference_values):.1f}")

# Save all QCD estimates to a single file
final_output = {
    "QCD": qcd_estimates
}

output_filename = f"QCD_ESTIMATE_ALL_HISTOGRAMS_from{region}.coffea"
save(final_output, output_filename)

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Successfully processed {len(qcd_estimates)} histograms:")
qcd_total_events = {} #to store as json file first create a dictionary 
for hist_name in qcd_estimates.keys():
    total_events = qcd_estimates[hist_name].sum().value
    print(f"  {hist_name}: {total_events:.1f} events")
    
    #fill the dictionary , structure : {hist_name : corresponding_total_events}
    qcd_total_events[hist_name] = total_events


#create the json filename
json_filename = f"region{region}_allHist.json"
json_filename1 = f"bin_by_bin_region{region}_diff_values.json"

#open the json file in the write mode, with specific indentation
with open(json_filename,'w') as json_file : 
    json.dump(qcd_total_events ,json_file,indent =4)

with open(json_filename1,'w') as json_file1:
    json.dump(bin_by_bin_diff, json_file1,indent=4)

#Print the summary
print(f"\nAll QCD estimates saved to: {output_filename}")
print("Individual plots saved as: QCD_hist_{histogram_name}.png")
print("QCD histograms are now compatible with your MC histograms!")
print(f"\nAll the events for each hist are saved to {json_filename}")
print(f"\nDifference values bin by bin for region{region} saved : {json_filename1}")
