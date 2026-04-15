import os
import argparse
import joblib
import awkward as ak
import pandas as pd
import xgboost as xgb
import yaml
# -------------------------------
# Load trained BDT bundle
# -------------------------------
booster = xgb.Booster()
booster.load_model("trained_model_for_ttbar.xgb")

features = ["FW1","Sxz","Szz","AL","p2in","planarity","pT_Sum","nJet","delta_R","dphi_lb"]
print("Loaded BDT model")
print("Features used:", features)

with open("xsec_ngen_config.yaml","r") as f:
    cfg = yaml.safe_load(f)

UL2018 = cfg[2017]

# Integrated luminosity (pb^-1)
lumi = 41479.6

# Per-sample normalization factor: (XS * lumi / Ngen)
factor = {
    sample: xs * lumi / ngen
    for sample, (xs, ngen) in UL2018.items()
}

factor["DATA"] = 1.0

print("Factors are : ", factor)
# -------------------------------
# Core function
# -------------------------------
def apply_to_dataset(file, tree_key, weight_branch="weight", out_dir="."):

    print(f"\nProcessing: {file}")

    # Load parquet
    data = ak.from_parquet(file)
    data_flat = data[tree_key]
    df = ak.to_dataframe(data_flat)
    

    print(df.head())
    print("Events loaded:", len(df))

    # Check weight branch
    if weight_branch not in df.columns:
        raise RuntimeError(
            f"Weight branch '{weight_branch}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Build feature matrix
    X = df[features]
    #X = imputer.transform(X)

    # Compute BDT score
    dmatrix = xgb.DMatrix(X)
    bdt_score = booster.predict(dmatrix)
    
    
    print(df[weight_branch].values)
    print(factor[tree_key])
    
    weight_normalised = df[weight_branch] * factor[tree_key]
    #weight_normalised = df[weight_branch]  # ---> Later you can add the lumi weight while running the python file for generating the data mc for the bdt scores 
    print(f"Tree Key is : {tree_key} and norm is {factor[tree_key]}")
    
    # Final minimal dataframe
    out_df = pd.DataFrame({
        "BDT_score": bdt_score,
        "PDG_ID":df["sum_pdgId"].values,
        "weights": weight_normalised
    })

    # Auto-generate output filename
    output_file = os.path.join(
        out_dir,
        f"{tree_key}_bdt_score_pdgid_weights.parquet"
    )

    # Write parquet
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_parquet(output_file)

    print(f"Saved: {output_file}")


# -------------------------------
# CLI
# -------------------------------
parser = argparse.ArgumentParser(
    description="Compute BDT score and save only BDT_score and weights"
)

parser.add_argument(
    "--file",
    default= "ttbar_SemiLeptonic.parquet",
    type=str,
    help="Input parquet file"
)

parser.add_argument(
    "--tree_key",
    default = "ttbar_SemiLeptonic",
    type=str,
    help="Dataset name (used for output file naming)"
)

parser.add_argument(
    "--weight_branch",
    default="weights",
    type=str,
    help="Name of the weight branch (default: weight)"
)

parser.add_argument(
    "--out_dir",
    default=".",
    type=str,
    help="Directory to store output parquets (default: current directory)"
)

args = parser.parse_args()

# -------------------------------
# Run
# -------------------------------
apply_to_dataset(
    file=args.file,
    tree_key=args.tree_key,
    weight_branch=args.weight_branch,
    out_dir=args.out_dir
)

