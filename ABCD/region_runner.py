# run_analysis.py
import sys
import dask
from coffea import util
from coffea.nanoevents import NanoAODSchema
from dask.diagnostics import ProgressBar
from coffea.dataset_tools import preprocess, apply_to_fileset, max_chunks
from dataset import get_fileset
from distributed import Client  
import os 
import importlib
from datetime import datetime
import argparse
from region_abcd_proc import ElectronChannel

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run ABCD method for QCD estimation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_analysis.py  --year 2017 --region A --btagWP M --pileUpWP M
  python3 run_analysis.py  --year 2018 --region B --btagWP M --pileUpWP M 
        """
    )

    parser.add_argument(
        '--year',
        type=str,
        required=True,
        choices=['2016postVFP','2016preVFP','2017','2018'],
        help='Data-taking eras/year'
    )

    parser.add_argument(
        '--region',
        type=str,
        required=True,
        choices=['A','B','C','D'],
        help='Region for ABCD'        
    )

    parser.add_argument(
        '--btagWP',
        type=str,
        required=True,
        choices=['L','M','T'],
        help='BTagging working point'
    )

    parser.add_argument(
        '--pileUpWP',
        type=str,
        required=True,
        choices=['L','M','T'],
        help='PileUp working point'
    )
    
    parser.add_argument(
        '--jetpt',
        type=float,
        required=True,
        default =20.0,
        help='Minimum Jet pt'
    )
    
    parser.add_argument(
        '--choice',
        type=int,
        required=True,
        choices=[1,2],
        help='1-->ID VS MT , 2-->ID VS NBTAGS'
    )
    return parser.parse_args()

def main():
    #Parse command line arguments
    args = parse_args()

    print("="*60)
    print(f"Welcome to ABCD Method, Here we shall estimate the QCD !!!")
    print("="*60)
    print(f"Configuration")
    print(f"  Era : {args.year}")
    print(f"  Region : {args.region}")
    print(f"  Btagging_wp :{args.btagWP} ")
    print(f"  PileUp_wp:{args.pileUpWP}")
    print(f"  Minimum Jet Pt:{args.jetpt}")
    print(f"  ABCD_CHOICE:{args.choice}")
    print("="*60)


    #Initialize processor with command line arguments
    processor_instance = ElectronChannel(year=args.year , region = args.region ,btagWP = args.btagWP ,pileUpWP = args.pileUpWP , jetPt=args.jetpt , choice = args.choice)

    client =Client()

    print("-----Fileset Loading---")
    fileset = get_fileset()
    
    print(f"\n DataSet summary for {args.year}:")
    print("-"*60)
    for key, info in fileset.items():
        print(f"{key:<20} : {len(info['files']):>4} files")
    

    print(f"\n{'='*60}")
    print(f"Hi! I am running analysis for the Region {args.region}, Year {args.year}....")
    print(f"{'='*60}\n")

    # Process the data
    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        files_per_batch=10,
        skip_bad_files=True,
        save_form=False,
    )
    
    to_compute = apply_to_fileset(
        processor_instance,
        max_chunks(dataset_runnable, 300),
        schemaclass=NanoAODSchema,
    )
    
    with ProgressBar():
        (out,) = dask.compute(to_compute, scheduler='threads')
    
    # Create output directory based on module name
    output_dir = f"/mnt/disk2/sanskar/REGIONS_ABCD/{args.year}Analysis/ABCD_NBTAGS_VS_ID/region{args.region}"
    #output_dir = "/nfs/home/sanskar/2016postVFP/NEW_SCRIPTS/ABCD_WITH_NBJETS_EXPT"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on module name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"{output_dir}/region{args.region}_ABCD_for_{args.year}_with_nbtags_vs_id.coffea"
    #output_path = "test.coffea"
    # Save the output
    util.save(out, output_path)
    
    print(f"✓ Analysis completed successfully!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
