#!/usr/bin/env python3
import os
import sys
from pyrosetta import *
import glob

# Define the chain IDs here as variables for easy modification
ACE2_CHAIN = "A"  # Default ACE2 chain
PEPTIDE_CHAIN = "G"  # Default peptide chain

def initialize_pyrosetta():
    """Initialize PyRosetta with appropriate options."""
    init(options="-mute all")
    print("PyRosetta initialized successfully.")

def extract_peptide(input_pdb, output_dir, ace2_chain=ACE2_CHAIN, peptide_chain=PEPTIDE_CHAIN):
    """
    Extract peptide chain from the input PDB file and save it to output directory.
    
    Parameters:
    - input_pdb: Path to input PDB file
    - output_dir: Directory to save the extracted peptide
    - ace2_chain: Chain ID for ACE2 (not used for extraction, just for logging)
    - peptide_chain: Chain ID for the peptide to extract
    """
    # Get the base filename without path and extension
    base_name = os.path.basename(input_pdb).replace('.pdb', '')
    
    # Load the PDB structure
    try:
        pose = pyrosetta.pose_from_pdb(input_pdb)
        print(f"Loaded: {input_pdb}")
    except Exception as e:
        print(f"Error loading {input_pdb}: {e}")
        return False
    
    # Create a selector for the peptide chain
    chain_selector = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(peptide_chain)
    
    # Get the residue indices for the peptide chain
    peptide_residues = pyrosetta.rosetta.core.select.get_residues_from_subset(chain_selector.apply(pose))
    
    if not peptide_residues:
        print(f"No Chain {peptide_chain} (peptide) found in {input_pdb}")
        return False
    
    # Create a new pose with only the peptide chain
    peptide_pose = pyrosetta.Pose()
    for res_idx in peptide_residues:
        peptide_pose.append_residue_by_jump(pose.residue(res_idx), 1)
    
    # Save the peptide pose to the output directory
    output_path = os.path.join(output_dir, f"{base_name}_peptide_{peptide_chain}.pdb")
    peptide_pose.dump_pdb(output_path)
    print(f"Saved peptide (Chain {peptide_chain}) to: {output_path}")
    
    return True

def main():
    # Check if command-line arguments are provided
    if len(sys.argv) != 3:
        print(f"Usage: python extract_peptides.py <input_directory> <output_directory>")
        print(f"       Processing files with ACE2 as Chain {ACE2_CHAIN} and peptide as Chain {PEPTIDE_CHAIN}")
        print(f"       (You can modify these chain IDs at the top of the script)")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Initialize PyRosetta
    initialize_pyrosetta()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process all PDB files in the input directory
    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return
    
    print(f"Found {len(pdb_files)} PDB files to process...")
    print(f"Extracting peptide chain {PEPTIDE_CHAIN} (ACE2 is chain {ACE2_CHAIN})")
    
    # Process each PDB file
    success_count = 0
    for pdb_file in pdb_files:
        if extract_peptide(pdb_file, output_dir, ACE2_CHAIN, PEPTIDE_CHAIN):
            success_count += 1
    
    print(f"Processing complete. Successfully extracted peptides from {success_count} out of {len(pdb_files)} files.")

if __name__ == "__main__":
    main()