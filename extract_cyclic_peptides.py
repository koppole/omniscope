#!/usr/bin/env python3
import os
import sys
from pyrosetta import *
import glob

# Define the chain ID to extract
PEPTIDE_CHAIN = "G"  # Chain G to extract

def initialize_pyrosetta():
    """Initialize PyRosetta with appropriate options."""
    init(options="-mute all")
    print("PyRosetta initialized successfully.")

def extract_chain_G(input_pdb, output_dir):
    """
    Extract chain G from the input PDB file and save it to output directory.
    Works with both linear and cyclic peptides.
    
    Parameters:
    - input_pdb: Path to input PDB file
    - output_dir: Directory to save the extracted chain
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
    
    # Extract chain G using proper chain selection
    try:
        # Create a selector for chain G
        chain_selector = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(PEPTIDE_CHAIN)
        chain_selection = chain_selector.apply(pose)
        
        # Convert boolean vector to residue indices vector
        residue_indices = pyrosetta.rosetta.utility.vector1_unsigned_long()
        for i in range(1, len(chain_selection) + 1):
            if chain_selection[i]:
                residue_indices.append(i)
        
        # Check if any residues were selected
        if len(residue_indices) == 0:
            print(f"No Chain {PEPTIDE_CHAIN} found in {input_pdb}")
            return False
        
        # Create a new pose with only the selected residues
        subset_pose = pyrosetta.Pose()
        pyrosetta.rosetta.core.pose.pdbslice(subset_pose, pose, residue_indices)
        
        # Save the extracted chain to the output directory
        output_path = os.path.join(output_dir, f"{base_name}_chain_{PEPTIDE_CHAIN}.pdb")
        subset_pose.dump_pdb(output_path)
        print(f"Saved Chain {PEPTIDE_CHAIN} to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting Chain {PEPTIDE_CHAIN} from {input_pdb}: {e}")
        return False

def main():
    # Check if command-line arguments are provided
    if len(sys.argv) != 3:
        print(f"Usage: python extract_chain_G.py <input_directory> <output_directory>")
        print(f"       This will extract Chain {PEPTIDE_CHAIN} from all PDB files")
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
    print(f"Extracting Chain {PEPTIDE_CHAIN} from each file")
    
    # Process each PDB file
    success_count = 0
    for pdb_file in pdb_files:
        if extract_chain_G(pdb_file, output_dir):
            success_count += 1
    
    print(f"Processing complete. Successfully extracted Chain {PEPTIDE_CHAIN} from {success_count} out of {len(pdb_files)} files.")

if __name__ == "__main__":
    main()