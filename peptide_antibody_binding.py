import os
import sys
import glob
import pyrosetta
import numpy as np
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.analysis import *
from pyrosetta.rosetta.core.pose import *
from pyrosetta.rosetta.protocols.minimization_packing import *
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.rigid import *
from pyrosetta.rosetta.protocols.relax import *

# Minimal initialization with essential options for stability
pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false -use_input_sc -ex1 -ex2")

# TCR data from the business case document
tcr_data = [
    {
        "CDR3": "CASSLDRGSEKLFF",
        "V": "TRBV5-1*01",
        "J": "TRBJ1-4*01",
        "Species": "HomoSapiens",
        "MHC A": "HLA-A*02:01",
        "MHC B": "B2M",
        "MHC class": "MHCI",
        "Epitope": "KLPDDFTGCV",
        "Epitope gene": "Spike"
    },
    {
        "CDR3": "CASSVGQGYEQYF",
        "V": "TRBV5-1*01",
        "J": "TRBJ2-7*01",
        "Species": "HomoSapiens",
        "MHC A": "HLA-A*24:02",
        "MHC B": "B2M",
        "MHC class": "MHCI",
        "Epitope": "NYNYLYRLF",
        "Epitope gene": "Spike"
    }
]

def load_structures(antibody_pdbs, peptide_directory):
    """
    Load antibody structures from specific files and peptide structures from a directory
    """
    antibodies = {}
    for name, pdb_file in antibody_pdbs.items():
        try:
            pose = pyrosetta.pose_from_pdb(pdb_file)
            antibodies[name] = pose
            print(f"Loaded antibody {name} from {pdb_file}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    # Load all peptides from the directory
    peptides = {}
    peptide_files = glob.glob(os.path.join(peptide_directory, "*.pdb"))
    
    if not peptide_files:
        print(f"No PDB files found in directory: {peptide_directory}")
        sys.exit(1)
    
    for pdb_file in peptide_files:
        try:
            name = os.path.basename(pdb_file).replace(".pdb", "")
            pose = pyrosetta.pose_from_pdb(pdb_file)
            peptides[name] = pose
            print(f"Loaded peptide {name} from {pdb_file}")
        except Exception as e:
            print(f"Failed to load {pdb_file}: {e}")
    
    return antibodies, peptides

def extract_peptide_sequence(peptide_pose):
    """
    Extract the amino acid sequence from a pose, skipping non-standard residues
    """
    sequence = ""
    for i in range(1, peptide_pose.total_residue() + 1):
        try:
            # Only extract sequence for protein residues
            if peptide_pose.residue(i).is_protein():
                aa_type = peptide_pose.residue(i).aa()
                one_letter = pyrosetta.rosetta.core.chemical.name_from_aa(aa_type)
                sequence += one_letter
            else:
                # For non-standard residues, add an X (or skip)
                pass
        except Exception as e:
            print(f"Warning: Could not extract sequence for residue {i}: {e}")
    return sequence

def is_protein_residue_with_ca(pose, res_idx):
    """Helper function to check if a residue is a standard protein residue with CA atom"""
    try:
        return pose.residue(res_idx).is_protein() and pose.residue(res_idx).has("CA")
    except Exception:
        return False

def identify_cdr3_from_sequence(antibody_pose, cdr3_sequences):
    """
    Identify CDR3 regions in an antibody based on provided CDR3 sequences
    """
    cdr3_regions = {}
    
    # Extract sequence from the antibody pose
    antibody_sequence = ""
    for i in range(1, antibody_pose.total_residue() + 1):
        if antibody_pose.residue(i).is_protein():
            aa_type = antibody_pose.residue(i).aa()
            one_letter = pyrosetta.rosetta.core.chemical.name_from_aa(aa_type)
            antibody_sequence += one_letter
    
    # Search for each CDR3 sequence
    for cdr3_info in cdr3_sequences:
        cdr3_seq = cdr3_info["CDR3"]
        
        # Remove the 'CASS' prefix which is common in CDR3 sequences but might not be in the structure
        search_seq = cdr3_seq.replace("CASS", "")
        
        # Find the sequence in the antibody
        start_idx = antibody_sequence.find(search_seq)
        if start_idx != -1:
            # Convert string index to residue numbers
            residue_ids = []
            for i in range(len(search_seq)):
                res_num = start_idx + i + 1
                if res_num <= antibody_pose.total_residue():
                    residue_ids.append(res_num)
            
            cdr3_regions[cdr3_seq] = residue_ids
            print(f"Found CDR3 sequence {cdr3_seq} at residues {residue_ids}")
        else:
            print(f"Could not find CDR3 sequence {cdr3_seq} in antibody")
    
    return cdr3_regions

def identify_cdr_regions(antibody_pose):
    """
    Identify CDR regions of the antibody using Chothia numbering scheme (simplified)
    This is a simplified approximation - in production code, you'd use a proper CDR detector
    """
    cdr_regions = {
        "CDR-H1": [],
        "CDR-H2": [],
        "CDR-H3": [],
        "CDR-L1": [],
        "CDR-L2": [],
        "CDR-L3": []
    }
    
    # Determine chain lengths
    chains = []
    for i in range(1, antibody_pose.num_chains() + 1):
        chain_begin = antibody_pose.chain_begin(i)
        chain_end = antibody_pose.chain_end(i)
        chains.append((chain_begin, chain_end))
    
    if len(chains) < 2:
        print("Warning: Less than 2 chains detected in antibody. CDR detection may be inaccurate.")
        return cdr_regions
    
    # Assuming first chain is heavy, second is light (common convention)
    heavy_begin, heavy_end = chains[0]
    light_begin, light_end = chains[1]
    
    # Rough CDR estimates based on relative positions in the chains
    # These are rough approximations - a real implementation would use proper CDR detection
    
    # Heavy chain CDRs
    h_chain_len = heavy_end - heavy_begin + 1
    cdr_h1_start = heavy_begin + int(h_chain_len * 0.2)  # ~20% into heavy chain
    cdr_h1_end = cdr_h1_start + 10  # ~10 residues long
    
    cdr_h2_start = heavy_begin + int(h_chain_len * 0.4)  # ~40% into heavy chain
    cdr_h2_end = cdr_h2_start + 17  # ~17 residues long
    
    cdr_h3_start = heavy_begin + int(h_chain_len * 0.7)  # ~70% into heavy chain
    cdr_h3_end = cdr_h3_start + 13  # Variable length, ~13 is typical
    
    # Light chain CDRs
    l_chain_len = light_end - light_begin + 1
    cdr_l1_start = light_begin + int(l_chain_len * 0.2)  # ~20% into light chain
    cdr_l1_end = cdr_l1_start + 15  # ~15 residues long
    
    cdr_l2_start = light_begin + int(l_chain_len * 0.4)  # ~40% into light chain
    cdr_l2_end = cdr_l2_start + 7   # ~7 residues long
    
    cdr_l3_start = light_begin + int(l_chain_len * 0.7)  # ~70% into light chain
    cdr_l3_end = cdr_l3_start + 10  # ~10 residues long
    
    # Add residue numbers to CDR regions
    for i in range(cdr_h1_start, cdr_h1_end + 1):
        if is_protein_residue_with_ca(antibody_pose, i):
            cdr_regions["CDR-H1"].append(i)
    
    for i in range(cdr_h2_start, cdr_h2_end + 1):
        if is_protein_residue_with_ca(antibody_pose, i):
            cdr_regions["CDR-H2"].append(i)
    
    for i in range(cdr_h3_start, cdr_h3_end + 1):
        if is_protein_residue_with_ca(antibody_pose, i):
            cdr_regions["CDR-H3"].append(i)
    
    for i in range(cdr_l1_start, cdr_l1_end + 1):
        if is_protein_residue_with_ca(antibody_pose, i):
            cdr_regions["CDR-L1"].append(i)
    
    for i in range(cdr_l2_start, cdr_l2_end + 1):
        if is_protein_residue_with_ca(antibody_pose, i):
            cdr_regions["CDR-L2"].append(i)
    
    for i in range(cdr_l3_start, cdr_l3_end + 1):
        if is_protein_residue_with_ca(antibody_pose, i):
            cdr_regions["CDR-L3"].append(i)
    
    print("Identified CDR regions:")
    for cdr, residues in cdr_regions.items():
        print(f"  {cdr}: {len(residues)} residues")
    
    return cdr_regions

def dock_peptide_to_specific_cdr(antibody_pose, peptide_pose, cdr_regions, tcr_data, output_prefix):
    """
    Dock peptide to the specific CDR3 regions based on TCR data with fallback to other CDRs
    """
    print(f"\nDocking peptide to specific CDR regions for {output_prefix}...")
    
    try:
        # Extract the peptide sequence
        peptide_sequence = extract_peptide_sequence(peptide_pose)
        print(f"Peptide sequence: {peptide_sequence}")
        
        # Check if the peptide contains any of the epitopes from TCR data
        has_epitope = False
        matching_epitope = None
        epitope_start_idx = -1
        
        for tcr in tcr_data:
            epitope = tcr["Epitope"]
            if epitope in peptide_sequence:
                has_epitope = True
                matching_epitope = epitope
                epitope_start_idx = peptide_sequence.find(epitope)
                print(f"Peptide contains epitope {epitope} at position {epitope_start_idx}")
                break
        
        # Validate input poses
        if antibody_pose.total_residue() == 0 or peptide_pose.total_residue() == 0:
            print("ERROR: Zero residue pose detected")
            return None
            
        print(f"Antibody has {antibody_pose.total_residue()} residues")
        print(f"Peptide has {peptide_pose.total_residue()} residues")
        
        # Save individual poses for reference
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        antibody_pose.dump_pdb(f"{output_prefix}_antibody.pdb")
        peptide_pose.dump_pdb(f"{output_prefix}_peptide.pdb")
        
        # Create scorefxn
        scorefxn = pyrosetta.create_score_function("ref2015")
        
        # First, try the default CDR target
        if has_epitope:
            target_cdr = "CDR-H3"  # Use CDR-H3 as primary target for epitope binding
            print(f"Using {target_cdr} as initial target for epitope binding")
        else:
            target_cdr = "CDR-H3"  # CDR-H3 is most commonly involved in binding
            print(f"No epitope match found, using {target_cdr} as initial target")
        
        # Order CDRs to try in case initial target fails
        # Start with initial target, then H2, H1, L3, L1, L2
        cdr_priority = [target_cdr]
        for cdr in ["CDR-H2", "CDR-H1", "CDR-L3", "CDR-L1", "CDR-L2"]:
            if cdr != target_cdr:
                cdr_priority.append(cdr)
        
        # Also include any custom CDR3 regions if found
        for cdr in cdr_regions.keys():
            if cdr.startswith("CDR3-") and cdr not in cdr_priority:
                cdr_priority.append(cdr)
                
        print(f"Will try CDRs in this order: {cdr_priority}")
        
        # Try each CDR in priority order
        best_result = None
        best_binding_cdr = None
        best_score = float('inf')
        
        for current_cdr in cdr_priority:
            print(f"\nAttempting to dock to {current_cdr}...")
            
            # Skip if this CDR has no residues
            if not cdr_regions.get(current_cdr, []):
                print(f"Skipping {current_cdr} - no residues defined")
                continue
                
            # Get center of current CDR region
            cdr_center_x, cdr_center_y, cdr_center_z = 0.0, 0.0, 0.0
            cdr_count = 0
            
            for res_id in cdr_regions[current_cdr]:
                try:
                    ca_xyz = antibody_pose.residue(res_id).xyz("CA")
                    cdr_center_x += ca_xyz[0]
                    cdr_center_y += ca_xyz[1]
                    cdr_center_z += ca_xyz[2]
                    cdr_count += 1
                except Exception as e:
                    print(f"Warning: Could not get coordinates for residue {res_id}: {e}")
            
            if cdr_count > 0:
                cdr_center_x /= cdr_count
                cdr_center_y /= cdr_count
                cdr_center_z /= cdr_count
                print(f"{current_cdr} center: ({cdr_center_x:.2f}, {cdr_center_y:.2f}, {cdr_center_z:.2f})")
            else:
                print(f"Error: Could not compute center for {current_cdr}")
                continue
            
            # Create combined PDB by positioning peptide near target CDR
            combined_pdb = f"{output_prefix}_{current_cdr}_combined.pdb"
            
            try:
                with open(f"{output_prefix}_antibody.pdb", 'r') as ab_file:
                    ab_lines = ab_file.readlines()
                
                with open(f"{output_prefix}_peptide.pdb", 'r') as pep_file:
                    pep_lines = pep_file.readlines()
                
                # Get peptide center of mass or epitope center if matched
                pep_com_x, pep_com_y, pep_com_z = 0.0, 0.0, 0.0
                pep_atom_count = 0
                
                # If we have a matching epitope, try to align it with the CDR
                epitope_center_residues = []
                if has_epitope and epitope_start_idx >= 0:
                    # Get residue numbers corresponding to the epitope
                    res_offset = 1  # Adjust if sequence numbering and residue numbering differ
                    for i in range(len(matching_epitope)):
                        epitope_res_num = epitope_start_idx + i + res_offset
                        if 1 <= epitope_res_num <= peptide_pose.total_residue():
                            epitope_center_residues.append(epitope_res_num)
                    
                    print(f"Epitope centers at residues: {epitope_center_residues}")
                    
                    # Calculate center of epitope residues
                    if epitope_center_residues:
                        epitope_center_x, epitope_center_y, epitope_center_z = 0.0, 0.0, 0.0
                        epitope_count = 0
                        
                        for res_id in epitope_center_residues:
                            try:
                                ca_xyz = peptide_pose.residue(res_id).xyz("CA")
                                epitope_center_x += ca_xyz[0]
                                epitope_center_y += ca_xyz[1]
                                epitope_center_z += ca_xyz[2]
                                epitope_count += 1
                            except Exception as e:
                                print(f"Warning: Could not get coordinates for epitope residue {res_id}: {e}")
                        
                        if epitope_count > 0:
                            epitope_center_x /= epitope_count
                            epitope_center_y /= epitope_count
                            epitope_center_z /= epitope_count
                            
                            # Use epitope center instead of peptide COM
                            pep_com_x = epitope_center_x
                            pep_com_y = epitope_center_y
                            pep_com_z = epitope_center_z
                            pep_atom_count = 1  # Set to 1 to indicate we have a valid center
                            
                            print(f"Using epitope center: ({pep_com_x:.2f}, {pep_com_y:.2f}, {pep_com_z:.2f})")
                
                # Fallback to peptide COM if no epitope alignment
                if pep_atom_count == 0:
                    for line in pep_lines:
                        if line.startswith('ATOM'):
                            try:
                                x_str = line[30:38]
                                y_str = line[38:46]
                                z_str = line[46:54]
                                
                                pep_com_x += float(x_str)
                                pep_com_y += float(y_str)
                                pep_com_z += float(z_str)
                                pep_atom_count += 1
                            except Exception as e:
                                print(f"Warning: Could not parse atom coordinates: {e}")
                                continue
                    
                    if pep_atom_count > 0:
                        pep_com_x /= pep_atom_count
                        pep_com_y /= pep_atom_count
                        pep_com_z /= pep_atom_count
                        print(f"Using peptide COM: ({pep_com_x:.2f}, {pep_com_y:.2f}, {pep_com_z:.2f})")
                
                # Position peptide near CDR (with initial offset)
                offset = 20.0  # Å
                x_pos = cdr_center_x + offset
                y_pos = cdr_center_y
                z_pos = cdr_center_z
                
                # Write the combined PDB
                with open(combined_pdb, 'w') as out_file:
                    # Write antibody as chain A
                    for line in ab_lines:
                        if line.startswith('ATOM'):
                            new_line = line[:21] + 'A' + line[22:]
                            out_file.write(new_line)
                        elif not line.startswith('END'):
                            out_file.write(line)
                    
                    out_file.write("TER\n")
                    
                    # Add peptide as chain B with translation to position
                    for line in pep_lines:
                        if line.startswith('ATOM'):
                            try:
                                x_str = line[30:38]
                                y_str = line[38:46]
                                z_str = line[46:54]
                                
                                x_val = float(x_str)
                                y_val = float(y_str)
                                z_val = float(z_str)
                                
                                # Translate relative to peptide COM or epitope center
                                x_offset = x_val - pep_com_x + x_pos
                                y_offset = y_val - pep_com_y + y_pos
                                z_offset = z_val - pep_com_z + z_pos
                                
                                x_new_str = f"{x_offset:8.3f}"
                                y_new_str = f"{y_offset:8.3f}"
                                z_new_str = f"{z_offset:8.3f}"
                                
                                # Update coordinates and chain ID
                                new_line = line[:21] + 'B' + line[22:30] + x_new_str + y_new_str + z_new_str + line[54:]
                                out_file.write(new_line)
                            except Exception as e:
                                print(f"Warning: Error translating atom: {e}")
                                new_line = line[:21] + 'B' + line[22:]
                                out_file.write(new_line)
                        elif not line.startswith('END'):
                            out_file.write(line)
                            
                    out_file.write("END\n")
            except Exception as e:
                print(f"Error creating combined PDB: {e}")
                continue
            
            # Load the combined pose
            try:
                complex_pose = pyrosetta.pose_from_pdb(combined_pdb)
                
                # Initial score
                initial_score = scorefxn(complex_pose)
                print(f"Initial score: {initial_score:.2f}")
                
                # Perform rigid body docking
                rigid_body_mover = RigidBodyTransMover(complex_pose, 2)  # 2 = chain B (peptide)
                
                # Try different orientations to find the best one
                best_local_score = initial_score
                best_local_pose = complex_pose.clone()
                
                # Define translation vectors
                directions = [
                    (1, 0, 0),    # +x
                    (-1, 0, 0),   # -x
                    (0, 1, 0),    # +y
                    (0, -1, 0),   # -y
                    (0, 0, 1),    # +z
                    (0, 0, -1),   # -z
                    (1, 1, 0),    # +x+y
                    (1, 0, 1),    # +x+z
                    (0, 1, 1),    # +y+z
                    (1, 1, 1)     # +x+y+z
                ]
                
                # If we have an epitope, do more extensive sampling to find optimal orientation
                sampling_steps = 5 if has_epitope else 3
                
                # Try different distances to approach the CDR
                for step in range(1, sampling_steps + 1):
                    for dx, dy, dz in directions:
                        try:
                            test_pose = complex_pose.clone()
                            
                            # Move peptide closer to CDR
                            trans_vec = xyzVector_double_t(dx * -5, dy * -5, dz * -5)  # Negative to move toward antibody
                            rigid_body_mover.trans_axis(trans_vec)
                            rigid_body_mover.apply(test_pose)
                            
                            # Score the new position
                            current_score = scorefxn(test_pose)
                            
                            if current_score < best_local_score:
                                best_local_score = current_score
                                best_local_pose = test_pose.clone()
                                complex_pose = best_local_pose.clone()
                                print(f"  Found better pose: {best_local_score:.2f}")
                        except Exception as e:
                            print(f"  Error in rigid body movement: {e}")
                
                # Save the best pose before minimization
                best_local_pose.dump_pdb(f"{output_prefix}_{current_cdr}_before_relax.pdb")
                
                # Check if binding to CDR before relaxation (quick check)
                pre_relax_binding = check_cdr_binding(best_local_pose, cdr_regions, current_cdr)
                
                # If not binding to this CDR, skip relaxation and try the next one
                if not pre_relax_binding:
                    print(f"Peptide does not bind to {current_cdr} before relaxation, trying next CDR")
                    continue
                
                # Now relax the complex to refine the binding
                print(f"Peptide binds to {current_cdr}, performing energy relaxation...")
                
                try:
                    # Create a MoveMap directly
                    movemap = MoveMap()
                    
                    # Mark all peptide residues as movable
                    peptide_start = best_local_pose.chain_begin(2)  # Chain B start
                    peptide_end = best_local_pose.chain_end(2)      # Chain B end
                    
                    for i in range(peptide_start, peptide_end + 1):
                        movemap.set_bb(i, True)
                        movemap.set_chi(i, True)
                    
                    # Mark target CDR residues as movable
                    for res_id in cdr_regions[current_cdr]:
                        if 1 <= res_id <= best_local_pose.total_residue():  # Ensure index is valid
                            movemap.set_bb(res_id, True)
                            movemap.set_chi(res_id, True)
                    
                    # Create energy log file
                    energy_log_file = f"{output_prefix}_{current_cdr}_relax_energy.log"
                    
                    # Perform relaxation with energy tracking
                    final_score = relax_with_energy_tracking(
                        best_local_pose, movemap, scorefxn, energy_log_file, max_cycles=3, min_iterations=5)
                    
                    print(f"Score after relaxation: {final_score:.2f}")
                    
                    # Save the relaxed complex
                    best_local_pose.dump_pdb(f"{output_prefix}_{current_cdr}_relaxed_complex.pdb")
                    
                    # Check if binding to CDR after relaxation
                    is_binding_to_cdr = check_cdr_binding(best_local_pose, cdr_regions, current_cdr)
                    
                    if is_binding_to_cdr:
                        print(f"Peptide binds to {current_cdr} after relaxation")
                        
                        # Save this as our best result if it has the best score so far
                        if final_score < best_score:
                            best_score = final_score
                            best_binding_cdr = current_cdr
                            
                            # Save result details
                            best_result = {
                                "binding_cdr": current_cdr,
                                "initial_score": best_local_score,
                                "relaxed_score": final_score,
                                "binding_to_cdr": is_binding_to_cdr,
                                "contains_epitope": has_epitope,
                                "matching_epitope": matching_epitope if has_epitope else None,
                                "output_pdb": f"{output_prefix}_{current_cdr}_relaxed_complex.pdb",
                                "energy_log": energy_log_file
                            }
                            
                            # If this is the first CDR in our priority list that binds, we could stop
                            # here to save time, but we'll continue to find the best binding CDR
                    else:
                        print(f"Peptide no longer binds to {current_cdr} after relaxation, trying next CDR")
                
                except Exception as e:
                    print(f"Error in relaxation: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # If there was an error in relaxation but the pre-relaxation binding was good
                    # consider this as a fallback solution
                    if pre_relax_binding and best_local_score < best_score:
                        best_score = best_local_score
                        best_binding_cdr = current_cdr
                        best_result = {
                            "binding_cdr": current_cdr,
                            "initial_score": best_local_score,
                            "relaxed_score": None,
                            "binding_to_cdr": pre_relax_binding,
                            "contains_epitope": has_epitope,
                            "matching_epitope": matching_epitope if has_epitope else None,
                            "output_pdb": f"{output_prefix}_{current_cdr}_before_relax.pdb"
                        }
                
            except Exception as e:
                print(f"Error loading combined PDB: {e}")
                continue
                
            # Clean up intermediate files
            try:
                os.remove(combined_pdb)
            except:
                pass
        
        # After trying all CDRs, return the best result
        if best_result:
            print(f"\nBest binding found with {best_binding_cdr}, score: {best_score:.2f}")
            
            # Also save a copy of the best result with a simplified name for easy access
            source_pdb = best_result["output_pdb"]
            dest_pdb = f"{output_prefix}_best_complex.pdb"
            try:
                # Copy the file
                with open(source_pdb, 'r') as src:
                    with open(dest_pdb, 'w') as dst:
                        dst.write(src.read())
                # Update the output PDB path to include both
                best_result["best_complex_pdb"] = dest_pdb
            except Exception as e:
                print(f"Warning: Could not create best complex copy: {e}")
            
            return best_result
        else:
            print("No binding found to any CDR region")
            return None
            
    except Exception as e:
        print(f"ERROR in docking protocol: {e}")
        return None

def relax_with_energy_tracking(pose, movemap, scorefxn, energy_log_file, max_cycles=3, min_iterations=5):
    """
    Perform relaxation with energy tracking for more convergence
    
    Parameters:
        pose: PyRosetta pose object to relax
        movemap: MoveMap defining which DOFs can move
        scorefxn: Score function to use
        energy_log_file: Path to output energy log file
        max_cycles: Maximum number of relax cycles
        min_iterations: Number of minimization iterations per cycle
    """
    # Create a copy of the pose to work with
    working_pose = pose.clone()
    
    # Open energy log file
    with open(energy_log_file, 'w') as f:
        f.write("Step\tEnergy\tEnergy_Change\n")
        
        # Log initial energy
        initial_energy = scorefxn(working_pose)
        print(f"Initial energy: {initial_energy:.2f}")
        f.write(f"0\t{initial_energy:.6f}\t-\n")
        
        step_counter = 0
        
        # Create minimization mover
        min_mover = MinMover()
        min_mover.score_function(scorefxn)
        min_mover.min_type('lbfgs_armijo_nonmonotone')
        min_mover.movemap(movemap)
        min_mover.tolerance(0.00001)  # Set tighter tolerance
        
        # Manual implementation of FastRelax cycles
        for cycle in range(1, max_cycles + 1):
            print(f"\nRelaxation cycle {cycle} of {max_cycles}")
            
            # Create packer task for repacking
            task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
            task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
            packer_task = task_factory.create_task_and_apply_taskoperations(working_pose)
            
            # Repack side chains
            packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, packer_task)
            packer.apply(working_pose)
            
            # Log energy after repacking
            step_counter += 100  # Increment by 100 steps
            repack_energy = scorefxn(working_pose)
            print(f"After repacking (step {step_counter}): Energy = {repack_energy:.2f}")
            f.write(f"{step_counter}\t{repack_energy:.6f}\t-\n")
            
            # Minimization with convergence tracking
            prev_energy = repack_energy
            curr_energy = prev_energy
            
            # Perform multiple minimization iterations for better convergence
            for min_step in range(1, min_iterations + 1):
                # Apply minimization
                min_mover.apply(working_pose)
                
                # Calculate energy
                curr_energy = scorefxn(working_pose)
                
                # Calculate energy change
                energy_diff = abs(prev_energy - curr_energy)
                
                # Log energy
                step_counter += 100
                print(f"Minimization step {min_step} (global step {step_counter}): "
                      f"Energy = {curr_energy:.2f}, Change = {energy_diff:.6f}")
                f.write(f"{step_counter}\t{curr_energy:.6f}\t{energy_diff:.6f}\n")
                
                # Update previous energy
                prev_energy = curr_energy
                
                # If energy is not changing much, we're near convergence
                if energy_diff < 0.01:
                    print(f"Minimization converging (energy diff < 0.01)")
                    if min_step >= 3:  # At least 3 steps
                        break
            
            # Log energy at the end of cycle
            cycle_energy = scorefxn(working_pose)
            print(f"End of cycle {cycle}: Energy = {cycle_energy:.2f}")
            
            # If the overall energy change is very small, we can stop
            if cycle > 1 and abs(cycle_energy - repack_energy) < 0.1:
                print(f"Reached energy convergence after {cycle} cycles")
                break
        
        # Get final energy
        final_energy = scorefxn(working_pose)
        print(f"\nFinal energy after relaxation: {final_energy:.2f}")
        step_counter += 1
        f.write(f"{step_counter}\t{final_energy:.6f}\t-\n")
        
        # Copy the result back to the original pose
        pose.assign(working_pose)
        
        return final_energy

def check_cdr_binding(complex_pose, cdr_regions, binding_cdr):
    """
    Check if the peptide is actually binding to the specified CDR region
    """
    print(f"Checking contacts between peptide and {binding_cdr}...")
    
    peptide_start = complex_pose.chain_begin(2)  # Chain B start
    peptide_end = complex_pose.chain_end(2)      # Chain B end
    
    # Get CDR residues
    cdr_residues = cdr_regions.get(binding_cdr, [])
    if not cdr_residues:
        print(f"No residues defined for {binding_cdr}")
        return False
    
    # Count contacts
    contact_count = 0
    cutoff_distance = 5.0  # Å
    
    for i in range(peptide_start, peptide_end + 1):
        if not is_protein_residue_with_ca(complex_pose, i):
            continue
            
        for j in cdr_residues:
            if not is_protein_residue_with_ca(complex_pose, j):
                continue
                
            ## Calculate CA-CA distance
            ca_i = complex_pose.residue(i).xyz("CA")
            ca_j = complex_pose.residue(j).xyz("CA")
            
            dx = ca_i[0] - ca_j[0]
            dy = ca_i[1] - ca_j[1]
            dz = ca_i[2] - ca_j[2]
            
            dist = (dx*dx + dy*dy + dz*dz)**0.5
            
            if dist < cutoff_distance:
                contact_count += 1
                
    binding_percent = (contact_count / len(cdr_residues)) * 100 if cdr_residues else 0
    
    print(f"Found {contact_count} contacts with {binding_cdr} ({binding_percent:.1f}% of CDR residues)")
    
    # Consider binding if at least 30% of CDR residues have contacts
    is_binding = binding_percent >= 30.0
    
    if is_binding:
        print(f"Peptide is binding to {binding_cdr}")
    else:
        print(f"Peptide is NOT binding to {binding_cdr}")
        
    return is_binding

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python peptide_antibody_docking.py <peptide_directory>")
        sys.exit(1)
    
    peptide_directory = sys.argv[1]
    if not os.path.isdir(peptide_directory):
        print(f"Error: {peptide_directory} is not a valid directory")
        sys.exit(1)
    
    # Define paths to your antibody structures
    antibody_pdbs = {
        "Ab0": "AB0_Antibody.pdb",
        "Ab1": "AB1_Antibody.pdb"
    }
    
    # Create output directory
    output_dir = "cyclic_peptides_docking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load structures
    antibodies, peptides = load_structures(antibody_pdbs, peptide_directory)
    
    # Extract peptide sequences
    peptide_sequences = {}
    for name, pose in peptides.items():
        try:
            peptide_sequences[name] = extract_peptide_sequence(pose)
            print(f"Extracted sequence for {name}: {peptide_sequences[name]}")
        except Exception as e:
            print(f"Error extracting sequence for {name}: {e}")
    
    # Process each antibody and peptide combination
    results = {}
    
    for peptide_name, peptide_pose in peptides.items():
        results[peptide_name] = {}
        print(f"\n===== Processing peptide: {peptide_name} =====")
        
        for ab_name, ab_pose in antibodies.items():
            print(f"\n----- Processing antibody {ab_name} with peptide {peptide_name} -----")
            
            # Identify CDR regions using the standard method
            cdr_regions = identify_cdr_regions(ab_pose)
            
            # Also try to identify CDR3 regions from the TCR sequences provided
            cdr3_regions = identify_cdr3_from_sequence(ab_pose, tcr_data)
            
            # If we found CDR3 regions from sequence, add them to our CDR regions dict
            if cdr3_regions:
                for cdr3_name, residues in cdr3_regions.items():
                    # Use a custom name to avoid overwriting standard CDRs
                    custom_name = f"CDR3-{cdr3_name}"
                    cdr_regions[custom_name] = residues
            
            # Dock peptide to specific CDR regions using knowledge from TCR data
            output_prefix = f"{output_dir}/{peptide_name}_{ab_name}"
            docking_result = dock_peptide_to_specific_cdr(ab_pose, peptide_pose, cdr_regions, tcr_data, output_prefix)
            
            if docking_result:
                results[peptide_name][ab_name] = docking_result
                print(f"Docking complete. Results: {docking_result}")
            else:
                print(f"Docking failed for {peptide_name} and {ab_name}")
    
    # Generate a summary report
    with open(f"{output_dir}/docking_summary.txt", "w") as f:
        f.write("PEPTIDE-ANTIBODY DOCKING SUMMARY\n")
        f.write("===============================\n\n")
        
        for peptide_name, ab_results in results.items():
            f.write(f"Peptide: {peptide_name}\n")
            f.write(f"Sequence: {peptide_sequences.get(peptide_name, 'Unknown')}\n")
            f.write("-" * 40 + "\n")
            
            for ab_name, result in ab_results.items():
                f.write(f"  Antibody: {ab_name}\n")
                
                if result:
                    f.write(f"  Binding CDR: {result.get('binding_cdr', 'Unknown')}\n")
                    f.write(f"  Initial Score: {result.get('initial_score', 'N/A')}\n")
                    f.write(f"  Relaxed Score: {result.get('relaxed_score', 'N/A')}\n")
                    f.write(f"  Binding to CDR: {result.get('binding_to_cdr', 'Unknown')}\n")
                    f.write(f"  Contains Epitope: {result.get('contains_epitope', 'No')}\n")
                    
                    if result.get('contains_epitope'):
                        f.write(f"  Matching Epitope: {result.get('matching_epitope', 'None')}\n")
                        
                    f.write(f"  Output PDB: {result.get('output_pdb', 'None')}\n")
                    f.write(f"  Energy Log: {result.get('energy_log', 'None')}\n")
                else:
                    f.write("  No successful docking\n")
                    
                f.write("\n")
            
            f.write("\n")
    
    print(f"\nDocking complete! Check the '{output_dir}' directory for all outputs.")

if __name__ == "__main__":
    main()