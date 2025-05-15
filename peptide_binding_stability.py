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
# Additional imports for peptide stability
from pyrosetta.rosetta.core.scoring.dssp import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.protocols.antibody import *

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

def identify_interface_residues(pose, chain1_begin, chain1_end, chain2_begin, chain2_end, cutoff=8.0):
    """
    Identify residues at the interface between two chains based on C-alpha distance
    
    Parameters:
        pose: PyRosetta pose
        chain1_begin/end: Start and end residue numbers for first chain
        chain2_begin/end: Start and end residue numbers for second chain
        cutoff: Distance cutoff in Angstroms
        
    Returns:
        List of residue numbers from first chain that are at the interface
    """
    interface_residues = []
    
    for i in range(chain1_begin, chain1_end + 1):
        if not pose.residue(i).is_protein() or not pose.residue(i).has("CA"):
            continue
            
        ca_i = pose.residue(i).xyz("CA")
        
        # Check if this residue is near any residue in the second chain
        for j in range(chain2_begin, chain2_end + 1):
            if not pose.residue(j).is_protein() or not pose.residue(j).has("CA"):
                continue
                
            ca_j = pose.residue(j).xyz("CA")
            
            # Calculate distance
            dx = ca_i[0] - ca_j[0]
            dy = ca_i[1] - ca_j[1]
            dz = ca_i[2] - ca_j[2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5
            
            if dist <= cutoff:
                interface_residues.append(i)
                break
    
    return interface_residues

def calculate_interface_energy(pose, chain1, chain2, scorefxn):
    """
    Calculate interface energy between two chains
    
    Parameters:
        pose: PyRosetta pose
        chain1: Chain number for first chain (1 = A, 2 = B, etc.)
        chain2: Chain number for second chain
        scorefxn: Score function to use
        
    Returns:
        Interface energy (negative is better binding)
    """
    # Score the complex
    complex_energy = scorefxn(pose)
    
    # Create a copy for chain 1 alone
    chain1_pose = pose.clone()
    chain1_begin = pose.chain_begin(chain1)
    chain1_end = pose.chain_end(chain1)
    
    # Delete all chains except chain 1
    for i in range(pose.num_chains(), 0, -1):
        if i != chain1:
            begin = pose.chain_begin(i)
            end = pose.chain_end(i)
            chain1_pose.delete_residue_range_slow(begin, end)
    
    # Score chain 1
    chain1_energy = scorefxn(chain1_pose)
    
    # Create a copy for chain 2 alone
    chain2_pose = pose.clone()
    chain2_begin = pose.chain_begin(chain2)
    chain2_end = pose.chain_end(chain2)
    
    # Delete all chains except chain 2
    for i in range(pose.num_chains(), 0, -1):
        if i != chain2:
            begin = pose.chain_begin(i)
            end = pose.chain_end(i)
            chain2_pose.delete_residue_range_slow(begin, end)
    
    # Score chain 2
    chain2_energy = scorefxn(chain2_pose)
    
    # Calculate interface energy
    interface_energy = complex_energy - (chain1_energy + chain2_energy)
    
    return interface_energy

def relax_complex(complex_pose, cdr_regions=None, binding_cdr=None, scorefxn=None, output_prefix=None, num_cycles=25):
    """
    Perform relaxation of a peptide-antibody or peptide-MHC complex
    with focus on the interface regions
    
    Parameters:
        complex_pose: PyRosetta pose of the complex
        cdr_regions: Dictionary of CDR regions (optional)
        binding_cdr: Name of the binding CDR (optional)
        scorefxn: Score function to use (will create ref2015 if None)
        output_prefix: Path prefix for output files
        num_cycles: Number of relaxation cycles to perform
        
    Returns:
        Relaxed complex pose
    """
    print(f"\n=== Relaxing Peptide-Antibody Complex ({num_cycles} cycles) ===")
    
    # Clone the input pose to avoid modifying the original
    relaxed_pose = complex_pose.clone()
    
    # Use provided scorefxn or create a default one
    if scorefxn is None:
        scorefxn = pyrosetta.create_score_function("ref2015")
    
    # Calculate initial score
    initial_score = scorefxn(relaxed_pose)
    print(f"Initial complex energy: {initial_score:.2f}")
    
    # Determine chains
    n_chains = relaxed_pose.num_chains()
    if n_chains < 2:
        print("Warning: Complex has fewer than 2 chains, may not be a valid complex")
        return relaxed_pose
    
    # Assume antibody/MHC is chain A, peptide is chain B
    antibody_chain = 1  # Chain A
    peptide_chain = 2   # Chain B
    
    antibody_begin = relaxed_pose.chain_begin(antibody_chain)
    antibody_end = relaxed_pose.chain_end(antibody_chain)
    peptide_begin = relaxed_pose.chain_begin(peptide_chain)
    peptide_end = relaxed_pose.chain_end(peptide_chain)
    
    print(f"Antibody/MHC residues: {antibody_begin}-{antibody_end}")
    print(f"Peptide residues: {peptide_begin}-{peptide_end}")
    
    # Create a MoveMap that focuses on the interface
    movemap = MoveMap()
    
    # Include peptide residues
    for i in range(peptide_begin, peptide_end + 1):
        movemap.set_bb(i, True)
        movemap.set_chi(i, True)
    
    # Include CDR region if specified
    if cdr_regions and binding_cdr:
        cdr_residues = cdr_regions.get(binding_cdr, [])
        print(f"Including {binding_cdr} residues in movemap: {len(cdr_residues)} residues")
        for res_id in cdr_residues:
            if 1 <= res_id <= relaxed_pose.total_residue():
                movemap.set_bb(res_id, True)
                movemap.set_chi(res_id, True)
    else:
        # If no CDR specified, include antibody/MHC residues near the peptide
        print("No CDR specified, identifying interface residues based on distance...")
        interface_residues = identify_interface_residues(relaxed_pose, antibody_begin, antibody_end, 
                                                       peptide_begin, peptide_end, cutoff=8.0)
        print(f"Identified {len(interface_residues)} interface residues")
        for res_id in interface_residues:
            movemap.set_bb(res_id, True)
            movemap.set_chi(res_id, True)
    
    # Energy log file if output_prefix is provided
    energy_log_file = None
    if output_prefix:
        energy_log_file = f"{output_prefix}_complex_relax_energy.log"
        with open(energy_log_file, 'w') as f:
            # Simple header for energy tracking
            f.write("Cycle\tProtocol\tEnergy\tImprovement\n")
            f.write(f"0\tInitial\t{initial_score:.2f}\t0.00\n")
    
    # Keep track of best pose and score
    best_pose = relaxed_pose.clone()
    best_score = initial_score
    best_cycle = 0
    
    # Try using FastRelax first
    try:
        print("\nAttempting FastRelax protocol for complex...")
        
        # Create FastRelax mover
        fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax()
        fast_relax.set_scorefxn(scorefxn)
        fast_relax.max_iter(num_cycles)
        fast_relax.set_movemap(movemap)
        
        # Apply FastRelax to a clone of the pose
        fr_pose = relaxed_pose.clone()
        fast_relax.apply(fr_pose)
        
        # Check if this improved the energy
        fr_score = scorefxn(fr_pose)
        fr_improvement = initial_score - fr_score
        
        print(f"FastRelax result: energy = {fr_score:.2f}, improvement = {fr_improvement:.2f}")
        
        # Log to file
        if energy_log_file:
            with open(energy_log_file, 'a') as f:
                f.write(f"1\tFastRelax\t{fr_score:.2f}\t{fr_improvement:.2f}\n")
        
        # Update best pose if improved
        if fr_score < best_score:
            best_pose.assign(fr_pose)
            best_score = fr_score
            best_cycle = 1
            print("FastRelax produced the best structure so far")
        
    except Exception as e:
        print(f"FastRelax for complex failed: {e}")
        print("Will use manual multi-cycle protocol")
    
    # Now perform manual multiple cycles of repacking and minimization
    print(f"\nPerforming {num_cycles} cycles of manual complex optimization...")
    
    # Create task for repacking
    task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
    task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    packer_task = task_factory.create_task_and_apply_taskoperations(relaxed_pose)
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, packer_task)
    
    # Create minimization mover
    min_mover = MinMover()
    min_mover.score_function(scorefxn)
    min_mover.min_type('lbfgs_armijo_nonmonotone')
    min_mover.movemap(movemap)
    min_mover.tolerance(0.0001)  # Tighter tolerance for better convergence
    
    # Multiple relaxation cycles
    for cycle in range(1, num_cycles + 1):
        print(f"\nComplex relaxation cycle {cycle}/{num_cycles}")
        
        # Repack sidechains
        packer.apply(relaxed_pose)
        repack_score = scorefxn(relaxed_pose)
        repack_improvement = initial_score - repack_score
        
        # Log to file
        if energy_log_file:
            with open(energy_log_file, 'a') as f:
                f.write(f"{cycle}\tRepack\t{repack_score:.2f}\t{repack_improvement:.2f}\n")
        
        print(f"After repacking: energy = {repack_score:.2f}, improvement = {repack_improvement:.2f}")
        
        # Minimization
        min_mover.apply(relaxed_pose)
        min_score = scorefxn(relaxed_pose)
        min_improvement = initial_score - min_score
        
        # Log to file
        if energy_log_file:
            with open(energy_log_file, 'a') as f:
                f.write(f"{cycle}\tMinimize\t{min_score:.2f}\t{min_improvement:.2f}\n")
        
        print(f"After minimization: energy = {min_score:.2f}, improvement = {min_improvement:.2f}")
        
        # Update best pose if improved
        if min_score < best_score:
            best_pose.assign(relaxed_pose)
            best_score = min_score
            best_cycle = cycle + 1  # +1 to differentiate from FastRelax cycle
            print(f"Cycle {cycle} produced the best structure so far")
        
        # Check for convergence every 3 cycles
        if cycle % 3 == 0 and cycle > 3:
            if best_cycle < cycle - 3:
                print(f"No improvement in last 3 cycles (best was cycle {best_cycle})")
                print("Continuing relaxation in case we can escape local minimum...")
            else:
                print(f"Still seeing improvements (best was cycle {best_cycle})")
    
    # Final score calculation and report
    final_score = scorefxn(best_pose)
    total_improvement = initial_score - final_score
    
    print("\n=== Complex Relaxation Summary ===")
    print(f"Initial energy: {initial_score:.2f}")
    print(f"Final energy: {final_score:.2f}")
    print(f"Total energy improvement: {total_improvement:.2f} ({(total_improvement/abs(initial_score))*100 if initial_score != 0 else 0:.1f}%)")
    print(f"Best structure was from cycle {best_cycle}")
    
    # Analyze structure changes
    try:
        # Calculate RMSD between initial and relaxed structures
        ca_rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(complex_pose, best_pose)
        print(f"Structure change: CA RMSD = {ca_rmsd:.2f} Å")
        
        # Calculate interface score - energy of complex minus energy of separated parts
        interface_energy = calculate_interface_energy(best_pose, antibody_chain, peptide_chain, scorefxn)
        print(f"Interface energy: {interface_energy:.2f}")
    except Exception as e:
        print(f"Error analyzing structure changes: {e}")
    
    # Save the relaxed structure if output_prefix is provided
    if output_prefix:
        relaxed_pdb = f"{output_prefix}_relaxed_complex.pdb"
        best_pose.dump_pdb(relaxed_pdb)
        print(f"Saved relaxed complex structure to {relaxed_pdb}")
    
    return best_pose

def relax_peptide_structure(peptide_pose, scorefxn=None, output_prefix=None, verbose=True, num_cycles=25):
    """
    Perform a comprehensive relaxation of the peptide structure
    including sidechain repacking and backbone minimization,
    with detailed energy tracking
    
    Parameters:
        peptide_pose: PyRosetta pose to relax
        scorefxn: Score function to use (will create ref2015 if None)
        output_prefix: Path prefix for output files
        verbose: Whether to print detailed energy tracking
        num_cycles: Number of relaxation cycles to perform
        
    Returns:
        Relaxed pose
    """
    print(f"\n=== Relaxing Peptide Structure ({num_cycles} cycles) ===")
    
    # Clone the input pose to avoid modifying the original
    relaxed_pose = peptide_pose.clone()
    
    # Use provided scorefxn or create a default one
    if scorefxn is None:
        scorefxn = pyrosetta.create_score_function("ref2015")
    
    # Report initial score
    initial_score = scorefxn(relaxed_pose)
    print(f"Initial total energy: {initial_score:.2f}")
    
    # Create a movemap for backbone and sidechains
    movemap = MoveMap()
    for i in range(1, relaxed_pose.total_residue() + 1):
        movemap.set_bb(i, True)
        movemap.set_chi(i, True)
    
    # Energy log file if output_prefix is provided
    energy_log_file = None
    if output_prefix:
        energy_log_file = f"{output_prefix}_relax_energy.log"
        with open(energy_log_file, 'w') as f:
            # Simple header for energy tracking
            f.write("Cycle\tProtocol\tEnergy\tImprovement\n")
            f.write(f"0\tInitial\t{initial_score:.2f}\t0.00\n")
    
    # Keep track of best pose and score
    best_pose = relaxed_pose.clone()
    best_score = initial_score
    
    # Try comprehensive FastRelax first
    try:
        print("\nAttempting comprehensive FastRelax protocol...")
        
        # Create FastRelax mover with more cycles
        fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax()
        fast_relax.set_scorefxn(scorefxn)
        fast_relax.max_iter(num_cycles)  # Set to the requested number of cycles
        fast_relax.set_movemap(movemap)
        
        # Apply FastRelax to a clone of the pose
        fr_pose = relaxed_pose.clone()
        fast_relax.apply(fr_pose)
        
        # Check if this improved the energy
        fr_score = scorefxn(fr_pose)
        fr_improvement = initial_score - fr_score
        
        print(f"FastRelax result: energy = {fr_score:.2f}, improvement = {fr_improvement:.2f}")
        
        # Log to file
        if energy_log_file:
            with open(energy_log_file, 'a') as f:
                f.write(f"1\tFastRelax\t{fr_score:.2f}\t{fr_improvement:.2f}\n")
        
        # Update best pose if improved
        if fr_score < best_score:
            best_pose.assign(fr_pose)
            best_score = fr_score
            print("FastRelax produced the best structure so far")
        
    except Exception as e:
        print(f"Comprehensive FastRelax failed: {e}")
        print("Will use manual multi-cycle protocol")
    
    # Now perform manual multiple cycles of repacking and minimization
    print(f"\nPerforming {num_cycles} cycles of manual optimization...")
    
    # Create task for repacking
    task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
    task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    packer_task = task_factory.create_task_and_apply_taskoperations(relaxed_pose)
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, packer_task)
    
    # Create minimization mover
    min_mover = MinMover()
    min_mover.score_function(scorefxn)
    min_mover.min_type('lbfgs_armijo_nonmonotone')
    min_mover.movemap(movemap)
    min_mover.tolerance(0.0001)  # Tighter tolerance for better convergence
    
    # Track best cycle
    best_cycle = 0
    
    # Multiple relaxation cycles
    for cycle in range(1, num_cycles + 1):
        print(f"\nRelaxation cycle {cycle}/{num_cycles}")
        
        # Repack sidechains
        packer.apply(relaxed_pose)
        repack_score = scorefxn(relaxed_pose)
        repack_improvement = initial_score - repack_score
        
        # Log to file
        if energy_log_file:
            with open(energy_log_file, 'a') as f:
                f.write(f"{cycle}\tRepack\t{repack_score:.2f}\t{repack_improvement:.2f}\n")
        
        print(f"After repacking: energy = {repack_score:.2f}, improvement = {repack_improvement:.2f}")
        
        # Minimization
        min_mover.apply(relaxed_pose)
        min_score = scorefxn(relaxed_pose)
        min_improvement = initial_score - min_score
        
        # Log to file
        if energy_log_file:
            with open(energy_log_file, 'a') as f:
                f.write(f"{cycle}\tMinimize\t{min_score:.2f}\t{min_improvement:.2f}\n")
        
        print(f"After minimization: energy = {min_score:.2f}, improvement = {min_improvement:.2f}")
        
        # Update best pose if improved
        if min_score < best_score:
            best_pose.assign(relaxed_pose)
            best_score = min_score
            best_cycle = cycle
            print(f"Cycle {cycle} produced the best structure so far")
        
        # Check for convergence every 5 cycles
        if cycle % 5 == 0:
            if best_cycle < cycle - 5:
                print(f"No improvement in last 5 cycles (best was cycle {best_cycle})")
                print("Continuing relaxation in case we can escape local minimum...")
            else:
                print(f"Still seeing improvements (best was cycle {best_cycle})")
    
    # Final score calculation and report
    final_score = scorefxn(best_pose)
    total_improvement = initial_score - final_score
    
    print("\n=== Relaxation Summary ===")
    print(f"Initial energy: {initial_score:.2f}")
    print(f"Final energy: {final_score:.2f}")
    print(f"Total energy improvement: {total_improvement:.2f} ({(total_improvement/abs(initial_score))*100 if initial_score != 0 else 0:.1f}%)")
    print(f"Best structure was from cycle {best_cycle}")
    
    # Analyze structure changes
    try:
        # Calculate RMSD between initial and relaxed structures
        ca_rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(peptide_pose, best_pose)
        all_atom_rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(peptide_pose, best_pose)
        
        print(f"Structure change: CA RMSD = {ca_rmsd:.2f} Å, All-atom RMSD = {all_atom_rmsd:.2f} Å")
        
        # Check for secondary structure changes
        dssp_before = pyrosetta.rosetta.core.scoring.dssp.Dssp(peptide_pose)
        dssp_before.insert_ss_into_pose(peptide_pose)
        ss_before = peptide_pose.secstruct()
        
        dssp_after = pyrosetta.rosetta.core.scoring.dssp.Dssp(best_pose)
        dssp_after.insert_ss_into_pose(best_pose)
        ss_after = best_pose.secstruct()
        
        print(f"Secondary structure before: {ss_before}")
        print(f"Secondary structure after:  {ss_after}")
        
        if ss_before != ss_after:
            print("Relaxation revealed new secondary structure elements!")
            
            # Count secondary structure elements
            before_h = ss_before.count('H')
            before_e = ss_before.count('E')
            before_l = ss_before.count('L')
            
            after_h = ss_after.count('H')
            after_e = ss_after.count('E')
            after_l = ss_after.count('L')
            
            print(f"Before: {before_h} helix, {before_e} sheet, {before_l} loop")
            print(f"After:  {after_h} helix, {after_e} sheet, {after_l} loop")
    except Exception as e:
        print(f"Error analyzing structure changes: {e}")
    
    # Save the relaxed structure if output_prefix is provided
    if output_prefix:
        relaxed_pdb = f"{output_prefix}_relaxed.pdb"
        best_pose.dump_pdb(relaxed_pdb)
        print(f"Saved relaxed structure to {relaxed_pdb}")
    
    return best_pose

def check_cdr_binding(complex_pose, cdr_regions, binding_cdr):
    """
    Check if the peptide is actually binding to the specified CDR region and analyze binding stability
    """
    print(f"Checking contacts between peptide and {binding_cdr}...")
    
    peptide_start = complex_pose.chain_begin(2)  # Chain B start
    peptide_end = complex_pose.chain_end(2)      # Chain B end
    
    # Get CDR residues
    cdr_residues = cdr_regions.get(binding_cdr, [])
    if not cdr_residues:
        print(f"No residues defined for {binding_cdr}")
        return {"is_binding": False}
    
    # Track different types of contacts
    contacts = {
        "total": 0,           # All contacts within cutoff distance
        "hydrogen_bonds": 0,  # Hydrogen bonds 
        "salt_bridges": 0,    # Salt bridges (charged interactions)
        "hydrophobic": 0      # Hydrophobic interactions
    }
    
    # Track specific interacting residue pairs
    interface_pairs = []
    
    # Default cutoff distances (Å)
    cutoff_distance = 5.0       # General contact cutoff
    hbond_cutoff = 3.5          # Hydrogen bond cutoff
    salt_bridge_cutoff = 4.0    # Salt bridge cutoff
    hydrophobic_cutoff = 4.5    # Hydrophobic interaction cutoff
    
    # Define charged and hydrophobic residues
    pos_charged = ["ARG", "LYS", "HIS"]
    neg_charged = ["ASP", "GLU"]
    hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"]
    
    # Check for contacts
    for i in range(peptide_start, peptide_end + 1):
        if not is_protein_residue_with_ca(complex_pose, i):
            continue
            
        peptide_res_name = complex_pose.residue(i).name()
        peptide_res_name3 = peptide_res_name[:3]  # 3-letter code
        
        for j in cdr_residues:
            if not is_protein_residue_with_ca(complex_pose, j):
                continue
            
            ab_res_name = complex_pose.residue(j).name()
            ab_res_name3 = ab_res_name[:3]  # 3-letter code
            
            # Calculate CA-CA distance
            ca_i = complex_pose.residue(i).xyz("CA")
            ca_j = complex_pose.residue(j).xyz("CA")
            
            dx = ca_i[0] - ca_j[0]
            dy = ca_i[1] - ca_j[1]
            dz = ca_i[2] - ca_j[2]
            
            dist = (dx*dx + dy*dy + dz*dz)**0.5
            
            # Check for general contact
            if dist < cutoff_distance:
                contacts["total"] += 1
                
                # Record the interacting pair
                peptide_aa = complex_pose.residue(i).name1()
                ab_aa = complex_pose.residue(j).name1()
                
                interface_pair = {
                    "peptide_res": i,
                    "peptide_aa": peptide_aa,
                    "antibody_res": j,
                    "antibody_aa": ab_aa,
                    "distance": dist
                }
                interface_pairs.append(interface_pair)
                
                # Check for potential hydrogen bonds (simplified)
                # In production code, you'd use Rosetta's HBond detection
                if dist < hbond_cutoff:
                    donor_acceptor_pairs = [
                        # Check for backbone-backbone H-bonds
                        ("N", "O"),
                        ("O", "N"),
                        # Check common sidechain donors and acceptors
                        ("OH", "O"),
                        ("OH", "OD1"),
                        ("OH", "OD2"),
                        ("OH", "OE1"),
                        ("OH", "OE2"),
                        ("NE", "O"),
                        ("NH1", "O"),
                        ("NH2", "O"),
                        ("ND1", "O"),
                        ("ND2", "O"),
                        ("NE2", "O"),
                        ("NZ", "O")
                    ]
                    
                    # Check atom-atom distances for potential H-bonds
                    for donor_name, acceptor_name in donor_acceptor_pairs:
                        try:
                            # Check if peptide residue has donor and antibody has acceptor
                            if (complex_pose.residue(i).has(donor_name) and 
                                complex_pose.residue(j).has(acceptor_name)):
                                donor = complex_pose.residue(i).xyz(donor_name)
                                acceptor = complex_pose.residue(j).xyz(acceptor_name)
                                
                                d_dx = donor[0] - acceptor[0]
                                d_dy = donor[1] - acceptor[1]
                                d_dz = donor[2] - acceptor[2]
                                
                                d_dist = (d_dx*d_dx + d_dy*d_dy + d_dz*d_dz)**0.5
                                
                                if d_dist < hbond_cutoff:
                                    contacts["hydrogen_bonds"] += 1
                                    interface_pair["interaction"] = "hydrogen_bond"
                                    break
                            
                            # Check if antibody residue has donor and peptide has acceptor
                            if (complex_pose.residue(j).has(donor_name) and 
                                complex_pose.residue(i).has(acceptor_name)):
                                donor = complex_pose.residue(j).xyz(donor_name)
                                acceptor = complex_pose.residue(i).xyz(acceptor_name)
                                
                                d_dx = donor[0] - acceptor[0]
                                d_dy = donor[1] - acceptor[1]
                                d_dz = donor[2] - acceptor[2]
                                
                                d_dist = (d_dx*d_dx + d_dy*d_dy + d_dz*d_dz)**0.5
                                
                                if d_dist < hbond_cutoff:
                                    contacts["hydrogen_bonds"] += 1
                                    interface_pair["interaction"] = "hydrogen_bond"
                                    break
                        except Exception:
                            # Skip if atoms don't exist
                            pass
                
                # Check for salt bridges (simplified)
                if ((peptide_res_name3 in pos_charged and ab_res_name3 in neg_charged) or
                   (peptide_res_name3 in neg_charged and ab_res_name3 in pos_charged)):
                    if dist < salt_bridge_cutoff:
                        contacts["salt_bridges"] += 1
                        interface_pair["interaction"] = "salt_bridge"
                
                # Check for hydrophobic interactions
                if (peptide_res_name3 in hydrophobic and ab_res_name3 in hydrophobic):
                    if dist < hydrophobic_cutoff:
                        contacts["hydrophobic"] += 1
                        interface_pair["interaction"] = "hydrophobic"
                
    # Calculate binding metrics
    binding_percent = (contacts["total"] / len(cdr_residues)) * 100 if cdr_residues else 0
    
    print(f"Found {contacts['total']} contacts with {binding_cdr} ({binding_percent:.1f}% of CDR residues)")
    print(f"  Hydrogen bonds: {contacts['hydrogen_bonds']}")
    print(f"  Salt bridges: {contacts['salt_bridges']}")
    print(f"  Hydrophobic interactions: {contacts['hydrophobic']}")
    
    # Consider binding if at least 30% of CDR residues have contacts
    is_binding = binding_percent >= 30.0
    
    # Estimate binding strength based on contact types
    binding_strength = "Unknown"
    if is_binding:
        # Calculate a simple binding score
        binding_score = (
            contacts["hydrogen_bonds"] * 2.5 +  # H-bonds contribute most to specificity
            contacts["salt_bridges"] * 3.0 +    # Salt bridges are strong but can be affected by solvation
            contacts["hydrophobic"] * 1.5       # Hydrophobic interactions contribute to affinity
        )
        
        # Normalize by the number of residues involved
        normalized_score = binding_score / max(1, len(cdr_residues))
        
        if normalized_score > 1.0:
            binding_strength = "Strong"
        elif normalized_score > 0.5:
            binding_strength = "Moderate"
        else:
            binding_strength = "Weak"
            
        print(f"Peptide is binding to {binding_cdr} with {binding_strength} estimated binding strength")
    else:
        print(f"Peptide is NOT binding to {binding_cdr}")
    
    # Return more detailed binding information
    binding_info = {
        "is_binding": is_binding,
        "contact_count": contacts["total"],
        "binding_percent": binding_percent,
        "hydrogen_bonds": contacts["hydrogen_bonds"],
        "salt_bridges": contacts["salt_bridges"],
        "hydrophobic": contacts["hydrophobic"],
        "binding_strength": binding_strength,
        "interface_pairs": interface_pairs
    }
        
    return binding_info

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
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Save individual poses for reference
        antibody_pdb = f"{output_prefix}_antibody.pdb"
        peptide_pdb = f"{output_prefix}_peptide.pdb"
        
        print(f"Saving antibody to {antibody_pdb}")
        antibody_pose.dump_pdb(antibody_pdb)
        
        print(f"Saving peptide to {peptide_pdb}")
        peptide_pose.dump_pdb(peptide_pdb)
        
        # Verify that the PDB files were created
        if not os.path.exists(antibody_pdb):
            print(f"ERROR: Failed to create antibody PDB file {antibody_pdb}")
            return None
            
        if not os.path.exists(peptide_pdb):
            print(f"ERROR: Failed to create peptide PDB file {peptide_pdb}")
            return None
        
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
            
            # ======================================================
            # Enhanced combined PDB file creation
            # ======================================================
            try:
                print(f"Creating combined PDB file: {combined_pdb}")
                
                # Read the antibody PDB file
                with open(antibody_pdb, 'r') as ab_file:
                    ab_lines = ab_file.readlines()
                
                # Read the peptide PDB file
                with open(peptide_pdb, 'r') as pep_file:
                    pep_lines = pep_file.readlines()
                
                # Get peptide center of mass or epitope center if matched
                pep_com_x, pep_com_y, pep_com_z = 0.0, 0.0, 0.0
                pep_atom_count = 0
                
                # Calculate peptide center of mass from PDB file
                atom_positions = []
                for line in pep_lines:
                    if line.startswith('ATOM'):
                        try:
                            x_str = line[30:38].strip()
                            y_str = line[38:46].strip()
                            z_str = line[46:54].strip()
                            
                            if x_str and y_str and z_str:  # Ensure values exist
                                x_val = float(x_str)
                                y_val = float(y_str)
                                z_val = float(z_str)
                                
                                pep_com_x += x_val
                                pep_com_y += y_val
                                pep_com_z += z_val
                                pep_atom_count += 1
                                
                                atom_positions.append((x_val, y_val, z_val))
                        except Exception as e:
                            print(f"Warning: Could not parse atom coordinates: {e} in line: {line}")
                
                if pep_atom_count == 0:
                    print("ERROR: Could not calculate peptide center of mass - no valid atom coordinates found")
                    continue
                
                # Calculate center of mass
                pep_com_x /= pep_atom_count
                pep_com_y /= pep_atom_count
                pep_com_z /= pep_atom_count
                print(f"Peptide center of mass: ({pep_com_x:.2f}, {pep_com_y:.2f}, {pep_com_z:.2f})")
                
                # Position peptide near CDR (with initial offset)
                offset = 20.0  # Å
                x_pos = cdr_center_x + offset
                y_pos = cdr_center_y
                z_pos = cdr_center_z
                
                print(f"Positioning peptide at: ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})")
                
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
                                x_str = line[30:38].strip()
                                y_str = line[38:46].strip()
                                z_str = line[46:54].strip()
                                
                                if x_str and y_str and z_str:  # Ensure values exist
                                    x_val = float(x_str)
                                    y_val = float(y_str)
                                    z_val = float(z_str)
                                    
                                    # Translate relative to peptide COM
                                    x_offset = x_val - pep_com_x + x_pos
                                    y_offset = y_val - pep_com_y + y_pos
                                    z_offset = z_val - pep_com_z + z_pos
                                    
                                    x_new_str = f"{x_offset:8.3f}"
                                    y_new_str = f"{y_offset:8.3f}"
                                    z_new_str = f"{z_offset:8.3f}"
                                    
                                    # Update coordinates and chain ID
                                    new_line = line[:21] + 'B' + line[22:30] + x_new_str + y_new_str + z_new_str + line[54:]
                                    out_file.write(new_line)
                                else:
                                    print(f"Warning: Invalid coordinate in line: {line}")
                            except Exception as e:
                                print(f"Warning: Error translating atom: {e} in line: {line}")
                                # Still write the line with chain B but don't modify coordinates
                                new_line = line[:21] + 'B' + line[22:]
                                out_file.write(new_line)
                        elif not line.startswith('END'):
                            out_file.write(line)
                            
                    out_file.write("END\n")
                
                # Verify that the combined PDB file was created
                if not os.path.exists(combined_pdb):
                    print(f"ERROR: Failed to create combined PDB file {combined_pdb}")
                    continue
                
                # Check file size to make sure it's not empty
                if os.path.getsize(combined_pdb) < 100:  # Very small file size suggests an error
                    print(f"WARNING: Combined PDB file {combined_pdb} is suspiciously small")
                
                print(f"Successfully created combined PDB file: {combined_pdb}")
                
            except Exception as e:
                print(f"Error creating combined PDB: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Load the combined pose
            try:
                print(f"Loading combined pose from {combined_pdb}")
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
                
                # Check if binding to CDR before relaxation (detailed check)
                binding_info = check_cdr_binding(best_local_pose, cdr_regions, current_cdr)
                
                # If not binding to this CDR, skip relaxation and try the next one
                if not binding_info["is_binding"]:
                    print(f"Peptide does not bind to {current_cdr} before relaxation, trying next CDR")
                    continue
                
                # Now relax the complex to refine the binding
                print(f"Peptide binds to {current_cdr}, performing energy relaxation...")
                
                try:
                    # Perform comprehensive complex relaxation
                    relaxed_complex = relax_complex(
                        best_local_pose, 
                        cdr_regions=cdr_regions, 
                        binding_cdr=current_cdr,
                        scorefxn=scorefxn, 
                        output_prefix=f"{output_prefix}_{current_cdr}",
                        num_cycles=25  # Perform 25 cycles of relaxation
                    )
                    
                    # Update best_local_pose with the relaxed complex
                    best_local_pose.assign(relaxed_complex)
                    
                    # Calculate final score
                    final_score = scorefxn(best_local_pose)
                    print(f"Score after complex relaxation: {final_score:.2f}")
                    
                    # Save the relaxed complex
                    best_local_pose.dump_pdb(f"{output_prefix}_{current_cdr}_relaxed_complex.pdb")
                    
                    # Check if binding to CDR after relaxation with detailed binding info
                    binding_info = check_cdr_binding(best_local_pose, cdr_regions, current_cdr)
                    
                    if binding_info["is_binding"]:
                        print(f"Peptide binds to {current_cdr} after relaxation")
                        
                        # Calculate interface energy
                        interface_energy = calculate_interface_energy(best_local_pose, 1, 2, scorefxn)
                        print(f"Interface energy: {interface_energy:.2f}")
                        
                        # Save this as our best result if it has the best score so far
                        if final_score < best_score:
                            best_score = final_score
                            best_binding_cdr = current_cdr
                            
                            # Save result details including binding info
                            best_result = {
                                "binding_cdr": current_cdr,
                                "initial_score": best_local_score,
                                "relaxed_score": final_score,
                                "interface_energy": interface_energy,
                                "binding_to_cdr": binding_info["is_binding"],
                                "binding_strength": binding_info["binding_strength"],
                                "hydrogen_bonds": binding_info["hydrogen_bonds"],
                                "salt_bridges": binding_info["salt_bridges"],
                                "hydrophobic": binding_info["hydrophobic"],
                                "contains_epitope": has_epitope,
                                "matching_epitope": matching_epitope if has_epitope else None,
                                "output_pdb": f"{output_prefix}_{current_cdr}_relaxed_complex.pdb"
                            }
                    else:
                        print(f"Peptide no longer binds to {current_cdr} after relaxation, trying next CDR")
                
                except Exception as e:
                    print(f"Error in relaxation: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # If there was an error in relaxation but the pre-relaxation binding was good
                    # consider this as a fallback solution
                    if binding_info["is_binding"] and best_local_score < best_score:
                        best_score = best_local_score
                        best_binding_cdr = current_cdr
                        best_result = {
                            "binding_cdr": current_cdr,
                            "initial_score": best_local_score,
                            "relaxed_score": None,
                            "binding_to_cdr": binding_info["is_binding"],
                            "binding_strength": binding_info["binding_strength"],
                            "hydrogen_bonds": binding_info["hydrogen_bonds"],
                            "salt_bridges": binding_info["salt_bridges"],
                            "hydrophobic": binding_info["hydrophobic"],
                            "contains_epitope": has_epitope,
                            "matching_epitope": matching_epitope if has_epitope else None,
                            "output_pdb": f"{output_prefix}_{current_cdr}_before_relax.pdb"
                        }
                
            except Exception as e:
                print(f"Error loading combined PDB: {e}")
                import traceback
                traceback.print_exc()
                continue
                
            # Clean up intermediate files if successful
            # Commented out to preserve files for debugging
            # try:
            #    os.remove(combined_pdb)
            # except:
            #    pass
        
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
        import traceback
        traceback.print_exc()
        return None

class PeptideStabilityAnalyzer:
    """
    Class for analyzing peptide stability using various metrics
    """
    def __init__(self, scorefxn=None):
        """Initialize with a score function"""
        self.scorefxn = scorefxn if scorefxn else pyrosetta.create_score_function("ref2015")
    
    def compute_stability_metrics(self, peptide_pose, output_prefix):
        """
        Compute multiple stability metrics for a peptide
        
        Parameters:
            peptide_pose: PyRosetta pose of the peptide
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary of stability metrics
        """
        print("\n=== Computing Peptide Stability Metrics ===")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        
        # Initialize results dictionary
        stability_metrics = {}
        
        # Corrected per-residue energy calculation in compute_stability_metrics:

        # 1. Overall energy score
        try:
            overall_energy = self.scorefxn(peptide_pose)
            stability_metrics["overall_energy"] = overall_energy
            print(f"Overall energy score: {overall_energy:.2f}")
            
            # Get per-residue energies using a simpler approach
            residue_energies = []
            
            # Use a direct approach to get residue-level energies from the energies object
            energies = peptide_pose.energies().residue_total_energies_array()
            
            # Extract per-residue energies
            for i in range(1, peptide_pose.total_residue() + 1):
                # Skip non-protein residues
                if not peptide_pose.residue(i).is_protein():
                    continue
                
                # Get residue type
                aa_type = peptide_pose.residue(i).aa()
                aa = pyrosetta.rosetta.core.chemical.name_from_aa(aa_type)
                
                try:
                    # Get the energy for this residue
                    res_energy = peptide_pose.energies().residue_total_energy(i)
                    residue_energies.append((i, res_energy, aa))
                except Exception as e:
                    print(f"Could not get energy for residue {i}: {e}")
                    # Use a default energy if calculation fails
                    residue_energies.append((i, 0.0, aa))
            
            # Sort residues by energy (highest/worst first)
            residue_energies.sort(key=lambda x: x[1], reverse=True)
            
            # If we have too few residues for statistics, just use a simple threshold
            if len(residue_energies) <= 3:
                # For very short peptides, consider the highest energy residue as unstable
                if residue_energies:
                    unstable_residues = [residue_energies[0]]
                else:
                    unstable_residues = []
            else:
                # Identify unstable residues (high energy outliers)
                energies_array = np.array([e for _, e, _ in residue_energies])
                mean_energy = np.mean(energies_array)
                std_energy = np.std(energies_array)
                threshold = mean_energy + 1.5 * std_energy
                
                unstable_residues = [(i, e) for i, e, _ in residue_energies if e > threshold]
            
            stability_metrics["residue_energies"] = residue_energies
            stability_metrics["unstable_residues"] = unstable_residues
            
            # Write residue energies to file
            energy_file = f"{output_prefix}_residue_energies.txt"
            with open(energy_file, 'w') as f:
                f.write("Residue\tEnergy\tAA\tIsUnstable\n")
                for i, energy, aa in residue_energies:
                    is_unstable = any(i == ui for ui, _ in unstable_residues)
                    f.write(f"{i}\t{energy:.2f}\t{aa}\t{is_unstable}\n")
            
            print(f"Wrote residue energies to {energy_file}")
            if unstable_residues:
                print(f"Unstable residues (high energy outliers): {[i for i, _ in unstable_residues]}")
        except Exception as e:
            print(f"Error calculating energy: {e}")
            import traceback
            traceback.print_exc()
            stability_metrics["overall_energy"] = None
        
        # 2. Secondary structure analysis
        try:
            dssp = DsspMover()
            dssp.apply(peptide_pose)
            
            ss_counts = {'H': 0, 'E': 0, 'L': 0}  # Helix, Sheet, Loop
            ss_string = ''
            
            for i in range(1, peptide_pose.total_residue() + 1):
                ss = peptide_pose.secstruct(i)
                ss_string += ss
                
                if ss == 'H':
                    ss_counts['H'] += 1
                elif ss == 'E':
                    ss_counts['E'] += 1
                else:
                    ss_counts['L'] += 1
            
            total_res = peptide_pose.total_residue()
            helix_percent = (ss_counts['H'] / total_res) * 100 if total_res > 0 else 0
            sheet_percent = (ss_counts['E'] / total_res) * 100 if total_res > 0 else 0
            loop_percent = (ss_counts['L'] / total_res) * 100 if total_res > 0 else 0
            
            stability_metrics["secondary_structure"] = ss_string
            stability_metrics["ss_counts"] = ss_counts
            stability_metrics["helix_percent"] = helix_percent
            stability_metrics["sheet_percent"] = sheet_percent
            stability_metrics["loop_percent"] = loop_percent
            
            print(f"Secondary structure: {ss_string}")
            print(f"Helix: {helix_percent:.1f}%, Sheet: {sheet_percent:.1f}%, Loop: {loop_percent:.1f}%")
            
            # Estimate stability from secondary structure
            # Higher helix/sheet content often correlates with more stable peptides
            structured_percent = helix_percent + sheet_percent
            stability_metrics["structured_percent"] = structured_percent
            
            if structured_percent >= 60:
                ss_stability = "High"
            elif structured_percent >= 30:
                ss_stability = "Medium"
            else:
                ss_stability = "Low"
                
            stability_metrics["ss_stability"] = ss_stability
            print(f"Structure-based stability estimate: {ss_stability} ({structured_percent:.1f}% structured)")
        except Exception as e:
            print(f"Error in secondary structure analysis: {e}")
            stability_metrics["secondary_structure"] = None
        
        # 3. Ramachandran analysis
        try:
            rama_favorable_count = 0
            rama_allowed_count = 0
            rama_outlier_count = 0
            rama_data = []
            
            for i in range(2, peptide_pose.total_residue()):
                # Skip if terminus or not protein
                if not peptide_pose.residue(i).is_protein():
                    continue
                
                # Get phi/psi angles
                phi = peptide_pose.phi(i)
                psi = peptide_pose.psi(i)
                
                # Convert to degrees
                phi_deg = np.degrees(phi)
                psi_deg = np.degrees(psi)
                
                # Simple classification based on common Ramachandran regions
                # This is a simplified version - Rosetta has more sophisticated methods
                is_favorable = False
                is_allowed = False
                
                # Alpha helix region
                if (-140 <= phi_deg <= -60) and (-60 <= psi_deg <= 60):
                    is_favorable = True
                # Beta sheet region
                elif ((-150 <= phi_deg <= -60) and (100 <= psi_deg <= 180)) or \
                     ((-150 <= phi_deg <= -60) and (-180 <= psi_deg <= -120)):
                    is_favorable = True
                # Left-handed helix and other allowed regions
                elif ((40 <= phi_deg <= 100) and (-60 <= psi_deg <= 60)) or \
                     ((-180 <= phi_deg <= -150) and (-60 <= psi_deg <= 60)) or \
                     ((-140 <= phi_deg <= -60) and (60 <= psi_deg <= 100)):
                    is_allowed = True
                
                if is_favorable:
                    rama_favorable_count += 1
                elif is_allowed:
                    rama_allowed_count += 1
                else:
                    rama_outlier_count += 1
                
                aa = peptide_pose.residue(i).name1()
                rama_data.append((i, aa, phi_deg, psi_deg, is_favorable, is_allowed))
            
            total_rama = len(rama_data)
            if total_rama > 0:
                favorable_percent = (rama_favorable_count / total_rama) * 100
                allowed_percent = (rama_allowed_count / total_rama) * 100
                outlier_percent = (rama_outlier_count / total_rama) * 100
                
                stability_metrics["rama_favorable_percent"] = favorable_percent
                stability_metrics["rama_allowed_percent"] = allowed_percent
                stability_metrics["rama_outlier_percent"] = outlier_percent
                stability_metrics["rama_data"] = rama_data
                
                print(f"Ramachandran analysis:")
                print(f"  Favorable: {favorable_percent:.1f}%")
                print(f"  Allowed: {allowed_percent:.1f}%")
                print(f"  Outlier: {outlier_percent:.1f}%")
                
                # Write Ramachandran data to file
                rama_file = f"{output_prefix}_ramachandran.txt"
                with open(rama_file, 'w') as f:
                    f.write("Residue\tAA\tPhi\tPsi\tFavorable\tAllowed\n")
                    for i, aa, phi, psi, favorable, allowed in rama_data:
                        f.write(f"{i}\t{aa}\t{phi:.2f}\t{psi:.2f}\t{favorable}\t{allowed}\n")
                
                # Estimate stability from Ramachandran plot
                if favorable_percent >= 80:
                    rama_stability = "High"
                elif favorable_percent >= 60:
                    rama_stability = "Medium"
                else:
                    rama_stability = "Low"
                    
                stability_metrics["rama_stability"] = rama_stability
                print(f"  Ramachandran-based stability estimate: {rama_stability}")
            else:
                print("Not enough residues for Ramachandran analysis")
                stability_metrics["rama_stability"] = "Unknown"
        except Exception as e:
            print(f"Error in Ramachandran analysis: {e}")
            stability_metrics["rama_stability"] = None
        
        # 4. Temperature-based stability estimation
        try:
            # Clone pose for simulation
            sim_pose = peptide_pose.clone()
            
            # Parameters for fast folding simulation
            n_cycles = 3
            kT_values = [0.8, 1.0, 1.5, 2.0, 3.0]  # Increasing temperature
            
            # Track rmsd at each temperature
            rmsd_values = []
            unfolding_temp = None
            
            # Create a MoveMap for backbone movement
            movemap = MoveMap()
            for i in range(1, sim_pose.total_residue() + 1):
                movemap.set_bb(i, True)
                movemap.set_chi(i, True)
            
            reference_pose = peptide_pose.clone()
            
            for kT in kT_values:
                # Create Monte Carlo mover with specific temperature
                mc = MonteCarlo(sim_pose, self.scorefxn, kT)
                
                # Small and shear moves for backbone perturbation
                small_mover = SmallMover(movemap, kT, 1)
                small_mover.angle_max(5)
                
                shear_mover = ShearMover(movemap, kT, 1)
                shear_mover.angle_max(5)
                
                # Create sequence mover for multiple moves per cycle
                seq = SequenceMover()
                seq.add_mover(small_mover)
                seq.add_mover(shear_mover)
                
                # Trials at this temperature
                n_moves = 10 * sim_pose.total_residue()
                
                for i in range(n_moves):
                    seq.apply(sim_pose)
                    mc.boltzmann(sim_pose)
                
                # Accept the final pose
                mc.recover_low(sim_pose)
                
                # Calculate RMSD to starting structure
                CA_rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(reference_pose, sim_pose)
                rmsd_values.append((kT, CA_rmsd))
                
                print(f"Temperature {kT:.1f}: CA_RMSD = {CA_rmsd:.2f} Å")
                
                # If RMSD exceeds threshold, consider this the unfolding temperature
                if CA_rmsd > 2.0 and unfolding_temp is None:
                    unfolding_temp = kT
                    print(f"Unfolding detected at temperature: {kT:.1f}")
            
            stability_metrics["temperature_rmsd"] = rmsd_values
            stability_metrics["unfolding_temperature"] = unfolding_temp
            
            # Write temperature data to file
            temp_file = f"{output_prefix}_temperature_stability.txt"
            with open(temp_file, 'w') as f:
                f.write("Temperature\tRMSD\n")
                for kT, rmsd in rmsd_values:
                    f.write(f"{kT:.1f}\t{rmsd:.2f}\n")
            
            # Estimate stability from unfolding temperature
            if unfolding_temp is None or unfolding_temp >= 2.5:
                temp_stability = "High"
            elif unfolding_temp >= 1.5:
                temp_stability = "Medium"
            else:
                temp_stability = "Low"
                
            stability_metrics["temperature_stability"] = temp_stability
            print(f"Temperature-based stability estimate: {temp_stability}")
            
            # Save the final unfolded state
            sim_pose.dump_pdb(f"{output_prefix}_unfolded.pdb")
        except Exception as e:
            print(f"Error in temperature-based stability analysis: {e}")
            stability_metrics["temperature_stability"] = None
        
        # 5. Calculate a combined stability score
        try:
            stability_scores = []
            
            # Energy-based score (normalized)
            if "overall_energy" in stability_metrics and stability_metrics["overall_energy"] is not None:
                energy = stability_metrics["overall_energy"]
                # Convert energy to 0-1 scale (lower energy is better)
                # Assuming -1.5 per residue is good, 0 is poor
                norm_energy = min(1.0, max(0.0, 
                    -energy / (1.5 * peptide_pose.total_residue())))
                stability_scores.append(norm_energy)
            
            # Secondary structure score
            if "structured_percent" in stability_metrics:
                structured_percent = stability_metrics["structured_percent"]
                # Convert percent to 0-1 scale
                norm_ss = min(1.0, max(0.0, structured_percent / 100))
                stability_scores.append(norm_ss)
            
            # Ramachandran score
            if "rama_favorable_percent" in stability_metrics:
                favorable_percent = stability_metrics["rama_favorable_percent"]
                # Convert percent to 0-1 scale
                norm_rama = min(1.0, max(0.0, favorable_percent / 100))
                stability_scores.append(norm_rama)
            
            # Temperature stability score
            if "unfolding_temperature" in stability_metrics and stability_metrics["unfolding_temperature"] is not None:
                # Higher temperature is better (0.8 to 3.0 range)
                unfolding_temp = stability_metrics["unfolding_temperature"]
                norm_temp = min(1.0, max(0.0, (unfolding_temp - 0.8) / 2.2))
                stability_scores.append(norm_temp)
            
            # Calculate overall stability score if we have at least 2 metrics
            if len(stability_scores) >= 2:
                overall_stability = np.mean(stability_scores) * 10  # Scale to 0-10
                stability_metrics["overall_stability_score"] = overall_stability
                
                # Stability classification
                if overall_stability >= 7.5:
                    stability_class = "Very High"
                elif overall_stability >= 6.0:
                    stability_class = "High"
                elif overall_stability >= 5.0:
                    stability_class = "Medium"
                elif overall_stability >= 3.5:
                    stability_class = "Low"
                else:
                    stability_class = "Very Low"
                    
                stability_metrics["stability_class"] = stability_class
                
                print(f"\nOverall peptide stability score: {overall_stability:.1f}/10 ({stability_class})")
            else:
                print("Not enough metrics to calculate overall stability")
                stability_metrics["overall_stability_score"] = None
                stability_metrics["stability_class"] = "Unknown"
        except Exception as e:
            print(f"Error calculating combined stability: {e}")
            stability_metrics["overall_stability_score"] = None
            stability_metrics["stability_class"] = "Unknown"
        
        # 6. Write a comprehensive stability report
        try:
            report_file = f"{output_prefix}_stability_report.txt"
            with open(report_file, 'w') as f:
                f.write("========================================\n")
                f.write("       PEPTIDE STABILITY REPORT         \n")
                f.write("========================================\n\n")
                
                # Basic info
                f.write(f"Peptide Length: {peptide_pose.total_residue()} residues\n")
                sequence = extract_peptide_sequence(peptide_pose)
                f.write(f"Sequence: {sequence}\n\n")
                
                # Overall stability
                if "overall_stability_score" in stability_metrics and stability_metrics["overall_stability_score"] is not None:
                    f.write(f"OVERALL STABILITY SCORE: {stability_metrics['overall_stability_score']:.1f}/10\n")
                    f.write(f"Stability Classification: {stability_metrics['stability_class']}\n\n")
                else:
                    f.write("OVERALL STABILITY: Could not be determined\n\n")
                
                # Energy details
                f.write("ENERGY ANALYSIS:\n")
                if "overall_energy" in stability_metrics and stability_metrics["overall_energy"] is not None:
                    f.write(f"Total Energy: {stability_metrics['overall_energy']:.2f}\n")
                    f.write(f"Energy per Residue: {stability_metrics['overall_energy']/peptide_pose.total_residue():.2f}\n")
                    
                    if stability_metrics.get("unstable_residues"):
                        f.write("Unstable Residues (high energy outliers):\n")
                        for i, energy in stability_metrics["unstable_residues"]:
                            aa = peptide_pose.residue(i).name1()
                            f.write(f"  {aa}{i}: {energy:.2f}\n")
                    else:
                        f.write("No unstable residues detected\n")
                else:
                    f.write("Energy could not be calculated\n")
                
                f.write("\n")
                
                # Secondary structure
                f.write("SECONDARY STRUCTURE ANALYSIS:\n")
                if "secondary_structure" in stability_metrics and stability_metrics["secondary_structure"] is not None:
                    f.write(f"Structure: {stability_metrics['secondary_structure']}\n")
                    f.write(f"Helix: {stability_metrics['helix_percent']:.1f}%\n")
                    f.write(f"Sheet: {stability_metrics['sheet_percent']:.1f}%\n")
                    f.write(f"Loop: {stability_metrics['loop_percent']:.1f}%\n")
                    f.write(f"Total Structured: {stability_metrics['structured_percent']:.1f}%\n")
                    f.write(f"Structure-based Stability: {stability_metrics['ss_stability']}\n")
                else:
                    f.write("Secondary structure could not be analyzed\n")
                
                f.write("\n")
                
                # Ramachandran
                f.write("RAMACHANDRAN ANALYSIS:\n")
                if "rama_favorable_percent" in stability_metrics:
                    f.write(f"Favorable: {stability_metrics['rama_favorable_percent']:.1f}%\n")
                    f.write(f"Allowed: {stability_metrics['rama_allowed_percent']:.1f}%\n")
                    f.write(f"Outlier: {stability_metrics['rama_outlier_percent']:.1f}%\n")
                    f.write(f"Ramachandran-based Stability: {stability_metrics['rama_stability']}\n")
                    
                    if stability_metrics["rama_outlier_percent"] > 10:
                        f.write("\nResidues with unfavorable backbone geometry:\n")
                        for i, aa, phi, psi, favorable, allowed in stability_metrics["rama_data"]:
                            if not favorable and not allowed:
                                f.write(f"  {aa}{i}: Phi={phi:.1f}, Psi={psi:.1f}\n")
                else:
                    f.write("Ramachandran analysis could not be performed\n")
                
                f.write("\n")
                
                # Temperature stability
                f.write("TEMPERATURE STABILITY:\n")
                if "unfolding_temperature" in stability_metrics and stability_metrics["unfolding_temperature"] is not None:
                    f.write(f"Unfolding Temperature: {stability_metrics['unfolding_temperature']:.1f}\n")
                    f.write(f"Temperature-based Stability: {stability_metrics['temperature_stability']}\n")
                    f.write("\nRMSD vs Temperature:\n")
                    for kT, rmsd in stability_metrics["temperature_rmsd"]:
                        f.write(f"  kT={kT:.1f}: RMSD={rmsd:.2f}Å\n")
                else:
                    f.write("Temperature stability could not be calculated\n")
                
                f.write("\n")
                
                # Summary and recommendations
                f.write("SUMMARY AND RECOMMENDATIONS:\n")
                
                # Overall assessment
                if "stability_class" in stability_metrics:
                    f.write(f"This peptide shows {stability_metrics['stability_class'].lower()} stability overall.\n\n")
                
                # Recommendations based on weak points
                recommendations = []
                
                if "unstable_residues" in stability_metrics and stability_metrics["unstable_residues"]:
                    recommendations.append("Consider mutating high-energy residues to more favorable amino acids.")
                
                if "structured_percent" in stability_metrics and stability_metrics["structured_percent"] < 30:
                    recommendations.append("The peptide has low secondary structure content. Consider adding helix or sheet stabilizing residues.")
                
                if "rama_outlier_percent" in stability_metrics and stability_metrics["rama_outlier_percent"] > 15:
                    recommendations.append("Several residues have unfavorable backbone geometry. Redesign these regions or add stabilizing interactions.")
                
                if "unfolding_temperature" in stability_metrics and stability_metrics["unfolding_temperature"] is not None and stability_metrics["unfolding_temperature"] < 1.5:
                    recommendations.append("The peptide unfolds at low temperature, indicating poor thermodynamic stability.")
                
                if recommendations:
                    f.write("Recommendations to improve stability:\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                else:
                    f.write("No specific recommendations for improving stability.")
                
            print(f"Wrote comprehensive stability report to {report_file}")
        except Exception as e:
            print(f"Error writing stability report: {e}")
        
        return stability_metrics
        
    def simulate_flexible_peptide(self, peptide_pose, output_prefix, n_samples=10):
        """
        Simulate the flexibility of the peptide to identify conformational variability
        
        Parameters:
            peptide_pose: PyRosetta pose of the peptide
            output_prefix: Prefix for output files
            n_samples: Number of conformational samples to generate
            
        Returns:
            List of ensemble poses and RMSD matrix
        """
        print("\n=== Simulating Peptide Flexibility ===")
        
        # Create a ScoreFunctionFactory
        sfxn = self.scorefxn
        
        # Create a MoveMap for backbone movement
        movemap = MoveMap()
        for i in range(1, peptide_pose.total_residue() + 1):
            movemap.set_bb(i, True)
            movemap.set_chi(i, True)
        
        # Monte Carlo object with room temperature kT
        kT = 1.0
        mc = MonteCarlo(peptide_pose, sfxn, kT)
        
        # Small and shear moves for backbone perturbation
        small_mover = SmallMover(movemap, kT, 1)
        small_mover.angle_max(5)
        
        shear_mover = ShearMover(movemap, kT, 1)
        shear_mover.angle_max(5)
        
        # Create sequence mover for multiple moves per cycle
        seq = SequenceMover()
        seq.add_mover(small_mover)
        seq.add_mover(shear_mover)
        
        # Create trial mover that accepts/rejects with Monte Carlo
        trial = TrialMover(seq, mc)
        
        # Create ensemble of structures
        ensemble = []
        for i in range(n_samples):
            # Start from original structure each time
            mc_pose = peptide_pose.clone()
            
            # Perform many trial moves
            n_moves = 10 * peptide_pose.total_residue()  # Scale with peptide size
            for j in range(n_moves):
                trial.apply(mc_pose)
            
            # Add the final pose to our ensemble
            ensemble.append(mc_pose.clone())
            
            # Save the structure
            mc_pose.dump_pdb(f"{output_prefix}_ensemble_{i+1}.pdb")
            
            print(f"Generated ensemble structure {i+1}/{n_samples}")
        
        # Calculate RMSD matrix between all ensemble members
        n_structures = len(ensemble)
        rmsd_matrix = np.zeros((n_structures, n_structures))
        
        for i in range(n_structures):
            for j in range(i+1, n_structures):
                rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(ensemble[i], ensemble[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        # Calculate average RMSD to measure flexibility
        upper_triangle = []
        for i in range(n_structures):
            for j in range(i+1, n_structures):
                upper_triangle.append(rmsd_matrix[i, j])
        
        avg_rmsd = np.mean(upper_triangle)
        max_rmsd = np.max(upper_triangle)
        
        print(f"Average pairwise RMSD: {avg_rmsd:.2f} Å")
        print(f"Maximum pairwise RMSD: {max_rmsd:.2f} Å")
        
        # Write RMSD matrix to file
        rmsd_file = f"{output_prefix}_ensemble_rmsd.txt"
        with open(rmsd_file, 'w') as f:
            f.write("# RMSD Matrix for Ensemble Structures\n")
            for i in range(n_structures):
                for j in range(n_structures):
                    f.write(f"{rmsd_matrix[i, j]:.2f}\t")
                f.write("\n")
        
        # Classify flexibility
        if avg_rmsd < 1.0:
            flexibility = "Low (Rigid)"
        elif avg_rmsd < 2.5:
            flexibility = "Medium"
        else:
            flexibility = "High (Flexible)"
        
        print(f"Peptide flexibility: {flexibility}")
        
        # Write a summary report
        summary_file = f"{output_prefix}_ensemble_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== Peptide Flexibility Analysis ===\n\n")
            f.write(f"Number of ensemble structures: {n_samples}\n")
            f.write(f"Average pairwise RMSD: {avg_rmsd:.2f} Å\n")
            f.write(f"Maximum pairwise RMSD: {max_rmsd:.2f} Å\n")
            f.write(f"Flexibility classification: {flexibility}\n\n")
            
            f.write("Interpretation:\n")
            if avg_rmsd < 1.0:
                f.write("This peptide is relatively rigid, maintaining a consistent structure\n")
                f.write("across different simulations. This suggests high structural stability.\n")
            elif avg_rmsd < 2.5:
                f.write("This peptide shows moderate flexibility, with some regions able to\n")
                f.write("adopt different conformations while maintaining an overall structure.\n")
            else:
                f.write("This peptide is highly flexible, with significant structural variation\n")
                f.write("across different simulations. This suggests it may not have a single\n")
                f.write("stable conformation and could be intrinsically disordered.\n")
        
        return ensemble, rmsd_matrix, avg_rmsd, flexibility

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
    
    # Create output directories
    output_dir = "docking_results"
    stability_dir = "stability_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stability_dir, exist_ok=True)
    
    # Load structures
    antibodies, peptides = load_structures(antibody_pdbs, peptide_directory)
    
    # Create stability analyzer
    stability_analyzer = PeptideStabilityAnalyzer()
    
    # Extract peptide sequences and analyze peptide stability
    peptide_sequences = {}
    peptide_stability_results = {}
    
    for name, pose in peptides.items():
        try:
            # Extract sequence
            peptide_sequences[name] = extract_peptide_sequence(pose)
            print(f"Extracted sequence for {name}: {peptide_sequences[name]}")
            
            # Analyze peptide stability
            print(f"\n===== Analyzing stability for peptide: {name} =====")
            output_prefix = f"{stability_dir}/{name}"
            
            # Save the original structure for reference
            pose.dump_pdb(f"{output_prefix}_original.pdb")
            
            # Create scorefxn for structure preparation and analysis
            scorefxn = pyrosetta.create_score_function("ref2015")
            
            # Perform comprehensive relaxation before stability analysis
            print(f"\nRelaxing structure for peptide: {name}")
            relaxed_pose = relax_peptide_structure(pose, scorefxn, output_prefix, verbose=True, num_cycles=25)
            
            # Compute stability metrics using the relaxed pose
            print(f"\nComputing stability metrics for relaxed peptide: {name}")
            stability_metrics = stability_analyzer.compute_stability_metrics(relaxed_pose, output_prefix)
            
            # Simulate flexibility also using relaxed pose
            print(f"\nSimulating flexibility for relaxed peptide: {name}")
            ensemble, rmsd_matrix, avg_rmsd, flexibility = stability_analyzer.simulate_flexible_peptide(
                relaxed_pose, output_prefix, n_samples=5)
            
            # Add flexibility metrics to stability results
            stability_metrics["flexibility"] = flexibility
            stability_metrics["avg_rmsd"] = avg_rmsd
            stability_metrics["rmsd_matrix"] = rmsd_matrix.tolist()  # Convert to list for storage
            
            # Add relaxation info to stability metrics
            stability_metrics["pre_relax_energy"] = scorefxn(pose)
            stability_metrics["post_relax_energy"] = scorefxn(relaxed_pose)
            stability_metrics["energy_improvement"] = stability_metrics["pre_relax_energy"] - stability_metrics["post_relax_energy"]
            
            peptide_stability_results[name] = stability_metrics
            
        except Exception as e:
            print(f"Error analyzing peptide {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Process each antibody and peptide combination for docking
    docking_results = {}
    
    for peptide_name, peptide_pose in peptides.items():
        docking_results[peptide_name] = {}
        print(f"\n===== Processing peptide: {peptide_name} =====")
        
        # Get stability information for this peptide
        stability_info = peptide_stability_results.get(peptide_name, {})
        stability_class = stability_info.get("stability_class", "Unknown")
        print(f"Peptide stability class: {stability_class}")
        
        # Get the relaxed peptide pose if available
        relaxed_peptide_path = f"{stability_dir}/{peptide_name}_relaxed.pdb"
        try:
            if os.path.exists(relaxed_peptide_path):
                print(f"Loading relaxed peptide structure for docking: {relaxed_peptide_path}")
                relaxed_peptide_pose = pyrosetta.pose_from_pdb(relaxed_peptide_path)
                # Use relaxed pose for docking
                peptide_pose_for_docking = relaxed_peptide_pose
            else:
                print("Relaxed peptide structure not found, using original structure for docking")
                peptide_pose_for_docking = peptide_pose
        except Exception as e:
            print(f"Error loading relaxed peptide, using original: {e}")
            peptide_pose_for_docking = peptide_pose
        
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
            docking_result = dock_peptide_to_specific_cdr(ab_pose, peptide_pose_for_docking, cdr_regions, tcr_data, output_prefix)
            
            if docking_result:
                # Add stability information to docking result
                docking_result["peptide_stability_class"] = stability_class
                if "overall_stability_score" in stability_info:
                    docking_result["peptide_stability_score"] = stability_info["overall_stability_score"]
                if "structured_percent" in stability_info:
                    docking_result["peptide_structured_percent"] = stability_info["structured_percent"]
                if "flexibility" in stability_info:
                    docking_result["peptide_flexibility"] = stability_info["flexibility"]
                
                docking_results[peptide_name][ab_name] = docking_result
                print(f"Docking complete. Results: {docking_result}")
            else:
                print(f"Docking failed for {peptide_name} and {ab_name}")
    
    # Generate a combined summary report with both docking and stability information
    with open(f"{output_dir}/combined_summary.txt", "w") as f:
        f.write("PEPTIDE-ANTIBODY DOCKING AND STABILITY SUMMARY\n")
        f.write("============================================\n\n")
        
        for peptide_name, ab_results in docking_results.items():
            f.write(f"Peptide: {peptide_name}\n")
            f.write(f"Sequence: {peptide_sequences.get(peptide_name, 'Unknown')}\n")
            
            # Add stability information
            stability_info = peptide_stability_results.get(peptide_name, {})
            f.write("\nSTABILITY INFORMATION:\n")
            f.write("-" * 40 + "\n")
            
            if stability_info:
                if "overall_stability_score" in stability_info:
                    f.write(f"Overall Stability Score: {stability_info['overall_stability_score']:.1f}/10\n")
                f.write(f"Stability Classification: {stability_info.get('stability_class', 'Unknown')}\n")
                
                if "secondary_structure" in stability_info:
                    f.write(f"Secondary Structure: {stability_info['secondary_structure']}\n")
                    f.write(f"Structured Content: {stability_info.get('structured_percent', 0):.1f}%\n")
                
                if "rama_favorable_percent" in stability_info:
                    f.write(f"Favorable Ramachandran: {stability_info['rama_favorable_percent']:.1f}%\n")
                
                if "flexibility" in stability_info:
                    f.write(f"Conformational Flexibility: {stability_info['flexibility']}\n")
                    f.write(f"Average Ensemble RMSD: {stability_info.get('avg_rmsd', 0):.2f} Å\n")
                
                if "energy_improvement" in stability_info:
                    f.write(f"Relaxation Energy Improvement: {stability_info['energy_improvement']:.2f}\n")
                
                if "unstable_residues" in stability_info and stability_info["unstable_residues"]:
                    f.write("Unstable Residues: ")
                    f.write(", ".join([f"{i}" for i, _ in stability_info["unstable_residues"]]))
                    f.write("\n")
            else:
                f.write("No stability information available\n")
            
            # Add docking information
            f.write("\nDOCKING RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for ab_name, result in ab_results.items():
                f.write(f"  Antibody: {ab_name}\n")
                
                if result:
                    f.write(f"  Binding CDR: {result.get('binding_cdr', 'Unknown')}\n")
                    f.write(f"  Initial Score: {result.get('initial_score', 'N/A')}\n")
                    f.write(f"  Relaxed Score: {result.get('relaxed_score', 'N/A')}\n")
                    
                    if "interface_energy" in result:
                        f.write(f"  Interface Energy: {result.get('interface_energy', 'N/A')}\n")
                    
                    f.write(f"  Binding to CDR: {result.get('binding_to_cdr', 'Unknown')}\n")
                    
                    # Include binding strength details if available
                    if "binding_strength" in result:
                        f.write(f"  Binding Strength: {result.get('binding_strength', 'Unknown')}\n")
                        f.write(f"  Hydrogen Bonds: {result.get('hydrogen_bonds', 0)}\n")
                        f.write(f"  Salt Bridges: {result.get('salt_bridges', 0)}\n")
                        f.write(f"  Hydrophobic Interactions: {result.get('hydrophobic', 0)}\n")
                    
                    f.write(f"  Contains Epitope: {result.get('contains_epitope', 'No')}\n")
                    
                    if result.get('contains_epitope'):
                        f.write(f"  Matching Epitope: {result.get('matching_epitope', 'None')}\n")
                        
                    f.write(f"  Output PDB: {result.get('output_pdb', 'None')}\n")
                    f.write(f"  Best Complex PDB: {result.get('best_complex_pdb', 'None')}\n")
                else:
                    f.write("  No successful docking\n")
                    
                f.write("\n")
            
            # Add combined stability and docking assessment
            f.write("\nCOMBINED ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            
            stability_score = stability_info.get("overall_stability_score", 0)
            if stability_score:
                if stability_score >= 7.0:
                    f.write("This peptide shows high intrinsic stability ")
                elif stability_score >= 5.0:
                    f.write("This peptide shows moderate intrinsic stability ")
                else:
                    f.write("This peptide shows low intrinsic stability ")
            
            best_docking_score = float('inf')
            best_binding_cdr = None
            best_interface_energy = None
            
            for ab_name, result in ab_results.items():
                if result and result.get('relaxed_score') is not None:
                    if result['relaxed_score'] < best_docking_score:
                        best_docking_score = result['relaxed_score']
                        best_binding_cdr = result.get('binding_cdr')
                        best_interface_energy = result.get('interface_energy')
            
            if best_binding_cdr:
                f.write(f"and binds best to the {best_binding_cdr} region ")
                f.write(f"with a docking score of {best_docking_score:.2f}")
                if best_interface_energy:
                    f.write(f" and interface energy of {best_interface_energy:.2f}")
                f.write(".\n")
            else:
                f.write("but did not successfully dock to any CDR region.\n")
            
            # Recommendations based on both stability and docking
            f.write("\nRecommendations:\n")
            
            if stability_score < 5.0 and best_binding_cdr:
                f.write("1. Consider improving peptide stability while maintaining binding affinity.\n")
                f.write("2. Focus modifications on non-binding regions to preserve the binding interface.\n")
            elif stability_score >= 5.0 and best_binding_cdr:
                f.write("1. This peptide shows good balance of stability and binding capability.\n")
                f.write(f"2. Minor optimizations could focus on the binding interface with {best_binding_cdr}.\n")
            elif stability_score >= 5.0 and not best_binding_cdr:
                f.write("1. The peptide is stable but doesn't dock well - redesign the binding interface.\n")
                f.write("2. Consider introducing complementary charges or hydrophobic patches.\n")
            else:
                f.write("1. Both stability and binding need significant improvement.\n")
                f.write("2. Consider a complete redesign or different peptide scaffold.\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
    
    # Also create an aggregated stability report across all peptides
    with open(f"{stability_dir}/aggregate_stability_report.txt", "w") as f:
        f.write("AGGREGATE PEPTIDE STABILITY REPORT\n")
        f.write("================================\n\n")
        
        # Table header
        f.write("Peptide\tSequence\tStability Score\tClass\tStructured %\tRama Favorable %\tFlexibility\tEnergy Improvement\n")
        
        # Sort peptides by stability score (if available)
        sorted_peptides = []
        for name, info in peptide_stability_results.items():
            score = info.get("overall_stability_score", 0)
            sorted_peptides.append((name, score))
        
        sorted_peptides.sort(key=lambda x: x[1], reverse=True)
        
        # Add data rows
        for name, _ in sorted_peptides:
            info = peptide_stability_results[name]
            sequence = peptide_sequences.get(name, "Unknown")
            
            score = info.get("overall_stability_score", "N/A")
            if score != "N/A":
                score = f"{score:.1f}"
                
            stability_class = info.get("stability_class", "Unknown")
            structured = info.get("structured_percent", "N/A")
            if structured != "N/A":
                structured = f"{structured:.1f}"
                
            rama = info.get("rama_favorable_percent", "N/A")
            if rama != "N/A":
                rama = f"{rama:.1f}"
                
            flexibility = info.get("flexibility", "Unknown")
            
            energy_improvement = info.get("energy_improvement", "N/A")
            if energy_improvement != "N/A":
                energy_improvement = f"{energy_improvement:.2f}"
            
            f.write(f"{name}\t{sequence}\t{score}\t{stability_class}\t{structured}\t{rama}\t{flexibility}\t{energy_improvement}\n")
        
        # Summary statistics if we have enough peptides
        if len(peptide_stability_results) >= 3:
            f.write("\nSUMMARY STATISTICS:\n")
            
            # Calculate averages for numeric metrics
            avg_metrics = {}
            for metric in ["overall_stability_score", "structured_percent", "rama_favorable_percent", "avg_rmsd", "energy_improvement"]:
                values = [info.get(metric) for info in peptide_stability_results.values() if info.get(metric) is not None]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
            
            if avg_metrics:
                f.write(f"Average Stability Score: {avg_metrics.get('overall_stability_score', 'N/A'):.1f}\n")
                f.write(f"Average Structured Content: {avg_metrics.get('structured_percent', 'N/A'):.1f}%\n")
                f.write(f"Average Favorable Ramachandran: {avg_metrics.get('rama_favorable_percent', 'N/A'):.1f}%\n")
                f.write(f"Average RMSD Flexibility: {avg_metrics.get('avg_rmsd', 'N/A'):.2f} Å\n")
                f.write(f"Average Energy Improvement: {avg_metrics.get('energy_improvement', 'N/A'):.2f}\n")
    
    print(f"\nAnalysis complete! Check the '{output_dir}' and '{stability_dir}' directories for all outputs.")

if __name__ == "__main__":
    main()