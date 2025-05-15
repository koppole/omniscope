import traceback
from pathlib import Path
from pyrosetta import *
from pyrosetta.rosetta.core.pose import Pose, PDBInfo, add_variant_type_to_pose_residue
from pyrosetta.rosetta.core.chemical import VariantType
from pyrosetta.rosetta.protocols.docking import setup_foldtree, DockingProtocol
from pyrosetta.rosetta.numeric import xyzVector_double_t as Vec
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.util import switch_to_residue_type_set
from pyrosetta.rosetta.core.scoring import cart_bonded, get_score_function
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
import numpy as np

init("-mute basic -mute core.optimization -detect_disulf false")

pdb_file = "/Users/koppole/Documents/Python_Files/Omniscope/pdb6m0j.ent"
peptide_length = 12  # 2 cysteines + (peptide_length - 2 central residues

def extract_chain(pose: Pose, chain_id: str) -> Pose:
    new_pose = Pose()
    first_residue_added = False
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(i) == chain_id:
            residue = pose.residue(i)
            if not first_residue_added:
                new_pose.append_residue_by_jump(residue, 1)
                first_residue_added = True
            else:
                try:
                    new_pose.append_residue_by_bond(residue)
                except RuntimeError:
                    new_pose.append_residue_by_jump(residue, new_pose.total_residue())
    return new_pose

def minimize_peptide(pose):
    #scorefxn = get_score_function(True)  # full-atom scorefxn
    #scorefxn.set_weight(cart_bonded, 1.0)  # ✅ Enable Cartesian scoring terms first
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart")
    relax = FastRelax()
    relax.cartesian(True)                  # ✅ Activate Cartesian mode
    relax.set_scorefxn(scorefxn)           # ✅ Set the updated scorefxn AFTER enabling cartesian
    relax.apply(pose)

def get_minimized_ace2(original_ace2_pose, cache_file="minimized_ace2.pdb"):
    cache_path = Path(cache_file)
    if cache_path.exists():
        print("📂 Loading minimized ACE2 from disk...")
        minimized_pose = pose_from_file(str(cache_path))
    else:
        print("🛠️  Minimizing ACE2 once and saving...")
        minimized_pose = original_ace2_pose.clone()
        minimize_peptide(minimized_pose)
        minimized_pose.dump_pdb(str(cache_path))
    return minimized_pose

def constrain_disulfide(pose):
    sg1 = AtomID(pose.residue(1).atom_index("SG"), 1)
    sg2 = AtomID(pose.residue(pose.total_residue()).atom_index("SG"), pose.total_residue())
    func = HarmonicFunc(2.02, 0.2)
    constraint = AtomPairConstraint(sg1, sg2, func)
    pose.add_constraint(constraint)

def get_interface_residues(partner1, partner2, cutoff=6.0):
    interface_residues = []
    for i in range(1, partner1.total_residue() + 1):
        res_i = partner1.residue(i)
        for j in range(1, partner2.total_residue() + 1):
            res_j = partner2.residue(j)
            for a in range(1, res_i.natoms() + 1):
                for b in range(1, res_j.natoms() + 1):
                    if res_i.xyz(a).distance(res_j.xyz(b)) < cutoff:
                        interface_residues.append(i)
                        break
                else:
                    continue
                break
    return sorted(set(interface_residues))

def extract_residue_subset(pose, residue_indices):
    subset_pose = Pose()
    for i, res_idx in enumerate(residue_indices):
        res = pose.residue(res_idx)
        if i == 0:
            subset_pose.append_residue_by_jump(res, 1)
        else:
            subset_pose.append_residue_by_bond(res)
    return subset_pose

def generate_disulfide_peptides(rbd_pose: Pose, window_size=12):
    seq = rbd_pose.sequence()
    peptides = []
    for i in range(len(seq) - window_size + 1):
        sub_seq = "C" + seq[i:i + window_size] + "C"
        peptides.append((i, sub_seq))
    return peptides

def align_peptide_to_template(peptide_pose: Pose, template_pose: Pose):
    def get_CA_coords(pose):
        return [pose.residue(i).xyz("CA") for i in range(1, pose.total_residue() + 1)]

    pep_coords = get_CA_coords(peptide_pose)
    tmpl_coords = get_CA_coords(template_pose)
    n = min(len(pep_coords), len(tmpl_coords))
    X = np.array([[v.x, v.y, v.z] for v in pep_coords[:n]])
    Y = np.array([[v.x, v.y, v.z] for v in tmpl_coords[:n]])

    X_cent = X.mean(axis=0)
    Y_cent = Y.mean(axis=0)
    X_centered = X - X_cent
    Y_centered = Y - Y_cent

    U, _, Vt = np.linalg.svd(X_centered.T @ Y_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Y_cent - X_cent @ R

    for i in range(1, peptide_pose.total_residue() + 1):
        res = peptide_pose.residue(i)
        for j in range(1, res.natoms() + 1):
            orig = res.xyz(j)
            orig_np = np.array([orig.x, orig.y, orig.z])
            new_pos = (orig_np - X_cent) @ R + t
            peptide_pose.set_xyz(AtomID(j, i), Vec(*new_pos))

def model_peptide_binding(peptide_seq, ace2_pose, template_pose):
    pep_pose = pose_from_sequence(peptide_seq, "fa_standard")

    # Patch the disulfide variants on terminal cysteines
    add_variant_type_to_pose_residue(pep_pose, VariantType.DISULFIDE, 1)
    add_variant_type_to_pose_residue(pep_pose, VariantType.DISULFIDE, pep_pose.total_residue())

    # Declare disulfide before minimization
    pep_pose.conformation().declare_chemical_bond(1, "SG", pep_pose.total_residue(), "SG")

    constrain_disulfide(pep_pose)
    minimize_peptide(pep_pose)
    align_peptide_to_template(pep_pose, template_pose)

    if not ace2_pose.pdb_info():
        ace2_pose.pdb_info(PDBInfo(ace2_pose))
    if not pep_pose.pdb_info():
        pep_pose.pdb_info(PDBInfo(pep_pose))

    ace2_pose.pdb_info().set_chains("A")
    pep_pose.pdb_info().set_chains("B")

    combined = Pose()
    combined.assign(ace2_pose)
    combined.append_pose_by_jump(pep_pose, 1)

    disulfide_res1 = ace2_pose.total_residue() + 1
    disulfide_res2 = ace2_pose.total_residue() + pep_pose.total_residue()
    combined.conformation().declare_chemical_bond(disulfide_res1, "SG", disulfide_res2, "SG")

    setup_foldtree(combined, "A_B", Vector1([1]))
    dock_protocol = DockingProtocol()
    dock_protocol.set_partners("A_B")
    dock_protocol.apply(combined)

    switch_to_residue_type_set(combined, 'fa_standard')
    scorefxn = get_fa_scorefxn()
    score = scorefxn(combined)

    pept_dir = Path("docked_peptides")
    pept_dir.mkdir(exist_ok=True)
    safe_name = peptide_seq.replace("/", "_")
    pept_path = pept_dir / f"docked_peptide_{safe_name}.pdb"
    combined.dump_pdb(str(pept_path))
    print(f"💾 Saved docked complex: {pept_path}")
    print(f"✅ Score for peptide {peptide_seq}: {score:.2f}")
    return score, peptide_seq

def main():
    full_pose = pose_from_file(pdb_file)
    print(f"✅ Loaded PDB '{pdb_file}' with {full_pose.total_residue()} residues.")

    raw_ace2_pose = extract_chain(full_pose, 'A')
    ace2_pose = get_minimized_ace2(raw_ace2_pose)
    rbd_pose = extract_chain(full_pose, 'E')

    print("🔍 Extracting RBD residues at ACE2 interface...")
    interface_indices = get_interface_residues(rbd_pose, ace2_pose)
    rbd_interface_pose = extract_residue_subset(rbd_pose, interface_indices)
    rbd_interface_pose.dump_pdb("rbd_interface_template.pdb")
    print(f"💾 Saved RBD interface template: rbd_interface_template.pdb")

    print("🧬 Generating disulfide-capped peptides from RBD interface")
    peptides = generate_disulfide_peptides(rbd_interface_pose, window_size=peptide_length - 2)

    results = []
    for idx, pep_seq in peptides:
        try:
            score, seq = model_peptide_binding(pep_seq, ace2_pose, rbd_interface_pose)
            results.append((seq, score))
        except Exception as e:
            print(f"❌ Failed peptide: {pep_seq}")
            traceback.print_exc()

    top5 = sorted(results, key=lambda x: x[1])[:5]
    print("\nTop 5 Peptide Immunogens based on ACE2 Binding Score:")
    for i, (seq, score) in enumerate(top5, 1):
        print(f"{i}. Sequence: {seq} | Score: {score:.2f}")

if __name__ == "__main__":
    main()
