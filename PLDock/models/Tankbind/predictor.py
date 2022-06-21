import os
import shutil
import logging
import torch
import pandas as pd
from tqdm import tqdm
import rdkit.Chem as Chem
from Bio.PDB import PDBParser
from torch_geometric.loader import DataLoader
from .tankbind.feature_utils import generate_sdf_from_smiles_using_rdkit
from .tankbind.feature_utils import get_protein_feature
from .tankbind.feature_utils import extract_torchdrug_feature_from_mol
from .tankbind.data import TankBind_prediction
from .tankbind.model import get_model
from .tankbind.generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords
# from utils import *
torch.set_num_threads(1)

def rdkit_sdf(pdb,structure_dir):
    #ligandFile = f"{pre}/{pdb}_ligand.sdf"
    ligandFile = os.path.join(structure_dir,pdb,f'{pdb}_ligand.sdf')
    smiles = Chem.MolToSmiles(Chem.MolFromMolFile(ligandFile))
    #rdkitMolFile = f"{pre}/{pdb}_rdkit_generated.sdf"
    rdkitMolFile = os.path.join(structure_dir,pdb,f'{pdb}_rdkit_generated.sdf')
    shift_dis = 0   # for visual only, could be any number, shift the ligand away from the protein.
    generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=shift_dis)

def protein_feature(pdb,structure_dir):
    proteinFile = os.path.join(structure_dir,pdb,f'{pdb}_protein.pdb')
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", proteinFile)
    res_list = list(s.get_residues())
    return get_protein_feature(res_list)

def compound_feature(pdb,structure_dir):
    rdkitMolFile = os.path.join(structure_dir,pdb,f'{pdb}_rdkit_generated.sdf')
    mol = Chem.MolFromMolFile(rdkitMolFile)
    return extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)

def p2rank(test_pdbs,structure_dir,p2rank_outdir,p2rank_path,threads:int = 1):
    test_file = os.path.join(test_dir, 'pdb_files.list')
    with open(test_file,'w') as f:
        f.writelines([os.path.join('structure',i,f'{i}_protein.pdb')+'\n' for i in test_pdbs if os.path.exists(os.path.join(structure_dir,i,f'{i}_protein.pdb'))])
    cmd = f"bash {p2rank_path} predict {test_file} -o {p2rank_outdir} -threads {threads}"
    os.system(cmd)


def get_protein_ligand_dicts(test_pdbs, structure_dir):
    protein_dict = {}
    compound_dict = {}

    for i in test_pdbs:
        rdkit_sdf(i,structure_dir)
        protein_dict[i] = protein_feature(i,structure_dir)
        compound_dict[i+"_rdkit_generated"] = compound_feature(i,structure_dir)
    return protein_dict,compound_dict


def get_protein_ligand_info(compound_dict,protein_dict,p2rank_outdir):
    info = []
    for compound_name in list(compound_dict.keys()):
        pdb = compound_name.split('_')[0]
        # use protein center as the block center.
        com = ",".join([str(a.round(3)) for a in protein_dict[pdb][0].mean(axis=0).numpy()])
        info.append([pdb, compound_name, "protein_center", com])
        p2rankFile = f"{p2rank_outdir}/{pdb}_protein.pdb_predictions.csv"
        pocket = pd.read_csv(p2rankFile)
        pocket.columns = pocket.columns.str.strip()
        pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
        for ith_pocket, com in enumerate(pocket_coms):
            com = ",".join([str(a.round(3)) for a in com])
            info.append([pdb, compound_name, f"pocket_{ith_pocket+1}", com])
    info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
    return info

def predict_affinity(dataset, modelFile, out_csv = None, batch_size:int=5, device: str='cpu'):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device= 'cpu'
    logging.basicConfig(level=logging.INFO)
    model = get_model(0, logging, device)
    # re-dock model
    # modelFile = "../saved_models/re_dock.pt"
    # self-dock model
    #modelFile = "../saved_models/self_dock.pt"
    #modelFile = '/root/workshop/docking/TankBind-test/saved_models/self_dock.pt'
    model.load_state_dict(torch.load(modelFile, map_location=device))
    _ = model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)
    affinity_pred_list = []
    y_pred_list = []
    for data in tqdm(data_loader):
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for i in range(data.y_batch.max() + 1):
            y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())

    affinity_pred_list = torch.cat(affinity_pred_list)
    info = dataset.data
    info['affinity'] = affinity_pred_list
    if out_csv is not None:
        info.to_csv(out_csv)
    return info, y_pred_list

def predict_poses(result_folder, dataset, re_info, y_pred_list, device):
    chosen = re_info.loc[re_info.groupby(['protein_name', 'compound_name'],sort=False)['affinity'].agg('idxmax')].reset_index()
    for i, line in chosen.iterrows():
        idx = line['index']
        protein_name = line['protein_name']
        pocket_name = line['pocket_name']
        compound_name = line['compound_name']
        ligandName = compound_name.split("_")[1]
        coords = dataset[idx].coords.to(device)
        protein_nodes_xyz = dataset[idx].node_xyz.to(device)
        n_compound = coords.shape[0]
        n_protein = protein_nodes_xyz.shape[0]
        y_pred = y_pred_list[idx].reshape(n_protein, n_compound).to(device)
        y = dataset[idx].dis_map.reshape(n_protein, n_compound).to(device)
        compound_pair_dis_constraint = torch.cdist(coords, coords)
        #rdkitMolFile = f"{pre}/{pdb}_{ligandName}_mol_from_rdkit.sdf"
        rdkitMolFile = os.path.join(structure_dir,protein_name,f'{protein_name}_rdkit_generated.sdf')
        mol = Chem.MolFromMolFile(rdkitMolFile)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
        info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, 
                                      LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                      n_repeat=1, show_progress=False)

        #result_folder = f'{pre}/{pdb}_result/'
        if not os.path.exists(result_folder): os.mkdirs(result_folder)
        #os.system(f'mkdir -p {result_folder}')
        # toFile = f'{result_folder}/{ligandName}_{pocket_name}_tankbind.sdf'
        toFile = f'{result_folder}/{ligandName}_tankbind.sdf'
        # print(toFile)
        new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
        write_with_new_coords(mol, new_coords, toFile)


def run_precdict(test_pdbs, out_dir, structure_dir, modelFile, p2rank_path, batch_size, device):
    p2rank_outdir = os.path.join(out_dir, 'p2rank')
    dataset_path = os.path.join(out_dir, 'dataset')
    out_csv = os.path.join(out_dir,'info_with_predicted_affinity.csv')
    # Predict pockets by p2rank
    p2rank(test_pdbs,structure_dir,p2rank_outdir,p2rank_path)
    #Prepare input info
    protein_dict,compound_dict = get_protein_ligand_dicts(test_pdbs, structure_dir)
    info = get_protein_ligand_info(compound_dict,protein_dict,p2rank_outdir)
    if os.path.exists(dataset_path): shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)
    # Make dataset
    dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)
    # Predict affinity and write to out_csv
    re_info,y_pred_list =  predict_affinity(dataset, modelFile, out_csv, batch_size, device)
    # Best scored poses
    #chosen = re_info.loc[info.groupby(['protein_name', 'compound_name'],sort=False)['affinity'].agg('idxmax')].reset_index()
    predict_poses(out_dir, dataset, re_info,y_pred_list, device)

if __name__ == '__main__':
    test_pdbs = ['3d4k']
    test_dir = '/root/workshop/docking/TankBind-test/examples/test'
    out_dir = os.path.join(test_dir, 'out')
    #p2rank_outdir = '/root/workshop/docking/TankBind-test/examples/test/p2rank_out'
    #dataset_path = '/root/workshop/docking/TankBind-test/examples/test/out/dataset'
    #dataset_path = os.path.join(test_dir, 'dataset')
    p2rank_path = '/root/workshop/docking/p2rank/prank'
    structure_dir = os.path.join(test_dir, 'structure')
    modelFile = '/root/workshop/docking/TankBind-test/saved_models/self_dock.pt'
    #out_csv = os.path.join(test_dir,'info_with_predicted_affinity.csv')
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_precdict(test_pdbs, out_dir, structure_dir, modelFile,p2rank_path, batch_size, device)