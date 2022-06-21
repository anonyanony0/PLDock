import os
import logging
import torch
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from .tankbind.feature_utils import get_protein_feature
from .tankbind.feature_utils import get_clean_res_list
from .tankbind.utils import construct_data_from_graph_gvp
from .tankbind.feature_utils import extract_torchdrug_feature_from_mol, get_canonical_smiles
from .tankbind.generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords
from .tankbind.model import get_model
torch.set_num_threads(1)
#logging.basicConfig(level=logging.INFO)

def protein_feature(pdb,structure_dir):
    proteinFile = os.path.join(structure_dir,pdb,f'{pdb}_protein.pdb')
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", proteinFile)
    res_list = list(s.get_residues())
    clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)
    return get_protein_feature(clean_res_list)

def p2rank(test_pdbs, structure_dir, p2rank_outdir, p2rank_path, threads:int = 1):
    if not os.path.exists(p2rank_outdir): os.mkdir(p2rank_outdir)
    test_file = os.path.join(p2rank_outdir, 'pdb_files.list')
    with open(test_file,'w') as f:
        f.writelines([os.path.join('structure',i,f'{i}_protein.pdb')+'\n' for i in test_pdbs if os.path.exists(os.path.join(structure_dir,i,f'{i}_protein.pdb'))])
    cmd = f"bash {p2rank_path} predict {test_file} -o {p2rank_outdir} -threads {threads}"
    os.system(cmd)

def get_inputs(test_pdbs,compound_list,protein_dict):
    d = pd.read_csv(compound_list, index_col=0)
    info = []
    for i in test_pdbs:
        for _, line in tqdm(d.iterrows(), total=d.shape[0]):
            smiles = line['smiles']
            compound_name = ""
            protein_name = i
            # use protein center as the pocket center.
            com = ",".join([str(a.round(3)) for a in protein_dict[protein_name][0].mean(axis=0).numpy()])
            info.append([protein_name, compound_name, smiles, "protein_center", com])
            # since WDR is actually small enough, and we are interested in finding a ligand binds to the central cavity.
            # the block centered at the centroid of the protein is enough.
            # we don't need additional p2rank predicted centers.
            if False:
                p2rankFile = f"{base_pre}/p2rank/{proteinName}.pdb_predictions.csv"
                pocket = pd.read_csv(p2rankFile)
                pocket.columns = pocket.columns.str.strip()
                pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
                for ith_pocket, com in enumerate(pocket_coms):
                    com = ",".join([str(a.round(3)) for a in com])
                    info.append([protein_name, compound_name, smiles, f"pocket_{ith_pocket+1}", com])
    info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'smiles', 'pocket_name', 'pocket_com'])
    return info



class MyDataset_VS(Dataset):
    def __init__(self, root, data=None, protein_dict=None, proteinMode=0, compoundMode=1,
                pocket_radius=20, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.proteinMode = proteinMode
        self.pocket_radius = pocket_radius
        self.compoundMode = compoundMode
        self.shake_nodes = shake_nodes
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        smiles = line['smiles']
        pocket_com = line['pocket_com']
        pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
        pocket_com = pocket_com.reshape((1, 3))
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False

        protein_name = line['protein_name']
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]

        try:
            smiles = get_canonical_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles)
            mol.Compute2DCoords()
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        except:
            print("something wrong with ", smiles, "to prevent this stops our screening, we repalce it with a placeholder smiles 'CCC'")
            smiles = 'CCC'
            mol = Chem.MolFromSmiles(smiles)
            mol.Compute2DCoords()
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        # y is distance map, instead of contact map.
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                              protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                              use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)

        return data


def get_model_eval(modelFile,device):
    logging.basicConfig(level=logging.INFO)
    model = get_model(0, logging, device)
    model.load_state_dict(torch.load(modelFile, map_location=device))
    _ = model.eval()
    return model

def predict_affinity(dataset, model, out_csv = None, batch_size:int=5, device: str='cpu'):
    #model = get_model(0, logging, device)
    #model.load_state_dict(torch.load(modelFile, map_location=device))
    #_ = model.eval()
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


def screening(dataset, model, out_dir, batch_size, device):
    #info = get_inputs(test_pdbs,compound_list,protein_dict)
    #os.system(f"rm -r {dataset_path}")
    #os.system(f"mkdir -p {dataset_path}")
    out_csv = os.path.join(out_dir, 'result_info.csv')
    #dataset = MyDataset_VS(dataset_path, data=info, protein_dict=protein_dict)
    info, y_pred_list = predict_affinity(dataset, model, out_csv, batch_size, device)
    chosen = info.loc[info.groupby(['protein_name', 'smiles'],sort=False)['affinity'].agg('idxmax')].reset_index()
    # pick one with affinity greater than 7.
    chosen = chosen.query("affinity > 7").reset_index(drop=True)
    return chosen


def get_one_pose(chosen, dataset, model, out_dir):
    line = chosen.iloc[0]
    idx = line['index']
    one_data = dataset[idx]
    data_with_batch_info = next(iter(DataLoader(dataset[idx:idx+1], batch_size=1, 
                             follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=1)))
    y_pred, affinity_pred = model(data_with_batch_info)

    coords = one_data.coords.to(device)
    protein_nodes_xyz = one_data.node_xyz.to(device)
    n_compound = coords.shape[0]
    n_protein = protein_nodes_xyz.shape[0]
    y_pred = y_pred.reshape(n_protein, n_compound).to(device).detach()
    #y = one_data.dis_map.reshape(n_protein, n_compound).to(device)
    compound_pair_dis_constraint = torch.cdist(coords, coords)
    smiles = line['smiles']
    print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol.Compute2DCoords()
    LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
    info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, 
                                  LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                  n_repeat=1, show_progress=False)
    #toFile = f'{base_pre}/one_tankbind.sdf'
    toFile = os.path.join(out_dir, 'one_tankbind.sdf')
    new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
    write_with_new_coords(mol, new_coords, toFile)

def run_screen(test_pdbs, modelFile, structure_dir, out_dir, p2rank_path, compound_list, batch_size, device):
    dataset_path = os.path.join(out_dir, 'dataset')
    model = get_model_eval(modelFile,device)
    protein_dict = {}
    for i in test_pdbs:
        protein_dict[i] = protein_feature(i, structure_dir)
    p2rank(test_pdbs, structure_dir, out_dir, p2rank_path)
    info = get_inputs(test_pdbs,compound_list,protein_dict)
    os.system(f"rm -r {dataset_path}")
    os.system(f"mkdir -p {dataset_path}")
    dataset = MyDataset_VS(dataset_path, data=info, protein_dict=protein_dict)
    chosen = screening(dataset, model, out_dir, batch_size, device)
    get_one_pose(chosen, dataset ,model, out_dir)

if __name__ == '__main__':
    test_pdbs = ['6dlo']
    base_pre = f"/root/workshop/docking/TankBind-test/examples/HTVS"
    compound_list = "/root/workshop/docking/TankBind-test/examples/HTVS/Mcule_10000.csv"

    outdir = '/root/workshop/docking/TankBind-test/examples/HTVS/out'
    p2rank_path = '/root/workshop/docking/p2rank/prank'
    #dataset_path = f"{base_pre}/dataset/"
    batch_size = 5
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device= 'cpu'
    # modelFile = "../saved_models/re_dock.pt"
    # self-dock model
    modelFile = "/root/workshop/docking/TankBind-test/saved_models/self_dock.pt"
    run_screen(test_pdbs, modelFile, base_pre, outdir, p2rank_path, compound_list, batch_size, device)
    