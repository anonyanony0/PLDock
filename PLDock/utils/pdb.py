from tabnanny import check
from PLDock.utils.utils import get_global_node_rank
from .constants import SPECIAL_RES, RCSB_AFFINITY_TYPES
from .io_tools import download, non_empty_file, check_download, check_dir, ungz
from .io_tools import dir_cwd, split_file, get_dict, write_list, read_dict
import pandas as pd
import numpy as np
from Bio.PDB import Select
from Bio.PDB import PDBList
from Bio.PDB import PDBIO
import Bio.PDB
import os
import re
import parallel as par
import contextlib
# import func_timeout
# from func_timeout import func_set_timeout
from molmass import Formula
from pypdb import get_pdb_file
from PLDock import log
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

try:
    from pymol import cmd
except ImportError:
    pass

logger =  log()

def get_formula(formula: str):
    """
    It takes a string and returns a Formula object if the string is a valid formula, and returns None
    otherwise
    
    :param formula: The formula to be parsed
    :return: A formula object
    """
    try:
        return Formula(formula)
    except Exception:
        return


def formula_info(formula_ins, mode='mw'):
    """
    It takes a Formula object and a mode, and returns the molecular weight, the mass number, the number of
    atoms, or the number of non-hydrogen atoms
    
    :param formula_ins: the formula to be analyzed
    :param mode: , defaults to mw (optional)
    :return: The mass of the formula
    """
    # can not use contextlib:
    if not formula_ins:
        return
    if mode == 'mw':
        try:
            return formula_ins.mass
        except Exception:
            return
    elif mode == 'mn':
        try:
            return formula_ins.isotope.massnumber
        except Exception:
            return
    elif mode == 'all':
        try:
            return sum(i[1] for i in formula_ins.composition())
        except Exception:
            return
    elif mode == 'nh':
        try:
            return sum(i[1] for i in formula_ins.composition() if i[0] != 'H')
        except Exception:
            return
    else:
        logger.error(f'Wrong mode: {mode}')


def filter_res_name(res_name, mode=True, speclal_res=None):
    """
    > This function takes a residue name and returns True if the residue is in the list of special
    residues, and False otherwise
    
    :param res_name: the name of the residue
    :param mode: False if you want to keep the special residues, True if you want to remove them,
    defaults to True (optional)
    :param speclal_res: a list of special residues
    :return: A boolean value. True means to remove, False means to keep.
    """
    if speclal_res is None:
        speclal_res = SPECIAL_RES
    return mode if res_name in speclal_res else not mode


def filter_mol_formula(formula_ins, limit=None, mode='nh'):
    """
    `filter_mol_formula` takes a molecular Formula object and returns a string if the formula's mass number,
    molecular weight, or number of non-hydrogen atoms is outside of a specified range
    
    :param formula: the chemical formula of the molecule
    :param limit: a tuple of two numbers, the lower and upper limits of the feature
    :param mode: 'mw' for molecular weight, 'mn' for mass number, 'all' for number of all atoms, 'nh'
    for number of non-hydrogen atoms, defaults to nh (optional)
    :return: A string with the formula feature.
    """
    formula_feature = formula_info(formula_ins, mode)
    if limit and formula_feature and (formula_feature < limit[0] or formula_feature > limit[1]):
        mode_dict = {'mw': 'molecular weight', 'mn': 'mass number', 'all': 'number of all atoms',
                     'nh': 'number of non-hydrogen atoms'}
        return f'{mode_dict[mode]}: {formula_feature}'


def filter_ligand(ligand_name, formula, speclal_res=None, atom_limit=None, nm_limit=None):
    """
    > If the residue name is not in the list of special residues, then check if the number of heavy
    atoms is out the atom limit. If it is, then return True. If not, then check if the molecular
    weight is out the molecular weight limit. If it is, then return True. If not, then return
    False
    
    :param ligand_name: The name of the ligand
    :param formula: the chemical formula of the ligand
    :param speclal_res: a list of residue names that are not considered ligands
    :param atom_limit: The minimum and maximum number of atoms limit in the ligand
    :param mn_limit: The minimum and maximum mass number limit of the ligand
    :return: A boolean value or a string. True or a string means to remove. False or None means to keep.
    """
    formula_ins = get_formula(formula)
    filt = filter_res_name(ligand_name, mode=True, speclal_res=speclal_res)
    if not filt:
        filt = filter_mol_formula(formula_ins, atom_limit, mode='nh')
    return filt or filter_mol_formula(formula_ins, nm_limit, mode='mn')


def check_pdb(pdb_file, format='pdb', remove=False):
    """
    It checks if a PDB file is valid
    
    :param pdb_file: the path to the pdb file
    :param format: the format of the PDB file. Can be either 'pdb' or 'cif', defaults to pdb
    (optional)
    :return: 1 if the file is a valid pdb file, 0 if it is not.
    """
    if non_empty_file(pdb_file):
        if format == 'pdb':
            func = Bio.PDB.PDBParser()
        elif format == 'cif':
            func = Bio.PDB.MMCIFParser()
        else:
            logger.warning(f'Unexists: {pdb_file}')
        try:
            return func.get_structure('X', pdb_file)[0]
        except Exception:
            if remove:
                os.remove(pdb_file)


#@func_set_timeout(10)
def biopdb_down(pdb, data_dir=None, format='pdb'):
    """
    It downloads a PDB file from the RCSB PDB database, and returns the path to the downloaded file
    
    :param pdb: The PDB ID of the structure you want to download
    :param data_dir: the directory where the downloaded PDB file will be saved
    :param format: the format of the file you want to download, defaults to pdb (optional)
    :return: A boolean value.
    """
    pdb = pdb.lower()
    data_dir = dir_cwd(data_dir)
    file_path = os.path.join(data_dir, f'{pdb}.{format}')
    # Can not use contextlib
    pdbl = PDBList()
    if format == 'cif':
        try:
            pdbl.retrieve_pdb_file(pdb, pdir=data_dir, file_format='mmCif')
        except EOFError:
            pass
    elif format == 'pdb':
        try:
            pdbl.retrieve_pdb_file(pdb, pdir=data_dir, file_format=format)
            downed_pdb = os.path.join(data_dir, f'pdb{pdb}.ent')
            os.rename(downed_pdb, file_path)
        except Exception:
            pass
    else:
        logger.warning(f'Given a wrong format when download {pdb}: {format}')
    return check_pdb(file_path, format, True)


# @func_set_timeout(10)
def pymol_down(pdb: str, data_dir=None, file_format: str = 'pdb'):
    """
    > This function downloads a PDB file from the RCSB PDB database and loads it into PyMOL
    
    :param pdb: The PDB ID of the structure you want to download
    :type pdb: str
    :param data_dir: The directory where the PDB files are stored
    :param format: The format of the file you want to download, defaults to pdb
    :type format: str (optional)
    :return: The pdb file is being returned.
    """
    pdb = pdb.lower()
    cmd.set('fetch_path', cmd.exp_path(data_dir), quiet=1)
    cmd.remove('all', quiet=1)
    cmd.fetch(pdb, type=file_format, quiet=1)
    return


def pymol_rm_ligands(in_file, out_file):
    cmd.remove('all', quiet=1)
    cmd.load(in_file)
    cmd.do('remove ligands')
    cmd.do('remove resn hoh')
    cmd.do('remove solvent')
    cmd.save(out_file)
    return


def pymol_cif_pdb(cif_file, pdb_file, remove_ligands=False):
    cmd.remove('all', quiet=1)
    cmd.load(cif_file)
    if remove_ligands:
        cmd.do('remove ligands')
        cmd.do('remove resn hoh')
        cmd.do('remove solvent')
    cmd.save(pdb_file)
    return


# @func_set_timeout(10)
def urlpdb_down(pdb, data_dir=None, file_format='pdb'):
    """
    It downloads a PDB file from the RCSB PDB website
    
    :param pdb: The PDB ID of the structure you want to download
    :param data_dir: the directory where the PDB files will be downloaded to
    :param format: The format of the file you want to download, defaults to pdb (optional)
    :return: A list of the PDB IDs of the structures that are in the PDB.
    """
    pdb = pdb.lower()
    data_dir = dir_cwd(data_dir)
    file_path = os.path.join(data_dir, f'{pdb}.{file_format}')
    if not non_empty_file(file_path):
        # Can not use contextlib.suppress
        try:
            pdb_file = get_pdb_file(
                pdb, filetype=file_format, compression=False)
            with open(file_path, 'r') as f:
                f.write(pdb_file)
        except Exception:
            pass
    urls = {'pdb': ['https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb}.ent',
                    'https://files.rcsb.org/download/{pdb}.pdb',
                    'https://files.rcsb.org/view/{pdb}.pdb'],
            'cif': ['https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}.cif',
                    'https://files.rcsb.org/download/{pdb}.cif',
                    'https://files.rcsb.org/view/{pdb}.cif']}

    for pdb_url in urls[file_format]:
        if not non_empty_file(file_path):
            res = download(pdb_url, file_path)
    return res


# @func_set_timeout(35)
def pdb_downer(pdb: str, data_dir=None, file_format: str = 'pdb'):
    """
    It downloads a PDB file from the RCSB PDB website, and if that fails, it downloads it from other websites
    
    :param data_dir: the directory where you want to save the PDB files
    :param pdb: The PDB ID of the structure you want to download
    :type pdb: str
    :param format: the format of the file you want to download, defaults to pdb
    :type format: str (optional)
    :return: 1 if the file is downloaded, 0 if not.
    """
    pdb = pdb.lower()
    data_dir = dir_cwd(data_dir)
    file_path = os.path.join(data_dir, f'{pdb}.{file_format}')
    if not non_empty_file(file_path):
        with contextlib.suppress(Exception):
            pymol_down(pdb, data_dir, file_format)
    if not non_empty_file(file_path):
        with contextlib.suppress(Exception):
            urlpdb_down(pdb, data_dir, file_format)
    if not non_empty_file(file_path):
        with contextlib.suppress(Exception):
            biopdb_down(pdb, data_dir, file_format)
    return 1 if non_empty_file(file_path) else 0


def down_pdb(pdb: str, data_dir=None, file_format: str = 'pdb', max_time: int = 2):
    """
    It downloads a PDB file from the RCSB PDB website, and saves it to a specified directory
    
    :param pdb: The PDB ID of the structure you want to download
    :type pdb: str
    :param data_dir: the directory where the data will be downloaded to
    :param format: the format of the file you want to download, defaults to pdb
    :type format: str (optional)
    :param max_time: the maximum number of times to try to download the file, defaults to 2
    :type max_time: int (optional)
    :return: 1 if the file is downloaded, 0 if not.
    """
    data_dir = dir_cwd(data_dir)
    file_path = os.path.join(data_dir, f'{pdb.lower()}.{file_format}')
    for _ in range(max_time):
        if not non_empty_file(file_path):
            with contextlib.suppress(Exception):
                pdb_downer(pdb, data_dir, file_format)
    if non_empty_file(file_path):
        return 1
    #logger.warning(f'Download {pdb}.{format} failed after {max_time} tries')
    return 0


def down_pdb_format(pdb: str, data_dir=None, file_format: str = 'all', max_time: int = 2):
    pdb_file = down_pdb(pdb, data_dir, file_format='pdb', max_time=2)
    if file_format == 'all':
        cif_file = down_pdb(pdb, data_dir, file_format='cif', max_time=2)
        if pdb_file and cif_file:
            return ('pdb', 'cif')
        elif pdb_file:
            return 'pdb'
        elif cif_file:
            return 'cif'
        else:
            return None
    elif file_format == 'one':
        if pdb_file:
            return 'pdb'
        else:
            cif_file = down_pdb(pdb, data_dir, file_format='cif', max_time=2)
            return 'cif' if cif_file else None


def down_ligand(pdb, chain, ligand, data_dir=None, format='sdf'):
    """
    > Download a ligand from the RCSB PDB website
    
    :param pdb: the PDB ID of the structure you want to download
    :param chain: The chain of the ligand
    :param ligand: the ligand name, e.g. "HEM"
    :param data_dir: the directory where you want to download the files to
    :param format: the format of the file. Can be sdf, mol, or mol2, defaults to sdf (optional)
    :return: 1 if succeeded, None if failed
    """
    data_dir = dir_cwd(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    url = f'https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain}&auth_comp_id={ligand}&encoding={format}&copy_all_categories=false&download=false'
    file_path = os.path.join(data_dir, f'{pdb}_ligand.{format}')
    return download(url, file_path)


def crawl_canonical_smiles(ligand: str, software: str = 'OpenEye OEToolkits'):
    """
    It takes a ligand name (e.g. 'ATP') and returns the canonical SMILES string for that ligand
    
    :param ligand: The ligand name, e.g. 'ATP'
    :type ligand: str
    :param software: The software used to generate the canonical SMILES, defaults to OpenEye OEToolkits
    :type software: str (optional)
    :return: The canonical smiles of the ligand
    """
    base_url = 'http://www.ebi.ac.uk/pdbe-srv/pdbechem/chemicalCompound/show/'
    url = f'{base_url}{ligand}'
    a = pd.read_html(url, header=1)[0]
    return a.loc[(a['Type'] == 'Canonical SMILES') & (a['Program'] == software), 'Descriptor'].values[0]


def ligand_chain(pdb, ligand):
    """
    It takes a PDB ID and a ligand ID and returns the chain ID of the ligand
    
    :param pdb: The PDB ID of the structure
    :param ligand: The ligand you want to find the chain for
    :return: The chain ID of the ligand
    """
    pdb = pdb.lower()
    url = f'http://www.ebi.ac.uk/pdbe/api/pdb/entry/ligand_monomers/{pdb}'
    ligands = get_dict(url)
    if ligands:
        return next((i['chain_id'] for i in ligands[pdb] if i['chem_comp_id'] == ligand), '')


def ligand_chembl(ligand):
    url = f'https://www.ebi.ac.uk/pdbe/api/pdb/compound/mappings/{ligand}'
    adict = get_dict(url)
    try:
        return adict[ligand][0]['chembl_id']
    except Exception:
        return ''


def down_ligand_smiles(data_dir: str = None, software: str = 'OpenEye'):
    """
    Download the SMILES file from the PDB Ligand Expo website
    
    :param data_dir: the directory where you want to download the file to
    :type data_dir: str
    :param software: The software used to generate the SMILES strings, defaults to OpenEye
    :type software: str (optional)
    :return: The file path of the downloaded file.
    """
    data_dir = dir_cwd(data_dir)
    name_dict = {'OpenEye': 'oe', 'CACTVS': 'cactvs'}
    file_name = f'Components-smiles-{name_dict[software]}.smi'
    file_path = os.path.join(data_dir, file_name)
    url = f'http://ligand-expo.rcsb.org/dictionaries/{file_name}'
    return download(url, file_path)


def get_ligand_by_name(residue_name, model, chains=None):
    """
    It takes a residue name and a model as input, and returns a dictionary of chains and their residues
    
    :param residue_name: The name of the ligand you want to extract
    :param model: The model you want to extract the residues from
    :return: A dictionary of chain:[residues].
    """
    ligands = {}
    chains_model = model.child_dict
    #if not chains:
    true_chains =  list(chains_model.keys())
    if not chains:
        chains = true_chains
    for c in chains:
        if c in true_chains:
            for protein_res in chains_model[c].child_list:
                if protein_res.resname == residue_name:
                    try:
                        ligands[c].append(protein_res)
                    except KeyError:
                        ligands[c] = [protein_res]
        else:
            logger.warning(f'Chain {c} is not found in the structure')
    return ligands


def save_complex(structure, ligands, chains, sites, filename):
    """
    It takes a structure, ligands, chains, and sites and saves them as separate files
    
    :param structure: the structure object from BioPython
    :param ligands: a dictionary of ligands, where the key is the ligand name and the value is a list of
    residues
    :param chains: a list of chains to be included in the protein file
    :param sites: a list of residues that are in the pocket
    :param filename: the name of the pdb file
    """
    if not chains:
        chains = structure.child_dict

    class ComplexSelect(Select):
        def accept_residue(self, residue):
            for group in ligands.values():
                con = (residue in group or residue.get_parent().get_id()
                       in chains) and not residue.resname == 'HOH'
                return 1 if con else 0

    class LigandSelect(Select):
        def accept_residue(self, residue):
            for group in ligands.values():
                return 1 if residue in group else 0

    class PocketSelect(Select):
        def accept_residue(self, residue):
            return 1 if residue in sites and residue.resname != 'HOH' else 0

    class ProteinSelect(Select):
        def accept_residue(self, residue):
            for group in ligands.values():
                if residue.get_parent().get_id() in chains and not (residue in group or residue.resname == 'HOH'):
                    return 1
                else:
                    return 0
    io = PDBIO()
    io.set_structure(structure)

    pf = filename+'_protein.pdb'
    try:
        io.save(pf, ProteinSelect())
        with open(pf, 'r') as f:
            plines = [i for i in f.readlines() if not (i.startswith(
                'HETATM') or i.startswith('TER') or i.startswith('END'))]
        with open(pf, 'w') as f:
            f.writelines(plines)
    except Exception:
        pass

    if ligands:
        lf = filename+'_ligand.pdb'
        try:
            io.save(lf, LigandSelect())
            with open(lf, 'r') as f:
                llines = [i for i in f.readlines() if not (
                    i.startswith('TER') or i.startswith('END'))]
        except Exception:
            pass

    if sites:
        pcf = filename+'_pockets.pdb'
        try:
            io.save(pcf, PocketSelect())
            with open(pcf, 'r') as f:
                pclines = [i for i in f.readlines() if not (i.startswith(
                    'HETATM') or i.startswith('TER') or i.startswith('END'))]
            with open(pcf, 'w') as f:
                f.writelines(pclines)
        except Exception:
            pass
    if ligands:
        cf = filename+'_complex.pdb'
        try:
            with open(cf, 'w') as f:
                f.writelines(plines+llines)
        except Exception:
            pass


def residue_dist_to_ligand(protein_residue, ligand_residue):
    """
    For each atom in the ligand, find the distance to each atom in the protein residue, and return the
    minimum distance.
    
    :param protein_residue: a single residue from the protein
    :param ligand_residue: the ligand residue
    :return: The minimum distance between the ligand and the protein.
    """
    #Returns distance from the protein C-alpha to the closest ligand atom
    dist = []
    for latom in ligand_residue:
        #if "CA" in protein_residue:
        for patom in protein_residue:
            vector = patom.coord - latom.coord
            dist.append(np.sqrt(np.sum(vector * vector)))
    return min(dist)


def active_site(ligands, distance, model):
    """
    For each ligand, find all residues within a given distance of the ligand
    
    :param ligands: a dictionary of ligands, where the key is the chain and the value is a list of
    residues
    :param distance: the distance from the ligand to the protein residue
    :param model: the PDB file
    :return: A set of chains and a list of residues
    """
    # Prints out residues located at a given distance from ligand
    chains = model.child_dict
    int_chains = []
    # sites = {i:[] for i in ligands.keys()}
    sites = []
    for group in ligands.values():
        for ligand_res in group:
            for c in chains:
                for protein_res in chains[c].child_list:
                    if protein_res not in group:
                        dist = residue_dist_to_ligand(protein_res, ligand_res)
                        if dist and dist <= distance:
                            sites.append(protein_res)
                            int_chains.append(c)
    return list(set(int_chains)), sites


def biopython_structure(pdb, data_dir):
    """
    If the file is not empty, try to read it with biopython, if it fails, try to download it again
    
    :param pdb: The PDB ID of the structure you want to download
    :param data_dir: the directory where the PDB files are stored
    :return: A structure object
    """
    structure = None
    for file_format in ['pdb', 'cif']:
        if not structure:
            parse = Bio.PDB.PDBParser() if file_format == 'pdb' else Bio.PDB.MMCIFParser()
            file = os.path.join(data_dir, f'{pdb}.{file_format}')
            if not non_empty_file(file):
                down_pdb(pdb, data_dir, file_format=file_format)
            if non_empty_file(file):
                try:
                    structure = parse.get_structure('X', file)
                except Exception:
                    structure = None
    return structure or logger.warning(f'Can not read {pdb} by biopython')


def save_first_ligand(pdb_file, ligand_name, filename, chains=None):
    """
    > This function takes a pdb file, a ligand name, and a filename as input, and saves the first ligand
    to the filename
    
    :param pdb_file: the path to the pdb file
    :param ligand_name: the name of the ligand you want to extract
    :param filename: the name of the file to save the ligand to
    """
    structure = 0
    fdir, code = os.path.split(pdb_file)
    code = os.path.splitext(code)[0]
    structure = biopython_structure(code, fdir)
    if structure:
        model = structure[0]
        ligands = get_ligand_by_name(ligand_name, model, chains=chains)
        if ligands:
            first = sorted(ligands)[0]
            ligands = {first: ligands[first]}
            int_chains, sites = active_site(ligands, 6, model)
            save_complex(model, ligands, int_chains, sites, filename)
            data_dir = os.path.dirname(filename)
            down_ligand(code, list(ligands.keys())[0],
                        ligand_name, data_dir, format='sdf')
            down_ligand(code, list(ligands.keys())[0],
                        ligand_name, data_dir, format='mol2')
        else:
            int_chains = model.child_dict.keys()
            ligands, sites = {}, 0
            save_complex(model, ligands, int_chains, sites, filename)
            logger.warning(f'Can not extract ligand from {pdb_file}')
    else:
        int_chains = ''
    return int_chains


def ligands_binding_affinity(pdb, ligands, filter_list=None):
    """
    It takes a PDB ID and a list of ligands and returns a dictionary of ligands and their binding
    affinities
    
    :param pdb: The PDB ID of the structure you want to get the ligands for
    :param ligands: list of ligands to get binding affinity for
    :param filter_list: a list of ligands to filter for. If you want to get the binding affinity for all
    ligands, leave this as None
    :return: A dictionary of ligands and their binding affinity.
    """
    base_url = 'https://data.rcsb.org/rest/v1/core/entry/'
    pdb_info = get_dict(f'{base_url}{pdb}')
    if pdb_info:
        with contextlib.suppress(Exception):
            #ligands = pdb_info['rcsb_entry_info']['nonpolymer_bound_components']
            ligands += pdb_info['rcsb_entry_info']['nonpolymer_bound_components']
    # act_types = ['Kd', 'Ki', 'IC50', 'EC50',
    #              '&Delta;G', '-T&Delta;S', '&Delta;H', 'Ka']
    afd = {ligand: {} for ligand in ligands}
    if pdb_info and 'rcsb_binding_affinity' in pdb_info:
        #afd = {ligand:{} for ligand in ligands}
        for af in pdb_info['rcsb_binding_affinity']:
            #assert af['unit'] == 'nM'
            if filter_list and af['comp_id'] not in afd and af['comp_id'] in filter_list:
                afd[af['comp_id']] = {}
            #if af['unit'] in ['nM', 'kJ/mol', 'M^-1'] and (af['type'] in act_types):
            if af['comp_id'] in afd:
                try:
                    afd[af['comp_id']][af['type']].append(af['value'])
                except Exception:
                    afd[af['comp_id']][af['type']] = [af['value']]
    for k, v in afd.items():
        if v:
            for name, value in v.items():
                afd[k][name] = sum(value)/len(value)
    #if filter:
    #    for i in list(afd.keys()):
    #        if filter_res_name(i, model=True, speclal_res=filter_list):
    #            del afd[i]
    return afd


def get_pdb_resolution_rfactor(pdb):
    """
    It takes a PDB ID as input and returns the resolution and R-factor of the structure
    
    :param pdb: The PDB ID of the structure you want to download
    :return: A tuple of the resolution and rfactor
    """
    base_url = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/experiment/'
    pdb_info = get_dict(f'{base_url}{pdb}')
    try:
        res = pdb_info[pdb][0]['resolution']
    except Exception:
        res = ''
    try:
        rf = pdb_info[pdb][0]['r_factor']
    except Exception:
        rf = ''
    
    if not res:
        base_url = 'https://data.rcsb.org/rest/v1/core/entry/'
        pdb_info = get_dict(f'{base_url}{pdb}')
        try:
            res = pdb_info['pdbx_vrpt_summary']['pdbresolution']
        except Exception:
            res = ''

    if not rf:
        base_url = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/electron_density_statistics/'
        pdb_info = get_dict(f'{base_url}{pdb}')
        try:
            rf = pdb_info[pdb][0]['r_factor']
        except Exception:
            rf = ''
    return (res,rf)


def get_pdb_ec(pdb,chain = None):
    """
    It takes a dictionary of PDB information, and returns the EC number of the enzyme that the PDB
    structure is of
    
    :param pdb_info: a dictionary containing the PDB information
    :return: The EC number of the pdb.
    """
    base_url = 'https://www.ebi.ac.uk/pdbe/api/mappings/ec/'
    pdb_info = get_dict(f'{base_url}{pdb}')
    try:
        ecs = pdb_info[pdb]['EC']
        if chain:
            ec = ''
            for i,j in ecs.items():
                ec_chain = j['mappings'][0]['chain_id']
                if chain == ec_chain:
                    ec = i
        else:
            ec = list(ecs.keys())[0]
    except Exception:
        ec = ''
    if not ec and not chain:
        try:
            ec = pdb_info['struct']['pdbx_descriptor'].split('(')[1].rstrip(')')
            m = r'\((E?\.?C?\.?\d\..{1,2}\..{1,2}\..{1,3})\)'
            ec = re.search(re.compile(m), ec)[1]
        except Exception:
            ec = ''
    return ec


def get_pdb_release_date(pdb):
    """
    It gets the release date of a PDB structure from the PDBe API, and if that fails, it gets it from
    the RCSB API
    
    :param pdb: The PDB ID of the structure you want to download
    :return: the release date(YYYY-MM-DD).
    """
    base_url = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/'
    pdb_info = get_dict(f'{base_url}{pdb}')
    try:
        rd = pdb_info[pdb][0]['release_date']
        rd = f'{rd[:4]}-{rd[4:6]}-{rd[6:]}'
    except Exception:
        rd = ''
    if not rd:
        base_url = 'https://data.rcsb.org/rest/v1/core/entry/'
        pdb_info = get_dict(f'{base_url}{pdb}')
        try:
            rd = pdb_info['rcsb_accession_info']['initial_release_date'].split('T')[0]
        except Exception:
            rd = ''
    return rd


def check_files(pdb, ligands, data_dir):
    """
    It checks if the files are present in the directory and if not, it downloads the pdb file and
    removes the ligands from it
    
    :param pdb: The PDB ID of the protein-ligand complex
    :param ligands: a list of ligands to be downloaded
    :param data_dir: the directory where the data is stored
    :return: A boolean value.
    """
    for ligand in ligands:
        file_dir = os.path.join(data_dir, 'structure', f'{pdb}_{ligand}')
        filename = os.path.join(file_dir, f'{pdb}')
        if non_empty_file(os.path.join(file_dir, 'pdb.csv')) and (non_empty_file(f'{filename}_ligand.sdf') or non_empty_file(f'{filename}_ligand.mol2')):
            if not non_empty_file(f'{filename}_protein.pdb'):
                tmp_dir = os.path.join(data_dir, 'tmp')
                file = down_pdb_format(pdb, tmp_dir, 'one')
                if file == 'pdb':
                    try:
                        pymol_rm_ligands(os.path.join(
                            tmp_dir, f'{pdb}.pdb'), f'{filename}_protein.pdb')
                    except Exception:
                        pass
                elif file == 'cif':
                    try:
                        pymol_cif_pdb(os.path.join(
                            tmp_dir, f'{pdb}.cif'), f'{filename}_protein.pdb', True)
                    except Exception:
                        pass
                else:
                    pass
        else:
            return False
    return True


def down_extract(pdb, ligands, data_dir=None, filter_list=None):
    """
    It downloads the PDB file, extracts ligands, and saves the ligands' binding affinity, resolution,
    EC number, and release date
    
    :param pdb: The PDB ID of the structure you want to download
    :param ligands: a list of ligands to extract from the PDB file
    :param data_dir: the directory where the data will be saved
    :param filter_list: a list of ligands to filter out
    :return: the ligand binding affinity for the ligands in the ligands list.
    """
    data_dir = dir_cwd(data_dir)
    #all_info = get_info(pdb)
    #logger.info(f'{pdb} START\n')
    if not check_files(pdb, ligands, data_dir):
        #base_url = 'https://data.rcsb.org/rest/v1/core/entry/'
        #all_info = get_dict(f'{base_url}{pdb}')

        res, rf = get_pdb_resolution_rfactor(pdb)
        ec = get_pdb_ec(pdb)
        rd = get_pdb_release_date(pdb)
        afd = ligands_binding_affinity(pdb, ligands, filter_list)

        #wlines = []
        #if afd:

        tmp_dir = os.path.join(data_dir, 'tmp')
        check_dir(tmp_dir)

        structure_dir = os.path.join(data_dir, 'structure')
        check_dir(structure_dir)

        pdb_file = os.path.join(tmp_dir, f'{pdb}.pdb')
        #if not os.path.exists(pdb_file):
        #    down_pdb(pdb, tmp_dir)
        for ligand, afs in afd.items():
            chain = ligand_chain(pdb, ligand)
            line = f'{pdb},{ligand},{chain}'

            for af_type in RCSB_AFFINITY_TYPES:
                try:
                    line += f',{afs[af_type]:.2f}'
                except KeyError:
                    line += ','
            
            #wlines.append(f'{line},{res},{ec},{rd}\n')
            ligand_dir = os.path.join(structure_dir, f'{pdb}_{ligand}')
            check_dir(ligand_dir)

            chains = [chain] if chain else None
            int_chains = save_first_ligand(
                pdb_file, ligand, os.path.join(ligand_dir, pdb), chains=chains)
            #wlines.append(f'{line},{res},{ec},{rd},{int_chains}\n')
            if int_chains:
                int_chains = '_'.join(int_chains)
            info_file = os.path.join(ligand_dir, 'pdb.csv')
            if not non_empty_file(info_file):
                write_list(info_file,
                           f'{line},{res},{ec},{rd},{rf},{int_chains}\n')
        #if os.path.exists(pdb_file):
        #    os.remove(pdb_file)

        #else:
        #    #wlines.append(f'{pdb},,,,,,,,,,,{res},{ec},{rd},\n')
        #    write_list(os.path.join(data_dir, 'pdb.csv'),
        #               f'{pdb},,,,,,,,,,,{res},{ec},{rd},\n', 'a+')
    with open('done.log','a+') as f:
        f.write(f'{pdb}\n')
    logger.info(f'{pdb} DONE')
    return afd


def write_pdb_files(pdb_dict, data_dir=None, filter_list=None, num_cpus: int = 1, rank: int = 0, size: int = 1):
    """
    `write_files` takes a list of PDB IDs, a directory to write the files to, and a number of CPUs to
    use, and writes a CSV file containing the PDB ID, the chain ID, and the sequence of the chain
    
    :param pdb_list: a list of pdb ids
    :param data_dir: the directory where you want to store the data
    :param num_cpus: number of cpus to use, defaults to 1
    :type num_cpus: int (optional)
    :param rank: the rank of the node in the cluster, defaults to 0
    :type rank: int (optional)
    :param size: the number of nodes in the cluster, defaults to 1
    :type size: int (optional)
    :return: A list of lines to be written to a file.
    """
    data_dir = dir_cwd(data_dir)
    rank = get_global_node_rank(rank, size)
    if rank == 0:
        inputs = [(pdb, ligands, data_dir, filter_list)
                  for pdb, ligands in pdb_dict.items()]
        logger.info(f"There are {len(pdb_dict)} pdbs to process")
        par.submit_jobs(down_extract, inputs, num_cpus)
        #wlines = []
        #for i in outs:
        #    wlines += i
        #with open(os.path.join(data_dir, 'pdb.csv'), 'w') as f:
        #    f.writelines(wlines)
    return

# TODO ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_pfam.tsv.gz
def pfam_uniport_pdbchain_dict(pfam_file):
    """
    > This function takes a file containing a mapping of Pfam domains to PDB chains and returns a
    dictionary with the Pfam domain as the key and the PDB chain as the value
    
    :param pfam_file: the file containing the pfam-pdb mappings
    :return: A dictionary with the key being a tuple of the pdb id and chain id and the value being a
    tuple of the pfam id and the uniprot id.
    """
    url = 'http://ftp.ebi.ac.uk/pub/databases/Pfam/mappings/pdb_pfam_mapping.txt'
    if check_download(pfam_file,url):
        pfam_dict = {}
        for line in split_file(pfam_file)[1:]:
            try:
                pf = line[3]
            except:
                pf = ''
            try:
                un = line[9]
            except:
                un = ''
            pfam_dict[(pf,un)] = (pf,un)
        return pfam_dict
        #return {(line[0],line[1]):(line[3],line[9]) for line in split_file(pfam_file)[1:]}
    else: 
        return {}


def sifts_pdbchain_uniport_dict(uniport_file):
    if not non_empty_file(uniport_file):
        url = 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz'
        tmp_file = os.path.join(os.path.dirname(uniport_file),'pdb_chain_uniprot.csv.gz')
        if check_download(tmp_file,url):
            ungz(tmp_file,uniport_file)
    if non_empty_file(uniport_file):
        return {(line[0],line[1]):line[2] for line in split_file(uniport_file,sep=',')[2:]}
    else:
        return {}


def uniport_pdbchain_dict(uniport_map_file):
    lines = split_file(uniport_map_file, sep='\t')
    return {(i[0],i[1]):i[2].split() for i in lines}


def pdbchain_uniport(pdb, chain, uniprot_dict, pfam_dict, sifts_dict):
    uniports = []
    with contextlib.suppress(Exception):
        uniports.extend(uniprot_dict[(pdb, chain)])
    with contextlib.suppress(Exception):
        s = sifts_dict[(pdb, chain)]
        if s not in uniports: 
            uniports.append(s)
    with contextlib.suppress(Exception):
        p = pfam_dict[(pdb, chain)]
        if p not in uniports: 
            uniports.append(p)
    return uniports


def uniprot_chembl_dict(chembl_file):
    url = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_uniprot_mapping.txt'
    if check_download(chembl_file,url):
        return {line[0]:line[1] for line in split_file(chembl_file,sep='\t')[1:]}
    else:
        return {}


def uniprot_chembl(uniport, chembl_dict):
    with contextlib.suppress(Exception):
        return chembl_dict[uniport]


