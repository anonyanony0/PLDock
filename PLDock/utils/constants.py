## Inspired by: https://github.com/BioinfoMachineLearning/DIPS-Plus
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder

# Cluster-specific limit to the number of compute nodes available to each Slurm job
MAX_NODES_PER_JOB = 4

# Dataset-global node count limits to restrict computational learning complexity
ATOM_COUNT_LIMIT = 17500  # Default filter for both datasets when encoding complexes at an atom-based level

# From where we can get bound PDB complexes
RCSB_BASE_URL = 'ftp://ftp.wwpdb.org/pub/pdb/data/biounit/coordinates/divided/'

# The PDB codes of structures added between DB4 and DB5 (to be used for testing dataset)
DB5_TEST_PDB_CODES = ['3R9A', '4GAM', '3AAA', '4H03', '1EXB',
                      '2GAF', '2GTP', '3RVW', '3SZK', '4IZ7',
                      '4GXU', '3BX7', '2YVJ', '3V6Z', '1M27',
                      '4FQI', '4G6J', '3BIW', '3PC8', '3HI6',
                      '2X9A', '3HMX', '2W9E', '4G6M', '3LVK',
                      '1JTD', '3H2V', '4DN4', 'BP57', '3L5W',
                      '3A4S', 'CP57', '3DAW', '3VLB', '3K75',
                      '2VXT', '3G6D', '3EO1', '4JCV', '4HX3',
                      '3F1P', '3AAD', '3EOA', '3MXW', '3L89',
                      '4M76', 'BAAD', '4FZA', '4LW4', '1RKE',
                      '3FN1', '3S9D', '3H11', '2A1A', '3P57']

# Postprocessing logger dictionary
DEFAULT_DATASET_STATISTICS = dict(num_of_processed_complexes=0, num_of_df0_residues=0, num_of_df1_residues=0,
                                  num_of_df0_interface_residues=0, num_of_df1_interface_residues=0,
                                  num_of_pos_res_pairs=0, num_of_neg_res_pairs=0, num_of_res_pairs=0,
                                  num_of_valid_df0_ss_values=0, num_of_valid_df1_ss_values=0,
                                  num_of_valid_df0_rsa_values=0, num_of_valid_df1_rsa_values=0,
                                  num_of_valid_df0_rd_values=0, num_of_valid_df1_rd_values=0,
                                  num_of_valid_df0_protrusion_indices=0, num_of_valid_df1_protrusion_indices=0,
                                  num_of_valid_df0_hsaacs=0, num_of_valid_df1_hsaacs=0,
                                  num_of_valid_df0_cn_values=0, num_of_valid_df1_cn_values=0,
                                  num_of_valid_df0_sequence_feats=0, num_of_valid_df1_sequence_feats=0,
                                  num_of_valid_df0_amide_normal_vecs=0, num_of_valid_df1_amide_normal_vecs=0)

# Parsing utilities for PDB files (i.e. relevant for sequence and structure analysis)
PDB_PARSER = PDBParser()
CA_PP_BUILDER = CaPPBuilder()

# Dict for converting three letter codes to one letter codes
D3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
RES_NAMES_LIST = list(D3TO1.keys())

# PSAIA features to encode as DataFrame columns
PSAIA_COLUMNS = ['avg_cx', 's_avg_cx', 's_ch_avg_cx', 's_ch_s_avg_cx', 'max_cx', 'min_cx']

# Constants for calculating half sphere exposure statistics
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY-'
AMINO_ACID_IDX = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

# Default fill values for missing features
HSAAC_DIM = 42  # We have 2 + (2 * 20) HSAAC values from the two instances of the unknown residue symbol '-'
DEFAULT_MISSING_FEAT_VALUE = np.nan
DEFAULT_MISSING_SS = '-'
DEFAULT_MISSING_RSA = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_RD = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_PROTRUSION_INDEX = [DEFAULT_MISSING_FEAT_VALUE for _ in range(6)]
DEFAULT_MISSING_HSAAC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(HSAAC_DIM)]
DEFAULT_MISSING_CN = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_SEQUENCE_FEATS = np.array([DEFAULT_MISSING_FEAT_VALUE for _ in range(27)])
DEFAULT_MISSING_NORM_VEC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(3)]

# Default number of NaN values allowed in a specific column before imputing missing features of the column with zero
NUM_ALLOWABLE_NANS = 5

# Features to be one-hot encoded during graph processing and what their values could be
FEAT_COLS = [
    # 'resname',  # By default, leave out one-hot encoding of residues' type to decrease feature redundancy
    'ss_value',
    'rsa_value',
    'rd_value'
]
FEAT_COLS.extend(
    PSAIA_COLUMNS +
    ['hsaac',
     'cn_value',
     'sequence_feats',
     'amide_norm_vec',
     # 'element'  # For atom-level learning only
     ])

ALLOWABLE_FEATS = [
    # By default, leave out one-hot encoding of residues' type to decrease feature redundancy
    # ["TRP", "PHE", "LYS", "PRO", "ASP", "ALA", "ARG", "CYS", "VAL", "THR",
    #  "GLY", "SER", "HIS", "LEU", "GLU", "TYR", "ILE", "ASN", "MET", "GLN"],
    ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'],  # Populated 1D list means restrict column feature values by list values
    [],  # Empty list means take scalar value as is
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [[]],  # Doubly-nested, empty list means take first-level nested list as is
    [],
    [[]],
    [[]],
    # ['C', 'O', 'N', 'S']  # For atom-level learning only
]


#SPECIAL_RES = ['HOH','CA','HG','K','NA','ZN','MG','CL','SO4','IOD','GOL','PO4',
#               'CU','MN','CO','SM','CU1','NO3','AZI','HG','OH','CD','NI','FMT',
#               'AF3','CMO','F3S','SF4','FE','CO3','O','PB','AG','NO','WO4','FE2',
#               'PT']

PDB_LIGAND_FORMULA_URL = \
    'http://ligand-expo.rcsb.org/dictionaries/cc-counts-extra.tdd'
LIGAND_PDB_URL = 'http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd'
STANDARD_UNITS = ['M','nM','uM','mM','pM','fM']
# 
BIOLIP_ARTIFACT = ['ACE', 'HEX', 'TMA', 'SOH', 'P25', 'CCN', 'PR', 'PTN',
                   'NO3', 'TCN', 'BU1', 'BCN', 'CB3', 'HCS', 'NBN', 'SO2',
                   'MO6', 'MOH', 'CAC', 'MLT', 'KR', '6PH', 'MOS', 'UNL',
                   'MO3', 'SR', 'CD3', 'PB', 'ACM', 'LUT', 'PMS', 'OF3',
                   'SCN', 'DHB', 'E4N', '13P', '3PG', 'CYC', 'NC', 'BEN',
                   'NAO', 'PHQ', 'EPE', 'BME', 'TB', 'ETE', 'EU', 'OES',
                   'EAP', 'ETX', 'BEZ', '5AD', 'OC2', 'OLA', 'GD3', 'CIT',
                   'DVT', 'OC6', 'MW1', 'OC3', 'SRT', 'LCO', 'BNZ', 'PPV',
                   'STE', 'PEG', 'RU', 'PGE', 'MPO', 'B3P', 'OGA', 'IPA',
                   'LU', 'EDO', 'MAC', '9PE', 'IPH', 'MBN', 'C1O', '1PE',
                   'YF3', 'PEF', 'GD', '8PE', 'DKA', 'RB', 'YB', 'GGD',
                   'SE4', 'LHG', 'SMO', 'DGD', 'CMO', 'MLI', 'MW2', 'DTT',
                   'DOD', '7PH', 'PBM', 'AU', 'FOR', 'PSC', 'TG1', 'KAI',
                   '1PG', 'DGA', 'IR', 'PE4', 'VO4', 'ACN', 'AG', 'MO4',
                   'OCL', '6UL', 'CHT', 'RHD', 'CPS', 'IR3', 'OC4', 'MTE',
                   'HGC', 'CR', 'PC1', 'HC4', 'TEA', 'BOG', 'PEO', 'PE5',
                   '144', 'IUM', 'LMG', 'SQU', 'MMC', 'GOL', 'NVP', 'AU3',
                   '3PH', 'PT4', 'PGO', 'ICT', 'OCM', 'BCR', 'PG4', 'L4P',
                   'OPC', 'OXM', 'SQD', 'PQ9', 'BAM', 'PI', 'PL9', 'P6G',
                   'IRI', '15P', 'MAE', 'MBO', 'FMT', 'L1P', 'DUD', 'PGV',
                   'CD1', 'P33', 'DTU', 'XAT', 'CD', 'THE', 'U1', 'NA',
                   'MW3', 'BHG', 'Y1', 'OCT', 'BET', 'MPD', 'HTO', 'IBM',
                   'D01', 'HAI', 'HED', 'CAD', 'CUZ', 'TLA', 'SO4', 'OC5',
                   'ETF', 'MRD', 'PT', 'PHB', 'URE', 'MLA', 'TGL', 'PLM',
                   'NET', 'LAC', 'AUC', 'UNX', 'GA', 'DMS', 'MO2', 'LA',
                   'NI', 'TE', 'THJ', 'NHE', 'HAE', 'MO1', 'DAO', '3PE',
                   'LMU', 'DHJ', 'FLC', 'SAL', 'GAI', 'ORO', 'HEZ', 'TAM',
                   'TRA', 'NEX', 'CXS', 'LCP', 'HOH', 'OCN', 'PER', 'ACY',
                   'MH2', 'ARS', '12P', 'L3P', 'PUT', 'IN', 'CS', 'NAW',
                   'SB', 'GUN', 'SX', 'CON', 'C2O', 'EMC', 'BO4', 'BNG',
                   'MN5', '__O', 'K', 'CYN', 'H2S', 'MH3', 'YT3', 'P22',
                   'KO4', '1AG', 'CE', 'IPL', 'PG6', 'MO5', 'F09', 'HO',
                   'AL', 'TRS', 'EOH', 'GCP', 'MSE', 'AKR', 'NCO', 'PO4',
                   'L2P', 'LDA', 'SIN', 'DMI', 'SM', 'DTD', 'SGM', 'DIO',
                   'PPI', 'DDQ', 'DPO', 'HCA', 'CO5', 'PD', 'OS', 'OH',
                   'NA6', 'NAG', 'W', 'ENC', 'NA5', 'LI1', 'P4C', 'GLV',
                   'DMF', 'ACT', 'BTB', '6PL', 'BGL', 'OF1', 'N8E', 'LMT',
                   'THM', 'EU3', 'PGR', 'NA2', 'FOL', '543', '_CP', 'PEK',
                   'NSP', 'PEE', 'OCO', 'CHD', 'CO2', 'TBU', 'UMQ', 'MES',
                   'NH4', 'CD5', 'HTG', 'DEP', 'OC1', 'KDO', '2PE', 'PE3',
                   'IOD', 'NDG', 'CL', 'HG', 'F', 'XE', 'TL', 'BA', 'LI',
                   'BR', 'TAU', 'TCA', 'SPD', 'SPM', 'SAR', 'SUC', 'PAM',
                   'SPH', 'BE7', 'P4G', 'OLC', 'OLB', 'LFA', 'D10', 'D12',
                   'DD9', 'HP6', 'R16', 'PX4', 'TRD', 'UND', 'FTT', 'MYR',
                   'RG1', 'IMD', 'DMN', 'KEN', 'C14', 'UPL', 'CMJ', 'ULI',
                   'MYS', 'TWT', 'M2M', 'P15', 'PG0', 'PEU', 'AE3', 'TOE',
                   'ME2', 'PE8', '6JZ', '7PE', 'P3G', '7PG', 'PG5', '16P',
                   'XPE', 'PGF', 'AE4', '7E8', '7E9', 'MVC', 'TAR', 'DMR',
                   'LMR', 'NER', '02U', 'NGZ', 'LXB', 'A2G', 'BM3', 'NAA',
                   'NGA', 'LXZ', 'PX6', 'PA8', 'LPP', 'PX2', 'MYY', 'PX8',
                   'PD7', 'XP4', 'XPA', 'PEV', '6PE', 'PEX', 'PEH', 'PTY',
                   'YB2', 'PGT', 'CN3', 'AGA', 'DGG', 'CD4', 'CN6', 'CDL',
                   'PG8', 'MGE', 'DTV', 'L44', 'L2C', '4AG', 'B3H', '1EM',
                   'DDR', 'I42', 'CNS', 'PC7', 'HGP', 'PC8', 'HGX', 'LIO',
                   'PLD', 'PC2', 'PCF', 'MC3', 'P1O', 'PLC', 'PC6', 'HSH',
                   'BXC', 'HSG', 'DPG', '2DP', 'POV', 'PCW', 'GVT', 'CE9',
                   'CXE', 'C10', 'CE1', 'SPJ', 'SPZ', 'SPK', 'SPW', 'HT3',
                   'HTH', '2OP', '3NI', 'BO3', 'DET', 'D1D', 'SWE', 'SOG']
BIOLIP_METAL = ['LA', 'NI', '3CO', 'K', 'CR', 'ZN', 'CD', 'PD', 'TB', 'YT3',
                'OS', 'EU', 'NA', 'RB', 'W', 'YB', 'HO3', 'CE', 'MN', 'TL',
                'LI', 'MN3', 'AU3', 'AU', 'EU3', 'AL', '3NI', 'FE2', 'PT',
                'FE', 'CA', 'AG', 'CU1', 'LU', 'HG', 'CO', 'SR', 'MG', 'PB',
                'CS', 'GA', 'BA', 'SM', 'SB', 'CU', 'MO', 'CU2']
BIOLIP_K_METR = ['UUU']
BIOLIP_DNA_RNA = ['NUC']
BIOLIP_DNA_PEPTIDE = ['III']
ION = ['NO3', 'PO4', 'OH', 'CL', 'SO4', 'CO3', 'BR', 'IOD', 'AZI', 'WO4',
       'SEK', 'NO2', 'I3M','BCT', 'VXA', 'ALF', 'BEF', 'MOO','FCO', 'AST',
       'SO3', '2HP']
METAL = ['RU']
INORGANIC = ['NO', 'O', 'FMT', 'AF3', 'CMO', 'F3S', 'SF4', 'HOH',  'BO4',
             'F', 'XE', 'SO2', 'FR', 'H2S', 'PO2', 'VN4', '6BP', 'CQ4']
LONG = ['12P', 'DAO', 'PLM', 'F09', '16P', 'R16', 'DKA']
SMALL = ['GOL', 'EDO', 'MAE', 'LAC', 'PPI', 'DMI', 'OXM', 'TLA', 'PIH',
        'ZN6', '1MZ', 'OXL', 'TFA', '2T8', 'VX']
COVALENT = ['13P', 'TAM', 'NAG', 'KDO', 'A2G', '3PG', 'EPE', 'LAC', 'PPI',
            'TLA', 'NGA', 'TAR', 'DKA']
POLYMER = ['NAG', 'LAC', 'PPI', 'DKA']
ADDITIVE = ['PIH']
UNKWOWN = ['NVP', 'KAI', 'FOL', 'CB3']
SPECIAL_RES = set(BIOLIP_METAL + ION + INORGANIC + BIOLIP_DNA_PEPTIDE + BIOLIP_K_METR + BIOLIP_ARTIFACT + BIOLIP_DNA_RNA)
# A list of the different affinity types that are used in the RCSB database.
RCSB_AFFINITY_TYPES = ['Ki', 'Kd', 'IC50', 'EC50',
                       '&Delta;G', '-T&Delta;S', '&Delta;H', 'Ka']
# Cluster-specific limit to the number of compute nodes available
# to each Slurm job
MAX_NODES_PER_JOB = 4
