"""
Source code (MIT-Licensed) inspired by DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus)
"""
import os
from PLDock import log
import PLDock.utils.atom3p.conservation as con
from PLDock.utils.utils import get_global_node_rank,get_psaia_dir,write_psaia_config

#def psaia_feature(psaia_dir: str, psaia_config: str, pdb_dataset: str, pkl_dataset: str,
#                  pruned_dataset: str, output_dir: str, source_type: str = 'pdbbind', rank: int = 0, size: int = 1):
def psaia_feature(pdb_dataset: str, precess_dir: str, output_dir: str, 
                  source_type: str = 'pdbbind', rank: int = 0, size: int = 1):
    """Run external programs for feature generation to turn raw PDB files into sequence or 
            structure-based residue features (saved in ../interim/external_feats by default).
        source_type: ['rcsb', 'db5', 'evcoupling', 'casp_capri','pdbbind']
    """
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:

        pkl_dataset = precess_dir
        pruned_dataset = precess_dir
        
        logger = log(os.path.join(output_dir, 'psaia_feature.log'))
        logger.info(
            f'Generating PSAIA features from PDB files in {pkl_dataset}')

        psaia_config = os.path.join(output_dir, source_type+'.txt')
        psaia_dir = get_psaia_dir()
        write_psaia_config(psaia_config, psaia_dir)
 
        # Generate protrusion indices
        con.map_all_protrusion_indices(psaia_dir, psaia_config, pdb_dataset, pkl_dataset,
                                        pruned_dataset, output_dir, source_type)

