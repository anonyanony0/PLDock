"""
Source code (MIT-Licensed) inspired by DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus)
"""

import os
from os import cpu_count
from PLDock import log
from PLDock.utils.atom3p.conservation import map_all_profile_hmms
from mpi4py import MPI
from PLDock.utils.utils import get_global_node_rank

def hhsuite_feature(pruned_dataset: str, output_dir: str, hhsuite_db: dir, rank: int = 0,
                    size: int = 1, num_cpu_jobs: int = cpu_count() // 2, num_cpus_per_job: int = 2,
                    num_iter: int = 2, source_type: str = 'pdbbind'):
    #               size: int = 1, num_cpu_jobs: int = cpu_count() // 2, num_cpus_per_job: int = 2, num_iter: int = 2, source_type: str = 'pdbbind', write_file: bool = True):
    """Run external programs for feature generation to turn raw PDB files from (../raw) into sequence or structure-based residue features (saved in ../interim/external_feats by default).
        source_type: ['rcsb', 'db5', 'evcoupling', 'casp_capri','pdbbind']
    """
    
    pkl_dataset = pruned_dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = log(os.path.join(output_dir, 'hhsuite_feature.log'))
    logger.info(
        f'Generating external features from PDB files in {pkl_dataset}')

    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    if 'dbCAN' in hhsuite_db:
        hhsuite_db = os.path.join(hhsuite_db, 'dbCAN-fam-V9')
    elif 'bfd' in hhsuite_db:
        # Determine true rank and size for a given node
        bfd_copy_ids = ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8",
                        "_9", "_10", "_11", "_12", "_17", "_21", "_25", "_29"]
        bfd_copy_id = bfd_copy_ids[rank]

        # Assemble true ID of the BFD copy to use for generating profile HMMs
        hhsuite_db = os.path.join(hhsuite_db + bfd_copy_id,
                                  'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')
        logger.info(f'Starting HH-suite for node {rank + 1} out of a global MPI world of size {size},'
                    f' with a local MPI world size of {MPI.COMM_WORLD.Get_size()}.'
                    f' This node\'s copy of the BFD is {hhsuite_db}')
    else:
        logging.warning('Can not recognise hhsuite_db')
    # Generate profile HMMs #
    # Run with --write_file=True using one node
    # Then run with --read_file=True using multiple nodes to distribute workload across nodes and their CPU cores
    map_all_profile_hmms(pkl_dataset, pruned_dataset, output_dir, hhsuite_db, num_cpu_jobs,
                         num_cpus_per_job, source_type, num_iter, rank, size, True)
    map_all_profile_hmms(pkl_dataset, pruned_dataset, output_dir, hhsuite_db, num_cpu_jobs,
                         num_cpus_per_job, source_type, num_iter, rank, size, False)


if __name__ == '__main__':
    input_dir = '/root/workshop/docking/PLDock/data/CASF-2016/coreset'
    output_dir = input_dir
    hhsuite_feature(input_dir, output_dir)
