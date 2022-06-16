"""
Source code (MIT-Licensed) inspired by DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus)
"""
import os
from mpi4py import MPI
from parallel import submit_jobs
from PLDock import log
from PLDock.utils.utils import get_global_node_rank
from PLDock.utils.utils import postprocess_pruned_pairs
from PLDock.utils.atom3p.database import get_structures_filenames, get_pdb_name, get_pdb_code
from PLDock.utils.atom3p.utils import slice_list

def postprocess(raw_pdb_dir: str, pruned_pairs_dir: str, external_feats_dir: str, output_dir: str,
         num_cpus: int=1, rank: int=0, size: int=1, source_type: str='pdbbind'):
    """Run postprocess_pruned_pairs on all provided complexes.
        source_type: ['rcsb', 'db5', 'evcoupling', 'casp_capri','pdbbind']
    """
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)
    logger = log(os.path.join(output_dir, 'psaia_feature.log'))
    logger.info(f'Starting postprocessing for node {rank + 1} out of a global MPI world of size {size},'
                f' with a local MPI world size of {MPI.COMM_WORLD.Get_size()}')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get work filenames
    logger.info(f'Looking for all pairs in {pruned_pairs_dir}')
    requested_filenames = get_structures_filenames(pruned_pairs_dir, extension='.dill')
    requested_filenames = [filename for filename in requested_filenames]
    requested_keys = [get_pdb_name(x) for x in requested_filenames]
    #requested_keys = [get_pdb_name(x,False) for x in requested_filenames]
    produced_filenames = get_structures_filenames(output_dir, extension='.dill')
    produced_keys = [get_pdb_name(x) for x in produced_filenames]
    #produced_keys = [get_pdb_name(x,False) for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    rscb_pruned_pair_ext = '.dill' if source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri'] else ''
    work_filenames = [os.path.join(pruned_pairs_dir, get_pdb_code(work_key)[1:3], work_key + rscb_pruned_pair_ext)
                      for work_key in work_keys]
    logger.info(f'Found {len(work_keys)} work pair(s) in {pruned_pairs_dir}')

    # Reserve an equally-sized portion of the full work load for a given rank in the MPI world
    work_filenames = list(set(work_filenames))  # Remove any duplicate filenames
    work_filename_rank_batches = slice_list(work_filenames, size)
    work_filenames = work_filename_rank_batches[rank]

    # Get filenames in which our threads will store output
    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        new_output_filename = sub_dir + '/' + get_pdb_name(pdb_filename) + ".dill" if \
            source_type in ['rcsb', 'evcoupling', 'casp_capri'] else \
            sub_dir + '/' + get_pdb_name(pdb_filename)
        output_filenames.append(new_output_filename)

    # Collect thread inputs
    inputs = [(raw_pdb_dir, external_feats_dir, i, o, source_type)
              for i, o in zip(work_filenames, output_filenames)]
    submit_jobs(postprocess_pruned_pairs, inputs, num_cpus)


if __name__ == '__main__':
    main()
