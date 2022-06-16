"""
Source code (MIT-Licensed) inspired by DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus)
"""
import os
import atom3.database as db
import numpy as np
import parallel as par
from PLDock import log
from PLDock.utils.utils import __load_to_keep_files_into_dataframe, process_pairs_to_keep
from PLDock.utils.utils import get_global_node_rank

def prune_pairs(pair_dir: str, to_keep_dir: str, output_dir: str, num_cpus: int=1, rank: int=0, size: int=1):
    """Run write_pairs on all provided complexes.   """
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = log(os.path.join(output_dir, 'prune_pairs.log'))
        to_keep_filenames = \
            db.get_structures_filenames(to_keep_dir, extension='.txt')
        if len(to_keep_filenames) == 0:
            logger.warning(f'There is no to_keep file in {to_keep_dir}.'
                           f' All pair files from {pair_dir} will be copied into {output_dir}')

        to_keep_df, always_keep_set = __load_to_keep_files_into_dataframe(to_keep_filenames)
        logger.info(f'There are {to_keep_df.shape} rows, cols in to_keep_df')

        # if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Get work filenames
        logger.info(f'Looking for all pairs in {pair_dir}')
        work_filenames = db.get_structures_filenames(pair_dir, extension='.dill')
        work_filenames = list(set(work_filenames))  # Remove any duplicate filenames
        work_keys = [db.get_pdb_name(x) for x in work_filenames]
        logger.info(f'Found {len(work_keys)} pairs in {output_dir}')

        # Get filenames in which our threads will store output
        output_filenames = []
        for pdb_filename in work_filenames:
            sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
            # if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
            output_filenames.append(
                sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".dill")

        # Collect thread inputs
        n_copied = 0
        inputs = [(i, o, to_keep_df, always_keep_set) for i, o in zip(work_filenames, output_filenames)]
        n_copied += np.sum(par.submit_jobs(process_pairs_to_keep, inputs, num_cpus))
        logger.info(f'{n_copied} out of {len(work_keys)} pairs was copied')


if __name__ == '__main__':

    prune_pairs()
