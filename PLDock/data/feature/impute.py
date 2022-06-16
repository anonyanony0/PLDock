"""
Source code (MIT-Licensed) inspired by DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus)
"""
import os
from pathlib import Path
from PLDock import log
from parallel import submit_jobs
from PLDock.utils.utils import get_global_node_rank, impute_missing_feature_values

def impute(output_dir: str, impute_atom_features: bool=False, advanced_logging: bool=False, num_cpus: int=1, rank: int=0, size: int=1):
    """Impute missing feature values."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = log(os.path.join(output_dir, 'impute.log'))
        logger.info('Imputing missing feature values for given dataset')

        # Make sure the output_dir exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Collect thread inputs
        inputs = [(pair_filename.as_posix(), pair_filename.as_posix(), impute_atom_features, advanced_logging)
                  for pair_filename in Path(output_dir).rglob('*.dill')]
        # Without impute_atom_features set to True, non-CA atoms will be filtered out after writing updated pairs
        submit_jobs(impute_missing_feature_values, inputs, num_cpus)


if __name__ == '__main__':
    impute()
