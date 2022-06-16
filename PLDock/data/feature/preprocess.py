"""
Source code (MIT-Licensed) inspired by DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus)
"""
import os
from PLDock import log
from PLDock.utils.atom3p import complex as comp
from PLDock.utils.atom3p import neighbors as nb
from PLDock.utils.atom3p import pair as pair
from PLDock.utils.atom3p import parse as pa
from PLDock.utils.utils import get_global_node_rank


def preprocess(input_dir: str, output_dir: str, num_cpus: int = 1, rank: int = 0, size: int = 1,
               neighbor_def: str = 'non_heavy_res', cutoff: int = 6, source_type: str = 'pdbbind', unbound: bool = False):
    """Run data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../interim).
    For reference, pos_idx indicates the IDs of residues in interaction with non-heavy atoms in a cross-protein residue.
        neighbor_def: ['non_heavy_res', 'non_heavy_atom', 'ca_res', 'ca_atom']
        source_type: ['rcsb', 'db5', 'evcoupling', 'casp_capri','pdbbind']
        pos_idx:  IDs of residues in interaction with non-heavy atoms in a cross-protein residue

    """
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = log(os.path.join(output_dir, 'preprocess.log'))
        logger.info('Making interim data set from raw data')


        parsed_dir = os.path.join(output_dir, 'parsed')
        pa.parse_all(input_dir, parsed_dir, num_cpus)

        # if complexes.dill not exist, generate it. If exist, ignore
        complexes_dill = os.path.join(output_dir, 'complexes/complexes.dill')
        comp.complexes(parsed_dir, complexes_dill, source_type)

        pairs_dir = os.path.join(output_dir, 'pairs')
        get_neighbors = nb.build_get_neighbors(neighbor_def, cutoff)
        get_pairs = pair.build_get_pairs(
            neighbor_def, source_type, unbound, get_neighbors, False)
        complexes = comp.read_complexes(complexes_dill)
        pair.all_complex_to_pairs(
            complexes, source_type, get_pairs, pairs_dir, num_cpus)
        logger.info('DONE')

if __name__ == '__main__':
    input_dir = '/root/workshop/docking/PLDock/data/CASF-2016/coreset'
    output_dir = input_dir
    preprocess(input_dir, output_dir)
