import os
import dill
from PLDock import log
from PLDock.data.feature.featurer import feature_adir

logger = log()

def data_feature(input_dir, output_dir, hhsuite_db, num_cpus):
    """
    It takes in a directory of PDB files, and outputs a directory of feature files
    
    :param input_dir: the directory containing the PDB files
    :param output_dir: the directory where the output files will be written
    :param hhsuite_db: the path to the hhsuite database
    :param num_cpus: number of CPUs to use
    """
    """
    It takes in a directory of PDB files, and outputs a directory of feature files
    
    :param input_dir: the directory containing the PDB files
    :param output_dir: the directory where the output files will be written
    :param hhsuite_db: the path to the hhsuite database
    :param num_cpus: number of CPUs to use
    """
    precess_dir = os.path.join(output_dir, 'parsed')
    psaia_dir = os.path.join(output_dir, 'external_feats')
    hhsuite_dir = os.path.join(
        output_dir, 'external_feats', 'HHSUITE', 'PDBBIND')
    pruned_pairs_dir = os.path.join(output_dir, 'pairs')
    post_dir = os.path.join(output_dir, 'final', 'raw')
    feature_adir(input_dir, output_dir, precess_dir, psaia_dir,
                 hhsuite_dir, pruned_pairs_dir, post_dir, hhsuite_db, num_cpus)
    logger.info('All done')

def dataloader(data_dir:str,data_list:str=None):
    """
    It takes a directory and returns a list of all the files in that directory
    
    :param data_dir: the directory where the data is stored
    :return: A list of dictionaries.
    """
    results = []
    if data_list is None:
        data_list = os.listdir(data_dir)
    for i in data_list:
        with open(os.path.join(data_dir,i),'rb') as f:
            results.append(dill.load(f))
    return results


