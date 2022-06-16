"""
Source code (MIT-Licensed) inspired by deepchem (https://github.com/deepchem/deepchem)
"""

from PLDock.utils.io_tools import read_list,write_list
from PLDock.utils.utils import get_global_node_rank
import oddt
from oddt import fingerprints
import parallel as par
from rdkit import DataStructs
import os
from typing import Any, Dict, List, Iterator, Optional, Sequence, Tuple
from loguru import logger
import numpy as np
import pandas as pd

class FingerprintSplitter():
  """Class for doing data splits based on the Tanimoto similarity between ECFP4
  fingerprints.
  This class tries to split the data such that the molecules in each dataset are
  as different as possible from the ones in the other datasets.  This makes it a
  very stringent test of models.  Predicting the test and validation sets may
  require extrapolating far outside the training data.
  The running time for this splitter scales as O(n^2) in the number of samples.
  Splitting large datasets can take a long time.
  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self):
    """Create a FingerprintSplitter."""
    super(FingerprintSplitter, self).__init__()

  def split(self,
            #dataset: Dataset,
            fps,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None
           ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits compounds into training, validation, and test sets based on the
    Tanimoto similarity of their ECFP4 fingerprints. This splitting algorithm
    has an O(N^2) run time, where N is the number of elements in the dataset.
    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use (ignored since this algorithm is deterministic).
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).
    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import AllChem
    except ModuleNotFoundError:
      raise ImportError("This function requires RDKit to be installed.")
    dataset = fps
    # Compute fingerprints for all molecules.

    #mols = [Chem.MolFromSmiles(smiles) for smiles in dataset.ids]
    #fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    # Split into two groups: training set and everything else.

    train_size = int(frac_train * len(dataset))
    valid_size = int(frac_valid * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_inds, test_valid_inds = _split_fingerprints(fps, train_size,
                                                      valid_size + test_size)

    # Split the second group into validation and test sets.

    if valid_size == 0:
      valid_inds = []
      test_inds = test_valid_inds
    elif test_size == 0:
      test_inds = []
      valid_inds = test_valid_inds
    else:
      test_valid_fps = [fps[i] for i in test_valid_inds]
      test_inds, valid_inds = _split_fingerprints(test_valid_fps, test_size,
                                                  valid_size)
      test_inds = [test_valid_inds[i] for i in test_inds]
      valid_inds = [test_valid_inds[i] for i in valid_inds]
    return train_inds, valid_inds, test_inds


def _split_fingerprints(fps: List, size1: int,
                        size2: int) -> Tuple[List[int], List[int]]:
  """This is called by FingerprintSplitter to divide a list of fingerprints into
  two groups.
  """
  assert len(fps) == size1 + size2
  from rdkit import DataStructs

  # Begin by assigning the first molecule to the first group.

  fp_in_group = [[fps[0]], []]
  indices_in_group: Tuple[List[int], List[int]] = ([0], [])
  remaining_fp = fps[1:]
  remaining_indices = list(range(1, len(fps)))
  max_similarity_to_group = [
      DataStructs.BulkTanimotoSimilarity(fps[0], remaining_fp),
      [0] * len(remaining_fp)
  ]
  while len(remaining_fp) > 0:
    # Decide which group to assign a molecule to.

    group = 0 if len(fp_in_group[0]) / size1 <= len(
        fp_in_group[1]) / size2 else 1

    # Identify the unassigned molecule that is least similar to everything in
    # the other group.

    i = np.argmin(max_similarity_to_group[1 - group])

    # Add it to the group.

    fp = remaining_fp[i]
    fp_in_group[group].append(fp)
    indices_in_group[group].append(remaining_indices[i])

    # Update the data on unassigned molecules.

    similarity = DataStructs.BulkTanimotoSimilarity(fp, remaining_fp)
    max_similarity_to_group[group] = np.delete(
        np.maximum(similarity, max_similarity_to_group[group]), i)
    max_similarity_to_group[1 - group] = np.delete(
        max_similarity_to_group[1 - group], i)
    del remaining_fp[i]
    del remaining_indices[i]
  return indices_in_group

def cal_fig(line,data_dir):
    i = line.split(',')
    if i[-3]:
        pdb = i[0]
        lig = i[1]
        filename = os.path.join(data_dir,f'{pdb}_{lig}',f'{pdb}_')
        pdb_file = f'{filename}protein.pdb'
        lig_file = f'{filename}ligand.sdf'
        try:
            protein = next(oddt.toolkit.readfile('pdb', pdb_file))
            protein.protein = True
            ligand = next(oddt.toolkit.readfile('sdf', lig_file))
            bitset = list(set(fingerprints.PLEC(ligand, protein, size=1024).tolist()))
            bv1 = DataStructs.ExplicitBitVect(1024)
            bv1.SetBitsFromList(bitset)
            line = f"{line.rstrip()},{bv1.ToBitString()}\n"
            #lines.append(line)
            write_list('/root/workshop/data/data_0513/pdb_map_interacion_0602.csv',[line],'a')
            #fps.append(bv1)
            logger.info(pdb)
            #print(bv1.ToBitString())
            return bv1,line
        except:
            logger.info(f'{pdb} wrong!')
            #wlist.append(pdb)

def get_figs(data_dir,lines,out_list,rank=0, size=1,num_cpus=150):
    rank = get_global_node_rank(rank, size)
    if rank == 0:
        inputs = [(j,data_dir) for j in lines]
        res = par.submit_jobs(cal_fig, inputs, num_cpus)
    fps = [n[0] for n in res if n]
    lines = [n[1] for n in res if n]
    sp = FingerprintSplitter()
    logger.info('Split...')
    train_inds, valid_inds, test_inds = sp.split(fps)
    logger.info('Split Done')
    for i in range(len(lines)):
        if i in train_inds:
            lines[i] = lines[i].rstrip()+',train\n'
        elif i in valid_inds:
            lines[i] = lines[i].rstrip()+',val\n'
        elif i in test_inds:
            lines[i] = lines[i].rstrip()+',test\n'
        else:
            print(f'wrong index!{i}')
    write_list(out_list,lines)
    logger.info('All Done')
if __name__ == '__main__':
  f0 = '/root/data/pdb.csv'
  data_dir = '/root/data//structure'
  out_list ='/root/data/pdb_split.csv'
  f0 =read_list(f0)
  get_figs(data_dir, f0, out_list, rank=0, size=1,num_cpus=150)
