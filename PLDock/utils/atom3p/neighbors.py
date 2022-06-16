"""Functions relating to pairs of regions."""

import numpy as np
import pandas as pd
import scipy.spatial as spa


def build_get_neighbors(criteria, cutoff):
    """Build a neighbor function to check if two subunits are neighboring."""

    def get_neighbors_param(df0, df1):
        return get_neighbors(df0, df1, criteria, cutoff)

    return get_neighbors_param


def get_neighbors(df0, df1, criteria, cutoff):
    """Find neighbors according to provided criteria and cutoff."""
    if criteria == 'non_heavy_res':
        return _get_non_heavy_neighbor_residues(df0, df1, cutoff)
    elif criteria == 'non_heavy_atom':
        return _get_non_heavy_neighbor_atoms(df0, df1, cutoff)
    elif criteria == 'ca_res':
        return _get_ca_neighbor_residues(df0, df1, cutoff)
    elif criteria == 'ca_atom':
        return _get_ca_neighbor_atoms(df0, df1, cutoff)
    else:
        raise RuntimeError("Unrecognized criteria {:}".format(criteria))


def _get_non_heavy_neighbor_residues(df0, df1, cutoff):
    """Get neighboring residues for non-heavy atom-based distance."""
    non_heavy0 = df0[df0['element'] != 'H']
    non_heavy1 = df1[df1['element'] != 'H']

    dist = spa.distance.cdist(non_heavy0[['x', 'y', 'z']], non_heavy1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    if len(pairs) == 0:
        return [], []
    # Use the found pairs to find unique pairings of residues.
    res0 = non_heavy0.iloc[pairs[:, 0]][['pdb_name', 'model', 'chain', 'residue']]
    res1 = non_heavy1.iloc[pairs[:, 1]][['pdb_name', 'model', 'chain', 'residue']]
    res0 = res0.reset_index(drop=True)
    res1 = res1.reset_index(drop=True)
    # We concatenate so that we can find unique _pairs_.
    res = pd.concat((res0, res1), axis=1)
    res = res.drop_duplicates()
    # # Split back out now that we have found duplicates.
    res0 = res.iloc[:, range(4)]
    res1 = res.iloc[:, range(4, 8)]
    res0 = res0.reset_index(drop=True)
    res1 = res1.reset_index(drop=True)

    return res0, res1


def _get_non_heavy_neighbor_atoms(df0, df1, cutoff):
    """Get neighboring atoms for non-heavy atom-based distance."""
    non_heavy0 = df0[df0['element'] != 'H']
    non_heavy1 = df1[df1['element'] != 'H']

    dist = spa.distance.cdist(non_heavy0[['x', 'y', 'z']], non_heavy1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    if len(pairs) == 0:
        return [], []
    # Use the found pairs to find unique pairings of atoms.
    atoms0 = non_heavy0.iloc[pairs[:, 0]]
    atoms1 = non_heavy1.iloc[pairs[:, 1]]
    atoms0 = atoms0.reset_index(drop=True)
    atoms1 = atoms1.reset_index(drop=True)
    # Concatenate so that we can find unique pairs.
    atoms = pd.concat((atoms0, atoms1), axis=1)
    atoms = atoms.drop_duplicates()
    # Split back out now that we have found duplicates.
    atoms0 = atoms.iloc[:, range(11)]
    atoms1 = atoms.iloc[:, range(11, 22)]
    atoms0 = atoms0.reset_index(drop=True)
    atoms1 = atoms1.reset_index(drop=True)
    return atoms0, atoms1


def _get_ca_neighbor_residues(df0, df1, cutoff):
    """Get neighboring residues for alpha-carbon based distance."""
    ca0 = df0[df0['atom_name'] == 'CA']
    ca1 = df1[df1['atom_name'] == 'CA']

    dist = spa.distance.cdist(ca0[['x', 'y', 'z']], ca1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    if len(pairs) == 0:
        return [], []
    res0 = ca0.iloc[pairs[:, 0]][['pdb_name', 'model', 'chain', 'residue']]
    res1 = ca1.iloc[pairs[:, 1]][['pdb_name', 'model', 'chain', 'residue']]
    res0 = res0.reset_index(drop=True)
    res1 = res1.reset_index(drop=True)
    return res0, res1


def _get_ca_neighbor_atoms(df0, df1, cutoff):
    """Get neighboring atoms for alpha-carbon based distance."""
    ca0 = df0[df0['atom_name'] == 'CA']
    ca1 = df1[df1['atom_name'] == 'CA']

    dist = spa.distance.cdist(ca0[['x', 'y', 'z']], ca1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    if len(pairs) == 0:
        return [], []
    atoms0 = ca0.iloc[pairs[:, 0]]
    atoms1 = ca1.iloc[pairs[:, 1]]
    atoms0 = atoms0.reset_index(drop=True)
    atoms1 = atoms1.reset_index(drop=True)
    return atoms0, atoms1
