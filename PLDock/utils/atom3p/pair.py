import collections as col
import multiprocessing as mp
import os

import dill
import numpy as np
import pandas as pd
import parallel as par

from . import case as ca
from . import complex as comp
from . import database as db
from . import neighbors as nb
from .structure import get_ca_pos_from_residues, get_ca_pos_from_atoms

from PLDock import log
log_pair = log()

# Add pocket df
Pair = col.namedtuple(
    'Pair', ['complex', 'df0', 'df1', 'df2', 'pos_idx', 'neg_idx', 'srcs', 'id', 'sequences'])  
#    'Pair', ['complex', 'df0', 'df1', 'pos_idx', 'neg_idx', 'srcs', 'id', 'sequences'])

PairWithLabels = col.namedtuple(
    'PairWithLabels', ['complex', 'df0', 'df1', 'df2', 'pos_idx', 'neg_idx', 'srcs', 'id', 'sequences', 'labels'])
#    'PairWithLabels', ['complex', 'df0', 'df1', 'pos_idx', 'neg_idx', 'srcs', 'id', 'sequences', 'labels'])

sem = mp.Semaphore()


def add_pairs_parser(subparsers, pp):
    """Add parser."""

    def all_complexes_to_pairs_main(args):
        all_complexes_to_pairs_full(args)

    ap = subparsers.add_parser(
        'pairs', description='complexes to pairs',
        help='uses output of the complex command to split pickled proteins',
        parents=[pp])
    ap.set_defaults(func=all_complexes_to_pairs_main)
    ap.add_argument('nbd', metavar='neighbor_def', type=str,
                    help='neighbor definition (i.e. non_heavy_res, non_heavy_atom, ca_res, or ca_atom)')
    ap.add_argument('complexes_dill', metavar='complexes.dill', type=str,
                    help='complexes file')
    ap.add_argument('output_dir', type=str,
                    help='directory to output to')
    ap.add_argument('-n', '--criteria', dest='criteria',
                    choices=['ca', 'heavy'],
                    default='ca', help='criteria for finding neighboring'
                                       ' residues (default: by alpha carbon distance)')
    ap.add_argument('-t', '--cutoff', dest='cutoff', type=float,
                    default=8, help='cutoff distance to be used with'
                                    ' neighbor criteria (default: 8)')
    ap.add_argument('-f', '--full', dest='full', action='store_true',
                    help='generate all possible negative examples, '
                         'as opposed to a sampling of same size as positive.')
    ap.add_argument(
        '-u', '--unbound', help='whether to use unbound data.',
        action="store_true")
    ap.add_argument('-c', metavar='cpus', default=mp.cpu_count(), type=int,
                    help='number of cpus to use for processing (default:'
                         ' number processors available on current machine)')


def all_complexes_to_pairs_full(args):
    complexes = comp.read_complexes(args.complexes_dill)
    get_neighbors = nb.build_get_neighbors(args.criteria, args.cutoff)
    get_pairs = build_get_pairs(args.nbd, complexes['type'], args.unbound, get_neighbors, args.full)
    all_complex_to_pairs(complexes, get_pairs, args.output_dir, args.c)


def all_complex_to_pairs(complexes, source_type, get_pairs, output_dir, num_cpus):
    """Reads in structures and produces appropriate pairings."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    requested_keys = complexes['data'].keys()
    produced_keys = complexes_from_pair_dir(output_dir)
    work_keys = [key for key in requested_keys if key not in produced_keys]

    inputs = [(complexes['data'][key], source_type, get_pairs, output_dir)
              for key in work_keys]
    log_pair.info("{:} requested keys, {:} produced keys, {:} work keys"
                 .format(len(requested_keys), len(produced_keys),
                         len(work_keys)))
    par.submit_jobs(complex_to_pairs, inputs, num_cpus)


def complexes_from_pair_dir(pair_dir):
    """Get all complex names from provided pair directory."""
    filenames = db.get_structures_filenames(pair_dir, extension='.dill')
    # Remove per-chain identifier.
    # TODO: This could cause issues when only some of the pairs have been
    # written.
    return ['_'.join(db.get_pdb_name(x).split('_')[:-1]) for x in filenames]


def complex_to_pairs(complex, source_type, get_pairs, output_dir):
    pairs_txt = output_dir + '/pairs.txt'
    name = complex.name
    log_pair.info("Working on {:}".format(name))
    pairs, num_subunits = get_pairs(complex)
    casp_capri_addon_message = '; selecting pair with most inter-chain interactions'
    log_pair_message = "For complex {:} found {:} pairs out of {:} chains"
    log_pair_message += casp_capri_addon_message if source_type == 'casp_capri' and num_subunits > 1 else ''
    log_pair.info(log_pair_message.format(name, len(pairs), num_subunits))
    sub_dir = output_dir + '/' + db.get_pdb_code(name)[1:3]
    f = name
    if ('mut' in f) and ('mut' not in db.get_pdb_code(name)):
        pdb = db.get_pdb_code(name)(f) + f[f.rfind('_') + 1: f.find('.')]
        sub_dir = output_dir + '/' + pdb
    with sem:
        if len(pairs) > 0:
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
            with open(pairs_txt, 'a') as f:
                f.write(name + '\n')

    if source_type == 'casp_capri':
        pair_with_most_interactions = pairs[0]
        for pair in pairs:
            if len(pair.pos_idx) > len(pair_with_most_interactions.pos_idx):
                pair_with_most_interactions = pair
        pairs = [pair_with_most_interactions]
        assert len(pairs) == 1, 'For CASP-CAPRI complexes, the max-interactions chain must be the only chain selected'
    for i, pair in enumerate(pairs):
        output_dill = "{:}/{:}_{:}.dill".format(sub_dir, name, i)
        write_pair_as_dill(pair, output_dill)


def write_pair_as_dill(pair, output_dill):
    """Write pair as dill file."""
    with open(output_dill, 'wb') as f:
        dill.dump(pair, f)


def read_pair_from_dill(input_dill):
    """Read pair from dill file."""
    with open(input_dill, 'rb') as f:
        return dill.load(f)


def build_get_pairs(neighbor_def, type, unbound, nb_fn, full):
    def get_pairs_param(complex):
        return get_pairs(neighbor_def, complex, type, unbound, nb_fn, full)

    return get_pairs_param


def get_pairs(neighbor_def, complex, type, unbound, nb_fn, full):
    """
    Get pairings for provided complex.

    A complex is a set of chains.  For our interface prediction problem, we
    currently only deal with pairs of chains.  Here, we find all possible such
    pairings, for a given definition of neighboring.
    """
    if type == 'rcsb':
        pairs, num_subunits = \
            _get_rcsb_pairs(neighbor_def, complex, unbound, nb_fn, full)
    elif type == 'db5' or type == 'db5mut' or type == 'hotspot':
        pairs, num_subunits = \
            _get_db5_pairs(complex, unbound, nb_fn, full)
    elif type == 'dockground':
        pairs, num_subunits = \
            _get_db5_pairs(complex, unbound, nb_fn, full)
    elif type == 'evcoupling':
        pairs, num_subunits = \
            _get_evcoupling_pairs(complex, unbound, nb_fn, full)
    elif type == 'casp_capri':
        pairs, num_subunits = \
            _get_casp_capri_pairs(neighbor_def, complex, unbound, nb_fn, full)
    elif type == 'input':
        pairs, num_subunits = \
            _get_input_pairs(complex)
    elif type == 'pdbbind':
        pairs, num_subunits = \
            _get_pdbbind_pairs(complex)
    else:
        raise RuntimeError("Unrecognized dataset type {:}".format(type))
    return pairs, num_subunits


def _get_rcsb_pairs(neighbor_def, complex, unbound, nb_fn, full):
    """
    Get pairs for rcsb type complex.

    For this type of complex, we assume that each chain is its own entity,
   and that two chains form a pair if at least one pair of residues spanning
    the two are considered neighbors.
    """
    if unbound:
        log_pair.error("Requested unbound pairs from RCSB type complex, "
                      "even though they don't have unbound data.")
        raise RuntimeError("Unbound requested for RCSB")
    (pkl_filename,) = complex.bound_filenames
    df = pd.read_pickle(pkl_filename)
    # TODO: Allow for keeping more than just first model.
    if df.shape[0] == 0:
        return [], 0
    df = df[df['model'] == df['model'][0]]
    pairs, num_chains = _get_all_chain_pairs(neighbor_def, complex, df, nb_fn, pkl_filename, full)
    return pairs, num_chains


def _get_db5_pairs(complex, unbound, nb_fn, full):
    """
    Get pairs for docking benchmark 5 type complex.

    For this type of complex, we assume that each file is its own entity,
    and that there is essentially one pair for each complex, with one side
    being all the chains of the ligand, and the other all the chains of the
    receptor.
    """
    (lb, rb) = complex.bound_filenames
    (lu, ru) = complex.unbound_filenames
    lb_df = pd.read_pickle(lb)
    rb_df = pd.read_pickle(rb)
    # Always use bound to get neighbors...
    lres, rres = nb_fn(lb_df, rb_df)
    if unbound:
        # ...but if unbound, we then use the actual atoms from unbound.
        ldf, rdf = pd.read_pickle(lu), pd.read_pickle(ru)

        # Convert residues' pdb_names to unbound.
        lres['pdb_name'] = lres['pdb_name'].map(
            lambda x: ca.find_of_type(x, ldf['pdb_name'].values, None, False, style='db5'))
        rres['pdb_name'] = rres['pdb_name'].map(
            lambda x: ca.find_of_type(x, rdf['pdb_name'].values, None, False, style='db5'))

        # Remove residues that we cannot map from bound structure to unbound.
        rres_index = rres[['pdb_name', 'model', 'chain', 'residue']]
        lres_index = lres[['pdb_name', 'model', 'chain', 'residue']]
        rdf_index = rdf[['pdb_name', 'model', 'chain', 'residue']]
        ldf_index = ldf[['pdb_name', 'model', 'chain', 'residue']]
        rgone = [i for i, x in rres_index.iterrows()
                 if not (np.array(x) == rdf_index).all(1).any()]
        lgone = [i for i, x in lres_index.iterrows()
                 if not (np.array(x) == ldf_index).all(1).any()]
        gone = list(set(lgone).union(set(rgone)))
        if len(gone) > 0:
            log_pair.warning("Dropping {:}/{:} residues from {:} that didn't map "
                            "to unbound from bound.".format(len(gone), len(lres), complex.name))
            lres = lres.drop(gone)
            rres = rres.drop(gone)

        lsrc, rsrc = lu, ru
    else:
        ldf, rdf = lb_df, rb_df
        lsrc, rsrc = lb, rb
    lpos = get_ca_pos_from_residues(ldf, lres)
    rpos = get_ca_pos_from_residues(rdf, rres)
    pos_idx, neg_idx = _get_residue_positions(ldf, lpos, rdf, rpos, full)
    srcs = {'src0': lsrc, 'src1': rsrc}
    pair = Pair(complex=complex.name, df0=ldf, df1=rdf, pos_idx=pos_idx, neg_idx=neg_idx, srcs=srcs, id=0, sequences={})
    return [pair], 2


def _get_evcoupling_pairs(complex, unbound, nb_fn, full):
    """
    Get pairs for EVCoupling type complex.

    For this type of complex, we assume that each chain is its own entity,
    and that two chains form a pair if at least one pair of residues spanning
    the two are considered neighbors.
    """
    (lb, rb) = complex.bound_filenames
    lb_df = pd.read_pickle(lb)
    rb_df = pd.read_pickle(rb)
    # Always use bound to get neighbors...
    lres, rres = nb_fn(lb_df, rb_df)
    ldf, rdf = lb_df, rb_df
    lsrc, rsrc = lb, rb
    lpos = get_ca_pos_from_residues(ldf, lres)
    rpos = get_ca_pos_from_residues(rdf, rres)
    pos_idx, neg_idx = _get_residue_positions(ldf, lpos, rdf, rpos, full)
    srcs = {'src0': lsrc, 'src1': rsrc}
    pair = Pair(complex=complex.name, df0=ldf, df1=rdf, pos_idx=pos_idx, neg_idx=neg_idx, srcs=srcs, id=0, sequences={})
    return [pair], 2


def _get_casp_capri_pairs(neighbor_def, complex, unbound, nb_fn, full):
    """
    Get pairs for CASP-CAPRI type complex.

    For this type of complex, we assume that each chain is its own entity,
    and that two chains form a pair if at least one pair of residues spanning
    the two are considered neighbors.
    """
    if unbound:
        log_pair.error("Requested unbound pairs from RCSB type complex, "
                      "even though they don't have unbound data.")
        raise RuntimeError("Unbound requested for RCSB")
    (pkl_filename,) = complex.bound_filenames
    df = pd.read_pickle(pkl_filename)
    # TODO: Allow for keeping more than just first model.
    if df.shape[0] == 0:
        return [], 0
    df = df[df['model'] == df['model'][0]]
    pairs, num_chains = _get_all_chain_pairs(neighbor_def, complex, df, nb_fn, pkl_filename, full)
    return pairs, num_chains


def _get_input_pairs(complex):
    """
    Get pairs for input type complex.

    For this type of complex, we assume that each file is its own entity,
    and that there is essentially one pair for each complex, with one side
    being all the chains of the ligand, and the other all the chains of the
    receptor.
    """
    (lu, ru) = complex.unbound_filenames
    ldf, rdf = pd.read_pickle(lu), pd.read_pickle(ru)
    lsrc, rsrc = lu, ru
    srcs = {'src0': lsrc, 'src1': rsrc}
    pos_idx, neg_idx = np.array([]), np.array([])
    pair = Pair(complex=complex.name, df0=ldf, df1=rdf, pos_idx=pos_idx, neg_idx=neg_idx, srcs=srcs, id=0, sequences={})
    return [pair], 2


def _get_all_chain_pairs(neighbor_def, complex, df, nb_fn, filename, full):
    """Get all possible chain pairs from provided dataframe."""
    pairs = []
    # We reset the index here so each chain's dataframe can be treated independently.
    groups = [(x[0], x[1].reset_index(drop=True))
              for x in df.groupby(['chain', 'model'])]
    num_chains = len(groups)
    num_pairs = 0
    pair_idx = 0
    for i in range(num_chains):
        (chain0, df0) = groups[i]
        for j in range(i + 1, num_chains):
            (chain1, df1) = groups[j]
            if 'atom' in neighbor_def:
                atoms0, atoms1 = nb_fn(df0, df1)
                if len(atoms0) == 0:
                    # No neighbors between these 2 chains.
                    continue
                else:
                    num_pairs += 1
                pos0 = get_ca_pos_from_atoms(df0, atoms0)
                pos1 = get_ca_pos_from_atoms(df1, atoms1)
                pos_idx, neg_idx = _get_atoms_positions(df0, pos0, df1, pos1, full)
            else:  # Use residue-level neighbor function
                res0, res1 = nb_fn(df0, df1)
                if len(res0) == 0:
                    # No neighbors between these 2 chains.
                    continue
                else:
                    num_pairs += 1
                pos0 = get_ca_pos_from_residues(df0, res0)
                pos1 = get_ca_pos_from_residues(df1, res1)
                pos_idx, neg_idx = _get_residue_positions(df0, pos0, df1, pos1, full)
            srcs = {'src0': filename, 'src1': filename}
            pair = Pair(complex=complex.name, df0=df0, df1=df1, pos_idx=pos_idx,
                        neg_idx=neg_idx, srcs=srcs, id=pair_idx, sequences={})
            pairs.append(pair)
            pair_idx += 1
    return pairs, num_chains


def _get_residue_positions(df0, pos_ca0, df1, pos_ca1, full):
    """Get negative pairings given positive pairings."""
    ca0 = df0[df0['atom_name'] == 'CA']
    ca1 = df1[df1['atom_name'] == 'CA']
    num0, num1 = ca0.shape[0], ca1.shape[0]
    num_pos = pos_ca0.shape[0]
    num_total = num0 * num1
    pos_idxs = []
    for p0, p1 in zip(pos_ca0.index, pos_ca1.index):
        #print(p0, p1)
        idx0 = ca0.index.get_loc(p0)
        idx1 = ca1.index.get_loc(p1)
        pos_idxs.append((idx0, idx1))
    pos_idxs = np.array(pos_idxs)
    #print(pos_idxs)
    pos_flat = np.ravel_multi_index(
        (pos_idxs[:, 0], pos_idxs[:, 1]), (num0, num1))
    neg_flat = np.arange(num_total)
    neg_flat = np.delete(neg_flat, pos_flat)
    if not full:
        np.random.shuffle(neg_flat)
        neg_flat = neg_flat[:num_pos]
    neg_idxs = np.array(np.unravel_index(neg_flat, (num0, num1))).T
    neg_ca0 = ca0.iloc[neg_idxs[:, 0]]
    neg_ca1 = ca1.iloc[neg_idxs[:, 1]]
    neg_ca_idxs = np.stack((neg_ca0.index.values, neg_ca1.index.values)).T
    pos_ca_idxs = np.stack((pos_ca0.index.values, pos_ca1.index.values)).T
    return pos_ca_idxs, neg_ca_idxs


def _get_atoms_positions(df0, pos_ca0, df1, pos_ca1, full):
    """Get negative pairings given positive pairings."""
    ca0 = df0[df0['atom_name'] == 'CA']
    ca1 = df1[df1['atom_name'] == 'CA']
    num0, num1 = ca0.shape[0], ca1.shape[0]
    num_pos = pos_ca0.shape[0]
    num_total = num0 * num1
    pos_idxs = []
    for p0, p1 in zip(pos_ca0.index, pos_ca1.index):
        idx0 = ca0.index.get_loc(p0)
        idx1 = ca1.index.get_loc(p1)
        pos_idxs.append((idx0, idx1))
    pos_idxs = np.array(pos_idxs)
    pos_flat = np.ravel_multi_index(
        (pos_idxs[:, 0], pos_idxs[:, 1]), (num0, num1))
    neg_flat = np.arange(num_total)
    neg_flat = np.delete(neg_flat, pos_flat)
    if not full:
        np.random.shuffle(neg_flat)
        neg_flat = neg_flat[:num_pos]
    neg_idxs = np.array(np.unravel_index(neg_flat, (num0, num1))).T
    neg_ca0 = ca0.iloc[neg_idxs[:, 0]]
    neg_ca1 = ca1.iloc[neg_idxs[:, 1]]
    neg_ca_idxs = np.stack((neg_ca0.index.values, neg_ca1.index.values)).T
    pos_ca_idxs = np.stack((pos_ca0.index.values, pos_ca1.index.values)).T
    return pos_ca_idxs, neg_ca_idxs


def _get_atom_positions(df0, pos_atoms0, df1, pos_atoms1, full):
    """Get negative atom pairings given positive atom pairings."""
    non_heavy0 = df0[df0['element'] != 'H']
    non_heavy1 = df1[df1['element'] != 'H']
    num0, num1 = non_heavy0.shape[0], non_heavy1.shape[0]
    num_pos = pos_atoms0.shape[0]
    num_total = num0 * num1
    pos_idxs = []
    for p0, p1 in zip(pos_atoms0.index, pos_atoms1.index):
        idx0 = non_heavy0.index.get_loc(p0)
        idx1 = non_heavy1.index.get_loc(p1)
        pos_idxs.append((idx0, idx1))
    pos_idxs = np.array(pos_idxs)
    pos_flat = np.array([0]) if pos_idxs.size == 0 else np.ravel_multi_index(
        (pos_idxs[:, 0], pos_idxs[:, 1]), (num0, num1))
    neg_flat = np.arange(num_total)
    neg_flat = np.delete(neg_flat, pos_flat)
    if not full:
        np.random.shuffle(neg_flat)
        neg_flat = neg_flat[:num_pos]
    neg_idxs = np.array(np.unravel_index(neg_flat, (num0, num1))).T
    neg_atoms0 = non_heavy0.iloc[neg_idxs[:, 0]]
    neg_atoms1 = non_heavy1.iloc[neg_idxs[:, 1]]
    neg_atom_idxs = np.stack((neg_atoms0.index.values, neg_atoms1.index.values)).T
    pos_atom_idxs = np.stack((pos_atoms0.index.values, pos_atoms1.index.values)).T
    return pos_atom_idxs, neg_atom_idxs

def _get_pdbbind_pairs(complex):
    """
    Get pairs for input type complex.

    For this type of complex, we assume that each file is its own entity,
    and that there is essentially one pair for each complex, with one side
    being all the chains of the ligand, and the other all the chains of the
    receptor.
    """
    (lb, rb, pb) = complex.bound_filenames
    #ldf, rdf, pdf  = pd.read_pickle(lb), pd.read_pickle(rb), pd.read_pickle(pb)
    ldf, rdf = pd.read_pickle(lb), pd.read_pickle(rb)
    pdf = pd.read_pickle(pb) if pb else None
    lsrc, rsrc, psrc= lb, rb, pb
    srcs = {'src0': lsrc, 'src1': rsrc, 'src2': psrc}
    pos_idx, neg_idx = np.array([]), np.array([])
    pair = Pair(complex=complex.name, df0=ldf, df1=rdf, df2=pdf, pos_idx=pos_idx, neg_idx=neg_idx, srcs=srcs, id=0, sequences={})
    return [pair], 2
