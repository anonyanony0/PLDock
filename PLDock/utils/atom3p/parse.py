import multiprocessing as mp
import os

import parallel as par

from . import database as db
from . import structure as struct

from PLDock import log
log_parse = log()

def add_parse_parser(subparsers, pp):
    """Add parser."""

    def parse_all_main(args):
        parse_all(args.pdb_dataset, args.output_dir, args.c)

    ap = subparsers.add_parser(
        'parse', description='pdb to dataframe',
        help='pdb dataset to pickled dataframes',
        parents=[pp])
    ap.set_defaults(func=parse_all_main)
    ap.add_argument('pdb_dataset', metavar='pdb', type=str,
                    help='pdb dataset')
    ap.add_argument('output_dir', metavar='output', type=str,
                    help='directory to output to')
    ap.add_argument('-c', metavar='cpus', default=mp.cpu_count(), type=int,
                    help='number of cpus to use for processing (default:'
                    ' number processors available on current machine)')


def parse_all(pdb_dataset, output_dir, num_cpus):
    """Parse pdb dataset (pdb files) to pandas dataframes."""
    requested_filenames = db.get_structures_filenames(pdb_dataset)
    requested_filenames += [i for i in db.get_structures_filenames(pdb_dataset,extension='.mol2') if 'ligand.mol2' in i]
    #print(requested_filenames)
    produced_filenames = db.get_structures_filenames(
        output_dir, extension='.pkl')
    # produced_filenames=[]
    requested_keys = [db.get_pdb_name(x) for x in requested_filenames]
    produced_keys = [db.get_pdb_name(x) for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    work_filenames = [x[0] for x in
                        db.get_all_filenames(work_keys, pdb_dataset, enforcement=2) if x]
    #                  db.get_all_filenames(work_keys, pdb_dataset, enforcement=2)]
    # get ligand mol2
    work_filenames += [x[0] for x in
                      db.get_all_filenames(work_keys, pdb_dataset, extension='.mol2', enforcement=2) if x]
    log_parse.info("{:} requested keys, {:} produced keys, {:} work keys"
                 .format(len(requested_keys), len(produced_keys),
                         len(work_keys)))

    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
        output_filenames.append(
            sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".pkl")

    inputs = [(key, output)
              for key, output in zip(work_filenames, output_filenames)]
    par.submit_jobs(parse, inputs, num_cpus)


def parse(pdb_filename, output_pkl):
    """Parse single pdb file to output directory."""
    # Get middle two letters of code.
    log_parse.info("Reading {:}".format(pdb_filename)) 
    df = struct.parse_structure(pdb_filename, one_model=False)
    log_parse.info("Writing {:} to {:}".format(pdb_filename, output_pkl))
    if df is not None:
        df.to_pickle(output_pkl)
    log_parse.info("Done writing {:} to {:}"
                 .format(pdb_filename, output_pkl))
