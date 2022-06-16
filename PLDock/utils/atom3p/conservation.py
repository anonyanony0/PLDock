import multiprocessing as mp
import os
import pickle
import subprocess
import timeit
from pathlib import Path

import pandas as pd
import parallel as par

from . import database as db
from . import sequence as sequ
from .utils import extract_hmm_profile, slice_list
from PLDock import log
log_conservation = log()


def add_conservation_parser(subparsers, pp):
    """Add parser."""

    def map_all_pssms_main(args):
        map_all_pssms(args.pkl_dataset, args.pruned_dataset, args.blastdb,
                      args.output_dir, args.c, args.source_type, args.rank, args.size)

    ap = subparsers.add_parser(
        'conservation', description='sequence conservation',
        help='compute sequence conservation features',
        parents=[pp])
    ap.set_defaults(func=map_all_pssms_main)
    ap.add_argument('pkl_dataset', metavar='pkl', type=str,
                    help='parsed dataset')
    ap.add_argument('blastdb', metavar='bdb', type=str,
                    help='blast database to do lookups on')
    ap.add_argument('output_dir', metavar='output', type=str,
                    help='directory to output to')
    ap.add_argument('-c', metavar='cpus', default=mp.cpu_count(), type=int,
                    help='number of cpus to use for processing (default:'
                         ' number processors available on current machine)')
    ap.add_argument('--source_type', metavar='complex_type', default='rcsb', type=str,
                    help='whether the source PDBs are for bound or unbound complexes (i.e. RCSB (e.g. DIPS) or DB5 complexes)')


def gen_protrusion_index(psaia_dir, psaia_config_file, file_list_file):
    """Generate protrusion index for file list of PDB structures."""
    log_conservation.info("PSAIA'ing {:}".format(file_list_file))
    _psaia(psaia_dir, psaia_config_file, file_list_file)


def gen_pssm(pdb_filename, blastdb, output_filename):
    """Generate PSSM and PSFM from sequence."""
    pdb_name = db.get_pdb_name(pdb_filename)
    out_dir = os.path.dirname(output_filename)
    work_dir = os.path.join(out_dir, 'work')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    fasta_format = work_dir + "/{:}.fa"
    id_format = work_dir + "/{:}.cpkl"
    chains, chain_fasta_filenames, id_filenames = sequ.pdb_to_fasta(
        pdb_filename, fasta_format, id_format, True)

    pssms = []
    for chain, chain_fasta_filename, id_filename in \
            zip(chains, chain_fasta_filenames, id_filenames):
        basename = os.path.splitext(chain_fasta_filename)[0]
        pssm_filename = "{}.pssm".format(basename)
        blast_filename = "{}.blast".format(basename)
        clustal_filename = "{}.clustal".format(basename)
        al2co_filename = "{}.al2co".format(basename)
        if not os.path.exists(pssm_filename):
            log_conservation.info("Blasting {:}".format(chain_fasta_filename))
            _blast(chain_fasta_filename, pssm_filename, blast_filename,
                   blastdb)

        if not os.path.exists(pssm_filename):
            log_conservation.warning("No hits for {:}".format(chain_fasta_filename))
            # Create empty file.
            open(pssm_filename, 'w').close()

        if not os.path.exists(clustal_filename):
            log_conservation.info("Converting {:}".format(blast_filename))
            _to_clustal(blast_filename, clustal_filename)

        if not os.path.exists(al2co_filename):
            log_conservation.info("Al2co {:}".format(al2co_filename))
            _al2co(clustal_filename, al2co_filename)

        if os.stat(pssm_filename).st_size != 0:
            pssm = pd.read_csv(
                pssm_filename, skiprows=2, skipfooter=6, delim_whitespace=True,
                engine='python', usecols=range(20), index_col=[0, 1])
            pssm = pssm.reset_index()
            del pssm['level_0']
            pssm.rename(columns={'level_1': 'orig'}, inplace=True)

            pscm = pd.read_csv(
                pssm_filename, skiprows=2, skipfooter=6, delim_whitespace=True,
                engine='python', usecols=range(20, 40), index_col=[0, 1])
            psfm = pscm.applymap(lambda x: x / 100.)
            psfm = psfm.reset_index()
            del psfm['level_0']
            psfm.columns = pssm.columns
            del psfm['orig']
            del pssm['orig']

            # Combine both into one.
            psfm = psfm.add_prefix('psfm_')
            pssm = pssm.add_prefix('pssm_')
            al2co = pd.read_csv(
                al2co_filename, delim_whitespace=True, usecols=[2],
                names=['al2co'])
            pssm = pd.concat([pssm, psfm, al2co], axis=1)

        else:
            log_conservation.warning("No pssm found for {:} (model {:}, chain {:})"
                            .format(pdb_name, chain[-2], chain[-1]))
            pssm, psfm = None, None

        pdb_name = db.get_pdb_name(pdb_filename)
        key = pdb_name + '-' + chain[-2] + '-' + chain[-1]
        pos_to_res = pickle.load(open(id_filename, 'rb'))[key]

        if pssm is not None:  # Skip if PSSM was not found
            pssm['pdb_name'] = db.get_pdb_name(pdb_filename)
            pssm['model'] = chain[0]
            pssm['chain'] = chain[1]
            pssm['residue'] = pos_to_res
            pssms.append(pssm)
    pssms = pd.concat(pssms)
    return pssms


def gen_profile_hmm(num_cpus, pkl_filename, output_filename, hhsuite_db, source_type, num_iter):
    """Generate profile HMM from sequence."""
    pdb_name = db.get_pdb_name(pkl_filename)
    out_dir = os.path.dirname(output_filename)
    work_dir = os.path.join(out_dir, 'work')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    fasta_format = work_dir + "/{:}.fa"
    id_format = work_dir + "/{:}.cpkl"

    # Get FASTA sequence-chain representations of PDB structures
    chains, chain_fasta_filenames, id_filenames = sequ.pdb_to_fasta(pkl_filename, fasta_format, id_format, True)

    # Process each profile HMM for a given PDB structure or complex
    num_chains = 0
    profile_hmms = []
    for chain, chain_fasta_filename, id_filename in zip(chains, chain_fasta_filenames, id_filenames):
        basename = os.path.splitext(chain_fasta_filename)[0]
        profile_hmm_filename = "{}.hhm".format(basename)
        hhblits_filename = "{}.a3m".format(basename)

        if not os.path.exists(profile_hmm_filename):
            log_conservation.info("HHblits'ing {:}".format(chain_fasta_filename))
            _hhsuite(num_cpus, chain_fasta_filename, hhblits_filename, profile_hmm_filename, hhsuite_db, num_iter)

        if not os.path.exists(profile_hmm_filename):
            log_conservation.warning("No hits for {:}".format(chain_fasta_filename))
            # Create empty file
            open(profile_hmm_filename, 'w').close()

        if os.stat(profile_hmm_filename).st_size != 0:
            with open(chain_fasta_filename, 'r') as fasta:
                with open(profile_hmm_filename, 'r') as hmm:
                    sequence = ''
                    for seq_line in fasta.readlines()[1:]:
                        sequence += " ".join(seq_line.splitlines())
                    profile_hmm = extract_hmm_profile(hmm.read(), sequence)
        else:
            log_conservation.warning("No profile HMM found for {:} (model {:}, chain {:})"
                            .format(pdb_name, chain[-2], chain[-1]))
            profile_hmm = None

        pdb_name = db.get_pdb_name(pkl_filename)
        key = pdb_name + '-' + chain[-2] + '-' + chain[-1]
        pos_to_res = pickle.load(open(id_filename, 'rb'))[key]

        if profile_hmm is not None:  # Skip if profile HMM was not found
            profile_hmm = pd.DataFrame(data=profile_hmm)
            profile_hmm.insert(0, 'pdb_name', db.get_pdb_name(pkl_filename))
            profile_hmm.insert(1, 'model', chain[0])
            profile_hmm.insert(2, 'chain', chain[1])
            profile_hmm.insert(3, 'residue', pos_to_res)
            profile_hmms.append(profile_hmm)
        # Keep track of how many chains have been processed
        num_chains += 1
    # Merge related DataFrames into a single one
    profile_hmms = pd.concat(profile_hmms)
    return profile_hmms, num_chains


def map_protrusion_indices(psaia_dir, psaia_config_file, file_list_file):
    start_time = timeit.default_timer()
    start_time_psaiaing = timeit.default_timer()
    gen_protrusion_index(psaia_dir, psaia_config_file, file_list_file)
    elapsed_psaiaing = timeit.default_timer() - start_time_psaiaing

    start_time_writing = timeit.default_timer()
    elapsed_writing = timeit.default_timer() - start_time_writing

    elapsed = timeit.default_timer() - start_time
    log_conservation.info(('For generating protrusion indices, spent {:05.2f} PSAIA\'ing,'
                  ' {:05.2f} writing, and {:05.2f} overall.').format(elapsed_psaiaing, elapsed_writing, elapsed))


def map_pssms(pdb_filename, blastdb, output_filename):
    pdb_name = db.get_pdb_name(pdb_filename)
    start_time = timeit.default_timer()
    start_time_blasting = timeit.default_timer()
    pis = gen_pssm(pdb_filename, blastdb, output_filename)
    num_chains = len(pis.groupby(['pdb_name', 'model', 'chain']))
    elapsed_blasting = timeit.default_timer() - start_time_blasting

    parsed = pd.read_pickle(pdb_filename)
    parsed = parsed.merge(pis, on=['model', 'pdb_name', 'chain', 'residue'])

    start_time_writing = timeit.default_timer()
    parsed.to_pickle(output_filename)
    elapsed_writing = timeit.default_timer() - start_time_writing

    elapsed = timeit.default_timer() - start_time
    log_conservation.info(('For {:d} pssms generated from {}, spent {:05.2f} blasting,'
                  ' {:05.2f} writing, and {:05.2f} overall.').format(num_chains, pdb_name, elapsed_blasting,
                                                                     elapsed_writing, elapsed))


def map_profile_hmms(num_cpus, pkl_filename, output_filename, hhsuite_db, source_type, num_iter):
    pdb_name = db.get_pdb_name(pkl_filename)
    start_time = timeit.default_timer()
    start_time_blitsing = timeit.default_timer()
    profile_hmms, num_chains = gen_profile_hmm(num_cpus, pkl_filename, output_filename,
                                               hhsuite_db, source_type, num_iter)
    elapsed_blitsing = timeit.default_timer() - start_time_blitsing

    start_time_writing = timeit.default_timer()
    profile_hmms.to_pickle(output_filename)
    elapsed_writing = timeit.default_timer() - start_time_writing

    elapsed = timeit.default_timer() - start_time
    log_conservation.info(('For {:d} profile HMMs generated from {}, spent {:05.2f} blitsing,'
                  ' {:05.2f} writing, and {:05.2f} overall.').format(num_chains, pdb_name, elapsed_blitsing,
                                                                     elapsed_writing, elapsed))


def _psaia(psaia_dir, psaia_config_file, file_list_file):
    """Run PSAIA on specified input."""
    psa_path = os.path.join(psaia_dir, 'psa')
    psaia_command = f"yes y | {psa_path} {psaia_config_file} {file_list_file}"  # PSA is the PSAIA's CLI (i.e., its GUI)
    log_out = "{}.out".format(file_list_file)
    log_err = "{}.err".format(file_list_file)
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            f_out.write('=================== CALL ===================\n')
            f_out.write(psaia_command + '\n')
            try:
                subprocess.check_call(psaia_command, shell=True, stderr=f_err, stdout=f_out)
            except subprocess.CalledProcessError as cpe:
                if cpe.returncode == 1:  # A return code of 1 indicates that PSA was successful
                    pass
                else:
                    raise cpe
            f_out.write('================= END CALL =================\n')


def _hhsuite(num_cpus, query, output_a3m, output_hhm, hhsuite_db, num_iter):
    """Run HH-suite3 on specified input."""
    hhsuite_command = "hhblits -cpu {:} -i {:} -d {:} -oa3m {:} -n {} && hhmake -i {:} -o {:}"
    log_out = "{}.out".format(output_hhm)
    log_err = "{}.err".format(output_hhm)
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = hhsuite_command.format(num_cpus, query, hhsuite_db, output_a3m, num_iter, output_a3m, output_hhm)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def _blast(query, output_pssm, output, blastdb):
    """Run PSIBlast on specified input."""
    psiblast_command = "psiblast -db {:} -query {:} -out_ascii_pssm {:} " + \
                       "-save_pssm_after_last_round -out {:}"
    log_out = "{}.out".format(output)
    log_err = "{}.err".format(output)
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = psiblast_command.format(blastdb, query, output_pssm, output)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def _to_clustal(psiblast_in, clustal_out):
    """Convert PSIBlast output to CLUSTAL format."""
    log_out = "{}.out".format(clustal_out)
    log_err = "{}.err".format(clustal_out)
    mview_command = "mview -in blast -out clustal {:} | tail -n+4 > {:}"
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = mview_command.format(psiblast_in, clustal_out)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def _al2co(clustal_in, al2co_out):
    """Use al2co on CLUSTAL format."""
    log_out = "{}.out".format(al2co_out)
    log_err = "{}.err".format(al2co_out)
    al2co_command = "al2co -i {:} -g 0.9 | head -n -12 > {:}"
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = al2co_command.format(clustal_in, al2co_out)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def map_all_protrusion_indices(psaia_dir, psaia_config_file, pdb_dataset, pkl_dataset,
                               pruned_dataset, output_dir, source_type):
    ext = '.pkl'
    if source_type.lower() == 'rcsb':
        # Filter out pairs that did not survive pruning previously to reduce complexity
        pruned_pdb_names = [db.get_pdb_name(filename)
                            for filename in db.get_structures_filenames(pruned_dataset, extension='.dill')]
        requested_filenames = [
            os.path.join(pkl_dataset, db.get_pdb_code(pruned_pdb_name)[1:3], pruned_pdb_name.split('_')[0] + ext)
            for pruned_pdb_name in pruned_pdb_names
        ]
    else:  # DB5 does not employ pair pruning, so there are no pairs to filter
        requested_filenames = [filename for filename in db.get_structures_filenames(pkl_dataset, extension=ext)]

    # Filter DB5 filenames to unbound type and get all work filenames
    requested_filenames = [filename for filename in requested_filenames
                           if (source_type.lower() == 'db5' and '_u_' in filename)
                           or (source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri', 'input','pdbbind'])]                       
    #                       or (source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri', 'input'])]
    requested_keys = [db.get_pdb_name(x) for x in requested_filenames]
    #print(requested_keys)
    requested_pdb_codes = [db.get_pdb_code(x) for x in requested_filenames]
    #produced_filenames_path = os.path.join(output_dir, 'PSAIA', source_type.upper())
    produced_filenames_path = os.path.join(output_dir, 'PSAIA')
    if not os.path.exists(produced_filenames_path): os.makedirs(produced_filenames_path)
    produced_filenames = [path.as_posix() for path in Path(produced_filenames_path).rglob('*.tbl')]
    produced_keys = [db.get_pdb_code(x) for x in produced_filenames]
    work_keys = [key for key, pdb_code in zip(requested_keys, requested_pdb_codes) if pdb_code not in produced_keys and 'ligand' not in key]
    """ format_pdb_code_for_inputs = lambda pdb_code, source_type: pdb_code[1:3] \ """
    """     if source_type.lower() in ['db5', 'input'] \ """
    """     else pdb_code.upper() """
    format_pdb_code_for_inputs = lambda pdb_code, source_type: pdb_code[1:3] \
        if source_type.lower() in ['db5', 'input'] \
        else pdb_code.lower()
    if source_type.lower() == 'rcsb' or source_type.lower() == 'casp_capri':
        work_filenames = [os.path.join(pdb_dataset, db.get_pdb_code(work_key)[1:3], work_key)
                          for work_key in work_keys]
    else:
        work_filenames = [os.path.join(pdb_dataset,
                                       format_pdb_code_for_inputs(db.get_pdb_code(work_key), source_type),
                                       work_key)
                          for work_key in work_keys]

    # Remove any duplicate filenames
    work_filenames = list(set(work_filenames))

    # Exit early if no inputs need to processed
    log_conservation.info("{:} PDB files to process with PSAIA".format(len(work_filenames)))

    # Create comprehensive filename list for PSAIA to single-threadedly process for requested features (e.g. protrusion)
    #file_list_file = os.path.join(output_dir, 'PSAIA', source_type.upper(), 'pdb_list.fls')
    #file_list_file = os.path.join(produced_filenames_path,'pdb_list.fls')
    file_list_file = os.path.join(output_dir,'pdb_list.fls')
    
    with open(file_list_file, 'w') as file:
        for requested_pdb_filename in work_filenames:
            file.write(f'{requested_pdb_filename}\n')

    inputs = [(psaia_dir, psaia_config_file, file_list_file)]
    par.submit_jobs(map_protrusion_indices, inputs, 1)  # PSAIA is inherently single-threaded in execution


def map_all_pssms(pkl_dataset, pruned_dataset, blastdb, output_dir, num_cpus, source_type, rank, size):
    ext = '.pkl'
    if source_type.lower() == 'rcsb':  # Filter out pairs that did not survive pruning previously to reduce complexity
        pruned_pdb_names = [db.get_pdb_name(filename)
                            for filename in db.get_structures_filenames(pruned_dataset, extension='.dill')]
        requested_filenames = [
            os.path.join(pkl_dataset, db.get_pdb_code(pruned_pdb_name)[1:3], pruned_pdb_name.split('_')[0] + ext)
            for pruned_pdb_name in pruned_pdb_names
        ]
    else:  # DB5 does not employ pair pruning, so there are no pairs to filter
        requested_filenames = [filename for filename in db.get_structures_filenames(pkl_dataset, extension=ext)]

    # Filter DB5 filenames to unbound type and get all work filenames
    requested_filenames = [filename for filename in requested_filenames
                           if (source_type.lower() == 'db5' and '_u_' in filename)
                           or (source_type.lower() == 'rcsb')
                           or (source_type.lower() == 'evcoupling')
                           or (source_type.lower() == 'casp_capri')]
    requested_keys = [db.get_pdb_name(x) for x in requested_filenames]
    produced_filenames = db.get_structures_filenames(output_dir, extension='.pkl')
    produced_keys = [db.get_pdb_name(x) for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    if source_type.lower() == 'rcsb' or source_type.lower() == 'casp_capri':
        work_filenames = [os.path.join(pkl_dataset, db.get_pdb_code(work_key)[1:3], work_key + ext)
                          for work_key in work_keys]
    else:
        work_filenames = [os.path.join(pkl_dataset, db.get_pdb_code(work_key)[1:3].upper(), work_key + ext)
                          for work_key in work_keys]

    # Reserve an equally-sized portion of the full work load for a given rank in the MPI world
    work_filenames = list(set(work_filenames))
    work_filename_rank_batches = slice_list(work_filenames, size)
    work_filenames = work_filename_rank_batches[rank]

    # Remove any duplicate filenames
    log_conservation.info("{:} requested keys, {:} produced keys, {:} work filenames".format(len(requested_keys),
                                                                                    len(produced_keys),
                                                                                    len(work_filenames)))

    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
        output_filenames.append(sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".pkl")

    inputs = [(key, blastdb, output)
              for key, output in zip(work_filenames, output_filenames)]
    par.submit_jobs(map_pssms, inputs, num_cpus)


def map_all_profile_hmms(pkl_dataset, pruned_dataset, output_dir, hhsuite_db, num_cpu_jobs,
                         num_cpus_per_job, source_type, num_iter, rank, size, write_file):
    ext = '.pkl'
    if write_file:
        if source_type.lower() == 'rcsb':
            # Filter out pairs that did not survive pruning previously to reduce complexity
            pruned_pdb_names = [db.get_pdb_name(filename)
                                for filename in db.get_structures_filenames(pruned_dataset, extension='.dill')]
            requested_filenames = [
                os.path.join(pkl_dataset, db.get_pdb_code(pruned_pdb_name)[1:3], pruned_pdb_name.split('_')[0] + ext)
                for pruned_pdb_name in pruned_pdb_names
            ]
        else:  # DB5 does not employ pair pruning, so there are no pairs to filter
            requested_filenames = [filename for filename in db.get_structures_filenames(pkl_dataset, extension=ext)]

        # Filter DB5 filenames to unbound type and get all work filenames
        requested_filenames = [filename for filename in requested_filenames
                               if (source_type.lower() == 'db5' and '_u_' in filename)
                               or (source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri', 'input', 'pdbbind'])]
        #                       or (source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri', 'input'])]
        requested_keys = [db.get_pdb_name(x) for x in requested_filenames]
        produced_filenames = db.get_structures_filenames(output_dir, extension='.pkl')
        produced_keys = [db.get_pdb_name(x) for x in produced_filenames]
        #work_keys = [key for key in requested_keys if key not in produced_keys]
        work_keys = [key for key in requested_keys if key not in produced_keys and 'ligand'not in key]
        """ establish_pdb_code_case = lambda pdb_code, source_type: pdb_code.lower() \ """
        """     if source_type.lower() == 'casp_capri' \ """
        """     else pdb_code.upper() """
        establish_pdb_code_case = lambda pdb_code, source_type: pdb_code.lower() \
            if source_type.lower() in ['casp_capri','pdbbind']\
            else pdb_code.upper()
        work_filenames = [os.path.join(pkl_dataset,
                                       establish_pdb_code_case(db.get_pdb_code(work_key), source_type)[1:3],
                                       work_key + ext)
                          for work_key in work_keys]

        # Remove any duplicate filenames
        work_filenames = list(set(work_filenames))
        log_conservation.info("{:} requested keys, {:} produced keys, {:} work filenames".format(len(requested_keys),
                                                                                        len(produced_keys),
                                                                                        len(work_filenames)))

        if source_type.lower() == 'input':
            # Directly generate profile HMM features after aggregating input filenames
            log_conservation.info("{:} work filenames".format(len(work_filenames)))

            output_filenames = []
            for pdb_filename in work_filenames:
                sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir, exist_ok=True)
                output_filenames.append(sub_dir + '/' + db.get_pdb_name(pdb_filename) + '.pkl')

            inputs = [(num_cpus_per_job, key, output, hhsuite_db, source_type, num_iter)
                      for key, output in zip(work_filenames, output_filenames)]
            par.submit_jobs(map_profile_hmms, inputs, num_cpu_jobs)
        else:
            # Write out a local file containing all work filenames
            temp_df = pd.DataFrame({'filename': work_filenames})
            temp_df.to_csv(f'{output_dir}/{source_type}_work_filenames.csv')
            log_conservation.info('File containing work filenames written to storage. Exiting...')

    # Read from previously-created work filenames CSV
    else:
        work_filenames = pd.read_csv(f'{output_dir}/{source_type}_work_filenames.csv').iloc[:, 1].to_list()
        work_filenames = list(set(work_filenames))  # Remove any duplicate filenames

        # Reserve an equally-sized portion of the full work load for a given rank in the MPI world
        work_filename_rank_batches = slice_list(work_filenames, size)
        work_filenames = work_filename_rank_batches[rank]

        log_conservation.info("{:} work filenames".format(len(work_filenames)))

        output_filenames = []
        for pdb_filename in work_filenames:
            sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
            output_filenames.append(sub_dir + '/' + db.get_pdb_name(pdb_filename) + '.pkl')

        inputs = [(num_cpus_per_job, key, output, hhsuite_db, source_type, num_iter)
                  for key, output in zip(work_filenames, output_filenames)]
        par.submit_jobs(map_profile_hmms, inputs, num_cpu_jobs)
