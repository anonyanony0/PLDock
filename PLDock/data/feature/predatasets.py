import os
import re
import parallel as par
from pypdb import get_info
from PLDock.PLDock.utils.pdb import down_pdb
from PLDock.utils.utils import get_global_node_rank, save_first_ligand
from PLDock import log


class predatasets():
    def __init__(self,
                 data_dir: str = os.getcwd(),
                 log_file: str = None):
        """Initialize

        Args:
            data_dir (str, optional): Dir to place data.
                Defaults to 'os.getcwd()'.
            log_file (str, optional): Path of logs. Defaults to None.
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.logger = log(os.path.join(
            self.data_dir, 'prepdb.log')) if log_file else log()

    def get_list_from_dir(self, list_dir):
        files = [os.path.join(list_dir, i) for i in os.listdir(
            list_dir) if os.path.isfile(os.path.join(list_dir, i))]
        all_pdb = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                all_pdb += [i.lower() for i in line.rstrip().split(',')]
        return all_pdb

    def down_extract(self, pdb):
        all_info = get_info(pdb)
        try:
            res = all_info['pdbx_vrpt_summary']['pdbresolution']
        except:
            res = ''
        try:
            ec = all_info['struct']['pdbx_descriptor'].split(
                '(')[1].rstrip(')')
            m = r'\((E?\.?C?\.?\d\..{1,2}\..{1,2}\..{1,3})\)'
            ec = re.search(re.compile(m), ec).group(1)
        except:
            ec = ''
        try:
            rd = all_info['rcsb_accession_info']['initial_release_date'].split('T')[0]
        except:
            rd = ''
        try:
            ligands = all_info['rcsb_entry_info']['nonpolymer_bound_components']
        except:
            ligands = []
        act_types = ['Kd', 'Ki', 'IC50', 'EC50',
                     '&Delta;G', '-T&Delta;S', '&Delta;H', 'Ka']
        afd = {ligand: {} for ligand in ligands}
        if 'rcsb_binding_affinity' in all_info:
            for af in all_info['rcsb_binding_affinity']:
                if not af['comp_id'] in afd:
                    afd[af['comp_id']] = {}
                if af['unit'] in ['nM', 'kJ/mol', 'M^-1'] and (af['type'] in act_types):
                    try:
                        afd[af['comp_id']][af['type']].append(af['value'])
                    except:
                        afd[af['comp_id']][af['type']] = [af['value']]
                else:
                    self.logger.warning(
                        f"{pdb},{af['comp_id']} : {af['type']} , {af['unit']} !")
        for k, v in afd.items():
            if v:
                for name, value in v.items():
                    afd[k][name] = sum(value)/len(value)
        #for i in list(afd.keys()):
        #    if fileter_res_by_name(i):
        #        del afd[i]
        wlines = []
        if afd:
            tmp_dir = os.path.join(self.data_dir, 'tmp')
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            structure_dir = os.path.join(self.data_dir, 'structure')
            if not os.path.exists(structure_dir):
                os.makedirs(structure_dir)
            down_pdb(pdb, tmp_dir)
            pdb_file = os.path.join(tmp_dir, f'{pdb}.pdb')
            for ligand, afs in afd.items():
                line = f'{pdb},{ligand}'
                for af_type in act_types:
                    try:
                        line += f',{afs[af_type]:.2f}'
                    except KeyError:
                        line += ','
                wlines.append(f'{line},{res},{ec},{rd}\n')
                if os.path.exists(pdb_file):
                    ligand_dir = os.path.join(structure_dir, f'{pdb}_{ligand}')
                    if not os.path.exists(ligand_dir):
                        os.makedirs(ligand_dir)
                    save_first_ligand(pdb_file, ligand,
                                      os.path.join(ligand_dir, pdb))
            if os.path.exists(pdb_file):
                os.remove(pdb_file)
        else:
            wlines.append(f'{pdb},,,,,,,,,,{res},{ec}{rd}\n')

        return wlines

    def write_files(self, pdb_list, num_cpus: int = 1, rank: int = 0, size: int = 1):
        rank = get_global_node_rank(rank, size)
        if rank == 0:
            inputs = [(pdb,) for pdb in pdb_list]
            outs = par.submit_jobs(self.down_extract, inputs, num_cpus)
            wlines = []
            for i in outs:
                wlines += i
            with open(os.path.join(self.data_dir, 'pdb.csv'), 'w') as f:
                f.writelines(wlines)
        return
