import os
from loguru import logger
from PLDock import log
from PLDock.utils.utils import get_global_node_rank
from PLDock.data.feature.preprocess import preprocess
from PLDock.data.feature.psaia_feature import psaia_feature
from PLDock.data.feature.hhsuite_feature import hhsuite_feature
from PLDock.data.feature.postprocess import postprocess
from PLDock.data.feature.impute import impute
logger = log()

def feature_adir(input_dir,output_dir,precess_dir,psaia_dir,hhsuite_dir,pruned_pairs_dir,post_dir,hhsuite_db,num_cpus):
    #if not os.path.join(output_dir): os.mkdir(output_dir)
    preprocess(input_dir=input_dir,output_dir=output_dir,num_cpus=num_cpus)
    psaia_feature(input_dir,precess_dir,psaia_dir)
    hhsuite_feature(precess_dir, hhsuite_dir, hhsuite_db, num_cpu_jobs=num_cpus)
    postprocess(input_dir,pruned_pairs_dir,psaia_dir,post_dir,num_cpus=num_cpus)
    impute(post_dir,num_cpus=num_cpus)
    logger.info(f'{input_dir} done')
    return

def feature(input_dir,output_dir,hhsuite_db,num_cpus):
    precess_dir = os.path.join(output_dir,'parsed')
    psaia_dir =  os.path.join(output_dir,'external_feats')
    hhsuite_dir =  os.path.join(output_dir,'external_feats','HHSUITE','PDBBIND')
    pruned_pairs_dir =  os.path.join(output_dir, 'pairs')
    post_dir = os.path.join(output_dir,'final','raw')
    feature_adir(input_dir,output_dir,precess_dir,psaia_dir,hhsuite_dir,pruned_pairs_dir,post_dir,hhsuite_db,num_cpus)
    logger.info('All done')
