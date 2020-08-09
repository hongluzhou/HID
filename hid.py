import os
import gzip
import pickle
import pdb
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from utils import upscaling
from utils import downscaling
from utils import hastook

import CDK
import CSDK
import NDM
import FOREST


hierMID_start = time.time()


# config
def create_config():
    config = {
        'corpus_path': './data/memetracker_dataset_1k_user/',   # {memetracker_dataset_1k_user, twitter7_dataset_small, digg_500user, synthetic_data_split}
        'output_path': './data/hierMID_output/',

        'num_scales': 1,
        'coarse_portion': 2,  # float greater than 1
        'user_emb_dim': 64,
        'max_epochs': 8000,

        'diffuser': 'CSDK',  # {'CDK', 'CSDK', 'FOREST'}
        'transition_matrix': {
            'A': True,
            'Anei': False,
            'Aori': False
                             },
        'up_operator': {
            'type': 'HAC',  # {'HAC', 'Kmeans', 'Spectral'}
            'affinity': 'l1',
            'linkage': 'average'
                       }
    }
    return config


config = create_config()


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, default=config['corpus_path'], help='dataset directory')
parser.add_argument('--output_path', type=str, default=config['output_path'], help='experiment output directory to save files to be generated')
parser.add_argument('--num_scales', type=int, default=config['num_scales'], help='the number of scales you wish to run (0 means without using HierMID)')
parser.add_argument('--coarse_portion', type=float, default=config['coarse_portion'], help='the portion to coarse')
parser.add_argument('--max_epochs', type=int, default=config['max_epochs'], help='maximun number of epochs (per scale)')
parser.add_argument('--diffuser', type=str, default=config['diffuser'], help='the baseline diffuser (currently support CDK, CSDK)')
parser.add_argument('--upoperator', type=str, default=config['up_operator']['type'], help='upscaling operator type (currently support HAC, Kmeans, Spectral)')
args = parser.parse_args()


# argument parser will overwrite the above default config
if args.corpus_path != config['corpus_path']:
    config['corpus_path'] = args.corpus_path
if args.output_path != config['output_path']:
    config['output_path'] = args.output_path
if args.num_scales != config['num_scales']:
    config['num_scales'] = args.num_scales
if args.num_scales != config['coarse_portion']:
    config['coarse_portion'] = args.coarse_portion
if args.max_epochs != config['max_epochs']:
    config['max_epochs'] = args.max_epochs
if args.diffuser != config['diffuser']:
    config['diffuser'] = args.diffuser
if args.upoperator != config['up_operator']['type']:
    config['up_operator']['type'] = args.upoperator

print(config)

if not os.path.exists(config['output_path']):
    os.makedirs(config['output_path'])

with open(os.path.join(config['output_path'], 'config.pickle'), 'wb') as f:
    pickle.dump(config, f)

"""
Obtain D0 train data
"""
with open(os.path.join(config['corpus_path'], 'D0_diffpath_user.pickle'), 'rb') as f:
    diffpath_user_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_diffpath_time.pickle'), 'rb') as f:
    diffpath_time_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_diffpath_info.pickle'), 'rb') as f:
    diffpath_info_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_diffpath_info_reverse.pickle'), 'rb') as f:
    diffpath_info_reverse_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_infoid_infoname.pickle'), 'rb') as f:
    infoid_infoname_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_infoid_infoname_reverse.pickle'), 'rb') as f:
    infoid_infoname_reverse_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_uid_uname.pickle'), 'rb') as f:
    uid_uname_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_uid_uname_reverse.pickle'), 'rb') as f:
    uid_uname_reverse_D0 = pickle.load(f)

corpus_D0 = (diffpath_user_D0, diffpath_time_D0,
             diffpath_info_D0, diffpath_info_reverse_D0,
             infoid_infoname_D0, infoid_infoname_reverse_D0,
             uid_uname_D0, uid_uname_reverse_D0)

print("the number of users: {}".format(len(uid_uname_D0)))
print("the number of information: {}".format(len(infoid_infoname_D0)))
print("the number of diffusion paths: {}".format(len(diffpath_user_D0)))

"""
Upscaling
"""
start_time = time.time()
num_user_list, diffp = upscaling(config, config['num_scales'], corpus_D0, config['up_operator'], config['transition_matrix'])
config['upscaling_time_in_seconds'] = time.time() - start_time
print("upscaling finished")
hastook(start_time)
print("num_user_list: {}".format(num_user_list))
print("num_diffp_list: {}".format(diffp))
config['num_user_list']  = num_user_list
config['num_diffp_list'] = diffp


"""
Learn embedding at Ds scale
"""
scale = config['num_scales']
print("learn embedding at D{} scale".format(scale))
with open(os.path.join(config['output_path'], 'D{}_diffpath_user.pickle'.format(scale)), 'rb') as f:
    diffpath_user_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_diffpath_time.pickle'.format(scale)), 'rb') as f:
    diffpath_time_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_diffpath_info.pickle'.format(scale)), 'rb') as f:
    diffpath_info_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_diffpath_info_reverse.pickle'.format(scale)), 'rb') as f:
    diffpath_info_reverse_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_infoid_infoname.pickle'.format(scale)), 'rb') as f:
    infoid_infoname_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_infoid_infoname_reverse.pickle'.format(scale)), 'rb') as f:
    infoid_infoname_reverse_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_uid_uname.pickle'.format(scale)), 'rb') as f:
    uid_uname_Ds = pickle.load(f)
with open(os.path.join(config['output_path'], 'D{}_uid_uname_reverse.pickle'.format(scale)), 'rb') as f:
    uid_uname_reverse_Ds = pickle.load(f)
corpus_Ds = (diffpath_user_Ds, diffpath_time_Ds,
             diffpath_info_Ds, diffpath_info_reverse_Ds,
             infoid_infoname_Ds, infoid_infoname_reverse_Ds,
             uid_uname_Ds, uid_uname_reverse_Ds)
print("the number of users at D{} scale: {}".format(scale, len(uid_uname_Ds)))
print("the number of information at D{} scale: {}".format(scale, len(infoid_infoname_Ds)))
print("the number of diffusion paths at D{} scale: {}".format(scale, len(diffpath_user_Ds)))

if config['diffuser'] == 'CDK':
    embed = CDK.embed
    embedding_vec = embed(config, scale, corpus_Ds)
elif config['diffuser'] == 'CSDK':
    embed = CSDK.embed
    embedding_vec = embed(config, scale, corpus_Ds)
elif config['diffuser'] == 'NDM':
    embed = NDM.embed
    embedding_vec = embed(config, scale, corpus_Ds)
elif config['diffuser'] == 'FOREST':
    embed = FOREST.embed
    embedding_vec = embed(config, scale, corpus_Ds, parser)
else:
    print("Undefined diffuser! Please check config!")
    pdb.set_trace()


"""
Downscaling and for loop embedding-learning
"""
corpus_c = corpus_Ds
while scale > 0:
    (corpus_f, embedding_vec_f) = downscaling(config, scale, corpus_c, embedding_vec)
    scale -= 1
    print("learn embedding at D{} scale".format(scale))
    if config['diffuser'] == 'FOREST':
        embedding_vec_f = embed(config, scale, corpus_f, parser, embedding_vec_f)
    else:
        embedding_vec_f = embed(config, scale, corpus_f, embedding_vec_f)

    embedding_vec = embedding_vec_f
    corpus_c = corpus_f

print("Finished!")
hastook(hierMID_start)
config['train_time_in_seconds'] = time.time() - hierMID_start - config['upscaling_time_in_seconds']
config['running_time_in_seconds'] = time.time() - hierMID_start
print(config['output_path'])
with open(os.path.join(config['output_path'], 'config.pickle'), 'wb') as f:
    pickle.dump(config, f)
