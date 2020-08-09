import warnings
warnings.filterwarnings("ignore")
import os
import gzip
import pickle
import pandas as pd
import datetime
from datetime import datetime as dt
import pdb
import numpy as np
import time
import random
import matplotlib.pyplot as plt
random.seed(0)

import CSDK
import CDK
import FOREST 

from utils import hastook


test_start = time.time()

config = {
        'corpus_path': './data/digg_500user/',  # {memetracker_dataset_1k_user, twitter7_dataset_small, digg_500user}
        'output_path': './data/digg_CDK_HAC_s2_p1dot2',

        'diffuser': 'CDK',  # {'CDK', 'CSDK', 'FOREST'}

        'num_scales': 2,
        'test_epoch':2666,
        'embedding_scale': 0   # no need to change everytime
            }

print(config)

"""
Obtain D0 test data (ground truth)
"""
with open(os.path.join(config['corpus_path'], 'diffpath_user_test.pickle'), 'rb') as f:
    diffpath_user_test = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'diffpath_time_test.pickle'), 'rb') as f:
    diffpath_time_test = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'diffpath_info_test.pickle'), 'rb') as f:
    diffpath_info_test = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'diffpath_info_reverse_test.pickle'), 'rb') as f:
    diffpath_info_reverse_test = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_infoid_infoname.pickle'), 'rb') as f:
    infoid_infoname_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_infoid_infoname_reverse.pickle'), 'rb') as f:
    infoid_infoname_reverse_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_uid_uname.pickle'), 'rb') as f:
    uid_uname_D0 = pickle.load(f)
with open(os.path.join(config['corpus_path'], 'D0_uid_uname_reverse.pickle'), 'rb') as f:
    uid_uname_reverse_D0 = pickle.load(f)


corpus_test = (diffpath_user_test, diffpath_time_test,
               diffpath_info_test, diffpath_info_reverse_test,
               infoid_infoname_D0, infoid_infoname_reverse_D0,
               uid_uname_D0, uid_uname_reverse_D0)

print("the number of users: {}".format(len(uid_uname_D0)))
print("the number of information: {}".format(len(infoid_infoname_D0)))
print("the number of diffusion paths: {}".format(len(diffpath_user_test)))


"""
Testing
"""
if config['diffuser'] == 'CDK':
    test = CDK.test
elif config['diffuser'] == 'CSDK':
    test = CSDK.test
elif config['diffuser'] == 'FOREST':
    test = FOREST.test
else:
    print("Undefined diffuser! Please check config!")
    pdb.set_trace()

test(config=config, save_dir=config['output_path'], test_epoch=config['test_epoch'], embedding_scale=config['embedding_scale'], corpus_test=corpus_test)

print("Finished!")
hastook(test_start)
