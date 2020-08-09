import os
import time
import pdb
import pickle
import numpy as np
import random
import string
import math
import logging
import datetime
from datetime import datetime as dt
from collections import Counter
from scipy.sparse import lil_matrix, hstack
from multiprocessing import Pool
import multiprocessing
num_cpu = multiprocessing.cpu_count()
# random.seed(0)

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


""" Refs:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec

https://scikit-learn.org/stable/modules/clustering.html
"""


def utc_timestamp(timestamp, string_time=False, time_format="%Y-%m-%dT%H:%M:%SZ"):
    if string_time:
        timestamp = dt.strptime(timestamp, time_format) 
    
    base_timestep = datetime.datetime(1970, 1, 1)
    curr_utc = int((timestamp - base_timestep).total_seconds())
    return curr_utc


def hastook(start_time):
    if time.time() - start_time > 3600:
        print("took {} h".format(round((time.time() - start_time) / 3600.0), 2))
    elif time.time() - start_time > 60:
        print("took {} m".format(round((time.time() - start_time) / 60.0), 2))
    else:
        print("took {} s".format(round(time.time() - start_time), 2))
    return


def set_logger(logger_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(logger_name, mode='w')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info(os.path.basename(__file__))
    logger.info(dt.now().strftime('%m/%d/%Y %I:%M:%S %p'))

    return logger


def print_and_log(logger, msg):
    print(msg)
    logger.info(msg)


"""
A_pk: adjacency matrix of diffusion path p with step k
"""
def adjacency_matrix(diffpath, num_user, step, uname_uid):
    adjmtx = lil_matrix((num_user, num_user), dtype=np.int8)
    for i in range(len(diffpath) - step):
        srt_node = uname_uid[diffpath[i]]
        dst_node = uname_uid[diffpath[i + step]]
        adjmtx[srt_node, dst_node] += 1
    return adjmtx


"""
A_p: adjacency matrix of diffusion path p with all accumulated steps
"""
def diffpath_adjacency_matrix_cooccur(diffpath, num_user, uname_uid):
    diffpath_adjmtx = lil_matrix((num_user, num_user), dtype=np.int8)
    max_step = len(diffpath) - 1
    for step in range(1, max_step + 1):
        adjmtx = adjacency_matrix(diffpath, num_user, step, uname_uid)
        diffpath_adjmtx += adjmtx
    return diffpath_adjmtx


"""
A_p_neighbor: adjacency matrix of diffusion path p with accumulated steps
              up to the neighboring threshold
"""
def diffpath_adjacency_matrix_thresh(diffpath, num_user, uname_uid, threshold):
    diffpath_adjmtx = lil_matrix((num_user, num_user), dtype=np.int8)
    max_step = min(len(diffpath) - 1, threshold)
    for step in range(1, max_step + 1):
        adjmtx = adjacency_matrix(diffpath, num_user, step, uname_uid)
        diffpath_adjmtx += adjmtx
    return diffpath_adjmtx


"""
A_p_origin: adjacency matrix of diffusion path p regarding origin
"""
def diffpath_adjacency_matrix_origin(diffpath, num_user, uname_uid):
    # diffpath_adjmtx = lil_matrix((num_user, num_user), dtype=np.int8)
    max_step = len(diffpath) - 1
    EA = uname_uid[diffpath[0]]
    iEA = lil_matrix((num_user, 1), dtype=np.int8)
    iEA[EA] = 1

    tmp = lil_matrix((num_user, num_user), dtype=np.int8)
    for step in range(1, max_step + 1):
        adjmtx = adjacency_matrix(diffpath, num_user, step, uname_uid)
        tmp += adjmtx
    diffpath_adjmtx = iEA.dot((tmp[EA]).reshape(1, num_user))
    return diffpath_adjmtx


"""
A_cooccur
"""
def multiprocess_A_cooccur(diffpaths_user, num_user, uname_uid):
    list_of_args_tuple = []
    for diffpath in diffpaths_user:
        list_of_args_tuple.append((diffpath, num_user, uname_uid))
    with Pool(num_cpu) as pool:
        diffpaths_adjmtx = pool.starmap(diffpath_adjacency_matrix_cooccur, list_of_args_tuple)
    A_cooccur = sum(diffpaths_adjmtx) + sum(diffpaths_adjmtx).T
    return A_cooccur


def singleprocess_A_cooccur(diffpaths_user, num_user, uname_uid):
    diffpaths_adjmtx = []
    for i in range(len(diffpaths_user)):
        print("A_cooccur", i + 1, len(diffpaths_user))
        diffpath = diffpaths_user[i]
        diffpaths_adjmtx.append(
            diffpath_adjacency_matrix_cooccur(diffpath, num_user, uname_uid))
    A_cooccur = sum(diffpaths_adjmtx) + sum(diffpaths_adjmtx).T
    return A_cooccur


"""
A_neighbor
"""
def multiprocess_A_neighbor(diffpaths_user, num_user, uname_uid, threshold=2):
    list_of_args_tuple = []
    for diffpath in diffpaths_user:
        list_of_args_tuple.append((diffpath, num_user, uname_uid, threshold))
    with Pool(num_cpu) as pool:
        diffpaths_adjmtx = pool.starmap(diffpath_adjacency_matrix_thresh, list_of_args_tuple)
    A_neighbor = sum(diffpaths_adjmtx)
    return A_neighbor


def singleprocess_A_neighbor(diffpaths_user, num_user, uname_uid, threshold=2):
    diffpaths_adjmtx = []
    for i in range(len(diffpaths_user)):
        print("A_neighbor", i + 1, len(diffpaths_user))
        diffpath = diffpaths_user[i]
        diffpaths_adjmtx.append(
            diffpath_adjacency_matrix_thresh(diffpath, num_user, uname_uid, threshold))
    A_neighbor = sum(diffpaths_adjmtx)
    return A_neighbor


"""
A_origin
"""
def multiprocess_A_origin(diffpaths_user, num_user, uname_uid):
    list_of_args_tuple = []
    for diffpath in diffpaths_user:
        list_of_args_tuple.append((diffpath, num_user, uname_uid))
    with Pool(num_cpu) as pool:
        diffpaths_adjmtx = pool.starmap(diffpath_adjacency_matrix_origin, list_of_args_tuple)
    A_origin = sum(diffpaths_adjmtx)
    return A_origin


def singleprocess_A_origin(diffpaths_user, num_user, uname_uid):
    diffpaths_adjmtx = []
    for i in range(len(diffpaths_user)):
        print("A_origin", i + 1, len(diffpaths_user))
        diffpath = diffpaths_user[i]
        diffpaths_adjmtx.append(
            diffpath_adjacency_matrix_origin(diffpath, num_user, uname_uid))
    A_origin = sum(diffpaths_adjmtx)
    return A_origin


def randomString(scale, num_total_user):
    stringLength = scale * (math.ceil(math.log2(num_total_user)))
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(stringLength))
    return 'scale_' + str(scale) + random_string


def upscaling(config, num_scales, corpus_D0, up_operator, transition_matrix):
    
    num_user_list = []
    diffp = []


    (diffpath_user, diffpath_time, diffpath_info, diffpath_info_reverse,
     infoid_infoname, infoid_infoname_reverse,
     uid_uname, uid_uname_reverse) = corpus_D0

    with open(os.path.join(config['output_path'], 'D0_diffpath_user.pickle'), 'wb') as f:
        pickle.dump(diffpath_user, f)
    with open(os.path.join(config['output_path'], 'D0_diffpath_time.pickle'), 'wb') as f:
        pickle.dump(diffpath_time, f)
    with open(os.path.join(config['output_path'], 'D0_diffpath_info.pickle'), 'wb') as f:
        pickle.dump(diffpath_info, f)
    with open(os.path.join(config['output_path'], 'D0_diffpath_info_reverse.pickle'), 'wb') as f:
        pickle.dump(diffpath_info_reverse, f)
    with open(os.path.join(config['output_path'], 'D0_infoid_infoname.pickle'), 'wb') as f:
        pickle.dump(infoid_infoname, f)
    with open(os.path.join(config['output_path'], 'D0_infoid_infoname_reverse.pickle'), 'wb') as f:
        pickle.dump(infoid_infoname_reverse, f)
    with open(os.path.join(config['output_path'], 'D0_uid_uname.pickle'), 'wb') as f:
        pickle.dump(uid_uname, f)
    with open(os.path.join(config['output_path'], 'D0_uid_uname_reverse.pickle'), 'wb') as f:
        pickle.dump(uid_uname_reverse, f)
    
    with open(os.path.join(config['output_path'], 'num_over_scales.pickle'), 'wb') as f:
        pickle.dump([num_user_list, diffp], f)

    num_total_user = len(uid_uname)
    print("num_total_user: {}".format(num_total_user))
    
    num_user_list.append(num_total_user)
    diffp.append(len(diffpath_user))

    if num_scales == 0:
        return num_user_list, diffp

    logger = set_logger(os.path.join(config['output_path'], 'upscaling.log'))

    print_and_log(logger, "starting upscaling")

    num_user = len(uid_uname_reverse)

    print_and_log(logger, "number of scales: {}".format(num_scales))

    for scale in range(1, num_scales + 1):
        print_and_log(logger, "\nupscaling at scale {}".format(scale))

        count_tramtx = 0

        if transition_matrix['A']:
            count_tramtx += 1

            start_time = time.time()
            A_cooccur = multiprocess_A_cooccur(diffpath_user, num_user, uid_uname_reverse)
            hastook(start_time)
            print_and_log(logger, "A_cooccur obtaind with shape: {}".format(A_cooccur.shape))

            with open(os.path.join(config['output_path'], "D{}_A_cooccur.pickle".format(scale - 1)), 'wb') as f:
                pickle.dump(A_cooccur, f)

        if transition_matrix['Anei']:
            count_tramtx += 1

            start_time = time.time()
            A_neighbor = multiprocess_A_neighbor(diffpath_user, num_user, uid_uname_reverse)
            hastook(start_time)
            print_and_log(logger, "A_neighbor obtaind with shape: {}".format(A_neighbor.shape))

            with open(os.path.join(config['output_path'], "D{}_A_neighbor.pickle".format(scale - 1)), 'wb') as f:
                pickle.dump(A_neighbor, f)

        if transition_matrix['Aori']:
            count_tramtx += 1

            start_time = time.time()
            A_origin = multiprocess_A_origin(diffpath_user, num_user, uid_uname_reverse)
            hastook(start_time)
            print_and_log(logger, "A_origin obtaind with shape: {}".format(A_origin.shape))

            with open(os.path.join(config['output_path'], "D{}_A_origin.pickle".format(scale - 1)), 'wb') as f:
                pickle.dump(A_origin, f)

        if count_tramtx == 1:
            if transition_matrix['A']:
                A = A_cooccur
            elif transition_matrix['Anei']:
                A = A_neighbor
            elif transition_matrix['Aori']:
                A = A_origin
            else:
                print_and_log(logger, 'check your transition matrix specified!')
                pdb.set_trace()
        else:
            start_time = time.time()

            if transition_matrix['A'] and transition_matrix['Anei'] and transition_matrix['Aori']:
                A = hstack([A_cooccur, A_neighbor, A_origin])
            elif transition_matrix['A'] and transition_matrix['Anei']:
                A = hstack([A_cooccur, A_neighbor])
            elif transition_matrix['A'] and transition_matrix['Aori']:
                A = hstack([A_cooccur, A_origin])
            elif transition_matrix['Anei'] and transition_matrix['Aori']:
                A = hstack([A_neighbor, A_origin])
            else:
                print_and_log(logger, 'check your transition matrix specified!')
                pdb.set_trace()
            # A = np.concatenate((A_cooccur, A_neighbor, A_origin), axis=1)

            hastook(start_time)
            print_and_log(logger, "A obtaind with shape: {}".format(A.shape))

        with open(os.path.join(config['output_path'], "D{}_A.pickle".format(scale - 1)), 'wb') as f:
            pickle.dump(A, f)

        """
        with open(os.path.join(config['output_path'], "D{}_A.pickle".format(scale - 1)), 'rb') as f:
            A = pickle.load(f)
        """

        # turn sparse matrix into numpy array
        A = A.toarray()

        # clustering
        print_and_log(logger, up_operator)
        start_time = time.time()
        if up_operator['type'] == 'HAC':
            print_and_log(logger, "HAC clustering")
            model = AgglomerativeClustering(n_clusters=math.ceil(num_user / config['coarse_portion']), affinity=up_operator['affinity'], linkage=up_operator['linkage'])
        elif up_operator['type'] == 'Kmeans':
            print_and_log(logger, "K-Means clustering")
            model = KMeans(n_clusters=math.ceil(num_user / 2), verbose=0, random_state=0, n_jobs=-1)
        elif up_operator['type'] == 'Spectral':
            print_and_log(logger, "Spectral clustering")
            model = SpectralClustering(n_clusters=math.ceil(num_user / 2), affinity='rbf',
                                       assign_labels="discretize", random_state=0, n_jobs=-1)
        else:
            print_and_log(logger, "{} not implemented yet!".format(up_operator['type']))
            os._exit(0)
        model = model.fit(A)
        num_hyperusers = max(model.labels_) + 1
        hastook(start_time)
        print_and_log(logger, "scale {} has {} hyperusers".format(scale, num_hyperusers))

        # save clustering model and label
        with open(os.path.join(config['output_path'], "D{}_HAC_model.sav".format(scale - 1)), 'wb') as f:
            pickle.dump(A, f)
        with open(os.path.join(config['output_path'], "D{}_HAC_label.sav".format(scale - 1)), 'wb') as f:
            pickle.dump(model.labels_, f)

        # find hyperusers that have 1 node, i.e. no coarsening effect
        print_and_log(logger, "find hyperusers that have 1 node, i.e. no coarsening effect")
        hyperuser_no_coarsen_effect = set()
        num_hyperusers_counter = Counter(model.labels_)
        for hyperuser in num_hyperusers_counter:
            if num_hyperusers_counter[hyperuser] == 1:
                hyperuser_no_coarsen_effect.add(hyperuser)
        print_and_log(logger, "{} such users".format(len(hyperuser_no_coarsen_effect)))

        # uid_multiscales_Di
        # diffpath_user_Di, diffpath_time_Di, diffpath_info_Di, diffpath_info_reverse_Di
        # infoid_infoname_Di, infoid_infoname_reverse_Di
        # uid_uname_Di, uid_uname_reverse_Di

        # 1. uid_uname_Di, uid_uname_reverse_Di, uname_multiscales_Di
        print("1. uid_uname_Di, uid_uname_reverse_Di, uname_multiscales_Di")
        uid_uname_Di = dict()
        uid_uname_reverse_Di = dict()
        uname_multiscales_Di = dict()

        uname_fine_coarse_map = dict()
        uname_coarse_fine_map = dict()

        if scale == 1:
            uname_multiscales = dict()
            for uname in uid_uname_reverse:
                uname_multiscales[uname] = [uname]

        for hyperuser in range(num_hyperusers):
            if hyperuser in hyperuser_no_coarsen_effect:
                tmp = np.where(model.labels_ == hyperuser)[0]
                idx = tmp[0]
                uid_uname_reverse_Di[uid_uname[idx]] = hyperuser
                uid_uname_Di[hyperuser] = uid_uname[idx]

                uname_multiscales_Di[uid_uname_Di[hyperuser]] = uname_multiscales[uid_uname[idx]]
                uname_fine_coarse_map[uid_uname[idx]] = uid_uname_Di[hyperuser]
                uname_coarse_fine_map[uid_uname_Di[hyperuser]] = set()
                uname_coarse_fine_map[uid_uname_Di[hyperuser]].add(uid_uname[idx])
            else:
                uid_uname_Di[hyperuser] = randomString(scale, num_total_user)
                uid_uname_reverse_Di[uid_uname_Di[hyperuser]] = hyperuser

                tmp = np.where(model.labels_ == hyperuser)[0]
                uname_multiscales_Di[uid_uname_Di[hyperuser]] = list(uname_multiscales[uid_uname[idx]] for idx in tmp)
                uname_coarse_fine_map[uid_uname_Di[hyperuser]] = set()
                for idx in tmp:
                    uname_fine_coarse_map[uid_uname[idx]] = uid_uname_Di[hyperuser]
                    uname_coarse_fine_map[uid_uname_Di[hyperuser]].add(uid_uname[idx])

        num_user_Di = len(uid_uname_Di)

        with open(os.path.join(config['output_path'], "D{}_uid_uname.pickle".format(scale)), 'wb') as f:
            pickle.dump(uid_uname_Di, f)
        with open(os.path.join(config['output_path'], "D{}_uid_uname_reverse.pickle".format(scale)), 'wb') as f:
            pickle.dump(uid_uname_reverse_Di, f)
        with open(os.path.join(config['output_path'], "D{}_uname_multiscales.pickle".format(scale)), 'wb') as f:
            pickle.dump(uname_multiscales_Di, f)

        with open(os.path.join(config['output_path'], "D{}_uname_fine_coarse_map.pickle".format(scale)), 'wb') as f:
            pickle.dump(uname_fine_coarse_map, f)
        with open(os.path.join(config['output_path'], "D{}_uname_coarse_fine_map.pickle".format(scale)), 'wb') as f:
            pickle.dump(uname_coarse_fine_map, f)

        # 2. diffpath_user_Di, diffpath_time_Di, diffpath_info_Di, diffpath_info_reverse_Di
        print("2. diffpath_user_Di, diffpath_time_Di, diffpath_info_Di, diffpath_info_reverse_Di")
        diffpath_user_Di = []
        diffpath_time_Di = []
        diffpath_info_Di = dict()
        diffpath_info_reverse_Di = dict()
        for path_idx in range(len(diffpath_user)):
            path_user = diffpath_user[path_idx]
            path_time = diffpath_time[path_idx]
            uname_time = dict()
            for uname_idx in range(len(path_user)):
                uname = path_user[uname_idx]
                uname_time[uname] = path_time[uname_idx]

            new_path = []
            new_path_time = []
            new_path_set = set()
            for uname in path_user:
                hyperuname = uid_uname_Di[model.labels_[uid_uname_reverse[uname]]]
                # remove repetitive users
                if hyperuname not in new_path_set:
                    new_path.append(hyperuname)
                    new_path_set.add(hyperuname)
                    new_path_time.append(uname_time[uname])  # just choose a timestamp 

            if len(new_path) < 3:  # at least need us, ui, uj
                continue

            diffpath_info_Di[len(diffpath_user_Di)] = diffpath_info[path_idx]
            diffpath_info_reverse_Di[diffpath_info[path_idx]] = len(diffpath_user_Di)
            diffpath_user_Di.append(new_path)
            diffpath_time_Di.append(new_path_time)

        with open(os.path.join(config['output_path'], "D{}_diffpath_user.pickle".format(scale)), 'wb') as f:
            pickle.dump(diffpath_user_Di, f)
        with open(os.path.join(config['output_path'], "D{}_diffpath_time.pickle".format(scale)), 'wb') as f:
            pickle.dump(diffpath_time_Di, f)
        with open(os.path.join(config['output_path'], "D{}_diffpath_info.pickle".format(scale)), 'wb') as f:
            pickle.dump(diffpath_info_Di, f)
        with open(os.path.join(config['output_path'], "D{}_diffpath_info_reverse.pickle".format(scale)), 'wb') as f:
            pickle.dump(diffpath_info_reverse_Di, f)

        # 3. infoid_infoname_Di, infoid_infoname_reverse_Di
        print("3. infoid_infoname_Di, infoid_infoname_reverse_Di")
        infoid_infoname_Di = infoid_infoname
        infoid_infoname_reverse_Di = infoid_infoname_reverse
        with open(os.path.join(config['output_path'], "D{}_infoid_infoname.pickle".format(scale)), 'wb') as f:
            pickle.dump(infoid_infoname_Di, f)
        with open(os.path.join(config['output_path'], "D{}_infoid_infoname_reverse.pickle".format(scale)), 'wb') as f:
            pickle.dump(infoid_infoname_reverse_Di, f)

        print_and_log(logger, "the number of users at scale {}: {}".format(scale, len(uid_uname_Di)))
        print_and_log(logger, "the number of information at scale {}: {}".format(scale, len(infoid_infoname_Di)))
        print_and_log(logger, "the number of diffusion paths at scale {}: {}".format(scale, len(diffpath_user_Di)))
        
        num_user_list.append(len(uid_uname_Di))
        diffp.append(len(diffpath_user_Di))
        
        with open(os.path.join(config['output_path'], 'num_over_scales.pickle'), 'wb') as f:
            pickle.dump([num_user_list, diffp], f)

        if len(uid_uname_Di) != len(uid_uname_reverse_Di):
            print_and_log(logger, "wrong! len(uid_uname_Di) != len(uid_uname_reverse_Di)")
            pdb.set_trace()

        (diffpath_user, diffpath_time, diffpath_info, diffpath_info_reverse,
         infoid_infoname, infoid_infoname_reverse,
         uid_uname, uid_uname_reverse, num_user, uname_multiscales) = (
            diffpath_user_Di, diffpath_time_Di, diffpath_info_Di, diffpath_info_reverse_Di,
            infoid_infoname_Di, infoid_infoname_reverse_Di,
            uid_uname_Di, uid_uname_reverse_Di, num_user_Di, uname_multiscales_Di)
    return num_user_list, diffp


def downscaling(config, scale, corpus_c, embedding_vec):
    (diffpath_user_c, diffpath_time_c, diffpath_info_c, diffpath_info_reverse_c,
     infoid_infoname_c, infoid_infoname_reverse_c,
     uid_uname_c, uid_uname_reverse_c) = corpus_c

    with open(os.path.join(config['output_path'], "D{}_uname_coarse_fine_map.pickle".format(scale)), 'rb') as f:
        uname_coarse_fine_map = pickle.load(f)

    user_embed_c = embedding_vec[0]

    """
    Obtain finer scale corpus
    """
    with open(os.path.join(config['output_path'], 'D{}_diffpath_user.pickle'.format(scale - 1)), 'rb') as f:
        diffpath_user_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_diffpath_time.pickle'.format(scale - 1)), 'rb') as f:
        diffpath_time_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_diffpath_info.pickle'.format(scale - 1)), 'rb') as f:
        diffpath_info_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_diffpath_info_reverse.pickle'.format(scale - 1)), 'rb') as f:
        diffpath_info_reverse_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_infoid_infoname.pickle'.format(scale - 1)), 'rb') as f:
        infoid_infoname_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_infoid_infoname_reverse.pickle'.format(scale - 1)), 'rb') as f:
        infoid_infoname_reverse_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_uid_uname.pickle'.format(scale - 1)), 'rb') as f:
        uid_uname_f = pickle.load(f)
    with open(os.path.join(config['output_path'], 'D{}_uid_uname_reverse.pickle'.format(scale - 1)), 'rb') as f:
        uid_uname_reverse_f = pickle.load(f)

    print("the number of users at D{} scale: {}".format(scale - 1, len(uid_uname_f)))
    print("the number of information at D{} scale: {}".format(scale - 1, len(infoid_infoname_f)))
    print("the number of diffusion paths at D{} scale: {}".format(scale - 1, len(diffpath_user_f)))

    user_embed_f = np.zeros((len(uid_uname_f), config['user_emb_dim']))

    for hyperuname in uname_coarse_fine_map:
        hyperuid = uid_uname_reverse_c[hyperuname]
        hyperuvec = user_embed_c[hyperuid]
        children_uname = uname_coarse_fine_map[hyperuname]
        for fineuname in children_uname:
            fineuid = uid_uname_reverse_f[fineuname]
            user_embed_f[fineuid] = hyperuvec

    embedding_vec_f = embedding_vec
    embedding_vec_f[0] = user_embed_f

    corpus_f = (diffpath_user_f, diffpath_time_f,
                diffpath_info_f, diffpath_info_reverse_f,
                infoid_infoname_f, infoid_infoname_reverse_f,
                uid_uname_f, uid_uname_reverse_f)
    return (corpus_f, embedding_vec_f)

