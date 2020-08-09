import os
import pickle
import pdb
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
num_cpu = multiprocessing.cpu_count()
random.seed(0)

from sklearn.metrics import pairwise_distances

from metrics import compute_AP
from metrics import compute_MAP
from metrics import compute_Patk
from metrics import compute_precision_recall_f1_auc


def print_time_elaspsed(time_start, msg):
    time_elaspsed = time.time() - time_start
    if time_elaspsed > 3600:
        print("{} took {} h".format(msg, round(time_elaspsed / 3600, 2)))
    elif time_elaspsed > 60:
        print("{} took {} min".format(msg, round(time_elaspsed / 60, 2)))
    else:
        print("{} took {} s".format(msg, round(time_elaspsed, 2)))


def perform_prediction_one_cascade(c_cascade, sc, idx_2_user, sc_pairwise_distance):
    # get predicted cascade
    predicted_cascade_user_index = np.argsort(sc_pairwise_distance)
    p_cascade = [sc]
    for i in range(1, len(c_cascade)):
        p_cascade.append(idx_2_user[predicted_cascade_user_index[i]])
    return p_cascade


def perform_prediction_multiprocessing(diffpath_user, uid_uname_reverse, uid_uname, Z):
    func_start = time.time()

    # compute pairwise distance for sc with current embedding space
    distance_matrix = pairwise_distances(Z, metric='euclidean', n_jobs=num_cpu)

    list_of_args_tuple = []
    for c_idx in range(len(diffpath_user)):
        c_cascade = diffpath_user[c_idx]
        sc = c_cascade[0]
        sc_idx = uid_uname_reverse[sc]
        sc_pairwise_distance = distance_matrix[sc_idx]

        arg = (c_cascade, sc, uid_uname, sc_pairwise_distance)
        list_of_args_tuple.append(arg)
    print("list of args are prepared. going to use multiprocessing (cores: {})".format(num_cpu))
    with Pool(num_cpu) as pool:
        predicted_cascades = pool.starmap(perform_prediction_one_cascade, list_of_args_tuple)
    print_time_elaspsed(func_start, "perform prediction")
    # pdb.set_trace()
    return predicted_cascades


def embed(config, scale, corpus, embedding_vec=None):

    # specify some hyperparameters here
    initalization = 'uniform'
    max_epochs = config['max_epochs']
    num_sample_per_scale = 5000
    lr = 0.01
    decay = 0.000001
    lr_min = 0  # 0.0001
    vali_diffpath_percentage = 0.2
    evaluation_epoch = None  # or None
    evaluate_first = False

    (diffpath_user, diffpath_time,
     diffpath_info, diffpath_info_reverse,
     infoid_infoname, infoid_infoname_reverse,
     uid_uname, uid_uname_reverse) = corpus

    if embedding_vec is None:
        # randomly initialize embeddings
        if initalization == 'uniform':
            Z = np.random.uniform(-1, 1, (len(uid_uname), config['user_emb_dim']))
        else:  # 'normal':
            Z = np.random.randn(len(uid_uname), config['user_emb_dim'])
    else:
        [Z] = embedding_vec

    """
    Split diffusion paths into training and validation
    """
    diffpath_idx = list(range(len(diffpath_user)))
    random.shuffle(diffpath_idx)
    train_num = int((1 - vali_diffpath_percentage) * len(diffpath_user))
    train_diffpath_idx = diffpath_idx[:train_num]
    valid_diffpath_idx = diffpath_idx[train_num:]

    diffpath_user_train = []
    diffpath_time_train = []
    diffpath_info_train = dict()
    diffpath_info_reverse_train = dict()
    for idx in train_diffpath_idx:
        diffpath_info_train[len(diffpath_user_train)] = diffpath_info[idx]
        diffpath_info_reverse_train[diffpath_info[idx]] = len(diffpath_user_train)
        diffpath_user_train.append(diffpath_user[idx])
        diffpath_time_train.append(diffpath_time[idx])

    diffpath_user_valid = []
    diffpath_time_valid = []
    diffpath_info_valid = dict()
    diffpath_info_reverse_valid = dict()
    for idx in valid_diffpath_idx:
        diffpath_info_valid[len(diffpath_user_valid)] = diffpath_info[idx]
        diffpath_info_reverse_valid[diffpath_info[idx]] = len(diffpath_user_valid)
        diffpath_user_valid.append(diffpath_user[idx])
        diffpath_time_valid.append(diffpath_time[idx])

    """
    Training
    """
    train_start = time.time()
    epoch = 1
    avg_loss_over_epochs = []
    train_map_over_epochs = []
    valid_map_over_epochs = []
    epochs_no_update = []
    lr_his = [lr]
    # https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/

    while (not epoch > max_epochs):
        print("\nscale: {}   Epoch: {} ".format(scale, epoch))
        # --- evaluaton
        if (evaluation_epoch is not None) and (
                (epoch % evaluation_epoch == 0) or (epoch == 1 and evaluate_first)):
            print("==============Evaluation START")
            print("obtaining prediction result for training cascades...")
            diffpath_user_train_predicted = perform_prediction_multiprocessing(
                    diffpath_user_train, uid_uname_reverse, uid_uname, Z)

            print("obtaining prediction result for validation cascades...")
            diffpath_user_valid_predicted = perform_prediction_multiprocessing(
                    diffpath_user_valid, uid_uname_reverse, uid_uname, Z)

            # MAP
            train_map = compute_MAP(diffpath_user_train, diffpath_user_train_predicted)
            print("Training MAP: {}".format(round(train_map, 4)))

            valid_map = compute_MAP(diffpath_user_valid, diffpath_user_valid_predicted)
            print("Validation MAP: {}".format(round(valid_map, 4)))

            train_map_over_epochs.append(train_map)
            valid_map_over_epochs.append(valid_map)

            with open(os.path.join(config['output_path'], "D{}_train_map_over_epochs.pickle".format(scale)), 'wb') as f:
                pickle.dump(train_map_over_epochs, f)
            with open(os.path.join(config['output_path'], "D{}_valid_map_over_epochs.pickle".format(scale)), 'wb') as f:
                pickle.dump(valid_map_over_epochs, f)

            plot_improvement_over_history(train_map_over_epochs,
                                          "train_map_over_epochs",
                                          "train_map_over_epochs",
                                          os.path.join(
                                                  config['output_path'],
                                                  "D{}_train_map_over_epochs.png".format(scale)))
            plot_improvement_over_history(valid_map_over_epochs,
                                          "valid_map_over_epochs",
                                          "valid_map_over_epochs",
                                          os.path.join(
                                                  config['output_path'],
                                                  "D{}_valid_map_over_epochs.png".format(scale)))

            write_result_with_gt(
                    diffpath_user_train,
                    diffpath_user_train_predicted,
                    os.path.join(config['output_path'],
                                 "D{}_evaluation_result_train_epoch{}.txt".format(scale, epoch)),
                    map_value=round(train_map, 4))

            write_result_with_gt(
                    diffpath_user_valid,
                    diffpath_user_valid_predicted,
                    os.path.join(config['output_path'],
                                 "D{}_evaluation_result_valid_epoch{}.txt".format(scale, epoch)),
                    map_value=round(valid_map, 4))

            print("==============Evaluation END")

        # --- SGD
        total_loss = 0
        print("run SGD...")
        Z_new = Z.copy()
        for train_n in range(num_sample_per_scale):
            # samplec cascade c
            c_idx = np.random.randint(len(diffpath_user_train))
            c_cascade = diffpath_user_train[c_idx]
            while len(c_cascade) < 3:
                c_idx = np.random.randint(len(diffpath_user_train))
            c_cascade = diffpath_user_train[c_idx]
            sc = c_cascade[0]
            z_sc = Z[uid_uname_reverse[sc]]
            # sample ui in cascade c (will not be the first user or the last user)
            ui_idx = np.random.randint(1, len(c_cascade) - 1)
            ui = c_cascade[ui_idx]
            z_ui = Z[uid_uname_reverse[ui]]
            # sample uj not in cascade c or in but and later than ui
            uj_in_c = random.choice([True, False])
            if uj_in_c:
                uj_idx = np.random.randint(ui_idx + 1, len(c_cascade))
                uj = c_cascade[uj_idx]
                while uj == ui:
                    uj_idx = np.random.randint(ui_idx + 1, len(c_cascade))
                    uj = c_cascade[uj_idx]
                    # print(ui, uj, ui_idx, uj_idx, len(c_cascade))
                z_uj = Z[uid_uname_reverse[uj]]
            else:
                c_another_idx = np.random.randint(len(diffpath_user_train))
                c_another_cascade = diffpath_user_train[c_another_idx]
                while (c_another_idx == c_idx) or (len(c_another_cascade) < 3):
                    c_another_idx = np.random.randint(len(diffpath_user_train))
                    c_another_cascade = diffpath_user_train[c_another_idx]
                uj_idx = np.random.randint(1, len(c_another_cascade))  # will not be the first user
                uj = c_another_cascade[uj_idx]
                while uj == ui:
                    uj_idx = np.random.randint(1, len(c_another_cascade))
                    uj = c_another_cascade[uj_idx]
                z_uj = Z[uid_uname_reverse[uj]]

            # dj
            dj = np.linalg.norm(z_sc - z_uj)
            # di
            di = np.linalg.norm(z_sc - z_ui)

            # loss
            loss = max(0, 1 - (dj - di))
            total_loss += loss

            # compare to 1
            """
            print("epoch: {}  processing: {} {}/{}  loss: {} "
                  "dj-di: {}  no need to update?: {}  ".format(
                          epoch, round(train_n/len(train_cascades), 2),
                          train_n, len(train_cascades),
                          round(loss, 2), round(dj-di, 2), (dj-di) >= 1))
            """
            if (dj - di) < 1:
                z_ui_new = z_ui + 2 * lr * (z_sc - z_ui)
                z_uj_new = z_uj - 2 * lr * (z_sc - z_uj)
                z_sc_new = z_sc + 2 * lr * (z_ui - z_uj)

                Z_new[uid_uname_reverse[ui]] = z_ui_new
                Z_new[uid_uname_reverse[uj]] = z_uj_new
                Z_new[uid_uname_reverse[sc]] = z_sc_new

        # prepare next epoch
        total_loss = total_loss / len(diffpath_user_train)
        avg_loss_over_epochs.append(total_loss)
        if total_loss > 100:
            print("epoch: {}  average loss: {:.2e}  lr: {:.2e}".format(epoch, round(total_loss, 4), round(lr, 8)))
        else:
            print("epoch: {}  average loss: {}  lr: {:.2e}".format(epoch, round(total_loss, 4), round(lr, 8)))
        print_time_elaspsed(train_start, "training has")
        with open(os.path.join(config['output_path'], "D{}_avg_loss_over_epochs.pickle".format(scale)), 'wb') as f:
            pickle.dump(avg_loss_over_epochs, f)
        plot_improvement_over_history(avg_loss_over_epochs,
                                      "avg_loss_over_epochs",
                                      "avg_loss_over_epochs",
                                      os.path.join(config['output_path'], "D{}_avg_loss_over_epochs.png".format(scale)))
        plot_improvement_over_history(lr_his,
                                      "lr_his",
                                      "lr_his",
                                      os.path.join(config['output_path'], "D{}_lr_his.png".format(scale)))
        np.save(os.path.join(config['output_path'], "D{}_Z_{}.npy".format(scale, epoch)), Z)
        if np.array_equal(Z, Z_new) and total_loss > 0:
            print("Z == Z_new. No update in this epoch!")
            epochs_no_update.append(epoch)
            continue
        np.save(os.path.join(config['output_path'], "D{}_epochs_no_update.npy".format(scale)), epochs_no_update)
        lr = lr * 1 / (1 + decay * epoch)
        if lr < lr_min:
            lr = lr_min
        epoch += 1
        lr_his.append(lr)
        Z = Z_new

    embedding_vec = [Z]
    return embedding_vec


def write_result_with_gt(gt_cascades, pred_cascades, file_path, map_value=None,
                         patk=None, precision=None, recall=None, f1=None, auc=None, acc=None):
    result = []
    for i in range(len(gt_cascades)):
        gt = gt_cascades[i][1:]
        pred = pred_cascades[i][1:]
        gt_line = 'ground truth: ' + ' '.join(gt) + '\n'
        pred_line = 'predicted: ' + ' '.join(pred) + '\n'
        ap1 = 'AP: {}\n'.format(compute_AP(
                gt_cascades[i], pred_cascades[i]))
        lines_this = 'source user: {}:\n'.format(
                gt_cascades[i][0]) + gt_line + pred_line + ap1
        result.append(lines_this)
    if map_value is not None:
        result.append("\n\nMAP metric value: {}".format(round(map_value, 4)))
    if patk is not None:
        for k in sorted(patk.keys()):
            result.append("P@{} metric value: {}".format(k, round(patk[k], 4)))

    if precision is not None:
        result.append("Precision metric value: {}".format(round(precision, 4)))
    if recall is not None:
        result.append("Recall metric value: {}".format(round(recall, 4)))
    if f1 is not None:
        result.append("F1 metric value: {}".format(round(f1, 4)))
    if auc is not None:
        result.append("AUC metric value: {}".format(round(auc, 4)))
    if acc is not None:
        result.append("Accuracy metric value: {}".format(round(acc, 4)))

    with open(file_path, "w") as f:
        lines = '\n'.join(result)
        f.write(lines + '\n')
    return


def plot_improvement_over_history(values, line_label, title, save_file_dir):
    # figsize_settitng = (20, 8)
    # plt.figure(figsize=figsize_settitng, dpi=150)
    plt.figure()

    plt.title(title)
    color_this = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(values, '-', marker='.', color=color_this,
             label=line_label)
    plt.xlabel('epoch')
    plt.ylabel('improvement indicator value')
    plt.legend()
    plt.savefig(save_file_dir, bbox_inches='tight')
    plt.close()


def test(config, save_dir, test_epoch, embedding_scale, corpus_test):

    (diffpath_user, diffpath_time,
     diffpath_info, diffpath_info_reverse,
     infoid_infoname, infoid_infoname_reverse,
     uid_uname, uid_uname_reverse) = corpus_test

    """
    Obtain Embedding
    """
    Z = np.load(os.path.join(save_dir, "D{}_Z_{}.npy".format(embedding_scale, test_epoch)))

    print("==============Testing START")
    print("obtaining prediction result for testing cascades...")
    diffpath_user_predicted = perform_prediction_multiprocessing(
            diffpath_user, uid_uname_reverse, uid_uname, Z)

    # MAP
    test_map = compute_MAP(diffpath_user, diffpath_user_predicted)
    print("Testing MAP: {}".format(round(test_map, 4)))

    # Patk
    patk = dict()
    for k in [3, 5, 10, 50, 100]:
        patk[k] = compute_Patk(diffpath_user, diffpath_user_predicted, k)
        print("P@{} metric value: {}".format(k, round(patk[k], 4)))

    # contagion
    (test_precision, test_recall, test_f1, test_auc, test_acc) = compute_precision_recall_f1_auc(
            diffpath_user, diffpath_user_predicted, uid_uname, uid_uname_reverse)
    print("Testing Precision: {}".format(round(test_precision, 4)))
    print("Testing Recall: {}".format(round(test_recall, 4)))
    print("Testing F1: {}".format(round(test_f1, 4)))
    print("Testing AUC: {}".format(round(test_auc, 4)))
    print("Testing Accuracy: {}".format(round(test_acc, 4)))

    write_result_with_gt(
            diffpath_user,
            diffpath_user_predicted,
            os.path.join(save_dir,
                         "D{}_evaluation_result_test_epoch{}.txt".format(embedding_scale, test_epoch)),
            map_value=round(test_map, 4),
            patk=patk, precision=test_precision, recall=test_recall, f1=test_f1, auc=test_auc, acc=test_acc
                )

    print("==============Testing END")

    return
