"""
Adapted from https://github.com/achristensen56/KepecsCode.git

Adaptation by Greg Knoll: Nov 2022
"""
import numpy as np
import os
import glob

def make_dir_if_not_exist(path):
    """create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def clear_obj_files(path):
    files = glob.glob(path + ".pkl")

def map_traces(behav_df, obj_list, matches):
    assert(len(obj_list) == matches.shape[1])

    n_total_neurons = matches.shape[0]
    n_trials = len(behav_df)

    _, _, il = obj_list[0].interp_traces.shape
    _, _, sl = obj_list[0].sa_traces.shape
    _, _, rl = obj_list[0].ca_traces.shape
    _, _, rwl = obj_list[0].ra_traces.shape

    interp = np.zeros([n_trials, n_total_neurons, il])*np.nan
    stim = np.zeros([n_trials, n_total_neurons, sl])*np.nan
    response = np.zeros([n_trials, n_total_neurons, rl])*np.nan
    reward = np.zeros([n_trials, n_total_neurons, rwl])*np.nan

    for i, obj in enumerate(obj_list):
        trials = behav_df[behav_df.session == obj.session].index.to_numpy()

        assert(len(trials) == obj.n_trials)
        assert(il == obj.interp_traces.shape[2])

        iids = matches[:, i][~np.isnan(matches[:, i])]
        mids = [np.where(matches[:, i] == c)[0][0] for c in iids]

        A = np.zeros_like(interp[trials])
        A[:, mids, :] = obj[:, iids, :]
        interp[trials] = A

        A = np.zeros_like(stim[trials])
        A[:, mids, :] = obj[:, iids, 'stimulus']
        stim[trials] = A

        A = np.zeros_like(response[trials])
        A[:, mids, :] = obj[:, iids, 'response']
        response[trials] = A

        A = np.zeros_like(reward[trials])
        A[:, mids, :] = obj[:, iids, 'reward']
        reward[trials] = A

    interp_dict = {'interp_traces': interp,
                   'stim_aligned': stim,
                   'response_aligned': response,
                   'reward_aligned': reward,
                   'interp_inds': obj.interp_inds,
                   'response_ind':obj.choice_ind,
                   'reward_ind': obj.reward_ind,
                   'stim_ind': obj.stim_ind}


    return interp_dict

def get_cluster_traces(obj_list, phase_indexer, filter_active=True, require_all_clusters=False, cluster_list=None):
    """
    Returns an iterable group of lists containing the cluster labels, behavioral dataframes, neural traces, rat names.
    """

    labels = np.array([obj.cluster_labels for obj in obj_list]).flatten()
    n_clusters = len(np.unique(labels))

    if require_all_clusters:
        labels_red = []
        objs_red = []
        if cluster_list is None:
            cluster_list = np.arange(0, n_clusters)
        for obj in obj_list:
            if set(cluster_list).issubset(obj.cluster_labels):
                if sum([(sum(obj.cluster_labels == c) > 2) for c in cluster_list]) == len(cluster_list):
                    print(obj.name)
                    objs_red.append(obj)
                    labels_red.append(obj.cluster_labels)

        obj_list = objs_red


    for cluster_id in np.unique(labels):
        behavior_df_list = []
        traces = []
        rat_names = []
        for obj in obj_list:
            if filter_active:
                clust = obj.active_neurons[obj.cluster_labels == cluster_id]
            else:
                clust = np.arange(0, obj.n_neurons)[obj.cluster_labels == cluster_id]

            if len(clust) > 0:
                behavior_df_list.append(obj.behav_df)
                traces.append(obj[:, clust, phase_indexer])
                rat_names.append(obj.name)
            else:
                print("uhoh the cluster was too small")
        yield(cluster_id, behavior_df_list, traces, rat_names)