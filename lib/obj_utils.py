"""
Adapted from https://github.com/achristensen56/KepecsCode.git

Adaptation by Greg Knoll: Nov 2022
"""
import numpy as np
import os
import glob

from neuropixels_preprocessing.lib.data_objs import TwoAFC, from_pickle
from neuropixels_preprocessing.session_params import load_session_metadata_from_csv, get_session_path


def make_dir_if_nonexistent(path, verbose=True):
    """create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            print(f"{path} created.", flush=True)
    else:
        if verbose:
            print(f"{path} already exists.", flush=True)


def combine_session_data_objects(data_root, rat_name_l, date_l):
    data_obj_l = []
    for sesh_i in range(len(date_l)):
        _metadata = load_session_metadata_from_csv(data_root, rat=rat_name_l[sesh_i], session_date=date_l[sesh_i])
        for probe_i in range(1, _metadata['n_probes']+1):
            _metadata['probe_num'] = probe_i
            _paths = get_session_path(_metadata, data_root, is_ephys_session=True)

            _obj = from_pickle(_paths['preprocess_dir'], probe_i, TwoAFC)
            _obj.n_neurons = len(_obj.metadata['nrn_phy_ids'])
            _obj.probe_dir = _paths['preprocess_dir'] + f"probe{probe_i}/"
            _obj.sort_dir = _paths['rec_dir'] + f"sorting_output/probe{probe_i}/sorter_output/"
            print(_obj, 'loaded from ', _obj.data_path)

            _obj.behavior_mat_path = _paths['behav_dir'] + _metadata['behavior_mat_file']
            data_obj_l.append(_obj)
    return data_obj_l

def clear_obj_files(path):
    files = glob.glob(path + ".pkl")


def get_cluster_traces(obj_list, alignment, filter_stable=True, require_all_clusters=False, cluster_list=None):
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
                    print(obj.metadata['rat_name'])
                    objs_red.append(obj)
                    labels_red.append(obj.cluster_labels)

        obj_list = objs_red


    for cluster_id in np.unique(labels):
        behavior_df_list = []
        traces = []
        rat_names = []
        for obj in obj_list:
            if filter_stable:
                clust = obj.stable_neurons[obj.cluster_labels == cluster_id]
            else:
                clust = np.arange(0, obj.n_neurons)[obj.cluster_labels == cluster_id]

            if len(clust) > 0:
                behavior_df_list.append(obj.behav_df)
                traces.append(obj[:, clust, alignment])
                rat_names.append(obj.metadata['rat_name'])
            else:
                print("uhoh the cluster was too small")
        yield(cluster_id, behavior_df_list, traces, rat_names)