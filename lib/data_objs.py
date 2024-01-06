"""
Data structures for handling spiking and behavioral data from Neuropixel experiments.

Adapted from https://github.com/achristensen56/KepecsCode.git

Adaptation by Greg Knoll: Nov 2022
"""

import pickle
import pandas as pd
from sqlalchemy import create_engine, update, delete
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import glob
import joblib

import neuropixels_preprocessing.lib.trace_utils as trace_utils
import neuropixels_preprocessing.lib.behavior_utils as bu

class DataContainer:
    def __init__(self, data_path, behav_df, metadata):
        self.data_path = data_path
        self.behav_df = behav_df
        self.metadata = metadata
        self.objID = f"{metadata['rat_name']}_{metadata['date']}_{metadata['task']}_probe{metadata['probe_num']}"
            
    def load_cluster_info(self, cluster_filename):
        cluster_dict = joblib.load(self.data_path + cluster_filename)
        self.cluster_neurons = cluster_dict['neurons']
        self.cluster_labels = cluster_dict['labels']
    
    def neurons_of_cluster(self, clust_i):
       return self.cluster_neurons[self.cluster_labels == clust_i]
        
    def load_traces(self, trace_type, downsample_dt):
        trace_dict = joblib.load(self.data_path + f'{trace_type}_aligned_traces_{downsample_dt}ms_bins')
        self.n_trials, self.n_neurons, _ = trace_dict['traces'].shape
        assert self.n_trials == len(self.behav_df)
        assert self.n_neurons == len(self.metadata['nrn_phy_ids'])
        
        if trace_type=='stimulus':        
            self.stimulus_traces = trace_dict['traces']
            self.stimulus_ind = trace_dict['ind']
        elif trace_type=='response':
            self.response_traces = trace_dict['traces']
            self.response_ind =  trace_dict['ind']
        elif trace_type=='reward':
            self.reward_traces = trace_dict['traces']
            self.reward_ind = trace_dict['ind']
        elif trace_type=='interp':
            self.interp_traces = trace_dict['traces']
            self.interp_inds = trace_dict['ind']
            if type(self.interp_inds) ==  list:
                self.interp_inds = [np.sum(self.interp_inds[0:i]) for i in np.arange(1, len(self.interp_inds) + 1)]
        else:
            raise NotImplementedError()

    def __getitem__(self, item):
        """
        :param item:
            - the first item is the trial "property", e.g., trial outcome
            - the second item can be a list of neurons, any valid neuron indexing
            - selects to what the traces are aligned.  Possible values are:
                'stimulus', 'response', 'reward', 'interp'
        :return:
        """
        # --------indexing arguments------------- #
        assert(len(item) == 3)
        trial_property_column = item[0]
        neuron_indexer = item[1]
        phase_indexer = item[2]
        # --------------------------------------- #

        # ---------- first get rows (trials) that match value of column of interest ------------ #
        if type(trial_property_column) == str:
            _col = self.behav_df[trial_property_column].copy()
            trial_id = _col.index[_col==True].to_list()
        elif type(trial_property_column) == list:
            if np.issubdtype(type(trial_property_column[0]), np.integer):
                trial_id = trial_property_column
            elif type(trial_property_column[0]) == str:
                idx = np.logical_and.reduce([self.behav_df[ti] for ti in trial_property_column])
                trial_id = self.behav_df[idx].index.to_list()
            else:
                raise "Not Implemented"
        elif type(trial_property_column) == slice:
            trial_id = trial_property_column
        elif type(trial_property_column) == dict:
            matched_rows = [(self.behav_df[ti] == trial_property_column[ti]).to_numpy() for ti in trial_property_column]
            idx = np.logical_and.reduce(matched_rows)
            trial_id = self.behav_df[idx].index.to_list()
        else:
            raise "Not Implemented"
        # ----------------------------------------------------------------------------- #

        # -----------------------Choose type of trace---------------------------------- #
        if type(phase_indexer) == slice:
            traces = self.interp_traces
            return traces[trial_id][:, neuron_indexer][:, :, phase_indexer]
        elif phase_indexer == "interp":
            traces = self.interp_traces
        elif phase_indexer == "response":
            traces = self.response_traces
        elif phase_indexer == "stimulus":
            traces = self.stimulus_traces
        elif phase_indexer == 'reward':
            traces = self.reward_traces
        else:
            raise "Not Implemented"
        if traces is None:
            raise "Not Implemented"
        # ----------------------------------------------------------------------------- #

        # -----Index the traces and return----- #
        return traces[trial_id][:, neuron_indexer]

    def __setitem__(self, key, value):
        raise "Not Implemented"

    def __str__(self):
        return str(self.objID)

    def iter_clusters(self,  phase_indexer = slice(None), return_labels = True):
        if return_labels:
            for c in np.unique(self.cluster_labels):
                tr = self[ :, self.stable_neurons[self.cluster_labels == c], phase_indexer]
                yield c, tr
        else:
            for c in np.unique(self.cluster_labels):
                tr = self[ :, self.stable_neurons[self.cluster_labels == c], phase_indexer]
                yield tr

    def get_alignment_index(self, alignment):
        if alignment == 'stimulus':
            return self.stim_ind
        elif alignment == 'response':
            return self.response_ind
        elif alignment == 'reward':
            return self.reward_ind
        elif alignment == 'interp':
            return self.interp_inds

    def get_feature_df(self, alignment='reward',
                             variables=['rewarded'],
                             force_new=False,
                             Gauss_filter_traces=[False, 40],
                             selected_neurons=[],
                             save=True):
        """
        A wrapper function to check whether a given feature dataframe has already been created and cached.
        If not, a new dataframe is created, and cached if desired.

        :param alignment: [string] behavioral event to which traces are aligned
        :param variables: [list of strings] desired behavior task variables (column names) to add to features
        :param force_new: create a new cache, even if one already exists
        :param apply_Gauss_filter_to_traces: if True, applies a Gaussian temporal smoothing filter to the traces
        :param selected_neurons: [numpy array or string] indices of neurons of interest
        :param save: [bool] whether to cache (save) the newly created dataframe
        :return: [pandas Dataframe] see docstring of get_trace_feature_df() in trace_utils.py
        """

        # matches = [(cache[0] == alignment) & (cache[1] == variables) for cache in self.feature_df_keys]
        # if (sum(matches) > 0) and not force_new:
        #     cache_ind = np.where(matches)[0]
        #     print(cache_ind[0])
        #     feature_df = pickle.load(open(self.feature_df_cache[int(cache_ind[0])], 'rb'))
        #     print('loaded a cached feature df')
        # else:
            
        # Get all traces from all neurons that match the prescribed alignment
        traces = self[:, :, alignment]
        if Gauss_filter_traces[0]:
            traces = gaussian_filter1d(traces, sigma=Gauss_filter_traces[1], axis=2)

        # the given list of selected neurons will be used, unless it is empty or if 'stable' is passed instead
        if type(selected_neurons)==str and selected_neurons=='stable':
            selected_neurons = self.stable_neurons
        elif not len(selected_neurons):
            selected_neurons = np.arange(self.n_neurons)

        feature_df = trace_utils.get_trace_feature_df(behav_df=self.behav_df,
                                                      selected_neurons=selected_neurons,
                                                      traces=traces,
                                                      rat_name=self.metadata['rat_name'],
                                                      session_date=self.metadata['date'],
                                                      probe_num=self.metadata['probe_num'],
                                                      behavior_variables=variables)

        # print("Created new feature df.")
        # if save:
        #     self.feature_df_keys.append([alignment, variables])
        #     fname = self.data_path + '_feature_df_' + str(len(self.feature_df_keys)) + '.pkl'
        #     with open (fname, 'wb') as f:
        #         pickle.dump(feature_df, f)

        #     print("New feature df cached.")
        #     self.feature_df_cache.append(fname)

        return feature_df
    
    
    def to_pickle(self, save_traces=True):
        print('Saving to ' + self.data_path + '...', end='')
        
        # ------------------------------------------------------------- #
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
            print('folder created...', end='')
        # ------------------------------------------------------------- #

        joblib.dump(self.behav_df, self.data_path + 'behav_df', compress=5)
        joblib.dump(self.metadata, self.data_path + 'metadata', compress=5)
        
        print('Done.')


def from_pickle(data_path, obj_class, sub_dir=''):
    metadata = joblib.load(data_path + 'metadata')
    behav_df =joblib.load(data_path + 'behav_df')
    
    for red_flag in ['no_matching_TTL_start_time', 'large_TTL_gap_after_start']:
        if red_flag in behav_df.keys() and behav_df[red_flag].sum()>0:
            print('Trials with' + red_flag + '!!!')

    return obj_class(data_path, behav_df=behav_df, metadata=metadata)


class TwoAFC(DataContainer):
    def __init__(self, data_path, behav_df, metadata):
        super().__init__(data_path, behav_df, metadata)


# def create_experiment_data_object(data_path, metadata, nrn_phy_ids, trialwise_binned_mat, cbehav_df):
#     traces_dict = trace_utils.create_traces_np(cbehav_df,
#                                                trialwise_binned_mat,
#                                                metadata,
#                                                aligned_ind=0,
#                                                filter_by_trial_num=False,
#                                                traces_aligned="TrialStart")
#     print('Creating data object...', end='')
#     # create and save data object
#     TwoAFC(data_path, cbehav_df, metadata).to_pickle()


# class Multiday_2AFC(DataContainer):
#     def __init__(self, dat_path, obj_list, cell_mapping,
#                  sps=None,
#                  name=None,
#                  cluster_labels=None,
#                  metadata=None,
#                  stable_neurons=None,
#                  record=True,
#                  feature_df_cache=[],
#                  feature_df_keys=[]):
#
#         assert('session' in obj_list[0].behav_df.keys())
#
#         self.behav_df = pd.concat([obj.behav_df for obj in obj_list])
#         self.n_sessions = len(obj_list)
#         traces_dict = map_traces(self.behav_df, obj_list, matches=cell_mapping)
#
#         self.name = name
#         self.metadata = metadata
#         self.sps = sps
#         self.dat_path = dat_path
#         self.feature_df_cache = feature_df_cache
#         self.feature_df_keys = feature_df_keys
#
#         self.linking_group = [obj.objID for obj in obj_list]
#
#         if objID is not None:
#             self.objID = objID
#         else:
#             self.objID = str(np.datetime64('now').astype('uint32'))
#
#         self.sa_traces = traces_dict['stim_aligned']
#         self.ca_traces = traces_dict['response_aligned']
#         self.ra_traces = traces_dict['reward_aligned']
#         self.interp_traces = traces_dict['interp_traces']
#
#         self.interp_inds = traces_dict['interp_inds']
#
#         self.choice_ind = traces_dict['response_ind']
#         self.reward_ind = traces_dict['reward_ind']
#         self.stim_ind = traces_dict['stim_ind']
#         self.n_trials, self.n_neurons, _ = self.sa_traces.shape
#
#         if stable_neurons is not None:
#             self.stable_neurons = stable_neurons
#         else:
#             self.stable_neurons = np.arange(self.n_neurons)
#
#         self.traces_dict = traces_dict
#
#         if (cluster_labels is not None):
#             self.cluster_labels = cluster_labels
#             self.cluster_ids = np.unique(cluster_labels)
#
#         if record:
#             self.record_experiment(self.linking_group, self.metadata['experiment_id'])


# def map_traces(behav_df, obj_list, matches):
#     assert(len(obj_list) == matches.shape[1])
#
#     n_total_neurons = matches.shape[0]
#     n_trials = len(behav_df)
#
#     _, _, il = obj_list[0].interp_traces.shape
#     _, _, sl = obj_list[0].sa_traces.shape
#     _, _, rl = obj_list[0].ca_traces.shape
#     _, _, rwl = obj_list[0].ra_traces.shape
#
#     interp = np.zeros([n_trials, n_total_neurons, il])*np.nan
#     stim = np.zeros([n_trials, n_total_neurons, sl])*np.nan
#     response = np.zeros([n_trials, n_total_neurons, rl])*np.nan
#     reward = np.zeros([n_trials, n_total_neurons, rwl])*np.nan
#
#     for i, obj in enumerate(obj_list):
#         trials = behav_df[behav_df.session == obj.session].index.to_numpy()
#
#         assert(len(trials) == obj.n_trials)
#         assert(il == obj.interp_traces.shape[2])
#
#         iids = matches[:, i][~np.isnan(matches[:, i])]
#         mids = [np.where(matches[:, i] == c)[0][0] for c in iids]
#
#         A = np.zeros_like(interp[trials])
#         A[:, mids, :] = obj[:, iids, :]
#         interp[trials] = A
#
#         A = np.zeros_like(stim[trials])
#         A[:, mids, :] = obj[:, iids, 'stimulus']
#         stim[trials] = A
#
#         A = np.zeros_like(response[trials])
#         A[:, mids, :] = obj[:, iids, 'response']
#         response[trials] = A
#
#         A = np.zeros_like(reward[trials])
#         A[:, mids, :] = obj[:, iids, 'reward']
#         reward[trials] = A
#
#     interp_dict = {'interp_traces': interp,
#                    'stim_aligned': stim,
#                    'response_aligned': response,
#                    'reward_aligned': reward,
#                    'interp_inds': obj.interp_inds,
#                    'response_ind':obj.choice_ind,
#                    'reward_ind': obj.reward_ind,
#                    'stim_ind': obj.stim_ind}
#
#
#     return interp_dict
