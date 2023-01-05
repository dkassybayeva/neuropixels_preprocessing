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
from joblib import load

import neuropixels_preprocessing.lib.trace_utils as tu
import neuropixels_preprocessing.lib.behavior_utils as bu
from neuropixels_preprocessing.lib.obj_utils import *

class DataContainer:
    def __init__(self, dat_path, behav_df, traces_dict, 
                 neuron_mask_df=None,
                 objID=None,
                 sps=None,
                 name=None,
                 cluster_labels=None,
                 metadata=None,
                 linking_group=None,
                 active_neurons=None,
                 record=False,
                 feature_df_cache=[],
                 feature_df_keys=[], 
                 behavior_phase=None):

        self.name = name
        self.metadata = metadata
        self.sps = sps
        self.dat_path = dat_path
        self.feature_df_cache = feature_df_cache
        self.feature_df_keys = feature_df_keys
        self.behavior_phase = behavior_phase
        self.pos_conf = []
        self.neg_conf = []


        if objID is not None:
            self.objID = objID
        else:
            self.objID = str(np.datetime64('now').astype('uint32'))
        self.behavior_phase = metadata['behavior_phase']

        if traces_dict is not None:
            self.sa_traces = traces_dict['stim_aligned']
            self.ca_traces = traces_dict['response_aligned']
            self.ra_traces = traces_dict['reward_aligned']
            self.interp_traces = traces_dict['interp_traces']

            self.interp_inds = traces_dict['interp_inds']
            if type(self.interp_inds) ==  list:
                self.interp_inds = [np.sum(self.interp_inds[0:i]) for i in np.arange(1, len(self.interp_inds) + 1)]

            self.choice_ind = traces_dict['response_ind']
            self.reward_ind = traces_dict['reward_ind']
            self.stim_ind = traces_dict['stim_ind']
            self.n_trials, self.n_neurons, _ = self.sa_traces.shape

            if active_neurons is not None:
                self.active_neurons = active_neurons
            else:
                self.active_neurons = np.arange(self.n_neurons)

            if neuron_mask_df is not None:
                self.neuron_mask_df = neuron_mask_df
            else:
                neurons = np.arange(0, self.n_neurons)
                df_dict = {'neurons':neurons}
                if active_neurons is not None and len(active_neurons) > 0:
                    active_mask = np.zeros_like(neurons)
                    active_mask[active_neurons] = 1
                    df_dict['active_mask'] = active_mask
                else:
                    df_dict['active_mask'] = [True]*self.n_neurons

                if len(cluster_labels) > 0:
                    cluster_mask = np.zeros_like(neurons)*np.nan
                    if len(cluster_labels) == self.n_neurons:
                        cluster_mask = cluster_labels
                    elif len(cluster_labels) == len(self.active_neurons):
                        cluster_mask[self.active_neurons] == cluster_labels
                    else:
                        raise("number of cluster labels is not the same as either "
                              "total number of neurons or number of active neurons")

                    df_dict['cluster_mask'] = cluster_mask
                self.neuron_mask_df = pd.DataFrame(df_dict, index=df_dict["neurons"], columns=["cluster_mask", "active_mask"])

            self.traces_dict = traces_dict
        self.behav_df = behav_df

        if (cluster_labels is not None):
            self.cluster_labels = cluster_labels
            self.cluster_ids = np.unique(cluster_labels)

        if record:
            self.record_experiment(linking_group, self.metadata['experiment_id'])

    def record_experiment(self, linking_group, experiment_id):
        tracking_df = pd.DataFrame.from_dict({'ratname': [self.name],
                                              'date': [self.metadata['date']],
                                              'experimenter': [self.metadata['experimenter']],
                                              'stimulus': [self.metadata['stimulus']],
                                              'time_investment': [self.metadata['time_investment']],
                                              'reward_bias': [self.metadata['reward_bias']],
                                              'prior':[self.metadata['prior']],
                                              'behavior_phase':[self.metadata['behavior_phase']],
                                              'recording_type': [self.metadata['recording_type']],
                                              'linking_group': [linking_group],
                                              'experiment_id': [experiment_id],
                                              'recording_region': [self.metadata['region']],
                                              'n_neurons': [self.n_neurons],
                                              'n_trials': [self.n_trials],
                                              'dat_path': [self.dat_path],
                                              'fps':[self.sps],
                                              'data_obj_version': ['TwoAFC_spikes_v1'],
                                              'objID': [self.objID]})

        #this is always the tracking file!
        engine = create_engine('sqlite:///D:\\tracking.db', echo=True)
        with engine.connect() as con:
            tracking_df.to_sql('2AFC_tracking', con=con, if_exists='append')

    def __getitem__(self, item):
        # --------indexing arguments------------- #
        assert(len(item) == 3)
        trial_property_column = item[0]  # e.g., trial outcome
        neuron_indexer = item[1]  # can be a list of neurons, any valid neuron indexing
        phase_indexer = item[2]  # selects to what the traces are aligned
        # --------------------------------------- #

        # ---------- first get rows (trials) that match value of column of interest ------------ #
        if type(trial_property_column) == str:
            trial_id = self.behav_df[trial_property_column].index.to_list()
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
            traces = self.ca_traces
        elif phase_indexer == "stimulus":
            traces = self.sa_traces
        elif phase_indexer == 'reward':
            traces = self.ra_traces
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
        return str(self.metadata)

    def iter_clusters(self,  phase_indexer = slice(None), return_labels = True):
        if return_labels:
            for c in np.unique(self.cluster_labels):
                tr = self[ :, self.active_neurons[self.cluster_labels == c], phase_indexer]
                yield c, tr
        else:
            for c in np.unique(self.cluster_labels):
                tr = self[ :, self.active_neurons[self.cluster_labels == c], phase_indexer]
                yield tr

    def to_pickle(self, remove_old=False):
        if remove_old:
            print("warning, deleting all old obj files")
            trace_files = glob.glob(self.dat_path+"*traces_dict.pkl")
            [os.remove(f) for f in trace_files]

            df_files = glob.glob(self.dat_path+"*behav_df.pkl")
            [os.remove(f) for f in df_files]

            info_files = glob.glob(self.dat_path + "*info.pkl")
            [os.remove(f) for f in info_files]

            feature_files = glob.glob(self.dat_path + "*feature_df*.pkl")
            [os.remove(f) for f in feature_files]

        save_dir = self.dat_path + '/' + self.objID + '/'
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        with open(save_dir + 'traces_dict.pkl', 'wb') as f:
            pickle.dump(self.traces_dict, f)

        with open(save_dir + "behav_df.pkl", 'wb') as f:
            pickle.dump(self.behav_df, f)

        with open(save_dir + "persistent_info.pkl", 'wb') as f:

            persistent_info = {'cluster_labels':self.cluster_labels,
                               'active_neurons':self.active_neurons,
                               'feature_df_cache': self.feature_df_cache,
                               'feature_df_keys':self.feature_df_keys,
                               'neuron_mask_df':self.neuron_mask_df,
                               'name': self.name,
                               'metadata': self.metadata,
                               'behavior_phase': self.behavior_phase,
                               'sps': self.sps}

            pickle.dump(persistent_info, f)

    def get_feature_df(self, alignment='reward',
                             variables=['rewarded'],
                             force_new=False,
                             apply_Gauss_filter_to_traces=False,
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

        matches = [(cache[0] == alignment) & (cache[1] == variables) for cache in self.feature_df_keys]
        if (sum(matches) > 0) and not force_new:
            cache_ind = np.where(matches)[0]
            print(cache_ind[0])
            feature_df = pickle.load(open(self.feature_df_cache[int(cache_ind[0])], 'rb'))
            print('loaded a cached feature df')
        else:
            # Get all traces from all neurons that match the prescribed alignment
            traces = self[:, :, alignment]
            if apply_Gauss_filter_to_traces:
                traces = gaussian_filter1d(traces, sigma=1, axis=2)

            # the given list of selected neurons will be used, unless it is empty or if 'active' is passed instead
            if selected_neurons == 'active':
                selected_neurons = self.active_neurons
            elif not len(selected_neurons):
                selected_neurons = np.arange(self.n_neurons)

            feature_df = tu.get_trace_feature_df(self.behav_df, selected_neurons,
                                              traces=traces, behavior_variables=variables, rat_name=self.name)

            print("Created new feature df.")
            if save:
                self.feature_df_keys.append([alignment, variables])
                fname = self.dat_path + self.objID + '_feature_df_' + str(len(self.feature_df_keys)) + '.pkl'
                with open (fname, 'wb') as f:
                    pickle.dump(feature_df, f)

                print("New feature df cached.")
                self.feature_df_cache.append(fname)

        return feature_df


class TwoAFC(DataContainer):
    def __init__(self, dat_path, behav_df, traces_dict, 
                 neuron_mask_df=None, 
                 objID=None, 
                 sps=None, 
                 name=None,
                 cluster_labels=None,
                 metadata=None,
                 linking_group=None, 
                 active_neurons=None, 
                 record=False, 
                 feature_df_cache=[], 
                 feature_df_keys=[], 
                 session=None, 
                 behavior_phase=None):

        super().__init__(dat_path, behav_df, traces_dict, neuron_mask_df, 
                         objID, sps, name, cluster_labels, metadata, 
                         linking_group, active_neurons, record, 
                         feature_df_cache, feature_df_keys, behavior_phase)

        self.session = session


class Multiday_2AFC(DataContainer):
    def __init__(self, dat_path, obj_list, cell_mapping, 
                 objID=None,
                 sps=None,
                 name=None,
                 cluster_labels=None,
                 metadata=None,
                 active_neurons=None,
                 record=True,
                 feature_df_cache=[],
                 feature_df_keys=[]):

        assert('session' in obj_list[0].behav_df.keys())

        self.behav_df = pd.concat([obj.behav_df for obj in obj_list])
        self.n_sessions = len(obj_list)
        traces_dict = map_traces(self.behav_df, obj_list, matches=cell_mapping)

        self.name = name
        self.metadata = metadata
        self.sps = sps
        self.dat_path = dat_path
        self.feature_df_cache = feature_df_cache
        self.feature_df_keys = feature_df_keys

        self.linking_group = [obj.objID for obj in obj_list]

        if objID is not None:
            self.objID = objID
        else:
            self.objID = str(np.datetime64('now').astype('uint32'))

        self.sa_traces = traces_dict['stim_aligned']
        self.ca_traces = traces_dict['response_aligned']
        self.ra_traces = traces_dict['reward_aligned']
        self.interp_traces = traces_dict['interp_traces']

        self.interp_inds = traces_dict['interp_inds']

        self.choice_ind = traces_dict['response_ind']
        self.reward_ind = traces_dict['reward_ind']
        self.stim_ind = traces_dict['stim_ind']
        self.n_trials, self.n_neurons, _ = self.sa_traces.shape

        if active_neurons is not None:
            self.active_neurons = active_neurons
        else:
            self.active_neurons = np.arange(self.n_neurons)

        self.traces_dict = traces_dict

        if (cluster_labels is not None):
            self.cluster_labels = cluster_labels
            self.cluster_ids = np.unique(cluster_labels)

        if record:
            self.record_experiment(self.linking_group, self.metadata['experiment_id'])


def from_pickle(dat_path, objID, obj_class):
    with open(dat_path + objID + "traces_dict.pkl", 'rb') as f:
        traces_dict = pickle.load(f)

    with open(dat_path + objID + "behav_df.pkl", 'rb') as f:
        behav_df = pickle.load(f)

    with open(dat_path + objID + "persistent_info.pkl", 'rb') as f:
        kwargs = pickle.load(f)

    return obj_class(dat_path, behav_df=behav_df, traces_dict=traces_dict, objID=objID, record=False, **kwargs)


def create_experiment_data_object(datapath, metadata, session_number, sps):
    # load neural data: [number of neurons x time bins in ms]
    spike_times = load(datapath + "spike_mat.npy")

    # make pandas behavior dataframe
    behav_df = load(datapath + 'behav_df')

    # format entries of dataframe for analysis (e.g., int->bool)
    cbehav_df = bu.convert_df(behav_df, session_type="SessionData", WTThresh=1, trim=False)

    # align spike times to behavioral data timeframe
    # spike_times_start_aligned = array [n_neurons x n_trials x longest_trial period in ms]
    spike_times_start_aligned, _ = tu.trial_start_align(cbehav_df, spike_times, sps=1000)

    # subsample (bin) data:
    # [n_neurons x n_trials x (-1 means numpy calculates: trial_len / dt) x ds]
    # then sum over the dt bins
    n_neurons = spike_times_start_aligned.shape[0]
    n_trials = spike_times_start_aligned.shape[1]
    timestep_ds = int(1000 / sps)
    spike_times_ds = spike_times_start_aligned.reshape(n_neurons, n_trials, -1, timestep_ds)
    spike_times_ds = spike_times_ds.sum(axis=-1)  # sum over bins

    # create trace alignments
    traces_dict = tu.create_traces_np(cbehav_df,
                                      spike_times_ds,
                                      sps=sps,
                                      aligned_ind=0,
                                      filter_by_trial_num=False,
                                      traces_aligned="TrialStart")

    cbehav_df['session'] = session_number
    # cbehav_df = bu.trim_df(cbehav_df)

    # create and save data object
    data_obj = TwoAFC(datapath, cbehav_df, traces_dict, name=metadata['rat_name'],
                      cluster_labels=[], metadata=metadata, sps=sps,
                      record=False, feature_df_cache=[], feature_df_keys=[])

    data_obj.to_pickle(remove_old=False)
