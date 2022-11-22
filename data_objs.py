import pickle
import pandas as pd
from sqlalchemy import create_engine, update, delete
import numpy as np
from trace_utils import get_feature_df
from scipy.ndimage import gaussian_filter1d
from obj_utils import *
import os
import glob

class DataContainer:
    def __init__(self, dat_path, behav_df, traces_dict, neuron_mask_df = None, objID  = None, sps = None, name = None,  cluster_labels = None,
                 metadata = None, linking_group = None, active_neurons = None, record = True, feature_df_cache = [],
                 feature_df_keys = [], behavior_phase = None):

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
                if len(active_neurons) > 0:
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
                self.neuron_mask_df = pd.DataFrame(df_dict, index = df_dict["neurons"], columns= ["cluster_mask", "active_mask"])

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
        assert(len(item) == 3)
        trial_indexer = item[0]
        phase_indexer = item[2]
        if phase_indexer == "interp":
            traces = self.interp_traces
            phi = slice(None, None, None)
        elif phase_indexer == "response":
            traces = self.ca_traces
            phi = slice(None, None, None)
        elif phase_indexer == "stimulus":
            traces = self.sa_traces
            phi = slice(None, None, None)
        elif phase_indexer == 'reward':
            traces = self.ra_traces
            phi = slice(None, None, None)
        elif type(phase_indexer) == slice:
            traces = self.interp_traces
            phi = phase_indexer
        else:
            raise("Not Implemented")
        if traces is None:
            raise("Not Implemented")
        if type(trial_indexer) == str:
            trial_id = self.behav_df[trial_indexer].index.to_list()
        elif type(trial_indexer) == list:
            if np.issubdtype(type(trial_indexer[0]), np.integer):
                trial_id = trial_indexer
            elif type(trial_indexer[0]) == str:
                trial_id = self.behav_df[np.logical_and.reduce([self.behav_df[ti] for ti in trial_indexer])].index.to_list()
            else:
                raise("not implemented")
        elif type(trial_indexer) == slice:
            trial_id = trial_indexer
        elif type(trial_indexer) == dict:
            trial_id = self.behav_df[np.logical_and.reduce([(self.behav_df[ti] == trial_indexer[ti]).to_numpy() for ti in trial_indexer])].index.to_list()
        else:
            raise("Not Implemented")

        traces = traces[trial_id]

        #neuron indexer can be a list of neurons, any valid neuron indexing
        neuron_indexer = item[1]
        traces = traces[:, neuron_indexer, :]

        traces = traces[:, :, phi]

        return traces

    def __setitem__(self, key, value):
        raise("Not Implemented")

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

    def to_pickle(self, remove_old = False):
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

        with open(self.dat_path + self.objID + "traces_dict.pkl", 'wb') as f:
            pickle.dump(self.traces_dict, f)

        with open(self.dat_path + self.objID + "behav_df.pkl", 'wb') as f:
            pickle.dump(self.behav_df, f)

        with open(self.dat_path + self.objID + "persistent_info.pkl", 'wb') as f:

            persistent_info = {'cluster_labels':self.cluster_labels,
                               'active_neurons':self.active_neurons,
                               'feature_df_cache': self.feature_df_cache,
                               'feature_df_keys':self.feature_df_keys,
                               'neuron_mask_df':self.neuron_mask_df,
                               'name': self.name,
                               'metadata': self.metadata,
                               'behavior_phase': self.behavior_phase}

            pickle.dump(persistent_info, f)

    def get_feature_df(self, alignment = 'reward', variables = ['rewarded'], force_new = False, save = True, filter_by = None,
                       filter = False, filter_active = False):
        '''
        :param alignment:
        :param variables:
        :return:
        '''
        matches = [(cache[0] == alignment) & (cache[1] == variables) for cache in self.feature_df_keys]
        if (sum(matches) > 0) and not force_new:
            cache_ind = np.where(matches)[0]
            print(cache_ind[0])
            feature_df = pickle.load(open(self.feature_df_cache[int(cache_ind[0])], 'rb'))
            print('loaded a cached feature df')
        else:
            traces = self[:, :, alignment]
            if filter:
                traces = gaussian_filter1d(traces, sigma = 1, axis = 2)

            if filter_active:
                feature_df = get_feature_df(self.behav_df, self.active_neurons, traces = traces, code_names = variables, rat_name = self.objID)
            else:
                if filter_by is not None:
                    feature_df = get_feature_df(self.behav_df, filter_by, traces=traces,
                                                code_names=variables,
                                                rat_name=self.objID)
                else:
                    feature_df = get_feature_df(self.behav_df, np.arange(self.n_neurons), traces=traces, code_names=variables,
                                            rat_name=self.objID)


            print("made a brand new feature df")
            if save:
                print("caching the new feature df")
                self.feature_df_keys.append([alignment, variables])
                fname = self.dat_path + self.objID + '_feature_df_' + str(len(self.feature_df_keys)) + '.pkl'
                with open (fname, 'wb') as f:
                    pickle.dump(feature_df, f)

                self.feature_df_cache.append(fname)

        return feature_df


class TwoAFC(DataContainer):
    def __init__(self, dat_path, behav_df, traces_dict, neuron_mask_df = None, objID  = None, sps = None, name = None,  cluster_labels = None,
                 metadata = None, linking_group = None, active_neurons = None, record = True, feature_df_cache = [],
                 feature_df_keys = [], session = None, behavior_phase = None):

        super().__init__(dat_path, behav_df, traces_dict, neuron_mask_df, objID, sps, name, cluster_labels,
                 metadata, linking_group, active_neurons, record, feature_df_cache, feature_df_keys,
                         behavior_phase)

        self.session = session

class Multiday_2AFC(DataContainer):
    def __init__(self, dat_path, obj_list, cell_mapping, objID = None, sps = None, name = None,  cluster_labels = None,
                 metadata = None, active_neurons = None, record = True, feature_df_cache = [],
                 feature_df_keys = []):

        assert('session' in obj_list[0].behav_df.keys())

        self.behav_df = pd.concat([obj.behav_df for obj in obj_list])
        self.n_sessions = len(obj_list)
        traces_dict = map_traces(self.behav_df, obj_list, matches = cell_mapping)

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

    return obj_class(dat_path, behav_df = behav_df, traces_dict = traces_dict, objID = objID, record = False, **kwargs)