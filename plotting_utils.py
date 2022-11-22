import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trace_utils import get_feature_df

def plot_condition_average(df, variables = [], markers = [], ax = None, n_boot = 100, legend = False, palette = 'husl'):
    '''
    wrapper for sns.lineplot
    :param df dataframe that contains the relevant variables to plot
    :param variables: a list of variables to condition plot by, must not be longer than 3 elements
    :param ax: axes to plot on
    :param n_boot: sets the number of bootstraps for seaborn to use. Smaller numbers are faster, seaborn default is 1000
    :return: fig, ax
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1)

    assert(len(variables) <= 2)

    param = ['hue', 'style']
    kwargs = {param[i]: variables[i] for i in range(len(variables))}

    sns.lineplot(data = df, x = 'time', y = 'activity', ax = ax, **kwargs, palette =palette, n_boot = n_boot, linewidth = 2, legend = legend)

    #todo make this more general for any alignment
    for m in markers:
        ax.axvline(m, color = 'k')

    return ax

def plot_all_neurons(traces, behav_df, variables, markers = [], title_str = ''):
    fig, axes = plt.subplots(5, 3, figsize = [20, 20])

    df = get_feature_df(behav_df, range(traces.shape[1]), traces, code_names = variables)

    ax = axes[0, 0]
    plot_condition_average(df, ['rewarded'], markers, ax)
    #eventually make this more alignments?

    axes = axes.reshape(-1)
    for i, n in enumerate(range(traces.shape[1])):
        if i +1 >= len(axes):
            #we only plot 15 neurons at most right now
            break
        else:
            plot_condition_average(df[df.neuron == n], ['rewarded'], markers, axes[i+1])
            axes[i+1].set_title("neuron: " + str(n) + title_str)
            sns.despine()
    plt.show()


from scipy.ndimage import gaussian_filter1d
from obj_utils import group_iter_clusters

def plot_clusters(data_objs, alignment = 'reward', FILTER_ACTIVE = False, markers = [], save = False, save_str = "",
                  PLOT_SINGLE_NEURONS = False, CLUSTS_TO_PLOT = [], MAX_TO_PLOT = 5, variables = ['rewarded']):

    labels = [obj.cluster_labels for obj in data_objs]
    labels = [l for lab in labels for l in lab]

    if FILTER_ACTIVE:
        objs = [data_objs[i] for i in range(len(data_objs)) if (len(data_objs[i].active_neurons) > 0)]
    else:
        objs = data_objs

    fig, axes = plt.subplots(int(len(np.unique(labels))/3 + 1), 3, figsize = [30, int(len(np.unique(labels))/3 + 1)*4])
    axes = axes.reshape(-1)

    for j, (c, dfs, traces, names) in enumerate(group_iter_clusters(objs, alignment, filter_active= FILTER_ACTIVE, require_all_clusters=False)):
        if (c in CLUSTS_TO_PLOT):

            t = [gaussian_filter1d(traces[i], sigma = 2, axis = 2) for i in range(len(traces))]

            df = pd.concat([get_feature_df(dfs[i].reset_index(drop = True), range(t[i].shape[1]),
                                           np.sqrt(t[i]), code_names = variables, rat_name = names[i])
                            for i in range(len(dfs))]).reset_index(drop = True)

            if j == len(np.unique(labels) - 1):
                legend = True
            else:
                legend = False

            plot_condition_average(df, variables, markers, axes[j], legend = legend)

            axes[j].set_title("cluster: " + str(c))
            sns.despine()

        if PLOT_SINGLE_NEURONS and (c in CLUSTS_TO_PLOT):

            uniqueID = list(zip(df.rat_name, df.neuron))

            df['uniqueID'] = uniqueID
            neurons = df['uniqueID'].unique()
            n_neurons = len(neurons)

            print(neurons, n_neurons)
            if n_neurons > MAX_TO_PLOT:
                inds = np.random.choice(np.arange(0, n_neurons), size = MAX_TO_PLOT, replace = False)
                n_neurons = MAX_TO_PLOT
                neurons = neurons[inds]

            fig2, axes2 =  plt.subplots(int(round((n_neurons + 1)/ 3)), 3, figsize= [10, int((n_neurons/10)*10)])
            axes2 = axes2.reshape(-1)

            for k, n in enumerate(neurons):
                print(k, n)
                plot_condition_average(df[df.uniqueID == n], variables, markers, axes2[k])
                axes2[k].set_title("cluster: " + str(c) + " uniqeID: " + str(n) )
                sns.despine()



            fig2.tight_layout()
            if save:
                plt.savefig("D:/figures/" + save_str + "single_neurons" + str(c) + ".pdf", dpi=600);
            fig2.show()

    fig.tight_layout()

    if save:
        plt.savefig("D:/figures/" + save_str + ".pdf", dpi=600);
    fig.show()

from sklearn.manifold import TSNE

def pretty_plot_tsne(clustermat, labels, perplexity = 35, n_iter = 250, plot_labels = True, save = False, save_str = ""):
    X_embedding = TSNE(metric='cosine', learning_rate='auto', perplexity=perplexity, n_iter=n_iter, init='pca').fit_transform(
       clustermat)

    colors = sns.color_palette(palette='twilight', n_colors=len(np.unique(labels)))#rns)))
    sort = np.argsort(labels)

    c = [colors[l] for l in sorted(labels)]

    plt.figure(figsize = [6, 6])
    plt.scatter(X_embedding[sort, 0], X_embedding[sort, 1], c=c, alpha=.5, linewidth=2, s = 80)

    if plot_labels:
        for c in sorted(np.unique(labels)):
            x = X_embedding[labels == c, 0].mean(axis = 0)
            y = X_embedding[labels == c, 1].mean(axis = 0)

            circle1 = plt.Circle((x, y), 3, color=colors[c])
            plt.gca().add_patch(circle1)

            plt.annotate(str(c), [x -1.5 ,y - 1.5],fontsize = 15, fontweight = "bold", fontfamily = "myriad" )


    plt.xlabel("tsne dim 1 au")
    plt.ylabel("tsne dim 2 au")
    sns.despine()

    if save:
        plt.savefig("D:/figures/" + save_str + ".pdf", dpi=600);
    plt.show()

    return X_embedding

from scipy.ndimage import gaussian_filter1d
from behavior_utils import map_p_correct
from scipy.stats import zscore

def plot_all_alignments(objs, neurs, save = False, save_str = ""):

    interp_traces = [gaussian_filter1d(obj[:, :, :], 3, axis = -1) for obj in objs]
    #reward_traces= [gaussian_filter1d(obj[:, :, 'reward'], 3, axis = -1) for obj in objs]
    choice_traces = [gaussian_filter1d(obj[:, :, 'response'], 3, axis = -1) for obj in objs]
    #stim_traces = [gaussian_filter1d(obj[:, :, :], 3, axis = -1) for obj in objs]

    df_interp = pd.concat([get_feature_df(obj.behav_df, ns, itt, code_names = ['rewarded']) for obj, itt, ns in zip(objs, interp_traces, neurs) if len(ns) > 0])
    #df_reward = pd.concat([get_feature_df(obj.behav_df, ns, rt, code_names = ['WT_qs', 'rewarded']) for obj, rt, ns in zip(objs, reward_traces, neurs) if len(ns) > 0])
    df_response = pd.concat([get_feature_df(obj.behav_df, ns, rwt, code_names = ['WT_qs', 'WT', 'correct', 'rewarded']) for obj, rwt, ns in zip(objs, choice_traces, neurs) if len(ns) > 0])
    #df_stim = pd.concat([get_feature_df(obj.behav_df, ns, st, code_names = ['prev_rewarded_1']) for obj, st, ns in zip(objs, stim_traces, neurs) if len(ns) > 0])

    ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2, rowspan=2)
    plot_condition_average(df_interp.reset_index(), markers = objs[0].interp_inds, variables = ['rewarded'], ax = ax1, legend = True, n_boot = 100)
    sns.despine()

    #ax2 = plt.subplot2grid(shape=(3, 3), loc=(2, 0), colspan=1)
    #plot_condition_average(df_stim.reset_index(), markers = objs[0].interp_inds, variables = ['prev_rewarded_1'], ax = ax2, n_boot = 100,
    #                       legend = False, palette='mako')
    #ax2.set_xlim(0, objs[0].interp_inds[2])
    #sns.despine()

    ax3 = plt.subplot2grid(shape=(4, 2), loc=(2, 0), colspan=1, rowspan = 2)
    plot_condition_average(df_response[df_response.WT > 2.5].reset_index(), markers = [objs[0].choice_ind], variables = ["correct"], ax = ax3, n_boot = 100,
                           legend = False, palette = 'vlag')

    ax3.set_xlim([0, objs[0].choice_ind + 80])
    sns.despine()

    ax4 = plt.subplot2grid(shape=(4, 2), loc=(2, 1), colspan=1, rowspan=2)
    plot_condition_average(df_response[df_response.rewarded == False].reset_index(), markers = [objs[0].choice_ind], variables = ['WT_qs'], ax = ax4, n_boot = 100,
                           legend = False, palette = "dark:b")

    ax4.set_xlim([0, objs[0].choice_ind + 80])
    sns.despine()

    plt.tight_layout()

    if save:
        plt.savefig("D:/figures/" + save_str + ".pdf", dpi=600);

    plt.show()

def plot_confidence(objs, inds, title_str = "", sps =40, save_str = None, save = False):
    dfs = [obj.behav_df[(obj.behav_df.before_switch == 0) & (obj.behav_df.WT > 2)] for obj in objs]

    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i].WT > 2.5]
        dfs[i]["easy"] = (map_p_correct(dfs[i])["p_correct"] >= .85)
        dfs[i]["uns_ev"] = abs(dfs[i].evidence)
        dfs[i]['WT_qs'] = pd.cut(dfs[i].WT, bins = 2,  labels =  ["short", "long"], retbins = False)
        dfs[i]["DV_bins"] =  pd.cut(dfs[i].evidence, bins = 5,  labels = np.linspace(-1, 1, 5), retbins = False)

    behav_df = pd.concat(dfs).reset_index()

    interp_traces = [gaussian_filter1d(obj[:, :, :][df.index.to_numpy()], 2, axis = -1) for df, obj in zip(dfs, objs)]
    df_interp = pd.concat([get_feature_df(df, ns, itt, code_names = ['probe', 'correct']) for obj, itt, ns, df in zip(objs, interp_traces, inds, dfs) if len(ns) > 0])

    choice_traces = [np.sqrt(gaussian_filter1d(obj[:, :, 'response'][:, :, obj.choice_ind:obj.choice_ind + int(2*sps)][df.index.to_numpy()], 2, axis = -1)) for df, obj in zip(dfs, objs)]

    df_response = pd.concat([get_feature_df(df, ns, rwt, code_names =['correct', 'DV_bins', 'WT', 'WT_qs', 'evidence', "easy", "uns_ev"]) for obj, rwt, ns, df in zip(objs, choice_traces, inds, dfs) if len(ns) > 0])

    df_response["norm_act"] = df_response["activity"]
    df_response["norm_act_qs"] = (df_response["activity"]*10).round(0)/10

    fig = plt.figure(figsize = [10, 10])
    fig.suptitle(title_str, fontsize=16)

    ax = plt.subplot2grid(shape = (4, 2),loc = (0, 0), colspan = 2)
    plot_condition_average(df_interp[df_interp.probe].reset_index(), markers = objs[0].interp_inds, variables = ['correct'], ax = ax, legend = True, n_boot = 100)
    sns.despine()


    ax = plt.subplot2grid(shape = (4, 2),loc = (1, 0), colspan = 1)
    sns.lineplot(data = df_response.reset_index(), x = "time", y = "activity", hue = "correct", size = "easy",  hue_order = [False, True],
                 palette = ["r", "g"],  ax = ax, n_boot = 100, size_order = [False, True])
    plt.axvline(objs[0].choice_ind, color = "k")
    sns.despine()

    ax = plt.subplot2grid(shape = (4, 2),loc = (1, 1), colspan = 1)
    sns.lineplot(data = df_response.reset_index(), x = "time", y = "norm_act", hue = "WT_qs", palette = "dark:b",
                 ax = ax, n_boot = 100)

    plt.axvline(objs[0].choice_ind, color = "k")
    sns.despine()

    ax = plt.subplot2grid(shape = (4, 2),loc = (2, 0), colspan = 1)
    sns.pointplot(data = df_response.reset_index(), x = "DV_bins", y = "norm_act", hue = "correct", hue_order = [False, True],
                 palette = ["r", "g"], ax = ax, n_boot = 100, x_bins = 5, x_estimator = np.nanmean)

    sns.despine()
    ax = plt.subplot2grid(shape = (4, 2),loc = (2, 1), colspan = 1)
    sns.lineplot(data = df_response.reset_index(), x = "norm_act_qs", y = "correct", ax = ax, n_boot = 100)
    sns.regplot(data = df_response.reset_index(), x = "norm_act_qs", y = "correct", ax = ax, n_boot = 100, x_estimator = np.nanmean,
                fit_reg = False,color = "b")

    sns.despine()

    ax = plt.subplot2grid(shape = (4, 2),loc = (3, 0), colspan = 1)
    sns.pointplot(data = behav_df[~behav_df.correct | behav_df.probe], x = "DV_bins", y = "WT", hue = "correct",hue_order = [False, True],
                 palette = ["r", "g"], ax = ax, n_boot = 100)

    sns.despine()
    ax = plt.subplot2grid(shape = (4, 2),loc = (3, 1), colspan = 1)
    sns.regplot(data = behav_df, x = "DV_bins", y = "resp_dir", ax = ax, n_boot = 100, logistic = True, x_bins = 8, x_estimator = np.nanmean)
    sns.despine()


    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    if save:
        plt.savefig("D:/figures/" + save_str +".pdf", dpi=300)
    plt.show()

def hue_regplot(data, x, y, hue, palette=None, **kwargs):
    from matplotlib.cm import get_cmap

    regplots = []

    levels = sorted(data[hue].unique())


    if palette is None:
        default_colors = get_cmap('tab10')
        palette = {k: default_colors(i) for i, k in enumerate(levels)}

    for key in levels:
        regplots.append(
            sns.regplot(
                x=x,
                y=y,
                data=data[data[hue] == key],
                color=palette[key],
                **kwargs
            )
        )

    return regplots

def make_updating_plots(behav_df, axes, vmin = -.2, vmax = .2, facecolor = 'white', row_title = "data", pad = 5, plot_conf = False):

    behav_df = behav_df[behav_df['prev_completed_1']].fillna(False)

    for ax in axes:
        ax.set_facecolor(facecolor)
        ax.set_clip_on(False)


    axes[0].annotate(row_title, xy=(0, .5),xytext= (-axes[0].yaxis.labelpad - pad, 0),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True, center = "dark")
    ax = axes[0]

    keys = sorted(behav_df.prior.unique())
    cmap1 = sns.color_palette('Dark2', 3)

    hue_regplot(data=behav_df, x='evidence', y='resp_dir', hue='prior',
                logistic=True, ax=ax, n_boot=1000, x_estimator = np.nanmean)

    ax.set_ylabel('Response Rightward (%)')
    ax.set_xlabel('Coherence')
    ax.set_yticks([0, 0.5, 1.0], ['0', '50', '100'])
    sns.despine()

    ax = axes[1]
    keys = sorted(behav_df.prev_resp_dir_1.unique())
    cmap1 = sns.color_palette('Set2', 3)

    hue_regplot(data=behav_df[behav_df.prev_correct_1 == True], x='evidence', y='resp_dir', hue='prev_resp_dir_1',
                logistic=True, ax=ax, n_boot=1000, palette={keys[i]: cmap1[i] for i in range(len(keys))}, x_estimator = np.nanmean)

    ax.set_ylabel('Response Rightward (%)')
    ax.set_xlabel('Coherence')
    ax.set_yticks([0, 0.5, 1.0], ['0', '50', '100'])
    sns.despine()


    if len(behav_df.prior.unique()) > 1:
        hm_prior_l = behav_df[(behav_df.prev_correct_1 == True) & (behav_df.prior < .5)].pivot_table(
            index=['evidence'], columns=['prev_evidence_1'],
            values='resp_dir', aggfunc=np.nanmean)

        av_psych_l = behav_df[(behav_df.prev_correct_1 == True) & (behav_df.prior < .5)].pivot_table(
            columns=["evidence"],
            values="resp_dir").to_numpy().reshape(
            [-1, 1])

        hm_prior_r = behav_df[(behav_df.prev_correct_1 == True) & (behav_df.prior > .5)].pivot_table(
            index=['evidence'],
            columns=['prev_evidence_1'],
            values='resp_dir',
            aggfunc=np.nanmean)

        av_psych_r = behav_df[(behav_df.prev_correct_1 == True) & (behav_df.prior > .5)].pivot_table(
            columns=["evidence"],
            values="resp_dir").to_numpy().reshape(
            [-1, 1])

        ax = axes[2]
        sns.heatmap(data=((hm_prior_l - av_psych_l) + (hm_prior_r - av_psych_r)) / 2, cmap=cmap, vmin=vmin, vmax=vmax,
                    ax=ax)
        sns.despine()
        ax.set_ylabel("evidence")
        ax.set_xlabel("previous evidence")
    else:
        hm_prior = behav_df[(behav_df.prev_correct_1 == True)].pivot_table(index=['evidence'],columns=['prev_evidence_1'],
                                                                            values='resp_dir', aggfunc=np.nanmean)

        av_psych = behav_df[(behav_df.prev_correct_1 == True)].pivot_table(columns=["evidence"],values='resp_dir',
                                                                             aggfunc=np.nanmean).to_numpy().reshape([-1, 1])

        ax = axes[2]
        sns.heatmap(data=(hm_prior - av_psych), cmap=cmap, vmin=vmin, vmax=vmax,
                    ax=ax)
        sns.despine()
        ax.set_ylabel("evidence")
        ax.set_xlabel("previous evidence")



    behav_df["abs_ev"] = abs(behav_df.evidence)
    if plot_conf:
        ax = axes[3]
        sns.pointplot(data = behav_df[(behav_df.probe )], x = 'abs_ev', y = 'confidence', hue = 'correct', ax = ax, ci = 68)
        ax.set_ylim([3, 7])
        ax.set_xlabel("evidence")
        ax.set_ylabel("confidence")