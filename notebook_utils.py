import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context='paper',
        style='whitegrid',
        font_scale=3,
        rc={"lines.linewidth": 5})

paths = {
    'macro': 'macro_results/',
    'micro': 'micro_results/',
}

ds_rename = {
    'citeseer': 'CIT',
    'cora': 'COR',
    'pubmed': 'MED',
    'cs': 'CS',
    'computers': 'CMP',
    'physics': 'PHY',
    'photo': 'PHO'
}


def transform_df_acc(df, threshold, selector='val_scores'):
    for seed in list(df['seed'].unique()):
        selected_seed = df['seed'] == seed
        df.loc[selected_seed, selector] = \
            df.loc[selected_seed, selector].apply(
                lambda x: 1.0 if x > threshold else 0.0)
        df.loc[selected_seed, selector] = \
            df.loc[selected_seed, selector].cumsum()
    return df


def load_evolution_results(seed=123, dataset='', macro=True, ev_type=""):
    # Load offspring results
    results_path = paths['macro'] if macro else paths['micro']
    results_path = os.path.join(results_path,
                                "_".join(["seed", str(seed)]))
    ev_results = pd.read_csv(os.path.join(results_path,
                                          "_".join(["parent",
                                                    "child",
                                                    dataset,
                                                    ev_type])),
                             names=['parent_arch', 'parent_acc',
                                    'child_arch', 'child_acc'],
                             header=None,
                             sep='>')
    arch_results = ev_results[['parent_arch', 'child_arch']].copy()
    arch_results['index'] = arch_results.index
    arch_results = pd.melt(arch_results, id_vars=['index'])
    ev_results = ev_results[['parent_acc', 'child_acc']]
    ev_results['index'] = ev_results.index

    ev_results = pd.melt(ev_results, id_vars=['index'])
    initial_population_scores = \
        pd.read_csv(os.path.join(results_path,
                                 "_".join(["initial",
                                           "population",
                                           dataset,
                                           ev_type])),
                    header=None,
                    names=['scores'])
    # Load timing results
    ev_time = open(os.path.join(results_path,
                                "_".join(["time", dataset, ev_type])),
                   'r').read()
    # Load population stats
    population_stats = pd.read_csv(os.path.join(results_path,
                                                "_".join(['population',
                                                          'stats',
                                                          dataset,
                                                          ev_type])))
    population_stats['index'] = population_stats.index
    population_stats = pd.melt(population_stats, id_vars=['index'])
    oom_count = pd.read_csv(os.path.join(results_path,
                                         "_".join(["oom",
                                                   dataset,
                                                   ev_type])),
                            names=['EA'],
                            dtype={'EA': np.int64})
    ev_results['seed'] = seed
    initial_population_scores['seed'] = seed
    population_stats['seed'] = seed
    arch_results['seed'] = seed
    oom_count['seed'] = seed
    return (ev_results,
            ev_time,
            initial_population_scores,
            population_stats,
            arch_results,
            oom_count)


def load_RL_results(seed=123, dataset='', macro=True):
    results_path = paths['macro'] if macro else paths['micro']
    results_path = os.path.join(results_path,
                                "_".join(["seed", str(seed)]))
    indexes = []
    val_scores = []
    store = False
    rl_times = []
    with open(os.path.join(results_path,
                           "_".join(['results',
                                     'RL',
                                     dataset])), 'r') as fp:
        for index, line in enumerate(fp):
            if 'took' in line:
                rl_times.append(float(line.split()[-1]))
            if 'val_score' in line and store:
                val_scores.append(float(line.split(',')[0].split(':')[1]))
            if 'training controller' in line:
                indexes.append(index)
                if 'over' in line:
                    store = False
                else:
                    store = True
    # Make times DataFrame
    rl_times = pd.DataFrame(np.array(rl_times), columns=['times'])
    rl_results = pd.DataFrame()
    rl_results['index'] = np.arange(len(val_scores))
    rl_results['val_scores'] = np.array(val_scores)
    rl_best_acc = pd.read_csv(os.path.join(results_path,
                                           "_".join(['best_acc_RL',
                                                     dataset])),
                              names=['arch',
                                     'accuracy',
                                     'error'],
                              sep='>')
    oom_count = pd.read_csv(os.path.join(results_path,
                                         "_".join(["oom",
                                                   dataset,
                                                   "RL"])),
                            names=['RL'],
                            dtype={'RL': np.int64})
    oom_count['seed'] = seed
    rl_results['seed'] = seed
    rl_times['seed'] = seed
    rl_best_acc['seed'] = seed
    return (rl_results,
            rl_times,
            rl_best_acc,
            oom_count)


def load_RS_results(seed=123, dataset='', macro=True):
    results_path = paths['macro'] if macro else paths['micro']
    results_path = os.path.join(results_path,
                                "_".join(["seed", str(seed)]))
    rs_results = pd.read_csv(os.path.join(results_path,
                             "_".join(['val_scores_RS',
                                       dataset])),
                             names=['val_scores'])
    rs_results['index'] = rs_results.index
    execution_time = open(os.path.join(results_path,
                                       "_".join(['time_RS',
                                                 dataset]))).read()
    best_accuracy_path = os.path.join(results_path,
                                      "_".join(['best_acc_RS',
                                                dataset]))
    rs_best_acc = pd.read_csv(best_accuracy_path,
                              names=['arch', 'accuracy'],
                              sep='>')
    oom_count = pd.read_csv(os.path.join(results_path,
                                         "_".join(["oom",
                                                   dataset,
                                                   "RS"])),
                            names=['RS'],
                            dtype={'RS': np.int64})
    oom_count['seed'] = seed
    rs_times = pd.DataFrame(np.array([float(execution_time)]),
                            columns=['times'])
    rs_results['seed'] = seed
    rs_times['seed'] = seed
    rs_best_acc['seed'] = seed
    return (rs_results,
            rs_times,
            rs_best_acc,
            oom_count)


def load_all_results(dataset='', macro=True, ev_type=""):
    results_path = paths['macro'] if macro else paths['micro']
    # EV
    ev_results_all = pd.DataFrame(columns=['index', 'variable',
                                           'value', 'seed'])
    pop_stats_all = pd.DataFrame(columns=['index', 'variable',
                                          'value', 'seed'])
    initial_pop_all = pd.DataFrame(columns=['scores', 'seed'])
    ev_arch_all = pd.DataFrame(columns=['index', 'variable',
                                        'value', 'seed'])
    ev_oom_all = pd.DataFrame(columns=['EA', 'seed'])
    ev_times_all = {}
    # RL
    rl_results_all = pd.DataFrame(columns=['index', 'val_scores', 'seed'])
    rl_times_all = pd.DataFrame(columns=['times', 'seed'])
    rl_best_acc_all = pd.DataFrame(columns=['accuracy', 'error',
                                            'seed', 'dataset'])
    rl_oom_all = pd.DataFrame(columns=['RL', 'seed'])
    # RS
    rs_results_all = pd.DataFrame(columns=['index', 'val_scores', 'seed'])
    rs_times_all = pd.DataFrame(columns=['times', 'seed'])
    rs_best_acc_all = pd.DataFrame(columns=['accuracy', 'seed', 'dataset'])
    rs_oom_all = pd.DataFrame(columns=['RS', 'seed'])
    for subdir in os.listdir(results_path):
        if 'seed' in subdir:
            seed = int(subdir.split('_')[-1])
            # LOAD EVOLUTION RESULTS
            (ev_results, ev_times,
             initial_pop_scores, population_stats, ev_arch, oom_count) = \
                load_evolution_results(seed=seed, dataset=dataset,
                                       macro=macro, ev_type=ev_type)
            ev_results_all = pd.concat([ev_results_all, ev_results],
                                       axis=0,
                                       ignore_index=True, sort=False)
            ev_arch_all = pd.concat([ev_arch_all, ev_arch],
                                    axis=0,
                                    ignore_index=True, sort=False)
            initial_pop_all = pd.concat([initial_pop_all, initial_pop_scores],
                                        axis=0,
                                        ignore_index=True, sort=False)
            pop_stats_all = pd.concat([pop_stats_all, population_stats],
                                      axis=0,
                                      ignore_index=True, sort=False)
            ev_oom_all = pd.concat([oom_count, ev_oom_all], axis=0,
                                   ignore_index=True, sort=False)
            ev_times_all[seed] = ev_times
            # LOAD RL RESULTS
            rl_results, rl_times, rl_best_acc, rl_oom = \
                load_RL_results(seed=seed, dataset=dataset, macro=macro)
            rl_results_all = pd.concat([rl_results_all, rl_results],
                                       axis=0,
                                       ignore_index=True, sort=False)
            rl_times_all = pd.concat([rl_times_all, rl_times],
                                     axis=0,
                                     ignore_index=True, sort=False)
            rl_best_acc['dataset'] = dataset
            rl_best_acc_all = pd.concat([rl_best_acc_all, rl_best_acc],
                                        axis=0,
                                        ignore_index=True,
                                        sort=False)
            rl_oom_all = pd.concat([rl_oom, rl_oom_all], axis=0,
                                   ignore_index=True, sort=False)
            # LOAD RS RESULTS
            rs_results, rs_times, rs_best_acc, rs_oom = \
                load_RS_results(seed=seed, dataset=dataset, macro=macro)
            rs_results_all = pd.concat([rs_results_all, rs_results],
                                       axis=0,
                                       ignore_index=True, sort=False)
            rs_times_all = pd.concat([rs_times_all, rs_times],
                                     axis=0,
                                     ignore_index=True, sort=False)
            rs_best_acc['dataset'] = dataset
            rs_best_acc_all = pd.concat([rs_best_acc_all, rs_best_acc],
                                        axis=0,
                                        ignore_index=True, sort=False)
            rs_oom_all = pd.concat([rs_oom, rs_oom_all], axis=0,
                                   ignore_index=True, sort=False)
    # return EA, RL and RS results
    return (ev_results_all, initial_pop_all, ev_times_all,
            pop_stats_all, ev_arch_all, ev_oom_all,
            rl_results_all, rl_times_all, rl_best_acc_all, rl_oom_all,
            rs_results_all, rs_times_all, rs_best_acc_all, rs_oom_all)


def draw_lineplots(datasets, ea, rl, rs, macro=True):
    nrows = len(datasets)
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(ncols * 10, nrows * 10),
                           sharey=True)
    for i, ds in enumerate(datasets):
        sns.lineplot(x='index', y='value', hue='variable',
                     data=ea[ds][ea[ds]['index'] < 400],
                     ax=ax[i][0])
        sns.lineplot(x='index', y='val_scores',
                     data=rl[ds][rl[ds]['index'] < 400],
                     ax=ax[i][1])
        sns.lineplot(x='index', y='val_scores',
                     data=rs[ds][rs[ds]['index'] < 400],
                     ax=ax[i][2])
        # Remove labels from legend
        handles, labels = ax[i][0].get_legend_handles_labels()
        ax[i][0].legend(handles=handles[1:], labels=labels[1:])
        ax[i][0].set_title('Evolution: ' + ds_rename[ds])
        ax[i][1].set_title('Reinforcement Learning: ' + ds_rename[ds])
        ax[i][2].set_title('Random Search: ' + ds_rename[ds])
        ax[i][0].set_xlabel('Iteration')
        ax[i][1].set_xlabel('Iteration')
        ax[i][2].set_xlabel('Iteration')
        ax[i][0].set_ylabel('Validation Score')
        ax[i][1].set_ylabel('Validation Score')
        ax[i][2].set_ylabel('Validation Score')
    results_path = paths['macro'] if macro else paths['micro']
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'plots',
                             'validation_score_per_epoch_200_all_seeds.pdf'))


def draw_cummmax(datasets, ev, rl, rs, macro=True):
    ncols = 2
    nrows = (len(datasets) // ncols)
    if (len(datasets) % ncols):
        nrows += 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 10),
                           sharey=True, squeeze=False)
    for idx, ds in enumerate(datasets):
        cummax_df = ev[ds].loc[ev[ds]['variable'] == 'Best']
        cummax_df.loc[:, 'variable'] = 'EA'
        for seed in rl[ds]['seed'].unique():
            seed_rl_cummax = rl[ds].loc[
                rl[ds]['seed'] == seed].cummax().rename(
                    columns={'val_scores': 'value'})
            seed_rl_cummax.loc[:, 'variable'] = 'RL'
            cummax_df = pd.concat(
                [cummax_df, seed_rl_cummax], axis=0, sort=False,
                ignore_index=True)
        for seed in rs[ds]['seed'].unique():
            seed_rs_cummax = rs[ds].loc[
                rs[ds]['seed'] == seed].cummax().rename(
                    columns={'val_scores': 'value'})
            seed_rs_cummax.loc[:, 'variable'] = 'RS'
            cummax_df = pd.concat(
                [cummax_df, seed_rs_cummax], axis=0, sort=False,
                ignore_index=True)
        cummax_df['value'] = cummax_df['value'].astype(np.float64)
        curax = ax[(idx // 2)][(idx % 2)]
        sns.lineplot(x='index', y='value', hue='variable', style='variable',
                     data=cummax_df, ax=curax)
        # Remove labels from legend
        handles, labels = curax.get_legend_handles_labels()
        curax.legend(handles=handles[1:], labels=labels[1:])
        curax.set_title(ds_rename[ds])
        curax.set_xlabel('Iteration')
        curax.set_ylabel('Validation Score')
    results_path = paths['macro'] if macro else paths['micro']
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'plots',
                             'validation_score_per_epoch_cummax.pdf'))


def draw_boxplots(ev_results_all, rl_results_all, all_seeds=False, macro=True):
    assert(len(ev_results_all.keys()) == len(rl_results_all.keys()))
    nrow = len(ev_results_all.keys())
    fig, ax = plt.subplots(nrow, 2, figsize=(18.75, nrow * 6), sharey=False)
    for i, db in enumerate(ev_results_all.keys()):
        sns.boxplot(x='variable', y='value',
                    data=ev_results_all[db], ax=ax[i][0])
        sns.boxplot(y='val_scores', data=rl_results_all[db], ax=ax[i][1])
        ax[i][0].set_title('Evolution')
        ax[i][1].set_title('Reinforcement Learning')
        ax[i][0].set_xlabel('')
        ax[i][0].set_ylabel('Validation Score')
        ax[i][1].set_ylabel('Validation Score')
    results_path = paths['macro'] if macro else paths['micro']
    if not all_seeds:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_boxplot.pdf'))
    else:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_boxplot_all_seeds.pdf'))


def draw_initial_pop_boxplot(datasets, initial_macro, initial_micro):
    full_initial_pop = pd.DataFrame(columns=['scores', 'seed',
                                             'dataset', 'mode'])
    for ds in datasets:
        initial_macro[ds]['dataset'] = ds
        initial_macro[ds]['mode'] = 'macro'
        full_initial_pop = pd.concat([full_initial_pop, initial_macro[ds]],
                                     ignore_index=True)
        initial_micro[ds]['dataset'] = ds
        initial_micro[ds]['mode'] = 'micro'
        full_initial_pop = pd.concat([full_initial_pop, initial_micro[ds]],
                                     ignore_index=True)
    full_initial_pop['dataset'] = \
        full_initial_pop['dataset'].apply(lambda x: ds_rename[x])
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x='dataset', y='scores', hue='mode',
                data=full_initial_pop, ax=ax)
    plt.legend(loc=2, title=None)
    ax.set_xlabel('')
    ax.set_ylabel('Validation Score')
    plt.tight_layout()
    plt.savefig(os.path.join(paths['macro'],
                             'plots',
                             'initial_pop_macro_micro.pdf'))


def draw_arch_count_over_threshold(datasets, rl, rs, ev, th, macro=True):
    transformed_rl, transformed_rs, transformed_ev = {}, {}, {}
    # Map for converting the name of the column contents
    map_ev = {
        'child_acc': 'EA',
        'RL': 'RL',
        'RS': 'RS'
    }
    # Number of rows and columns in the plot
    ncols = 2
    nrows = (len(datasets) // ncols)
    if len(datasets) % ncols:
        nrows += 1
    # Define figure aspects
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols * 10, nrows * 10),
                           sharey=True, squeeze=False)
    for idx, ds in enumerate(datasets):
        # Transform the results from this dataset
        # RL
        transformed_rl[ds] = transform_df_acc(rl[ds], th)
        # RS
        transformed_rs[ds] = transform_df_acc(rs[ds], th)
        # EV
        transformed_ev[ds] = ev[ds].loc[
            ev[ds]['variable'] == 'child_acc']
        transformed_ev[ds] = transform_df_acc(
            transformed_ev[ds],
            threshold=th,
            selector='value')
        temp_df = transformed_rl[ds].rename(
            columns={'val_scores': 'RL'}).drop(columns='seed')
        temp_df = temp_df.merge(transformed_rs[ds].rename(
            columns={'val_scores': 'RS'}).drop(columns='seed'), on='index')
        temp_df = temp_df.melt(id_vars='index')
        temp_df = pd.concat([temp_df, transformed_ev[ds]],
                            axis=0, ignore_index=True, sort=False)
        temp_df['variable'] = temp_df['variable'].apply(lambda x: map_ev[x])
        i = (idx // 2)
        j = (idx % ncols)
        curax = ax[i][j]
        sns.lineplot(x='index', y='value', hue='variable', style='variable',
                     data=temp_df, ax=curax)
        curax.set_ylabel('# Archs w/ Val. Score > ' + str(th))
        curax.set_xlabel('Iterations')
        curax.set_title(ds_rename[ds])
        # Remove labels from legend
        handles, labels = curax.get_legend_handles_labels()
        curax.legend(handles=handles[1:], labels=labels[1:], loc=2)
    plt.tight_layout()
    path_var = paths['macro'] if macro else paths['micro']
    plt.savefig(os.path.join(path_var, 'plots',
                             'EV+RL+RS_both' + str(th) + '.pdf'))


def draw_population_stats_chart(pop_stats_all, datasets, macro=True):
    ncols = 2
    nrows = (len(datasets) // ncols)
    if len(datasets) % ncols:
        nrows += 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols * 10, nrows * 10),
                           sharey=True, squeeze=False)
    for i, ds in enumerate(datasets):
        curax = ax[i // 2][(i % 2)]
        sns.lineplot(x='index', y='value', hue='variable',
                     style='variable',
                     data=pop_stats_all[ds], ax=curax)
        # Remove labels from legend
        handles, labels = curax.get_legend_handles_labels()
        curax.legend(handles=handles[1:], labels=labels[1:])
        curax.set_title(ds_rename[ds])
        curax.set_xlabel('Iterations')
        curax.set_ylabel('Validation Score')
    results_path = paths['macro'] if macro else paths['micro']
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'plots', 'population_stats.pdf'))


def draw_boxplot_time(rl, ea, rs, datasets, macro=True):
    full_times = pd.DataFrame(columns=['dataset', 'seed',
                                       'variable', 'value'])
    for ds in datasets:
        # Fix RL times on DF
        full_times_ds = rl[ds].groupby('seed').sum()
        full_times_ds = full_times_ds.rename(columns={'times': 'RL'})
        # Fix evolution times on DF
        ev_times = []
        for seed, times in sorted(ea[ds].items()):
            ev_times.append(float(times.strip().split('\n')[-1].split()[-1]))
        full_times_ds['EV'] = ev_times
        full_times_ds['seed'] = full_times_ds.index.values
        full_times_ds['dataset'] = ds_rename[ds]
        # Fix RS times
        rs[ds] = rs[ds].groupby('seed').sum()
        full_times_ds['RS'] = rs[ds]['times']
        full_times_ds = pd.melt(full_times_ds, id_vars=['dataset', 'seed'])
        full_times = pd.concat([full_times, full_times_ds], ignore_index=True)
    # Draw boxplot
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x='dataset', y='value', hue='variable', data=full_times)
    # Remove labels from legend
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        # handles=handles[1:],
        # labels=labels[1:],
        loc=2,
        title=None)
    ax.set_title('Execution Time Boxplot')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Time (s)')
    plt.tight_layout()
    results_path = paths['macro'] if macro else paths['micro']
    plt.savefig(os.path.join(results_path, 'plots', 'execution_time.pdf'))


def draw_best_acc_boxplot(rl, rs, ea, datasets, macro=True):
    full_best_acc = pd.DataFrame(columns=['dataset', 'seed',
                                          'variable', 'value'])
    for ds in datasets:
        # Fix RL ACC on DF
        full_best_acc_ds = rl[ds].drop(columns=['error', 'arch'])
        full_best_acc_ds = full_best_acc_ds.rename(columns={'accuracy': 'RL'})
        # Fix RS ACC
        rs_best_acc_ds = rs[ds].drop(columns=['arch']).rename(
            columns={'accuracy': 'RS'})
        # Fix Ev ACC
        grouped_by_seed = \
            ea[ds][ea[ds]['variable'] == 'Best'].groupby('seed').max()
        grouped_by_seed = grouped_by_seed.drop(columns=['index',
                                                        'variable'])
        grouped_by_seed = grouped_by_seed.reset_index()
        grouped_by_seed = grouped_by_seed.rename(columns={'value': 'EV'})
        grouped_by_seed['dataset'] = ds
        grouped_by_seed = grouped_by_seed.merge(rs_best_acc_ds,
                                                on=['dataset',
                                                    'seed'])
        # Merge!
        full_best_acc_ds = full_best_acc_ds.merge(grouped_by_seed,
                                                  on=['dataset',
                                                      'seed'])
        full_best_acc_ds = pd.melt(full_best_acc_ds,
                                   id_vars=['dataset',
                                            'seed'])
        full_best_acc = pd.concat([full_best_acc, full_best_acc_ds],
                                  axis=0, ignore_index=True)
    # Draw boxplot
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x='dataset', y='value', hue='variable', data=full_best_acc)
    ax.set_title('Best Accuracy Boxplot')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Method')
    plt.tight_layout()
    results_path = paths['macro'] if macro else paths['micro']
    plt.savefig(os.path.join(results_path, 'plots', 'best_accuracy.pdf'))


def get_oom_dataframe(datasets, rl, rs, ea):
    full_oom = pd.DataFrame(columns=['seed', 'dataset', 'variable', 'value'])
    for ds in datasets:
        merged = rl[ds].merge(rs[ds], on='seed')
        merged = merged.merge(ea[ds], on='seed')
        merged = merged.melt(id_vars=['seed'])
        merged['value'] = merged['value'] / 1000
        merged['dataset'] = ds_rename[ds]
        full_oom = pd.concat([full_oom, merged],
                             axis=0, ignore_index=True, sort=False)
    full_oom['value'] = full_oom['value'].astype(np.float64)
    return full_oom


def draw_oom_boxplot(datasets, rl, rs, ea, macro=True):
    full_oom = get_oom_dataframe(datasets, rl, rs, ea)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x='dataset', y='value', hue='variable',
                data=full_oom, ax=ax)
    plt.legend(loc=2, title=None)
    ax.set_xlabel('')
    ax.set_ylabel("# of Archs. that OOM'd the GPU")
    plt.tight_layout()
    results_path = paths['macro'] if macro else paths['micro']
    plt.savefig(os.path.join(results_path,
                             'plots',
                             'oom_plot.pdf'))
