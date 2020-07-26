import os
import pickle
import numpy as np
import pandas as pd
from math import ceil
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


def load_evolution_results(seed=123, database='', macro=True, ev_type=""):
    # Load offspring results
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    ev_results = pd.read_csv(os.path.join(results_path, 'seed_' + str(seed),
                                          'parent_child_' + database +
                                          "_" + ev_type),
                             names=['parent_arch', 'parent_acc',
                                    'child_arch', 'child_acc'],
                             header=None,
                             sep='>')
    arch_results = ev_results[['parent_arch', 'child_arch']].copy()
    arch_results['index'] = arch_results.index
    arch_results = pd.melt(arch_results, id_vars=['index'])
    ev_results = ev_results[['parent_acc', 'child_acc']]
    # ev_results['diff_child_parent'] = ev_results['child_acc'] - ev_results['parent_acc']
    ev_results['index'] = ev_results.index
    ev_results = pd.melt(ev_results, id_vars=['index'])
    initial_population_scores = \
        pd.read_csv(os.path.join(results_path,
                                 'seed_' + str(seed),
                                 'initial_population_' +
                                 database + "_" + ev_type),
                    header=None,
                    names=['scores'])
    # Load timing results
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    ev_time = open(os.path.join(results_path,
                                'seed_' + str(seed),
                                'time_' + database +
                                "_" + ev_type), 'r').read()
    # Load population stats
    population_stats = pd.read_csv(os.path.join(results_path,
                                                'seed_' + str(seed),
                                                'population_stats_' +
                                                database + "_" + ev_type))
    population_stats['index'] = population_stats.index
    population_stats = pd.melt(population_stats, id_vars=['index'])
    return (ev_results,
            ev_time,
            initial_population_scores,
            population_stats,
            arch_results)


def load_RL_results(seed=123, database='', macro=True):
    indexes = []
    val_scores = []
    store = False
    times = []
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    with open(os.path.join(results_path, 'seed_' + str(seed),
                           'results_RL_' + database), 'r') as fp:
        for index, line in enumerate(fp):
            if 'took' in line:
                times.append(float(line.split()[-1]))
            if 'val_score' in line and store:
                val_scores.append(float(line.split(',')[0].split(':')[1]))
            if 'training controller' in line:
                indexes.append(index)
                if 'over' in line:
                    store = False
                else:
                    store = True
    df = pd.DataFrame()
    df['index'] = np.arange(len(val_scores))
    df['val_scores'] = np.array(val_scores)
    rl_best_acc = pd.read_csv(os.path.join(results_path, 'seed_' + str(seed),
                                           'best_acc_RL_' + database),
                              names=['arch',
                                     'accuracy',
                                     'error'],
                              sep='>')
    return (df, pd.DataFrame(np.array(times), columns=['times']),
            rl_best_acc)


def load_RS_results(seed=123, database='', macro=True):
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    df = pd.read_csv(os.path.join(results_path, 'seed_' + str(seed),
                                  'val_scores_RS_' + database),
                     names=['val_scores'])
    df['index'] = df.index
    execution_time = open(os.path.join(results_path, 'seed_' + str(seed),
                                       'time_RS_' + database)).read()
    best_accuracy_path = os.path.join(results_path, 'seed_' + str(seed),
                                      'best_acc_RS_' + database)
    best_accuracy = pd.read_csv(best_accuracy_path,
                                names=['arch', 'accuracy'],
                                sep='>')
    return (df,
            pd.DataFrame(np.array([float(execution_time)]),
                         columns=['times']),
            best_accuracy)


def load_all_results(database='', macro=True, ev_type=""):
    # EV
    ev_results_all = pd.DataFrame(columns=['index', 'variable',
                                           'value', 'seed'])
    pop_stats_all = pd.DataFrame(columns=['index', 'variable',
                                          'value', 'seed'])
    initial_pop_all = pd.DataFrame(columns=['scores', 'seed'])
    ev_arch_all = pd.DataFrame(columns=['index', 'variable',
                                        'value', 'seed'])
    # RL
    rl_results_all = pd.DataFrame(columns=['index', 'val_scores', 'seed'])
    rl_times_all = pd.DataFrame(columns=['times', 'seed'])
    rl_best_acc_all = pd.DataFrame(columns=['accuracy', 'error',
                                            'seed', 'dataset'])
    # RS
    rs_results_all = pd.DataFrame(columns=['index', 'val_scores', 'seed'])
    rs_times_all = pd.DataFrame(columns=['times', 'seed'])
    rs_best_acc_all = pd.DataFrame(columns=['accuracy', 'seed', 'dataset'])
    ev_times_all = {}
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    for subdir in os.listdir(results_path):
        if 'seed' in subdir:
            seed = subdir.split('_')[-1]
            # LOAD EVOLUTION RESULTS
            ev_results, ev_times, initial_pop_scores, population_stats, ev_arch = \
                load_evolution_results(seed=seed, database=database,
                                       macro=macro, ev_type=ev_type)
            ev_results['seed'] = seed
            ev_arch['seed'] = seed
            initial_pop_scores['seed'] = seed
            population_stats['seed'] = seed
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
            ev_times_all[seed] = ev_times
            # LOAD RL RESULTS
            rl_results, rl_times, rl_best_acc = \
                load_RL_results(seed=seed, database=database, macro=macro)
            rl_results['seed'] = seed
            rl_results_all = pd.concat([rl_results_all, rl_results],
                                       axis=0,
                                       ignore_index=True, sort=False)
            rl_times['seed'] = seed
            rl_times_all = pd.concat([rl_times_all, rl_times],
                                     axis=0,
                                     ignore_index=True, sort=False)
            rl_best_acc['seed'] = seed
            rl_best_acc['dataset'] = database
            rl_best_acc_all = pd.concat([rl_best_acc_all, rl_best_acc],
                                        axis=0,
                                        ignore_index=True,
                                        sort=False)
            # LOAD RS RESULTS
            rs_results, rs_times, rs_best_acc = \
                load_RS_results(seed=seed, database=database, macro=macro)
            rs_results['seed'] = seed
            rs_results_all = pd.concat([rs_results_all, rs_results],
                                       axis=0,
                                       ignore_index=True, sort=False)
            rs_times['seed'] = seed
            rs_times_all = pd.concat([rs_times_all, rs_times],
                                     axis=0,
                                     ignore_index=True, sort=False)
            rs_best_acc['seed'] = seed
            rs_best_acc['dataset'] = database
            rs_best_acc_all = pd.concat([rs_best_acc_all, rs_best_acc],
                                        axis=0,
                                        ignore_index=True, sort=False)
    return (ev_results_all, initial_pop_all, ev_times_all, pop_stats_all, ev_arch_all,
            rl_results_all, rl_times_all, rl_best_acc_all,
            rs_results_all, rs_times_all, rs_best_acc_all)


def draw_lineplots(ev_results, rl_results, rs_results, all_seeds=False, macro=True):
    assert(len(ev_results.keys()) == len(rl_results.keys()))
    assert(len(rs_results.keys()) == len(rl_results.keys()))
    nrows = len(ev_results.keys())
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols* 10, nrows * 10), sharey=True)
    for i, db in enumerate(ev_results.keys()):
        sns.lineplot(x='index', y='value', hue='variable',
                     data=ev_results[db][ev_results[db]['index'] < 400], ax=ax[i][0])
        sns.lineplot(x='index', y='val_scores',
                     data=rl_results[db][rl_results[db]['index'] < 400], ax=ax[i][1])
        sns.lineplot(x='index', y='val_scores',
                     data=rs_results[db][rs_results[db]['index'] < 400], ax=ax[i][2])
        ax[i][0].legend(loc=2, title=None).remove()
        ax[i][0].set_title('Evolution: ' + db.capitalize())
        ax[i][1].set_title('Reinforcement Learning: ' + db.capitalize())
        ax[i][2].set_title('Random Search: ' + db.capitalize())
        ax[i][0].set_xlabel('Iteration')
        ax[i][1].set_xlabel('Iteration')
        ax[i][2].set_xlabel('Iteration')
        ax[i][0].set_ylabel('Validation Score')
        ax[i][1].set_ylabel('Validation Score')
        ax[i][2].set_ylabel('Validation Score')
    results_path = paths['macro'] if macro else paths['micro']
    plt.tight_layout()
    if not all_seeds:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_per_epoch_' + '.pdf'))
    else:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_per_epoch_200_all_seeds.pdf'))
        
def draw_cummmax(datasets, pop_stats_all, rl_results_all, rs_results_all, macro=True):
    ncols = 2
    nrows = (len(datasets) // ncols) if not (len(datasets) % ncols) else (len(datasets) // ncols) + 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols* 10, nrows * 10),
                           sharey=True, squeeze=False)
    for idx, ds in enumerate(datasets):
        cummax_df = pop_stats_all[ds][pop_stats_all[ds]['variable'] == 'Best']
        cummax_df.loc[:, 'variable'] = 'EA'
        for seed in rl_results_all[ds]['seed'].unique():
            seed_rl_cummax = rl_results_all[ds].loc[rl_results_all[ds]['seed'] == seed].cummax().rename(columns={'val_scores': 'value'})
            seed_rl_cummax.loc[:, 'variable'] = 'RL'
            cummax_df = pd.concat(
                [cummax_df, seed_rl_cummax], axis=0, sort=False,
                ignore_index=True)
        for seed in rs_results_all[ds]['seed'].unique():
            seed_rs_cummax = rs_results_all[ds].loc[rs_results_all[ds]['seed'] == seed].cummax().rename(columns={'val_scores': 'value'})
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
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    if not all_seeds:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_boxplot.pdf'))
    else:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_boxplot_all_seeds.pdf'))


def draw_initial_pop_boxplot(initial_pop_all, macro=True):
    full_initial_pop = pd.DataFrame(columns=['scores', 'seed', 'dataset'])
    for db in initial_pop_all.keys():
        initial_pop_all[db]['dataset'] = db
        full_initial_pop = pd.concat([full_initial_pop, initial_pop_all[db]],
                                     ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='dataset', y='scores', data=full_initial_pop, ax=ax)
    ax.set_title('Initial Population Boxplot')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Validation Score')
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    plt.savefig(os.path.join(results_path, 'plots',
                             'initial_population_boxplot_.pdf'))


def draw_arch_count_over_threshold(datasets, rl_results_all, rs_results_all, ev_results_all, threshold, macro=True):
    transformed_rl, transformed_rs, transformed_ev = {}, {}, {}
    # Map for converting the name of the column contents
    map_ev = {
        'child_acc': 'EA',
        'RL': 'RL',
        'RS': 'RS'
    }
    # Number of rows and columns in the plot
    ncols = 2
    nrows = (len(datasets) // ncols) + 1
    # Define figure aspects
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols * 10, nrows * 10),
                           sharey=True, squeeze=False)
    print('ax.shape: ', ax.shape)
    for idx, ds in enumerate(datasets):
        # Transform the results from this dataset
        transformed_rl[ds] = transform_df_acc(rl_results_all[ds], threshold)  # RL
        transformed_rs[ds] = transform_df_acc(rs_results_all[ds], threshold)  # RS
        transformed_ev[ds] = transform_df_acc(                                # EV
            ev_results_all[ds].loc[ev_results_all[ds]['variable'] ==
                                   'child_acc'],
            threshold=threshold,
            selector='value')
        temp_df = transformed_rl[ds].rename(columns={'val_scores':
                                                     'RL'}).drop(columns='seed')
        temp_df = temp_df.merge(transformed_rs[ds].rename(columns={'val_scores':
                                                                   'RS'}).drop(columns='seed'),
                                on='index')
        temp_df = temp_df.melt(id_vars='index')
        temp_df = pd.concat([temp_df,
                             transformed_ev[ds][transformed_ev[ds]['variable'] != 'parent_acc']],
                            axis=0,
                            ignore_index=True, sort=False)
        temp_df['variable'] = temp_df['variable'].apply(lambda x: map_ev[x])
        i = (idx // 2)
        j = (idx % ncols)
        curax = ax[i][j]
        sns.lineplot(x='index', y='value', hue='variable', style='variable',
                     data=temp_df, ax=curax)
        curax.set_ylabel('# Archs w/ Val. Score > ' + str(threshold))
        curax.set_xlabel('Iterations')
        curax.set_title(ds_rename[ds])
        # Remove labels from legend
        handles, labels = curax.get_legend_handles_labels()
        curax.legend(handles=handles[1:], labels=labels[1:], loc=2)
    plt.tight_layout()
    path_var = paths['macro'] if macro else paths['micro']
    plt.savefig(os.path.join(path_var, 'plots', 'EV+RL+RS_both' + str(threshold) + '.pdf'))


def draw_population_stats_chart(pop_stats_all, datasets, macro=True):
    ncols = 2
    nrows = (len(datasets) // ncols)
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


def draw_boxplot_time(rl_times_all, ev_times_all, rs_times_all, macro=True):
    assert(len(ev_times_all.keys()) == len(rl_times_all.keys()))
    assert(len(rs_times_all.keys()) == len(rl_times_all.keys()))
    full_times = pd.DataFrame(columns=['dataset', 'seed',
                                       'variable', 'value'])
    for db in ev_times_all.keys():
        # Fix RL times on DF
        full_times_db = rl_times_all[db].groupby('seed').sum()
        full_times_db = full_times_db.rename(columns={'times': 'RL'})
        # Fix evolution times on DF
        ev_times = []
        for seed, times in sorted(ev_times_all[db].items()):
            ev_times.append(float(times.strip().split('\n')[-1].split()[-1]))
        full_times_db['EV'] = ev_times
        full_times_db['seed'] = full_times_db.index.values
        full_times_db['dataset'] = db
        # Fix RS times
        rs_times_all[db] = rs_times_all[db].groupby('seed').sum()
        full_times_db['RS'] = rs_times_all[db]['times']
        full_times_db = pd.melt(full_times_db, id_vars=['dataset', 'seed'])
        full_times = pd.concat([full_times, full_times_db], ignore_index=True)
    # Draw boxplot
    fig, ax = plt.subplots(figsize=(15, 7.5))
    sns.boxplot(x='dataset', y='value', hue='variable', data=full_times)
    ax.set_title('Execution Time Boxplot')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Time (s)')
    plt.tight_layout()
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    plt.savefig(os.path.join(results_path, 'plots', 'execution_time.pdf'))


def draw_best_acc_boxplot(rl_best_acc, rs_best_acc, pop_stats_all, datasets, macro=True):
    full_best_acc = pd.DataFrame(columns=['dataset', 'seed',
                                          'variable', 'value'])
    for db in datasets:
        # Fix RL ACC on DF
        full_best_acc_db = rl_best_acc[db].drop(columns=['error', 'arch'])
        full_best_acc_db = full_best_acc_db.rename(columns={'accuracy': 'RL'})
        # Fix RS ACC
        rs_best_acc_db = rs_best_acc[db].drop(columns=['arch']).rename(columns={'accuracy':
                                                                                 'RS'})
        # Fix Ev ACC
        grouped_by_seed = \
            pop_stats_all[db][pop_stats_all[db]['variable'] ==
                              'Best'].groupby('seed').max()
        grouped_by_seed = grouped_by_seed.drop(columns=['index',
                                                        'variable'])
        grouped_by_seed = grouped_by_seed.reset_index()
        grouped_by_seed = grouped_by_seed.rename(columns={'value': 'EV'})
        grouped_by_seed['dataset'] = db
        grouped_by_seed = grouped_by_seed.merge(rs_best_acc_db,
                                                on=['dataset',
                                                    'seed'])
        # Merge!
        full_best_acc_db = full_best_acc_db.merge(grouped_by_seed,
                                                  on=['dataset',
                                                      'seed'])
        full_best_acc_db = pd.melt(full_best_acc_db,
                                   id_vars=['dataset',
                                            'seed'])
        full_best_acc = pd.concat([full_best_acc, full_best_acc_db],
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
