import os
import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

paths = {
    'macro': 'macro_results/',
    'micro': 'micro_results/',
}


def load_evolution_results(seed=123, database='', macro=True, ev_type=""):
    # Load offspring results
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    ev_results = pd.read_csv(os.path.join(results_path, 'seed_' + str(seed),
                                          'parent_child_' + database +
                                          "_" + ev_type),
                             names=['parent', 'child'])
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
    return ev_results, ev_time, initial_population_scores, population_stats


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
    return df, pd.DataFrame(np.array(times), columns=['times'])


def draw_lineplots(ev_results, rl_results, all_seeds=False, macro=True):
    assert(len(ev_results.keys()) == len(rl_results.keys()))
    nrows = len(ev_results.keys())
    fig, ax = plt.subplots(nrows, 2, figsize=(18.75, nrows * 6), sharey=True)
    for i, db in enumerate(ev_results.keys()):
        sns.lineplot(x='index', y='value', hue='variable',
                     data=ev_results[db], ax=ax[i][0])
        sns.lineplot(x='index', y='val_scores',
                     data=rl_results[db], ax=ax[i][1])
        ax[i][0].set_title('Evolution: ' + db.capitalize())
        ax[i][1].set_title('Reinforcement Learning: ' + db.capitalize())
        ax[i][0].set_xlabel('epoch')
        ax[i][1].set_xlabel('epoch')
        ax[i][0].set_ylabel('Validation Score')
        ax[i][1].set_ylabel('Validation Score')
    plt.tight_layout()
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    if not all_seeds:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_per_epoch_' + '.pdf'))
    else:
        plt.savefig(os.path.join(results_path, 'plots',
                                 'validation_score_per_epoch_all_seeds.pdf'))


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


def load_all_results(database='', macro=True, ev_type=""):
    ev_results_all = pd.DataFrame(columns=['index', 'variable',
                                           'value', 'seed'])
    pop_stats_all = pd.DataFrame(columns=['index', 'variable',
                                          'value', 'seed'])
    initial_pop_all = pd.DataFrame(columns=['scores', 'seed'])
    rl_results_all = pd.DataFrame(columns=['index', 'val_scores', 'seed'])
    rl_times_all = pd.DataFrame(columns=['times', 'seed'])
    ev_times_all = {}
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    for subdir in os.listdir(results_path):
        if 'seed' in subdir:
            seed = subdir.split('_')[-1]
            # LOAD EVOLUTION RESULTS
            ev_results, ev_times, initial_pop_scores, population_stats = \
                load_evolution_results(seed=seed, database=database,
                                       macro=macro, ev_type=ev_type)
            ev_results['seed'] = seed
            initial_pop_scores['seed'] = seed
            population_stats['seed'] = seed
            ev_results_all = pd.concat([ev_results_all, ev_results],
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
            rl_results, rl_times = \
                load_RL_results(seed=seed, database=database, macro=macro)
            rl_results['seed'] = seed
            rl_results_all = pd.concat([rl_results_all, rl_results],
                                       axis=0,
                                       ignore_index=True, sort=False)
            rl_times['seed'] = seed
            rl_times_all = pd.concat([rl_times_all, rl_times],
                                     axis=0,
                                     ignore_index=True, sort=False)
    return ev_results_all, initial_pop_all, ev_times_all, \
        rl_results_all, rl_times_all, pop_stats_all


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


def draw_population_stats_chart(pop_stats_all, macro=True):
    nrow = ceil(len(pop_stats_all.keys()) / 2)
    print('nrows: ', nrow)
    fig, ax = plt.subplots(nrows=nrow, ncols=2, figsize=(18.75, nrow * 6),
                           sharey=True, squeeze=False)
    for i, db in enumerate(pop_stats_all.keys()):
        sns.lineplot(x='index', y='value', hue='variable',
                     data=pop_stats_all[db], ax=ax[i // 2][(i % 2)])
        ax[i // 2][(i % 2)].set_title('Population Stats: ' + db.capitalize())
        ax[i // 2][(i % 2)].set_xlabel('Epoch')
        ax[i // 2][(i % 2)].set_ylabel('Validation Score')
    plt.tight_layout()
    if macro:
        results_path = paths['macro']
    else:
        results_path = paths['micro']
    plt.savefig(os.path.join(results_path, 'plots', 'population_stats.pdf'))


def draw_boxplot_time(rl_times_all, ev_times_all, macro=True):
    assert(len(ev_times_all.keys()) == len(rl_times_all.keys()))
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
        full_times_db['evolution'] = ev_times
        full_times_db['seed'] = full_times_db.index.values
        full_times_db['dataset'] = db
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

