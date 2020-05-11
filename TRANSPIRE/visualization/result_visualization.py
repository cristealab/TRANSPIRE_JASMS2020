import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_GO_enrichment_results(GO_df, orient = 'vertical', figsize = (8.5, 11)):

    if GO_df.shape[0] > 100:
        print('WARNING: Dataframe is relatively large... consider paring down GO terms to a more resonable number for efficient visualization.')

    fig, ax = plt.subplots(figsize=figsize)

    for_sns = GO_df.copy()
    for_sns['ratio_in_study'] = for_sns['ratio_in_study'].str.split('/', expand=True).astype(int).apply(lambda x: (x[0])/x[1]*100, axis=1)
    for_sns['ratio_in_pop'] = for_sns['ratio_in_pop'].str.split('/', expand=True).astype(int).apply(lambda x: (x[0])/x[1]*100, axis=1)
    for_sns['fold_enrichment'] = for_sns['ratio_in_study']/for_sns['ratio_in_pop']

    if orient == 'horizontal':
        
        sns.barplot(data=for_sns, x='fold_enrichment', y='name', hue='NS', dodge=False, ax= ax)
        ax.set_xlabel('fold enrichment')
        ax.set_ylim((len(for_sns)-0.5, -.5))
        ax.set_ylabel('')

        ax2 = ax.twiny()
        ax2.plot(for_sns['ratio_in_study'], ax.get_yticks(), ls='--', marker='o', c='grey')
        ax2.set_xlabel('percent of study population')

    elif orient == 'vertical':

        sns.barplot(data=for_sns, x='name', y='fold_enrichment', hue='NS', dodge=False, ax= ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel('fold enrichment')
        ax.set_xlabel('')

        ax2 = ax.twinx()
        ax2.plot(ax.get_xticks(), for_sns['ratio_in_study'], ls='--', marker='o', c='grey')
        ax2.set_ylabel('percent of study population')

    for patch in ax.patches:
        patch.set_edgecolor('white')
        patch.set_linewidth(.01)
        patch.set_alpha(0.5)

    ax.legend(bbox_to_anchor = (1.1, 1, 0, 0), loc=2, frameon=False)

    return fig