import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plots = [   {"fn": 'validation\\20221007_120006.csv', "title": "RootNet-65"},
            {"fn": 'validation\\20221007_105128.csv', "title": "RootNet-129"},
            {"fn": 'validation\\20221007_013703.csv', "title": "RootNet-257"}]

for plot in plots:
    data = pd.read_csv(plot['fn'], sep=';', index_col=0)

    data_to_plot = data.loc[["accuracy", "precision", "recall", "F1"]]
    data_to_plot = data_to_plot.drop(columns=["10"]).T
    # print(data_to_plot)
    sns.set_theme()
    rel = sns.relplot(data=data_to_plot, kind='line', lw=3)
    rel.set_xticklabels([str(x/10) for x in range(1,10)])
    plt.xlabel(r'$\sigma$ Threshold', fontsize=16)
    plt.ylabel('Percentage', fontsize=16)
    plt.title(plot['title'], fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0.7, 1.0))
    plt.savefig(plot['title']+".png", format="png", bbox_inches='tight')
