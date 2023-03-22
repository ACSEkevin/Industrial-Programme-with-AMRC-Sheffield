from itpma3_utils import *

"""
This is a MA3 data analysis, visualization and preprocessing script, 
all the auxiliary classes and functions can be found in package itpma3_utils
"""


def load_process(plot=False):
    print('Hello MA3')
    data = load_data()
    attribute = data.columns.tolist()[1:]
    print(attribute)

    # check missing value
    # print(data[data.isnull().T.any()])

    # correlation - Pearson/Spearman coefficient
    pearson, spearman = dict(), dict()
    target = data['target']

    # calculate and store correlation coefficients
    for att in attribute:
        pearson[att] = data[att].corr(target, method='pearson')
        spearman[att] = data[att].corr(target, method='spearman')

    # correlations visualization
    if plot is True:
        plot_correlation(pearson, spearman)
        plot_heatmap(data)

    # standardization, normalizer is not recommended
    normed_data = DataScaler(method='std', split_target=True).fit(data)
    # print(normed_data.shape)
    # print(DataScaler().available_methods)

    # principal component analysis
    pca = PrincipalComponentAnalysisWrapper(n_components=normed_data.shape[-1] - 1)
    pca.fit(normed_data[:, 1:])
    # print(pca.explained_variance_ratio)

    if plot is True:
        pca.visualize_variance_ratio()
        pca.plot_data_loss()

    feature = normed_data[:, 1:]
    target = normed_data[:, 0]
    print(feature.shape, target.shape)

    return DatasetSplitWrapper(test_size=0.2).split(feature, target)


if __name__ == '__main__':
    print('Hello MA3')
    # main()
