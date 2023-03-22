"""
ITP Group MA3
This file is to store some commonly used functions and classes
"""

import math
import pandas as pd
import seaborn as sns
import numpy as np
from keras.layers import Layer
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


def load_data(file_path=None):
    file_path = './data/ma3.csv' if file_path is None else file_path
    file = pd.read_csv(file_path)
    return file


def plot_correlation(pearson, spearman):
    plt.figure()
    plt.barh(range(len(pearson.keys())), list(pearson.values())[:: -1], color='deepskyblue')
    plt.title('Pearson correlation coefficient')
    plt.yticks(range(len(pearson.keys())), list(spearman.keys())[:: -1])
    plt.xticks(())
    plt.show()
    # plt.close()

    plt.figure()
    plt.barh(range(len(spearman.keys())), list(spearman.values())[:: -1], color='deeppink')
    plt.yticks(range(len(pearson.keys())), list(spearman.keys())[:: -1])
    plt.title('Spearman correlation coefficient')
    plt.xticks(())
    plt.show()
    plt.close()


def plot_heatmap(data):
    corr = data.corr(method='pearson')
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, cmap='rainbow')
    plt.title('feature heatmap')
    plt.show()
    plt.close()


def plot_model_curve(history):
    train_acc, train_loss = history.history['binary_accuracy'], history.history['loss']
    val_acc, val_loss = history.history['val_binary_accuracy'], history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, color='purple')
    plt.plot(val_acc, color='red')
    plt.xlabel('$epochs$')
    plt.ylabel('$accuracy$')
    plt.legend(['train_accuracy', 'val_accuracy'])

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, color='deeppink')
    plt.plot(val_loss, color='deepskyblue')
    plt.xlabel('$epochs$')
    plt.ylabel('$loss$')
    plt.legend(['train_loss', 'val_loss'])

    plt.show()
    plt.close()


def plot_confusion_matrix(y_val, y_hat_integer, cmap="Wistia"):
    cm = confusion_matrix(y_val, y_hat_integer, normalize="pred")
    plt.figure(1).set_figwidth(10)
    sns.heatmap(cm, annot=True, fmt=".2%", cmap=cmap, )
    plt.title("Confusion Matrix", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.show()


def plot_roc_auc_curve(y_val, y_hat, y_train=None, y_hat_train=None, colors=None, train_linestyle='-'):
    if colors is None:
        colors = ['dodgerblue', 'mediumvioletred']
    elif not hasattr(colors, '__iter__'):
        colors = [colors, colors]

    plt.plot(roc_curve(y_val, y_hat)[0], roc_curve(y_val, y_hat)[1], color=colors[0])
    if y_train.any() and y_hat_train.any():
        plt.plot(roc_curve(y_train, y_hat_train)[0], roc_curve(y_train, y_hat_train)[1],
                 color=colors[-1], linestyle=train_linestyle)
    plt.legend(['validation', 'training'])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC-AUC curve')
    plt.xticks(np.linspace(0, 1, 3))
    plt.yticks(np.linspace(0, 1, 3))
    plt.grid(visible=.01)
    plt.show()





class DatasetSplitWrapper:
    def __init__(self, test_size=0.2, shuffle=True, stratify=None):
        self.test_size = test_size
        self.shuffle = shuffle
        self.stratify = stratify

    def split(self, feature, target):
        return train_test_split(feature, target,
                                test_size=self.test_size,
                                shuffle=self.shuffle,
                                stratify=self.stratify)


class DataScaler:
    def __init__(self, method='std', feature_range=(0, 1), target=None, split_target=False):
        """
        Data scaler class
        :param method: /
        :param target: if target has been separated from the dataset,
                        the target will be combined back to the data when the arg is defined
        :param split_target: /
        """
        self.target = target
        self.method = method
        self.feature_range = feature_range
        self.split_target = split_target
        self.choices_map = {'std': StandardScaler(), 'norm': Normalizer(),
                            'minmax': MinMaxScaler(feature_range=feature_range), 'mabs': MaxAbsScaler()}
        assert self.method in self.choices_map.keys()
        self.scaler = self.choices_map[self.method]

    def fit(self, data):
        if self.split_target is True:
            data, self.target = data.drop(['target'], axis=1), data['target']
        normed = self.scaler.fit_transform(data)
        normed_data = np.hstack((np.array([self.target]).T, normed)) \
            if self.split_target is True else normed

        return normed_data

    @property
    def available_methods(self):
        return self.choices_map.keys()


class PrincipalComponentAnalysisWrapper:
    def __init__(self, n_components, random_state=None):
        self.n_components = n_components
        self.pca = PCA(n_components, svd_solver='auto', random_state=random_state)

    def fit(self, data):
        return self.pca.fit_transform(data)

    def visualize_variance_ratio(self):
        plt.figure(figsize=(8, 4))
        plt.bar(range(self.pca.explained_variance_ratio_.shape[-1]),
                self.pca.explained_variance_ratio_, color='deeppink')
        plt.xlabel('$principal components$')
        plt.xticks(range(self.pca.explained_variance_ratio_.shape[-1]))
        plt.ylabel('$cumulative explained variance ratio$')
        plt.title('PCA dimension vs. data loss')
        plt.show()
        plt.close()

    def plot_data_loss(self):
        cum_ratio = np.array([self.pca.explained_variance_ratio_]).cumsum()
        plt.figure(figsize=(8, 4))
        plt.plot(range(cum_ratio.shape[-1]), cum_ratio, color='deeppink')
        plt.plot(range(self.pca.explained_variance_ratio_.shape[-1]),
                 self.pca.explained_variance_ratio_, color='red')
        plt.xlabel('$principal components$')
        plt.xticks(range(cum_ratio.shape[-1]))
        plt.ylabel('$cumulative explained variance ratio$')
        plt.legend(['cumulative vr', 'vr'])
        plt.title('PCA dimension vs. data loss')
        plt.grid(visible=.01)
        plt.show()
        plt.close()

    def inverse_transform(self, data):
        return self.pca.inverse_transform(data)

    @property
    def explained_variance(self):
        return self.pca.explained_variance_

    @property
    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_

    @property
    def components(self):
        return self.pca.components_

    @property
    def mean(self):
        return self.pca.mean_

    @property
    def features_in(self):
        return self.pca.n_features_in_

    @property
    def samples(self):
        return self.pca.n_samples_


class LearningRateSchedulers:
    def __init__(self, epochs, init_lr=0.001, step=2):
        self.init_lr = init_lr
        self.epochs = epochs
        self.step = step

    def linear_scheduler(self, epoch, decay_rate=0.01):
        if epoch % self.step == 0:
            rate = 1 / (1 + decay_rate * epoch)
            self.init_lr = self.init_lr * rate
        return self.init_lr

    def exponential_scheduler(self, epoch, drop_rate=0.5):
        if epoch % self.step == 0:
            rate = np.power(drop_rate, np.floor((1 + epoch) / self.step))
            self.init_lr = self.init_lr * rate
        return self.init_lr

    def cosine_scheduler(self, epoch, end_lr=0.001):
        if epoch % self.step == 0:
            rate = ((1 + math.cos(epoch * math.pi / self.epochs)) / 2) * (1 - end_lr) + end_lr  # cosine
            self.init_lr = self.init_lr * rate
        return self.init_lr


class Accuracy(Layer):
    def __init__(self,
                 normalize=True,
                 sample_weight=None):
        super().__init__()
        self.normalize = normalize
        self.sample_weight = sample_weight

    def call(self, y_ture=None, y_hat=None, *args, **kwargs):
        return accuracy_score(y_ture, y_hat,
                              normalize=self.normalize,
                              sample_weight=self.sample_weight)


class Precision(Layer):
    def __init__(self, labels=None,
                 pos_label=1,
                 average='macro',
                 sample_weight=None,
                 zero_division=0):
        super().__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def call(self, y_ture=None, y_hat=None, *args, **kwargs):
        return precision_score(y_ture, y_hat,
                               labels=self.labels,
                               pos_label=self.pos_label,
                               average=self.average,
                               sample_weight=self.sample_weight,
                               zero_division=self.zero_division)


class Recall(Layer):
    def __init__(self,
                 labels=None,
                 pos_label=1,
                 average='macro',
                 sample_weight=None,
                 zero_division=0):
        super().__init__()
        self.fit_result = None
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def call(self, y_ture=None, y_hat=None, *args, **kwargs):
        self.fit_result = recall_score(y_ture, y_hat,
                                       labels=self.labels,
                                       pos_label=self.pos_label,
                                       average=self.average,
                                       sample_weight=self.sample_weight,
                                       zero_division=self.zero_division)
        return self.fit_result


class F1Score(Layer):
    def __init__(self,
                 labels=None,
                 pos_label=1,
                 average='macro',
                 sample_weight=None,
                 zero_division=0):
        super().__init__()
        self.fit_result = None
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def call(self, y_ture=None, y_hat=None, *args, **kwargs):
        return f1_score(y_ture, y_hat,
                        labels=self.labels,
                        pos_label=self.pos_label,
                        average=self.average,
                        sample_weight=self.sample_weight,
                        zero_division=self.zero_division)


class AreaUnderCurve(Layer):
    def __init__(self,
                 pos_label=1,
                 sample_weight=None,
                 drop_intermediate=True,
                 thresholds=False,
                 direct_cal=True
                 ):
        super().__init__()
        self.pos_label = pos_label
        self.sample_weight = sample_weight
        self.drop_intermediate = drop_intermediate
        self.thresholds = thresholds
        self.direct_cal = direct_cal

    def call(self, y_ture=None, y_hat_row=None, *args, **kwargs):
        if not self.direct_cal:
            fpr, tpr, thresholds = roc_curve(y_ture, y_hat_row,
                                             pos_label=self.pos_label,
                                             sample_weight=self.sample_weight,
                                             drop_intermediate=self.drop_intermediate)
            if self.thresholds is True:
                return auc(fpr, tpr), thresholds
            else:
                return auc(fpr, tpr)
        else:
            return roc_auc_score(y_ture, y_hat_row, sample_weight=self.sample_weight)


class TotalMeanMetricWrapper(Layer):
    def __init__(self,
                 model_name,
                 argmax=False,
                 normalize=True,
                 labels=None,
                 pos_label=1,
                 average='macro',
                 sample_weight=None,
                 zero_division=0,
                 drop_intermediate=True,
                 auc_include=True,
                 direct_cal=True,
                 auc_thresholds=False):

        super(TotalMeanMetricWrapper, self).__init__()
        self.model_name = model_name
        self.argmax = argmax
        self.average = average
        self.auc_include = auc_include
        self.result_dic = dict()
        self.metric_list = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
        assert self.argmax is False or auc_include is False

        self.accuracy = Accuracy(normalize=normalize, sample_weight=sample_weight)
        self.precision = Precision(labels=labels, pos_label=pos_label, average=average,
                                   sample_weight=sample_weight, zero_division=zero_division)
        self.recall = Recall(labels=labels, pos_label=pos_label, average=average,
                             sample_weight=sample_weight, zero_division=zero_division)
        self.f_score = F1Score(labels=labels, pos_label=pos_label, average=average,
                               sample_weight=sample_weight, zero_division=zero_division)
        self.auc = AreaUnderCurve(pos_label=pos_label, sample_weight=sample_weight, direct_cal=direct_cal,
                                  drop_intermediate=drop_intermediate, thresholds=auc_thresholds)
        self.metrics_wrapper = [self.accuracy, self.precision, self.recall, self.f_score, self.auc]

    def call(self, y_ture=None, y_hat=None, *args, **kwargs):
        if self.argmax is False:
            y_pred = np.floor(np.array(y_hat) + .5) if self.average == 'binary' else np.argmax(y_hat, axis=-1)
        else:
            y_pred = y_hat
        for index, metric in enumerate(self.metrics_wrapper):
            if index == len(self.metrics_wrapper) - 1:
                if self.auc_include is True:
                    if self.average != 'binary':
                        y_hat = np.max(y_hat, axis=-1)
                    self.result_dic[self.metric_list[index]] = metric(y_ture, y_hat)
                else:
                    self.result_dic[self.metric_list[index]] = 'score not included'
            else:
                self.result_dic[self.metric_list[index]] = metric(y_ture, y_pred)

        x = PrettyTable(field_names=self.result_dic.keys())
        x.title = self.model_name
        x.add_row(self.result_dic.values())
        print(x)

        return self.result_dic

    @property
    def result_scores(self):
        return self.result_dic

    @property
    def engaged_metrics(self):
        return self.metric_list
