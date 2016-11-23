# helpers for hw4 (applied machine learning) - applied data analysis course

import matplotlib.pyplot as plt
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import itertools
import random
import seaborn as sns


def viz_y_distribution(y1, y2):
    """ displays a visualization of y and ybin's distribution as a histogram"""
    fig = plt.figure(figsize=(3,3))
    y2.hist(alpha=0.5)
    y1.hist()
    fig.suptitle('Skin darkness histogram\n(blue: continuous, orange: binary)', fontsize=12, y=1.11)
    plt.xlabel("Skin darkness")
    plt.ylabel("Number of players")
    plt.show()


def viz_features_importance(model, featLabels):
    """ displays a visualization of a model's features relative importance"""
    
    # sort the model's features according to their relative importances
    d = dict(zip(featLabels, model.feature_importances_))
    sorted_features = sorted(d.items(), key=operator.itemgetter(1))[::-1]
    
    # visualization
    x = range(len(featLabels))
    y = [y for _,y in sorted_features]
    my_xticks = [x for x,_ in sorted_features]
    plt.figure(figsize=(10,3))
    plt.title("Features relative importance", y=1.08)
    plt.xticks(x, my_xticks, rotation=60)
    plt.plot(x, y)
    plt.show()

def viz_feat_imp(sorted_feat):
    """ visualize feature importances of a single model """
    
    # visualization
    x = range(len([x for x,_ in sorted_feat]))
    y = [y for _,y in sorted_feat]
    my_xticks = [x for x,_ in sorted_feat]
    plt.figure(figsize=(10,3))
    plt.title("Features relative importance", y=1.08)
    plt.xticks(x, my_xticks, rotation=90)
    plt.plot(x, y)
    plt.show()
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, score_metric=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring = score_metric, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def viz_complexity(Xtr, Xte, ytr, yte, estimators_grid, max_depth_grid, max_nodes_grid):
    """ visualize train/test errors when increasing the number of estimators for a RFC """
    
    test_errors = []
    train_errors = []
    
    for n_estimators_, max_depth_, max_nodes_ in list(zip(estimators_grid, max_depth_grid, max_nodes_grid)) :
        rfc = RandomForestClassifier(max_depth=max_depth_, 
                                     n_estimators=n_estimators_,
                                     max_leaf_nodes=max_nodes_,
                                     random_state=0)
        rfc = rfc.fit(X=Xtr, y=ytr)
        y_pred_tr = rfc.predict(Xtr)
        y_pred_te = rfc.predict(Xte)
        tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(ytr, y_pred_tr).ravel()
        tn_te, fp_te, fn_te, tp_te = confusion_matrix(yte, y_pred_te).ravel()
        train_errors.append(1-BACC(tn_tr,tp_tr,fn_tr,fp_tr))
        test_errors.append(1-BACC(tn_te,tp_te,fn_te,fp_te))
        
        
    # additional smoothing
    ma_te = pd.DataFrame(test_errors).rolling(window=10, center=False).mean().dropna().values
    ma_tr = pd.DataFrame(train_errors).rolling(window=10, center=False).mean().dropna().values
    
    # plot the results
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    test_line = plt.plot(range(len(ma_te)), ma_te, color="r", label="test error")
    train_line = plt.plot(range(len(ma_tr)), ma_tr, color="g", label="train error")
    ax.set_xlabel("Model complexity \n (growing n_estimators, max_depth and max_nodes)")
    ax.set_ylabel("Balanced Error")
    plt.legend(bbox_to_anchor=(0.4, 0.4))
    plt.xscale("log")
    plt.show()

    
def BACC(tn,tp,fn,fp):
    """ computes a balanced accuracy from true negatives, true positives, false negatives and false positives """
    bacc = 1/2 * (tp/(tp+fn) + tn/(tn+fp))
    return bacc


def score_BACC(y_truth, y_pred):
    """ computes the score of balanced accuracy from y and y_pred """
    cnf_matrix = confusion_matrix(y_truth, y_pred)
    tn, fp, fn, tp = cnf_matrix.ravel()
    score = BACC(tn, fp, fn, tp)
    return score


def load_x_y():
    """ import the data and put it into numpy arrays """
    full_df = pd.read_csv("df_agg.csv", index_col=0)
    X_ = np.array(full_df.drop("rater", axis=1))
    y_ = np.array(full_df.rater)
    return X_, y_

def remove_random_features(df_, y):
    """ this function enables to remove a random feature of a df """

    group_of_feat = list(df_.columns)
    importance_list = []
    for n in range(0,len(group_of_feat)):

        num_to_select = n               
        list_of_random_feat = random.sample(group_of_feat, num_to_select)
    
        df_cut = df_.drop(list_of_random_feat, axis=1)
        X_cut = np.array(df_cut)
        X_cut_stdzd = (X_cut - X_cut.mean()) / X_cut.std()
        X_cut_labels = df_cut.columns
    
        rfc_f = RandomForestClassifier(n_estimators=10, max_depth=20, min_samples_split=8, max_leaf_nodes=25, random_state=0)
        rfc_f.fit(X_cut_stdzd,y)
    
        d = dict(zip(X_cut_labels, rfc_f.feature_importances_))
        sorted_x = sorted(d.items(), key=operator.itemgetter(1))
        importance_list.append(sorted_x[::-1])
    return importance_list


def viz_random_heatmap(importance_list):
    """ this function displays a 'heatmap' table to highlight which features are the most important
    by removing randomly some of them """
    
    rank_df = pd.DataFrame.from_records(importance_list[0], columns=['Features','Score']).set_index('Features').rank(axis=0, ascending=False)

    for n in range(1,len(importance_list)):
    
        df_n = pd.DataFrame.from_records(importance_list[n], columns=['Features','Score'])
        rank_n = df_n.set_index('Features').rank(axis=0, ascending=False)
        rank_df = pd.merge(rank_df, rank_n, left_index=True, right_index=True, how='outer')

    rank_df.columns = range(0,len(importance_list))
    cm = sns.light_palette("green", as_cmap=True, reverse=True)

    return (rank_df.loc[:]
        .style
        .background_gradient(cmap=cm)
        .highlight_null('red'))

def remove_less_important(df_,y):
    """ this function enables to remove the least important feature of a df """
    
    keep_list = []
    for n in range(0,len(df_.columns)):
        if n == 0:
            X_ = np.array(df_)
            X_stdzd = (X_-X_.mean()) / X_.std()
            X_labels = df_.columns
    
        if n != 0:
            imp_list = sorted_x[:-1]
            list_of_feat = [x for x,_ in imp_list]
            df_cut = df_[list_of_feat]
            X_ = np.array(df_cut)
            X_stdzd = (X_- X_.mean()) / X_.std()
            X_labels = df_cut.columns
    
        # fit the classifier
        rfc = RandomForestClassifier(n_estimators=10, max_depth=20, min_samples_split=8, max_leaf_nodes=25, random_state=0)
        rfc.fit(X_stdzd,y)
    
        # find and sort the important features
        d = dict(zip(X_labels, rfc.feature_importances_))
        sorted_x = sorted(d.items(), key=operator.itemgetter(1))[::-1]
    
        keep_list.append(sorted_x)
    return keep_list

def viz_lessImportant_heatmap(keep_list):
    """ this function displays a 'heatmap' table to highlight which features are the most important
    by removing the least important iteratively """

    rank_df = pd.DataFrame.from_records(keep_list[0], columns=['Features','Score']).set_index('Features').rank(axis=0, ascending=False)

    for n in range(1,len(keep_list)):
    
        df_n = pd.DataFrame.from_records(keep_list[n], columns=['Features','Score'])
        rank_n = df_n.set_index('Features').rank(axis=0, ascending=False)
        rank_df = pd.merge(rank_df, rank_n, left_index=True, right_index=True, how='outer')

    rank_df.columns = range(0,len(keep_list))
    rank_df['full_no_nan_count'] = rank_df.apply(lambda x: x.count(), axis=1)

    rank_df = rank_df.sort_values(by='full_no_nan_count', ascending=True)
    rank_df = rank_df.drop('full_no_nan_count', axis=1)
    cm = sns.light_palette("green", as_cmap=True, reverse=True)

    return (rank_df.loc[:]
        .style
        .background_gradient(cmap=cm)
        .highlight_null('red'))


if __name__ == "__main__":
	pass