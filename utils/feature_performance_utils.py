from sklearn.metrics import precision_recall_curve, \
                            confusion_matrix, \
                            accuracy_score, \
                            roc_auc_score, \
                            roc_curve, \
                            f1_score, \
                            precision_score, auc

from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from .feature_utils import dummy_labelize_swk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import pickle


import warnings
warnings.filterwarnings("ignore")

def feature_selection_swk(feature_data, feature_label, k, return_names = True):
    selector = SelectKBest(mutual_info_classif, k = k)

    if return_names : 
        selected_features = selector.fit(feature_data, feature_label).get_support()

        return selector.fit_transform(feature_data, feature_label), selected_features
    
    else:

        return selector.fit_transform(feature_data, feature_label)

def performance_swk(clf, X_train, X_test, y_train, y_test):

    # Performance Check on Training data
    train_performance = clf.score(X_train, y_train)

    # Performance Check on Test data
    test_performance = clf.score(X_test, y_test)

    return train_performance, test_performance

def save_roc_curve(clf, X_test, y_test, roc_figure_save = True, n_classes = 2):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # test
    # y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    y_pred_proba_tr = np.amax(y_pred_proba, axis=1)

    # evaluation
    y_test_dummy = dummy_labelize_swk(y_test, n_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_dummy[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    best_tr = thresholds[0][np.argmax(tpr[0] - fpr[0])]
    best_dot = fpr[0][np.argmax(tpr[0] - fpr[0])], tpr[0][np.argmax(tpr[0] - fpr[0])]

    if roc_figure_save:

        plt.figure(figsize=(10, 10))

        lw = 2
        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot(*best_dot, 'ro')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating curve for normal&penia vs. porosis')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.jpg')

        # plt.show()
        plt.close()

    # print('done')

    return roc_auc[1], best_tr

def save_confusion_matrix(confusion):

    df_cm = pd.DataFrame(confusion, index = ['negative label', 'positive label'],
                    columns = ['negative prediction', 'positive prediction'])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.jpg')
    # plt.show()
    plt.close()

    return

def performance_all_swk(clf, X_test, y_test, roc_figure_save = True, confusion_matrix_save = True):

    y_pred = clf.predict(X_test)

    # auc score
    test_roc_score = save_roc_curve(clf, X_test, y_test, roc_figure_save, 2)

    # f1 score
    test_f1_score = f1_score(y_test, y_pred)

    # precision score
    test_precision_score = precision_score(y_test, y_pred)

    # accuracy score
    test_acc = accuracy_score(y_test, y_pred)

    if confusion_matrix_save : 
        confusion = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(confusion)

    return [test_roc_score, test_precision_score, test_f1_score, test_acc]

def return_selected_feature_names(selected_feature_index, feature_names):
    selected_features = list(np.array(feature_names)[np.where(selected_feature_index == True)])
    return selected_features

def load_texture_feature(feature_file, label_file):
    
    whole_feature = pd.read_excel(feature_file)
    label_dict = pickle.load(open(label_file, 'rb'))

    no_label = [a for a in whole_feature.to_dict('split')['index'] if a not in label_dict.keys()]
    whole_feature = whole_feature.drop(no_label)

    feature_data = whole_feature.to_dict('split')['data'] # This is input x
    feature_names = whole_feature.to_dict('split')['columns']
    feature_index = whole_feature.to_dict('split')['index']

    label_values = []

    for index in feature_index:

        label_values.append(label_dict[index])

    binary_feature_label = np.array(label_values)
    
    return np.array(feature_data), feature_names, feature_index, binary_feature_label

def load_clinical_data(clinical_data_file, feature_index):
    
    # read clinical data
    
    clinical_values = []

    clinical_data = pickle.load(open(clinical_data_file, 'rb'))
    for subject in feature_index:
        clinical_values.append(clinical_data[subject])
        
    clinical_data = np.array(clinical_values)
    
    feature_names_clinical = ['sex', 'age', 'weight']
    return clinical_data, feature_names_clinical

def save_confusion_matrix(confusion, index, columns, title):

    df_cm = pd.DataFrame(confusion, index = index,
                    columns = columns)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.title(title)
#     plt.savefig('confusion_matrix.jpg')
    plt.show()
    plt.close()

    return


def binarize_threshold_swk(x_, tr):
    x = x_.T[1]
    x[x>tr] = 1
    x[x<tr] = 0
    return x


def Validation(clf, X_test, y_test):
    print('================================')
    print("Validation")
    
#     y_test, y_pred = normal_porosis(y_test, y_pred)

    # y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    test_auc_score, best_tr = save_roc_curve(clf=clf, X_test=X_test, y_test=y_test, roc_figure_save=False, n_classes=2)

    y_pred = binarize_threshold_swk(y_pred_proba, best_tr)
    
    if len(Counter(y_test).keys()) == 2:
        average_method = 'binary'
    else:
        average_method = None
    
    confusion = confusion_matrix(y_test, y_pred)
    test_precision_score = precision_score(y_test, y_pred, average=average_method)
    test_f1_score = f1_score(y_test, y_pred, average=average_method)
    test_acc = accuracy_score(y_test, y_pred)

    print('confusion : ', confusion)
    print('Accuracy : %.3f' %test_acc)
    print('roc-auc score : %.3f' %test_auc_score)
    print('f1 score : %.3f' %test_f1_score)
    print('precision : %.3f' %test_precision_score)
    
    return test_acc, test_auc_score, test_precision_score, test_f1_score, confusion
    
def normal_porosis(true, pred):
    
    new_true = []; new_pred = []
    
    for t, p in zip(true, pred):
        
        if t == 1 or p == 1:
            continue
        else:
            new_true.append(t); new_pred.append(p)
            
    new_true = np.array(new_true); new_true[new_true == 2] = 1
    new_pred = np.array(new_pred); new_pred[new_pred == 2] = 1
    
    return new_true, new_pred

def save_predicition(feature_index, y_pred, y_pred_proba, filename):

    data = {
        'label' : y_pred,
        'confidence score for class 0' : np.asarray(y_pred_proba).T[0],
        'confidence score for class 1' : np.asarray(y_pred_proba).T[1]
    }

    pd.DataFrame(index=feature_index, data=data).to_excel(filename)

    print('saved')
    
    return