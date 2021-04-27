from utils.feature_classifier_utils import ensemble_voting, Lasso_feature_selection
from utils.feature_oversampling_utils import random_oversampling
from utils.feature_performance_utils import save_confusion_matrix, Validation, save_roc_curve

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import pickle as pkl
import os


class SfmConversion:
    def __init__(self, sfm):
        self.sfm = sfm

    def _repeat(self, sfm_support, num_repeat):
        sfm_support = np.expand_dims(sfm_support, 0)
        return np.repeat(sfm_support, num_repeat, axis=0)

    def transform(self, x):
        num_feature = self.sfm.get_support().shape[0] // 2

        left_sfm_support = self.sfm.get_support()[:num_feature]
        right_sfm_support = self.sfm.get_support()[num_feature:]

        common_sfm_support = np.where(left_sfm_support == right_sfm_support, left_sfm_support, False)
        common_sfm_support = self._repeat(common_sfm_support, x.shape[0])

        left_selected_x = x[:, :num_feature][common_sfm_support]
        right_selected_x = x[:, num_feature:][common_sfm_support]

        left_selected_x = left_selected_x.reshape(x.shape[0], left_selected_x.shape[0] // x.shape[0])
        right_selected_x = right_selected_x.reshape(x.shape[0], right_selected_x.shape[0] // x.shape[0])

        return np.append(left_selected_x, right_selected_x, axis=1)


def result2csv(results, filename='./result.csv'):
    means = []
    stds = []
    for key, value in results.items():
        means.extend(np.mean(value, axis=0))
        stds.extend(np.std(value, axis=0))

    columns = ['accuracy', 'auroc', 'precision', 'f1']
    keys = list(results.keys())
    index = []
    for key in keys:
        for col in columns:
            index.append((key, col))

    index = pd.MultiIndex.from_tuples(index, names=['type', 'metric'])
    pd.DataFrame(np.array([means, stds]).T, index=index, columns=['mean', 'std']).to_csv(filename)


reporting_results = dict(
    v1=[],
    v2=[],
    v3=[],
    v4=[],
    v5=[]
)


def resplit_left_right(data):
    assert data.shape[1] == 256

    lefts = data[:, :128]
    rights = data[:, 128:]

    merged_array = []
    for l, r in zip(lefts, rights):
        merged_array.append(l)
        merged_array.append(r)

    merged_array = np.array(merged_array)
    return merged_array


def DLR(version, DEEP_TR=0.03, TEXTURE_TR=0.01, RANDOM_NUMBER=42, data=None, test_ratio=0.2, num_df=256):
    if version == 'v1':
        CLINIC, DEEP, TEXTURE = True, False, False
    elif version == 'v2':
        CLINIC, DEEP, TEXTURE = False, False, True
    elif version == 'v3':
        CLINIC, DEEP, TEXTURE = False, True, False
    elif version == 'v4':
        CLINIC, DEEP, TEXTURE = True, True, False
    elif version == 'v5':
        CLINIC, DEEP, TEXTURE = True, True, True

    # Data Loading
    X, Y = data

    if test_ratio != 0:
        X_train, y_train = X, Y

        # Standard Scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        """
            Feature Selection
        """

        # only Clinic | version 1
        if CLINIC & (not TEXTURE) & (not DEEP):
            sfm = None

        # only texture | version 2
        elif (not CLINIC) & TEXTURE & (not DEEP):
            sfm = Lasso_feature_selection(X_train, y_train, tr=TEXTURE_TR)
            X_train = sfm.transform(X_train)

        # only deep | version 3
        elif (not CLINIC) & (not TEXTURE) & DEEP:
            sfm = Lasso_feature_selection(X_train, y_train, tr=DEEP_TR)
            X_train = sfm.transform(X_train)

        # deep + clinic | version 4
        elif CLINIC & (not TEXTURE) & DEEP:
            sfm = Lasso_feature_selection(X_train[:, :-4], y_train, tr=DEEP_TR)
            X_train = np.hstack([sfm.transform(X_train[:, :-4]), X_train[:, -4:]])

        # texture + deep + clinic | version 5
        elif CLINIC & TEXTURE & DEEP:
            # texture + clinic + deep
            clinical_features = X_train[:, -4:]
            deep_features = X_train[:, -(num_df + 4):-4]
            texture_features = X_train[:, :-(num_df + 4)]

            deep_sfm = Lasso_feature_selection(deep_features, y_train, tr=DEEP_TR)
            texture_sfm = Lasso_feature_selection(texture_features, y_train, tr=TEXTURE_TR)

            X_train = np.hstack([texture_sfm.transform(texture_features),
                                 deep_sfm.transform(deep_features),
                                 clinical_features])

            sfm = [texture_sfm, deep_sfm]

        # LASSO feature Selection -> Random OverSampling
        """
            Threshold > 0.1, 0.15, 0.2
        """
        X_train, y_train = random_oversampling(X_train, y_train, random_state=RANDOM_NUMBER)

        # model
        classifier = ensemble_voting(X_train, y_train, random_state=RANDOM_NUMBER, cv=5)

        # Training
        classifier.fit(X_train, y_train)

        output = dict(
            model=classifier,
            version=version,
            scaler=scaler,
            sfm=sfm,
            random_number=RANDOM_NUMBER,
            output_dict=None
        )
        return output

    elif test_ratio > 0:
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            test_size=test_ratio,
                                                            random_state=RANDOM_NUMBER)
        # Standard Scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print("num features before LASSO : ", X_train.shape[1])

        """
            Feature Selection
        """

        # only Clinic | version 1
        if CLINIC & (not TEXTURE) & (not DEEP):
            sfm = None

        # only texture | version 2
        elif (not CLINIC) & TEXTURE & (not DEEP):
            sfm = Lasso_feature_selection(X_train, y_train, tr=TEXTURE_TR)
            X_train = sfm.transform(X_train)
            X_test = sfm.transform(X_test)

        # only deep | version 3
        elif (not CLINIC) & (not TEXTURE) & DEEP:
            sfm = Lasso_feature_selection(X_train, y_train, tr=DEEP_TR)
            X_train = sfm.transform(X_train)
            X_test = sfm.transform(X_test)

        # deep + clinic | version 4
        elif CLINIC & (not TEXTURE) & DEEP:
            sfm = Lasso_feature_selection(X_train[:, :-4], y_train, tr=DEEP_TR)
            X_train = np.hstack([sfm.transform(X_train[:, :-4]), X_train[:, -4:]])
            X_test = np.hstack([sfm.transform(X_test[:, :-4]), X_test[:, -4:]])

        # texture + deep + clinic | version 5
        elif CLINIC & TEXTURE & DEEP:
            # texture + clinic + deep
            clinical_features = X_train[:, -4:]
            deep_features = X_train[:, -(num_df + 4):-4]
            texture_features = X_train[:, :-(num_df + 4)]

            deep_sfm = Lasso_feature_selection(deep_features, y_train, tr=DEEP_TR)
            texture_sfm = Lasso_feature_selection(texture_features, y_train, tr=TEXTURE_TR)

            X_train = np.hstack([texture_sfm.transform(texture_features),
                                 deep_sfm.transform(deep_features),
                                 clinical_features])

            clinical_features_test = X_test[:, -4:]
            deep_features_test = X_test[:, -(num_df + 4):-4]
            texture_features_test = X_test[:, :-(num_df + 4)]

            X_test = np.hstack([texture_sfm.transform(texture_features_test),
                                deep_sfm.transform(deep_features_test),
                                clinical_features_test])

            sfm = [texture_sfm, deep_sfm]

        print("num features after LASSO : ", X_train.shape[1])

        # LASSO feature Selection -> Random OverSampling
        """
            Threshold > 0.1, 0.15, 0.2
        """
        X_train, y_train = random_oversampling(X_train, y_train, random_state=RANDOM_NUMBER)

        # model
        classifier = ensemble_voting(X_train, y_train, random_state=RANDOM_NUMBER, cv=5)

        # Training
        classifier.fit(X_train, y_train)

        # Testing
        test_result = Validation(classifier, X_test, y_test)

        # Save Result
        reporting_results[version].append(list(test_result[:4]))

        output = dict(
            model=classifier,
            version=version,
            scaler=scaler,
            sfm=sfm,
            random_number=RANDOM_NUMBER,
            output_dict=dict(
                preds=classifier.predict(X_test),
                output_list=classifier.predict_proba(X_test)
            )
        )
        return output


def sfm_transform(sfm, data):
    return data[:, sfm.get_support() is True]


def save_model_bag(model_bag, save_path):
    """
        save model bag (pickle bag)
    """
    with open(save_path, 'wb') as file:
        pkl.dump(model_bag, file)
        assert os.path.isfile(save_path)


def DLR_inference(train_module, X_test, num_df=256):
    version = train_module['version']
    classifier = train_module['model']
    scaler = train_module['scaler']
    sfm = train_module['sfm']
    RANDOM_NUMBER = train_module['random_number']

    if version == 'v1':
        CLINIC, DEEP, TEXTURE = True, False, False
    elif version == 'v2':
        CLINIC, DEEP, TEXTURE = False, False, True
    elif version == 'v3':
        CLINIC, DEEP, TEXTURE = False, True, False
    elif version == 'v4':
        CLINIC, DEEP, TEXTURE = True, True, False
    elif version == 'v5':
        CLINIC, DEEP, TEXTURE = True, True, True

    X_test = scaler.transform(X_test)

    """
        Feature Selection
    """

    # only Clinic | version 1
    if CLINIC & (not TEXTURE) & (not DEEP):
        pass

    # only texture | version 2
    elif (not CLINIC) & TEXTURE & (not DEEP):
        X_test = sfm.transform(X_test)

    # only deep | version 3
    elif (not CLINIC) & (not TEXTURE) & DEEP:
        X_test = sfm.transform(X_test)

    # deep + clinic | version 4
    elif CLINIC & (not TEXTURE) & DEEP:
        X_test = np.hstack([sfm.transform(X_test[:, :-4]), X_test[:, -4:]])

    # texture + deep + clinic | version 5
    elif CLINIC & TEXTURE & DEEP:

        texture_sfm, deep_sfm = sfm

        # texture + clinic + deep
        clinical_features_test = X_test[:, -4:]
        deep_features_test = X_test[:, -(num_df + 4):-4]
        texture_features_test = X_test[:, :-(num_df + 4)]

        X_test = np.hstack([texture_sfm.transform(texture_features_test),
                            deep_sfm.transform(deep_features_test),
                            clinical_features_test])

    output_dict = dict(
        preds=classifier.predict(X_test),
        output_list=classifier.predict_proba(X_test)
    )

    return output_dict


if __name__ == '__main__':
    pass
