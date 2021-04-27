import numpy as np

from utils.feature_classifier_utils import Lasso_feature_selection
from main_utils.utils_train import SfmConversion


class FeatureSelector:
    def __init__(self, args):
        self.args = args

    def select(self, version, x_train, y_train):
        selected_output = fs_version(version=version,
                                     X_train=x_train,
                                     y_train=y_train,
                                     same_feature_each_side=self.args["same_feature_each_side"],
                                     resume=self.args["resume"],
                                     sfm=self.args["sfm"],
                                     deep_tr=self.args["deep_tr"],
                                     texture_tr=self.args["texture_tr"],
                                     num_df=self.args["num_df"])
        if self.args["resume"]:
            return selected_output
        else:
            selected_output, sfm = selected_output
            return selected_output, sfm


def fs_version(version,
               X_train,
               y_train,
               same_feature_each_side,
               resume,
               sfm,
               deep_tr=0.3,
               texture_tr=0.1,
               num_df=256):

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

    if deep_tr is not None:
        DEEP_TR = deep_tr
    if texture_tr is not None:
        TEXTURE_TR = texture_tr

    print("Deep TR : {}".format(deep_tr))
    print("Texture TR : {}".format(texture_tr))

    if resume:
        X_test = X_train
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
        return X_test

    elif same_feature_each_side:
        # only Clinic | version 1
        if CLINIC & (not TEXTURE) & (not DEEP):
            sfm = None

        # only texture | version 2
        elif (not CLINIC) & TEXTURE & (not DEEP):
            sfm = Lasso_feature_selection(X_train, y_train, tr=TEXTURE_TR)
            sfm = SfmConversion(sfm)
            X_train = sfm.transform(X_train)

        # only deep | version 3
        elif (not CLINIC) & (not TEXTURE) & DEEP:
            sfm = Lasso_feature_selection(X_train, y_train, tr=DEEP_TR)
            sfm = SfmConversion(sfm)
            X_train = sfm.transform(X_train)

        # deep + clinic | version 4
        elif CLINIC & (not TEXTURE) & DEEP:
            sfm = Lasso_feature_selection(X_train[:, :-4], y_train, tr=DEEP_TR)
            sfm = SfmConversion(sfm)
            X_train = np.hstack([sfm.transform(X_train[:, :-4]), X_train[:, -4:]])

        # texture + deep + clinic | version 5
        elif CLINIC & TEXTURE & DEEP:
            # texture + clinic + deep
            clinical_features = X_train[:, -4:]
            deep_features = X_train[:, -(num_df + 4):-4]
            texture_features = X_train[:, :-(num_df + 4)]

            deep_sfm = Lasso_feature_selection(deep_features, y_train, tr=DEEP_TR)
            texture_sfm = Lasso_feature_selection(texture_features, y_train, tr=TEXTURE_TR)

            deep_sfm = SfmConversion(deep_sfm)
            texture_sfm = SfmConversion(texture_sfm)

            X_train = np.hstack([texture_sfm.transform(texture_features),
                                 deep_sfm.transform(deep_features),
                                 clinical_features])
            return X_train, [texture_sfm, deep_sfm]

        print(X_train.shape)
        return X_train, sfm

    else:
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
            return X_train, [texture_sfm, deep_sfm]

        print(X_train.shape)

        return X_train, sfm
