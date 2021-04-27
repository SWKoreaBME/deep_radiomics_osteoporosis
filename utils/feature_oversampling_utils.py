from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def random_oversampling(feature_data, feature_label, random_state):

    X_resampled, y_resampled = \
        RandomOverSampler(random_state=random_state).fit_resample(feature_data, feature_label)

    return X_resampled, y_resampled

def random_undersampling(feature_data, feature_label, random_state):

    X_resampled, y_resampled = \
        RandomUnderSampler(random_state = random_state).fit_resample(feature_data, feature_label)

    return X_resampled, y_resampled

def smote(feature_data, feature_label, random_state):

    X_resampled, y_resampled = \
        SMOTE(random_state = random_state).fit_resample(feature_data, feature_label)

    return X_resampled, y_resampled