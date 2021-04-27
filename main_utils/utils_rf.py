from main_utils.utils_train import *


def train_rf(version,
             train_x,
             train_y,
             texture_tr,
             deep_tr,
             random_number,
             test_ratio,
             num_df):

    train_module = DLR(version=version,
                       DEEP_TR=deep_tr,
                       TEXTURE_TR=texture_tr,
                       RANDOM_NUMBER=random_number,
                       test_ratio=test_ratio,
                       data=[train_x, train_y],
                       num_df=num_df)

    return train_module


if __name__ == '__main__':
    pass
