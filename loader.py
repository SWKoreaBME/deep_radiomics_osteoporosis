from main_utils.utils_data import *
from tqdm import tqdm

import torch


class Loader:
    def __init__(self,
                 deep_feature_dir,
                 texture_feature_dir,
                 clinical_feature_file,
                 oneside=False,
                 label_file=None):

        self.deep_feature_dir = deep_feature_dir
        self.texture_feature_dir = texture_feature_dir
        self.clinical_feature_file = read_pickle(clinical_feature_file)
        self.label_file = read_pickle(label_file)
        self.oneside = oneside

        self.error_subject = ['2010_0386',
                              '2011_0664',
                              '2014_0066',
                              '2012_0462',
                              '2014_0075',
                              '2014_0480',
                              '2014_0491',
                              '2018_1037',
                              '2019_1019']

        self.subjects = dict()
        self.columns = dict()

    def get_data(self, phase="development"):

        df_data = read_csv(os.path.join(self.deep_feature_dir, f"{phase}.csv"))
        tf_data = read_csv(os.path.join(self.texture_feature_dir, f"original_{phase}.csv"))

        df_data = df_data.fillna(df_data.mean())
        tf_data = tf_data.fillna(tf_data.mean())

        total_subjects = get_subject_names(tf_data, phase)
        self.subjects[phase] = list()

        tf_cols = [f"{x}_left" for x in get_column_names(tf_data)] + [f"{x}_right" for x in get_column_names(tf_data)]
        df_cols = [f"{x}_left" for x in get_column_names(df_data)] + [f"{x}_right" for x in get_column_names(df_data)]
        clinical_cols = ["male", "female", "age", "weight"]

        self.columns[phase] = tf_cols + df_cols + clinical_cols

        X, Y = list(), list()

        for subject in tqdm(total_subjects):
            if subject in self.error_subject:
                continue

            if phase == "brmh":
                pairs = [subject.split('_')[0]+'_{}'.format(i) for i in range(2)]
            else:
                pairs = [subject + '_{}'.format(i) for i in range(2)]

            if phase == "development":
                texture_features = tf_data.loc[tf_data["Unnamed: 0"].isin(pairs)].values[:, 2:]
                deep_features = df_data.loc[df_data["Unnamed: 0"].isin(pairs)].values[:, 1:]
            else:
                texture_features = tf_data.loc[tf_data["Unnamed: 0"].isin(pairs)].values[:, 1:]
                deep_features = df_data.loc[df_data["Unnamed: 0"].isin(pairs)].values[:, 1:]

            clinical_features = np.array([self.clinical_feature_file[subject]])

            both_texture = texture_features.reshape((1, texture_features.shape[0] * texture_features.shape[1]))
            both_deep = deep_features.reshape((1, deep_features.shape[0] * deep_features.shape[1]))

            if self.oneside:
                num_tf = both_texture.shape[1] // 2
                num_df = both_deep.shape[1] // 2

                subject_left_features = np.hstack([both_texture[:, :num_tf], both_deep[:, :num_df], clinical_features])
                subject_right_features = np.hstack([both_texture[:, num_tf:], both_deep[:, num_df:], clinical_features])

                subject_left_features = subject_left_features.squeeze()
                subject_right_features = subject_right_features.squeeze()

                X.append(list(subject_left_features))
                X.append(list(subject_right_features))

                self.subjects[phase].append(f"{subject}_left")
                self.subjects[phase].append(f"{subject}_right")

            else:
                subject_total_features = np.hstack([both_texture, both_deep, clinical_features])
                subject_total_features = subject_total_features.squeeze()
                X.append(list(subject_total_features))
                self.subjects[phase].append(subject)

            if phase == "development":
                label = self.label_file[subject]
                if self.oneside:
                    Y.append(label)
                    Y.append(label)
                else:
                    Y.append(label)

        if phase == "development":
            x_values = np.array(X)
            y_values = np.array(Y)
            return x_values, y_values
        else:
            x_values = np.array(X)
            return x_values, self.subjects[phase]


# MLP dataloader
def mlp_loader(X, Y, batch_size):
    dataloader = []
    batch_num = (X.shape[0] // batch_size) + 1

    for i in range(batch_num):
        batch_index = i * batch_size
        try:
            if Y is not None:
                single_batch_x = X[batch_index:batch_index + batch_size]
                single_batch_y = Y[batch_index:batch_index + batch_size]
            else:
                single_batch_x = X[batch_index:batch_index + batch_size]

        except:
            if Y is not None:
                single_batch_x, single_batch_y = X[batch_index:], Y[batch_index:]
            else:
                single_batch_x = X[batch_index:]

        if Y is not None:
            dataloader.append([torch.tensor(single_batch_x).float(), torch.tensor(single_batch_y)])
        else:
            dataloader.append(torch.tensor(single_batch_x).float())

    return dataloader


# RF dataloader

def rf_loader(X, Y, batch_size):
    dataloader = []
    batch_num = (X.shape[0] // batch_size) + 1

    for i in range(batch_num):
        batch_index = i * batch_size
        try:
            if Y is not None:
                single_batch_x = X[batch_index:batch_index + batch_size]
                single_batch_y = Y[batch_index:batch_index + batch_size]
            else:
                single_batch_x = X[batch_index:batch_index + batch_size]

        except:
            if Y is not None:
                single_batch_x, single_batch_y = X[batch_index:], Y[batch_index:]
            else:
                single_batch_x = X[batch_index:]

        if Y is not None:
            dataloader.append([torch.tensor(single_batch_x).float(), torch.tensor(single_batch_y)])
        else:
            dataloader.append(torch.tensor(single_batch_x).float())
    return dataloader


if __name__ == '__main__':

    # Example

    deep_feature_dir = "/sda1/AI_osteoporosis/df_cropped"
    texture_feature_dir = "/sda1/AI_osteoporosis/texture_feature"
    snuh_clinic_feature_file = "/sda1/AI_osteoporosis/whole_clinic_final.pickle"
    brmh_clinic_feature_file = "/sda1/AI_osteoporosis/brmh_clinic_final_ver2.pickle"
    snuh_brmh_clinic_feature_file = "/sda1/AI_osteoporosis/snuh_brmh_clinic_total.pickle"
    label_file = "/sda1/AI_osteoporosis/label_dict_final.pickle"
    
    # merge_two_dictionaries
    if not os.path.isfile(snuh_brmh_clinic_feature_file):
        snuh_pkl, brmh_pkl = read_pickle(snuh_clinic_feature_file), read_pickle(brmh_clinic_feature_file)
        total_pkl = {**snuh_pkl, **brmh_pkl}
        save_pickle(total_pkl, snuh_brmh_clinic_feature_file)

    loader = Loader(deep_feature_dir=deep_feature_dir,
                    texture_feature_dir=texture_feature_dir,
                    clinical_feature_file=snuh_brmh_clinic_feature_file,
                    label_file=label_file)
    
    dev_X, dev_y = loader.get_data("development")
    test_X, test_subjects = loader.get_data("test")
    brmh_X, brmh_subjects = loader.get_data("brmh")

    pd.DataFrame(data=dev_X,
                 index=loader.subjects["development"],
                 columns=loader.columns["development"]).to_csv("/sda1/AI_osteoporosis/dev.csv")
    pd.DataFrame(data=test_X,
                 index=loader.subjects["test"],
                 columns=loader.columns["test"]).to_csv("/sda1/AI_osteoporosis/test.csv")
    pd.DataFrame(data=brmh_X,
                 index=loader.subjects["brmh"],
                 columns=loader.columns["brmh"]).to_csv("/sda1/AI_osteoporosis/brmh.csv")

    pass
