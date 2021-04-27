from main_utils.utils_train import *
from main_utils.utils_mlp import *
from main_utils.utils_rf import *
from main_utils.utils_fs import *

from loader import *
from pickle import load


class Inferencer:
    def __init__(self, args):
        self.version_functions = dict(
            v1=lambda x: x[:, -4:],  # only clinic
            v2=lambda x: x[:, :-260],  # only texture
            v3=lambda x: x[:, -260:-4],  # only deep
            v4=lambda x: x[:, -260:],  # deep + clinic
            v5=lambda x: x[:, :]
        )

        self.test_function_dict = dict(
            mlp=train_mlp,
            rf=train_rf
        )

        self.test_loader_dict = dict(
            mlp=mlp_loader,
            rf=rf_loader
        )

        self.infer_dict = dict(
            mlp=infer_mlp,
            rf=rf_loader
        )

        # todo: call from model package
        self.model_bag = self.load_pickle(args["model_path"])

        self.test_function = self.test_function_dict[args["model_type"]]
        self.scaler = self.model_bag["scaler"]
        self.sfm = self.model_bag["sfm"]
        self.model = self.model_bag["model"]
        self.loader = self.test_loader_dict[args["model_type"]]
        self.data_type = args["data_type"]
        self.infer_function = self.infer_dict[args["model_type"]]
        self.batch_size = args["batch_size"]
        self.subjects = args["subjects"]

        fs_args = dict(
            resume=True,
            sfm=self.sfm,
            deep_tr=None,
            texture_tr=None,
            same_feature_each_side=False,
            num_df=args["num_df"]
        )
        self.feature_selector = FeatureSelector(fs_args)
        self.test_output = None

    def preprocess(self, data):
        data = self.scaler.fit_transform(data)
        return data

    def run(self, test_x, version):
        version_function = self.version_functions[version]
        test_x = version_function(test_x)

        test_x = self.preprocess(test_x)
        test_x = self.feature_selector.select(version, test_x, None)

        test_loader = self.loader(X=test_x,
                                  Y=None,
                                  batch_size=self.batch_size)
        test_output = self.infer_function(test_loader)
        self.test_output = test_output
        return test_output

    def save(self):
        assert self.test_output is not None, "Try to save before running inference, You must run inference first"
        save_infer_result(self.test_output,
                          self.data_type,
                          self.subjects,
                          self.data_type,
                          save_path=self.args["save_path"])

    @staticmethod
    def load_pickle(file, mode='rb'):
        with open(file, mode) as f:
            a = load(f)
        return a


if __name__ == '__main__':
    pass
