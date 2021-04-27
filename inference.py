from main_utils.utils_train import *
from main_utils.utils_mlp import *
from main_utils.utils_rf import *
from main_utils.utils_fs import *

from loader import *

from tqdm import tqdm
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

        self.test_loader_dict = dict(
            mlp=osteodataloader,
            rf=rf_loader
        )

        self.infer_dict = dict(
            mlp=infer_mlp,
            rf=DLR_inference
        )

        # todo: call from model package
        self.model_type = args["model_type"]
        self.loader = self.test_loader_dict[args["model_type"]]
        self.test_type = args["test_type"]
        self.model_path = args["model_path"]
        self.infer_function = self.infer_dict[args["model_type"]]
        self.batch_size = args["batch_size"]
        self.subjects = args["subjects"]
        self.versions = args["versions"]
        self.same_feature_each_side = args["same_feature_each_side"]
        self.input_x = args["input_x"]
        self.save_path = args["save_dir"]
        self.model_type = args["model_type"]
        self.num_df = args["num_df"]
        self.test_output = None

    @staticmethod
    def preprocess(data, scaler):
        data = scaler.transform(data)
        return data

    def infer(self, test_x, version, model_bag):
        if self.model_type == "mlp":
            version_function = self.version_functions[version]

            fs_args = dict(
                resume=True,
                sfm=model_bag["sfm"],
                deep_tr=None,
                texture_tr=None,
                num_df=self.num_df,
                same_feature_each_side=self.same_feature_each_side
            )

            test_x = version_function(test_x)
            feature_selector = FeatureSelector(fs_args)
            test_x = self.preprocess(test_x, model_bag["scaler"])
            test_x = feature_selector.select(version, test_x, None)

            test_loader = self.loader(X=test_x,
                                      Y=None,
                                      batch_size=self.batch_size)
            test_output = self.infer_function(model_bag["model"],
                                              test_loader,
                                              device="cuda")
            self.test_output = test_output
            return test_output

        elif self.model_type == "rf":
            test_output = self.infer_function(model_bag,
                                              test_x,
                                              self.num_df)
            self.test_output = test_output
            return test_output

    def run(self):
        pbar = tqdm(total=len(self.versions))
        for version in self.versions:
            model_bag = self.load_pickle(os.path.join(self.model_path, f"model_{version}.pkl"))
            output = self.infer(self.input_x, version, model_bag)
            if output is not None:
                self.save(output, version)
            pbar.update(1)

    def save(self, test_output, version):
        assert test_output is not None, "Try to save before running inference, You must run inference first"
        save_infer_result(test_output,
                          version,
                          self.subjects,
                          self.test_type,
                          self.model_type,
                          self.save_path)

    @staticmethod
    def load_pickle(file, mode='rb'):
        with open(file, mode) as f:
            return_obj = load(f)
        return return_obj


if __name__ == '__main__':
    pass
