from main_utils.utils_train import *
from main_utils.utils_mlp import *
from main_utils.utils_rf import *
from main_utils.utils_fs import *
from main_utils.utils_data import osteodataloader


class Trainer:
    def __init__(self, args):
        self.version_functions = dict(
            v1=lambda x: x[:, -4:],  # only clinic
            v2=lambda x: x[:, :-260],  # only texture
            v3=lambda x: x[:, -260:-4],  # only deep
            v4=lambda x: x[:, -260:],  # deep + clinic
            v5=lambda x: x[:, :]
        )

        self.train_function_dict = dict(
            mlp=train_mlp,
            rf=train_rf
        )

        self.model_type = args["model_type"]
        self.train_function = self.train_function_dict[args["model_type"]]
        self.test_ratio = args["test_ratio"]
        # self.test_type = "valid" if self.test_ratio > 0 else "train"
        self.random_state = args["random_state"]
        self.batch_size = args["batch_size"]
        self.same_feature_each_side = args["same_feature_each_side"]
        self.versions = args["versions"]
        self.epochs = args["epochs"]
        self.save_dir = args["save_dir"]
        self.num_df = args["num_df"]
        self.deep_tr = args["deep_tr"]
        self.texture_tr = args["texture_tr"]

        self.model_bag = dict()
        self.fs_args = dict(
            resume=False,
            sfm=None,
            deep_tr=self.deep_tr,
            texture_tr=self.texture_tr,
            same_feature_each_side=self.same_feature_each_side,
            num_df=args["num_df"]
        )

    @staticmethod
    def preprocess(data, scaler):
        data = scaler.fit_transform(data)
        return data, scaler

    def run(self, train_x, train_y):
        for version in self.versions:
            print(f"\n>> version : {version}")
            version_function = self.version_functions[version]
            version_save_path = os.path.join(self.save_dir, f"model_{version}.pkl")

            # MLP
            if self.model_type == "mlp":
                version_train_x, version_train_y = version_function(train_x), train_y
                feature_selector = FeatureSelector(self.fs_args)

                if self.test_ratio > 0:
                    version_train_x, version_valid_x, version_train_y, version_valid_y = train_test_split(
                        version_train_x,
                        version_train_y,
                        test_size=self.test_ratio,
                        random_state=self.random_state)

                # Preprocessing
                scaler = StandardScaler()
                version_train_x, scaler = self.preprocess(version_train_x, scaler)
                version_valid_x = scaler.transform(version_valid_x)
                print("Preprocessing Done")

                # Feature Selection
                version_train_x, feature_selection_module = feature_selector.select(version,
                                                                                    version_train_x,
                                                                                    version_train_y)
                if self.test_ratio > 0:
                    version_valid_x = fs_version(version,
                                                 version_valid_x,
                                                 version_valid_y,
                                                 same_feature_each_side=self.same_feature_each_side,
                                                 resume=True,
                                                 sfm=feature_selection_module,
                                                 num_df=self.num_df)

                print("Feature Selection Done")
                # Random Over Sampling
                version_train_x, version_train_y = random_oversampling(version_train_x,
                                                                       version_train_y,
                                                                       random_state=self.random_state)

                # wrap data with dataloader
                train_loader = osteodataloader(version_train_x, version_train_y, batch_size=self.batch_size)
                valid_loader = osteodataloader(version_valid_x, version_valid_y, batch_size=self.batch_size) \
                    if self.test_ratio > 0 else train_loader

                # Train MLP module
                model = self.train_function(train_loader,
                                            valid_loader,
                                            bw=version_train_x.shape[1],
                                            epochs=self.epochs,
                                            device="cuda")

                self.model_bag[version] = dict(
                    model=model,
                    sfm=feature_selection_module,
                    scaler=scaler
                )

                # save model bag
                save_model_bag(self.model_bag[version], version_save_path)

            # RF
            elif self.model_type == "rf":
                train_module = self.train_function(version=version,
                                                   train_x=train_x,
                                                   train_y=train_y,
                                                   deep_tr=self.deep_tr,
                                                   texture_tr=self.texture_tr,
                                                   random_number=self.random_state,
                                                   test_ratio=self.test_ratio,
                                                   num_df=self.num_df)

                self.model_bag[version] = train_module
                save_model_bag(train_module, version_save_path)


if __name__ == '__main__':
    pass
