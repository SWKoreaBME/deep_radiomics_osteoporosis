from loader import Loader
from train import Trainer
from inference import Inferencer
from main_utils.utils_main import log_args

import argparse
from script.config import args


def main():
    log_args(args)

    loader = Loader(deep_feature_dir=args["deep_feature_dir"],
                    texture_feature_dir=args["texture_feature_dir"],
                    clinical_feature_file=args["snuh_brmh_clinic_feature_file"],
                    oneside=args["oneside"],
                    label_file=args["label_file"])

    for data_type in args["data_type"]:
        if phase == "train":
            input_x, input_y = loader.get_data(data_type)
            run_args = dict(
                **args
            )
            trainer = Trainer(run_args)
            trainer.run(input_x, input_y)

        elif phase == "test":
            input_x, subjects = loader.get_data(data_type)
            run_args = dict(
                subjects=subjects,
                input_x=input_x,
                test_type=data_type,
                **args
            )
            inferencer = Inferencer(run_args)
            inferencer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Osteoporosis Detection')
    parser.add_argument('--test', action='store_true', dest='test', help='Test module')
    p_args = parser.parse_args()

    phase = "test" if p_args.test else "train"
    args = args[phase]
    main()
