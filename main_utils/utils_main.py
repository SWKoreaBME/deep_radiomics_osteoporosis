from os.path import isfile, join
from os import makedirs


def args_save_to_txt(save_file_path, args):
    from datetime import date
    with open(save_file_path, 'w') as f:
        f.write(f"Date: {date.today()}\n\n")
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
        f.close()

    assert isfile(save_file_path)


def log_args(args, show=True):

    # Show all running arguments
    if show:
        for key, value in args.items():
            print(f"{key}: {value}")

    # Save argument parse arguments
    makedirs(args["save_dir"], exist_ok=True)

    # Save version log
    args_save_to_txt(join(args["save_dir"], "args.txt"), args)
