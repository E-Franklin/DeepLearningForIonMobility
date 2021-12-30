import argparse
from pathlib import Path

import torch

import wandb
from DataUtils import k0_to_ccs, ccs_to_k0

from predict import predict_values


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an existing model to predict ion mobility values."

    )

    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.0.1"
    )

    parser.add_argument('data_file',
                        help='The file containing the data.')
    parser.add_argument('model_path',
                        help='The path to the model to use for prediction.')
    parser.add_argument('config_file',
                        help='A .yaml containing the parameters for '
                             'constructing the model to load.')
    parser.add_argument('output_dir',
                        help='The directory where the output will be written.')
    parser.add_argument('--scaler',
                        help='The path to the scaler pkl file that was used '
                             'during training.')
    parser.add_argument('--transform',
                        help='The function to use to transform the predicted '
                             'values. Either k0_t0_ccs or ccs_to_k0',
                        choices=[ccs_to_k0, k0_to_ccs])

    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir).resolve()

    # config-defaults.yaml will be loaded automatically and any parameters may
    # be added or overwritten in the config file specified by the user.
    run = wandb.init(project="DeepLearningForIonMobility",
                     config=args.config_file,
                     reinit=True)

    # Check if CUDA is available on the system and use it if so.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update({'device': device})

    if args.transform == 'k0_to_ccs':
        args.transform = k0_to_ccs
    elif args.transform == 'ccs_to_k0':
        args.transform = ccs_to_k0

    predict_values(args.data_file, args.model_path, args.scaler,
                   args.output_dir, args.transform)

    run.finish()


if __name__ == "__main__":
    main()
