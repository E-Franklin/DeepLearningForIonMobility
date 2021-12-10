import argparse
from datetime import datetime

import torch

import wandb
from train import train_model
from pathlib import Path


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trains a deep learning model according to the parameters "
                    "specified in config-defaults.yaml and CONFIG_FILE."

    )

    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.0.1"
    )

    parser.add_argument('training_data',
                        help='The file containing the training data')
    parser.add_argument('testing_data',
                        help='The file containing the testing data')
    parser.add_argument('output_dir',
                        help='The directory where the output will be written.')
    parser.add_argument('config_file',
                        help='A .yaml containing the parameters for training.')
    parser.add_argument('--model_name',
                        help='The filename to use for saving the model.',
                        default=f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")

    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    # make necessary output directories if they don't exist
    model_path = Path(f'{args.output_dir}/models').resolve()
    if not model_path.exists():
        model_path.mkdir()

    outpath = Path(f'{args.output_dir}/output_data').resolve()
    if not outpath.exists():
        outpath.mkdir()

    # config-defaults.yaml will be loaded automatically and any parameters may
    # be added or overwritten in the config file specified by the user.
    run = wandb.init(project="DeepLearningForIonMobility",
                     name=args.model_name,
                     config=args.config_file,
                     reinit=True)

    # add the model_name to the config
    wandb.config.update({'model_name': args.model_name})
    # add command line arguments to wandb.config
    wandb.config.update(args)

    # Check if CUDA is available on the system and use it if so.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update({'device': device})

    print(wandb.config)
    
    print(f"Start training at {datetime.now()}")
    train_model()
    print(f"Training completed at {datetime.now()}")

    run.finish()


if __name__ == "__main__":
    main()
