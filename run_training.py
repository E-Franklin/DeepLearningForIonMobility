import argparse
import wandb
from datetime import datetime
from train import train_model
import torch


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
    parser.add_argument('output_folder',
                        help='The directory where the output will be written.')
    parser.add_argument('config_file',
                        help='A .yaml containing the parameters for training.')

    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    # create a name for the model. Default is model_datetime
    model_name = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    # config-defaults.yaml will be loaded automatically and any parameters may
    # be added or overwritten in the config file specified by the user.
    run = wandb.init(project="DeepLearningForIonMobility",
                     name=model_name,
                     config=args.config_file,
                     reinit=True)

    # add the model_name to the config
    wandb.config.update({'model_name': model_name})
    # add command line arguments to wandb.config
    wandb.config.update(args)

    # Check if CUDA is available on the system and use it if so.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update({'device': device})

    print(f"Start training at {datetime.now()}")
    train_model()
    print(f"Training completed at {datetime.now()}")

    run.finish()


if __name__ == "__main__":
    main()
