import argparse
import yaml
import lightning as L
from FreshTwin_lightning import FreshTwin_lightning
from torch.utils.data import DataLoader
from FreshTwinCuvisDataset import FreshTwinCuvisDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
#from torch_pca import PCA
from sklearn.decomposition import IncrementalPCA as PCA
import torch


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = get_arguments()
    config = parse_args(args)

    full_dataset = FreshTwinCuvisDataset(Path(config["data_path"]), white_path=config["white_path"], dark_path=config["dark_path"], in_channels=config["in_channels"])

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0, persistent_workers=False)

    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0, persistent_workers=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="train/epoch_loss",  # Metric to monitor
        dirpath=config["ckpt_dir"] + "/" + config["name"],  # Directory to save checkpoints
        filename=config["name"] + "-{epoch:02d}-{train/epoch_loss:.2f}",  # Filename format
        save_top_k=-1,  # Save all checkpoints
        mode="min",
        verbose=True,
    )
    logger = TensorBoardLogger(save_dir=config["logger_dir"], log_graph=True, name=config['name'])

    model = FreshTwin_lightning(config)

    trainer = L.Trainer(logger=logger,
                        max_steps=config["max_steps"],
                        benchmark=True,
                        precision='16-mixed',
                        gradient_clip_val=0.5,
                        callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    main()