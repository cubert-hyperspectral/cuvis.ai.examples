from pathlib import Path

import argparse
import yaml
import lightning as L

from Strawberry_lightning import Strawberry_lightning

from EfficientAD.reporting import Report
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config
if __name__ == '__main__':
    args = get_arguments()
    config = parse_args(args)
    model = Strawberry_lightning.load_from_checkpoint(config["checkpoint_to_load"], config=config)
    trainer = L.Trainer(inference_mode=True, precision='16-mixed')
    rep = Report(config, model, trainer, Path("../data/FT_reporting/"))
    rep.generate_report()
