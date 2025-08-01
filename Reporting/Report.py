import torch
import lightning as L
import torchmetrics
import tqdm
from pathlib import Path
import os
import yaml
from sympy.physics.control.control_plots import matplotlib
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
from enum import Enum
from sklearn.metrics import roc_curve, auc

from collections import defaultdict

from torch.xpu import device


class Metrics(Enum):
    ROC = "auroc"
    DICE = "dice"
    PER_CLASS_ANOMALY_ROC = "per_class_anomaly_auroc"
    PER_CLASS_DICE = "per_class_dice"


class Report:
    def __init__(self,
                 config: dict,
                 model: torch.nn.Module,
                 trainer: L.Trainer,
                 reporting_root_folder: Path,
                 dataset: torch.utils.data.Dataset,
                 dataset_path: Path):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.reporting_root_folder = reporting_root_folder
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.name = config["name"]
        self.reporting_run_folder = reporting_root_folder / self.name
        self.task = config["task"]
        if self.task == "binary":
            self.num_classes = 2
        else:
            self.num_classes = config["num_classes"]

        self.plot_thresholds = config['plot_thresholds']

        self.metrics = {
            Metrics.ROC: self.roc,
            Metrics.DICE: self.dice,
            Metrics.PER_CLASS_ANOMALY_ROC: self.per_class_anomaly_auroc,
            Metrics.PER_CLASS_DICE: self.per_class_dice
        }
        self.metrics_to_calc = config['metrics_to_calc'] if 'metrics_to_calc' in config else []
        self.create_images = config['create_images'] if 'create_images' in config else True
        self.class_map = config['class_map'] if 'class_map' in config else []
        self.dpi = 150

    def generate_report(self):
        if not os.path.exists(self.reporting_run_folder):
            os.makedirs(self.reporting_run_folder)
        with open(self.reporting_run_folder / "reporting_config.yaml", "w") as f:
            yaml.dump(self.config, f)

        metrics = {}
        dataset_name = self.dataset_path.name
        test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        predictions = self.trainer.predict(self.model, test_loader)
        target_folder = self.reporting_run_folder / dataset_name
        if not target_folder.is_dir():
            target_folder.mkdir(parents=True, exist_ok=True)
        if self.create_images:
            for j, batch in enumerate(tqdm.tqdm(test_loader, desc=f"creating images")):
                pred = predictions[j]
                if self.task == "anomaly":
                    pass
                    # TODO: labeling for anomalies
                if self.task == "segmentation":
                    inference_image, _ = self.create_inference_png(batch, pred, task=self.task)
                    inference_image.savefig(target_folder / (batch["name"][0] + ".png"))
                    plt.close(inference_image)

        # calculate metrics and save them into a yaml file
        for metric in self.metrics_to_calc:
            if Metrics(metric) not in self.metrics:
                raise ValueError(f"Metric {metric} not defined")
            else:
                metric_result, metric_plt = self.metrics[Metrics(metric)](test_loader, predictions)
                metrics[metric] = metric_result
                if metric_plt:
                    metric_plt.savefig(target_folder / (metric + ".png"))
        with open(self.reporting_run_folder / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

    def create_inference_png(self, batch, pred, task: str):
        rgb_images = {}
        image = batch["image"].squeeze().detach().cpu().permute(1, 2, 0).numpy()
        base_cmap = cm.get_cmap('tab20')  # , self.num_classes)
        colors = base_cmap(np.linspace(0, 1, self.num_classes))
        cmap = ListedColormap(base_cmap.colors[0:self.num_classes])
        cmap = base_cmap
        cmap = ListedColormap(colors)
        for rep in self.config["representations"]:
            rgb_images[rep] = (image[:, :, self.config["representations"][rep]])
            rgb_images[rep] = rgb_images[rep] / rgb_images[rep].max()
        nrows = len(rgb_images) + len(self.plot_thresholds) + 1
        fig_height = self.config["cube_size"][1] / self.dpi * (int(len(rgb_images) / 2 + 1) + len(self.plot_thresholds)) + 1
        fig_width = self.config["cube_size"][0] / self.dpi + 4
        fig, ax = plt.subplots(nrows, 2, tight_layout=True, dpi=self.dpi, figsize=(fig_width, fig_height))
        for axis in ax.flat:
            axis.set_visible(False)
            axis.get_xaxis().set_visible(False)
        for num, i in enumerate(rgb_images):
            ax[int(num / 2)][num % 2].imshow(rgb_images[i])
            ax[int(num / 2)][num % 2].set_title(i)
            ax[int(num / 2)][num % 2].set_visible(True)

        ax[int(len(rgb_images) / 2 + 1)][0].imshow(batch["mask"].squeeze().detach().cpu().numpy(), cmap=cmap, vmin=0, vmax=self.num_classes - 1)
        ax[int(len(rgb_images) / 2 + 1)][0].set_visible(True)
        ax[int(len(rgb_images) / 2 + 1)][0].set_title("Ground Truth")
        for num, threshold in enumerate(self.plot_thresholds):
            if task == "segmentation":

                mask = torch.softmax(pred, dim=0)
                mask[mask < threshold] = 0
                threshold_img = torch.argmax(mask, dim=0)

            elif task == "anomaly":
                threshold_img = torch.softmax(pred, dim=0)
                threshold_img[threshold_img < threshold] = 0
                threshold_img[threshold_img > threshold] = 1

            num = num + int(len(rgb_images) / 2) + 2
            ax[num][0].imshow(threshold_img, cmap=cmap, vmin=0, vmax=self.num_classes - 1)
            ax[num][0].set_title(f'Threshold image ({threshold})')
            ax[num][0].set_visible(True)

            ax[num][1].imshow(rgb_images['RGB'])
            ax[num][1].imshow(threshold_img, cmap=cmap, alpha=0.3, vmin=0, vmax=self.num_classes - 1)
            ax[num][1].set_title(f'Overlay image (TH: {threshold})')
            ax[num][1].set_visible(True)
        return fig, ax

    def roc(self, data_loader, predictions) -> [float, plt.figure]:
        if self.task == "segmentation":
            roc = torchmetrics.classification.MulticlassROC(self.num_classes)
            auroc = torchmetrics.classification.MulticlassAUROC(self.num_classes)
            for i, batch in tqdm.tqdm(enumerate(data_loader), desc="Calculating Auroc", total=len(data_loader)):
                roc.update(torch.softmax(predictions[i].unsqueeze(0).cuda(), 1), batch["mask"])
                auroc.update(torch.softmax(predictions[i].unsqueeze(0).cuda(), 1), batch["mask"])
            roc_fig, _ = roc.plot(score=True, labels=self.class_map)
            auroc_res = auroc.compute()
            return float(auroc_res), roc_fig

        elif self.task == "anomaly":
            pass
            # TODO: implement this case

    def dice(self, data_loader, predictions) -> [float, plt.figure]:
        dice_func = torchmetrics.Dice(ignore_index=0).to("cuda")
        cum_sum = 0
        total_sum = 0
        for i, batch in tqdm.tqdm(enumerate(data_loader), desc="Calculating Dice", total=len(data_loader)):
            pred = torch.argmax(predictions[i], dim=0, keepdim=False).unsqueeze(0).cuda()
            one_hot_pred = torch.nn.functional.one_hot(pred.type(torch.long), num_classes=self.num_classes).movedim(-1, 1)
            one_hot_gt = torch.nn.functional.one_hot(batch["mask"].type(torch.long), num_classes=self.num_classes).movedim(-1, 1)
            dice_func.update(one_hot_pred, one_hot_gt)
            diff = one_hot_gt != one_hot_pred.cuda()
            sum = torch.sum(diff)
            cum_sum = cum_sum + sum
            total_sum = total_sum + one_hot_pred.numel()

        dice_score = dice_func.compute()
        dice_func.reset()
        return float(dice_score), None

    def per_class_anomaly_auroc(self, data_loader, predictions) -> [float, plt.figure]:
        roc = torchmetrics.classification.MulticlassROC(num_classes=self.num_classes).to("cuda")
        auroc = torchmetrics.classification.MulticlassAUROC(num_classes=self.num_classes).to("cuda")
        for label, label_name in enumerate(self.class_map):
            for batch_num, batch in enumerate(tqdm.tqdm(data_loader, desc=f'calculating per class AUROC for {label_name}', total=len(data_loader))):
                mask = torch.zeros_like(batch['mask'], device="cuda")
                mask[batch["mask"] == label] = label
                prediction = torch.zeros_like(predictions[batch_num])
                prediction[mask == 1] = predictions[batch_num][mask == 1]
                roc.update(prediction.unsqueeze(0).cuda(), mask)
                auroc.update(prediction.unsqueeze(0).cuda(), mask)
        result = auroc.compute()
        auroc.reset()
        plot = roc.plot()
        roc.reset()
        return float(result), plot

    # TODO: implement over all anomaly auroc

    def per_class_dice(self, data_loader, predictions) -> [float, plt.figure]:
        all_dice = {}
        for label, label_name in enumerate(self.class_map):
            dice = torchmetrics.Dice(ignore_index=0).to("cuda")
            for batch_num, batch in enumerate(tqdm.tqdm(data_loader, desc=f'calculating per class Dice for {label_name}', total=len(data_loader))):
                mask = torch.zeros_like(batch['mask'], device="cuda")
                mask[batch["mask"] == label] = label if label > 0 else 1
                pred = torch.argmax(predictions[batch_num], 0).to("cuda")
                prediction = torch.zeros_like(pred, device="cuda")
                prediction[pred == label] = label if label > 0 else 1
                dice.update(prediction, mask)
            all_dice[label_name] = float(dice.compute())
            dice.reset()

        return all_dice, None
