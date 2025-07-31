import torch
import lightning as L
import torchmetrics
import tqdm
from pathlib import Path
import os
import yaml
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
from enum import Enum
from sklearn.metrics import roc_curve, auc

from collections import defaultdict

class Metrics(Enum):
    ROC = "auroc"
    DICE = "dice"


class Report:
    def __init__(self, config: dict, model: torch.nn.Module, trainer: L.Trainer, reporting_root_folder: Path, dataset: torch.utils.data.Dataset, dataset_path: Path):
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
            Metrics.ROC : self.roc,
            Metrics.DICE : self.dice
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
        cubes = self.dataset_path.glob("*.cu3s")
        cube_names = [Path(cube).stem for cube in cubes]
        dataset_name = self.dataset_path.name
        test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        predictions = self.trainer.predict(self.model, test_loader)
        target_folder = self.reporting_run_folder / dataset_name
        if not target_folder.is_dir():
            target_folder.mkdir(parents=True, exist_ok=True)
        for j, batch in enumerate(tqdm.tqdm(test_loader, desc=f"creating images")):
            pred = predictions[j]
            if self.task == "anomaly":
                pass
                # TODO: labeling for anomalies
            if self.task == "segmentation":
                pass
            if self.create_images:
                inference_image, _ = self.create_inference_png(batch,
                                                               pred,
                                                               task=self.task)
                inference_image.savefig(target_folder / (batch["name"][0] + ".png"))
                plt.close(inference_image)
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
            rgb_images[rep] = (image[:,:,self.config["representations"][rep]])
            rgb_images[rep] = rgb_images[rep] / rgb_images[rep].max()
        nrows = len(rgb_images) + len(self.plot_thresholds) + 1
        fig_height = self.config["cube_size"][1] / self.dpi * (int(len(rgb_images)/2 +1) + len(self.plot_thresholds)) + 1
        fig_width = self.config["cube_size"][0] / self.dpi + 4
        fig, ax = plt.subplots(nrows, 2, tight_layout=True, dpi=self.dpi, figsize=(fig_width, fig_height))
        for axis in ax.flat:
            axis.set_visible(False)
            axis.get_xaxis().set_visible(False)
        for num, i in enumerate(rgb_images):
            ax[int(num/2)][num%2].imshow(rgb_images[i])
            ax[int(num/2)][num%2].set_title(i)
            ax[int(num/2)][num%2].set_visible(True)

        ax[int(len(rgb_images)/2 + 1)][0].imshow(batch["mask"].squeeze().detach().cpu().numpy(), cmap=cmap, vmin= 0, vmax=self.num_classes -1)
        ax[int(len(rgb_images)/2 + 1)][0].set_visible(True)
        ax[int(len(rgb_images)/2 + 1)][0].set_title("Ground Truth")
        for num, threshold in enumerate(self.plot_thresholds):
            if task == "segmentation":

                mask = torch.softmax(pred,dim=0)
                mask[mask < threshold] = 0
                threshold_img = torch.argmax(mask, dim=0)

            elif task == "anomaly":
                threshold_img = torch.softmax(pred,dim=0)
                threshold_img[threshold_img < threshold] = 0
                threshold_img[threshold_img > threshold] = 1


            num = num + int(len(rgb_images) / 2) + 2
            ax[num][0].imshow(threshold_img, cmap = cmap, vmin= 0, vmax=self.num_classes -1)
            ax[num][0].set_title(f'Threshold image ({threshold})')
            ax[num][0].set_visible(True)

            ax[num][1].imshow(rgb_images['RGB'])
            ax[num][1].imshow(threshold_img, cmap = cmap, alpha=0.3, vmin= 0, vmax=self.num_classes -1)
            ax[num][1].set_title(f'Overlay image (TH: {threshold})')
            ax[num][1].set_visible(True)
        return fig, ax

    def roc(self, test_loader, predictions):
        class_scores = defaultdict(list)
        class_truths = defaultdict(list)
        present_classes = set()
        if self.task == "segmentation":
            # TODO: validate result, seems way to good
            roc = torchmetrics.classification.MulticlassROC(self.num_classes)
            auroc = torchmetrics.classification.MulticlassAUROC(self.num_classes)
            for i, batch in enumerate(test_loader):
                roc.update(predictions[i].unsqueeze(0).cuda(),batch["mask"])
                auroc.update(predictions[i].unsqueeze(0).cuda(),batch["mask"])
            roc_fig, _ = roc.plot(score=True, labels=self.class_map)
            auroc_res = auroc.compute()
            return float(auroc_res), roc_fig

        # TODO: test this case
        elif self.task == "anomaly":
            for i, batch in enumerate(test_loader):
                pred = predictions[i].cuda()
                gt_mask = batch["mask"]

                for cls in torch.unique(gt_mask):
                    if cls == 0:
                        continue # skip background
                present_classes.add(cls)
                mask = (gt_mask == cls)
                class_scores[cls].append(pred[mask])
                class_truths[cls].append(np.ones(np.count_nonzero(mask)))
                # Add negative samples
                neg_mask = (gt_mask == 0)
                class_scores[cls].append(pred[neg_mask])
                class_truths[cls].append(np.zeros(np.count_nonzero(neg_mask)))

        plt.figure(figsize=(7, 7))
        aucs = {}

        for cls in sorted(present_classes):
            if cls not in class_scores:
                continue

            y_scores = np.concatenate(class_scores[cls])
            y_true = np.concatenate(class_truths[cls])

            if len(np.unique(y_true)) < 2:
                print(f"Skipping class {cls}: insufficient positive/negative pixels")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            aucs[cls] = roc_auc

            cls_label = cls
            plt.plot(fpr, tpr, label=f"{cls_label} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Per-Class Pixel-Level ROC Curve")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(self.reporting_run_folder, exist_ok=True)
        plt.savefig(f'{self.reporting_run_folder}/AUROC_Class.png', dpi=300, bbox_inches="tight")
        plt.close()
    def dice(self, test_loader, predictions):
        ground_truths = torch.tensor([], device="cuda")
        for batch in test_loader:
            ground_truths = torch.cat((ground_truths, batch['mask']), 0)
        pred = torch.argmax(torch.tensor(np.array(predictions)), dim=1, keepdim=False)
        one_hot_pred = torch.nn.functional.one_hot(pred.type(torch.long), num_classes=self.num_classes).movedim(-1, 1)
        one_hot_gt = torch.nn.functional.one_hot(ground_truths.type(torch.long), num_classes=self.num_classes).movedim(-1, 1)
        dice_score = torchmetrics.functional.dice(one_hot_pred.cuda(), one_hot_gt,ignore_index=0)
        return float(dice_score), None
    def per_class_auroc(self, test_loader, predictions):
        # TODO: implement per class anomaly auroc
        for label, label_name in enumerate(self.class_map):
            for batch_num, batch in enumerate(tqdm.tqdm(test_loader, desc=f'calculating per class AUROC for {label_name}')):
                mask = batch['mask']
                mask[mask != label] = 0
                prediction = predictions[batch_num]
                prediction[prediction != label] = 0
        pass


    # TODO: implement over all anomaly auroc


    def per_class_dice(self, test_loader, predictions):
        # TODO: implemente per class dice
        pass

