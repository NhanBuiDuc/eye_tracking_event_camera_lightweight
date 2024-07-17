import torch
import torch.nn as nn
import numpy as np
from sinabs import SNNAnalyzer
from typing import Tuple, List

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

class YoloLoss(nn.Module):
    """
    Calculate the loss for Yolo (v1) model
    """

    def __init__(self, dataset_params, training_params):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (WiderFace is 1),
        """
        self.S = training_params["SxS_Grid"]
        self.B = training_params["num_boxes"]
        self.C = training_params["num_classes"]
        self.bbox_w = training_params["bbox_w"]
        self.img_width = dataset_params["img_width"]

        # Losses from Yolo Original Paper
        self.w_box_loss = training_params["w_box_loss"]
        self.w_conf_loss = training_params["w_conf_loss"]
        self.w_euclidian_loss = training_params["w_euclidian_loss"]
        self.w_iou_loss = 0

        self.box_loss = 0
        self.conf_loss = 0
        self.iou_loss = 0
        self.point_loss = 0
        self.total_loss = 0

        # Save last predictions and targets for loggings
        self.memory = {
            "distance": None,
            "points": {"target": None, "pred": None},
            "box": {"target": None, "pred": None},
        }

    def square_results(self, predictions):
        norm_pred1 = torch.zeros_like(predictions)
        point_1 = (
            predictions[..., :2] + (predictions[..., 2:] - predictions[..., :2]) / 2
        )
        norm_pred1[..., :2] = point_1 - self.bbox_w / self.img_width
        norm_pred1[..., 2:] = point_1 + self.bbox_w / self.img_width
        return norm_pred1

    def forward(self, predictions, target):
        if len(target.shape) == 5:
            target = target.flatten(end_dim=1)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Fix the bbox size
        predictions[..., (self.C + 1) : (self.C + 5)] = self.square_results(
            predictions[..., (self.C + 1) : (self.C + 5)]
        )
        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(
            predictions[..., (self.C + 1) : (self.C + 5)],
            target[..., (self.C + 1) : (self.C + 5)],
        )
        exists_box = target[..., self.C : (self.C + 1)]  # in paper this is Iobj_i
        box_targets = exists_box * target[..., (self.C + 1) : (self.C + 5)]

        if self.B == 2:
            predictions[..., (self.C + 6) : (self.C + 10)] = self.square_results(
                predictions[..., (self.C + 6) : (self.C + 10)]
            )
            iou_b2 = intersection_over_union(
                predictions[..., (self.C + 6) : (self.C + 10)],
                target[..., (self.C + 1) : (self.C + 5)],
            )
            ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
            iou_maxes, bestbox = torch.max(ious, dim=0)
            box_predictions = exists_box * (
                (
                    bestbox * predictions[..., (self.C + 6) : (self.C + 10)]
                    + (1 - bestbox) * predictions[..., (self.C + 1) : (self.C + 1 + 4)]
                )
            )
            conf_score = (
                bestbox * predictions[..., (self.C + 5) : (self.C + 6)]
                + (1 - bestbox) * predictions[..., self.C : (self.C + 1)]
            )
        else:
            box_predictions = exists_box * (
                predictions[..., (self.C + 1) : (self.C + 5)]
            )
            conf_score = predictions[..., self.C : (self.C + 1)]

        # bbox loss
        self.box_loss = (
            self.mse(
                torch.flatten(box_predictions, end_dim=-2),
                torch.flatten(box_targets, end_dim=-2),
            )
            .sum(1)
            .mean()
        )

        # conf_score is the confidence score for the bbox with highest IoU
        self.conf_loss = self.mse(
            torch.flatten(exists_box * conf_score),
            torch.flatten(exists_box * target[..., self.C : (self.C + 1)]),
        ).mean()

        # summary predictions
        pred_box = box_predictions.sum(-2).sum(
            -2
        )  # this works because we multiply by bestbox before
        pred_point = pred_box[..., :2] + (pred_box[..., 2:] - pred_box[..., :2]) / 2

        # summary target
        target_box = box_targets.sum(-2).sum(
            -2
        )  # this works because of the target transform
        target_point = (
            target_box[..., :2] + (target_box[..., 2:] - target_box[..., :2]) / 2
        )
        self.point_loss = torch.nn.PairwiseDistance(p=2)(
            pred_point, target_point
        ).mean()

        self.loss = {
            "box_loss": self.box_loss * self.w_box_loss,
            "conf_loss": self.conf_loss * self.w_conf_loss,
            "distance_loss": self.point_loss * self.w_euclidian_loss,
        }

        self.memory["box"]["target"] = target_box
        self.memory["box"]["pred"] = pred_box
        self.memory["points"]["target"] = target_point
        self.memory["points"]["pred"] = pred_point
        self.memory["distance"] = self.point_loss

        return sum(self.loss.values())
