import torch.nn as nn
import torch


class MetricTracker:
    """
    Metric Tracker class to store relevant metrics during training or
    validation. Intermediate metrics or network outputs can be accumulated over
    a desired period of time and then turned into a metric representation for
    e.g. tensorboard logging. This can happen e.g. according to a specified
    frequency or at the end of one epoch.
    """
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.loss = 0
        self.loss_updates = 0

    def update(self, prediction, target, loss):
        """
        Update tracker state with new values.
        """

        sm_predictions = nn.Softmax(dim=1)(prediction)
        cls_predictions = torch.argmax(sm_predictions, dim=1)

        self.correct += torch.sum(cls_predictions == target).float()
        self.incorrect += torch.sum(cls_predictions != target).float()

        self.loss += loss
        self.loss_updates += 1

    def get_metrics(self):
        """
        Turn tracker state into metric representation for e.g. logging purposes
        or result visualization.
        """

        accuracy = self.correct / (self.correct + self.incorrect)
        loss = self.loss / self.loss_updates

        return {'acc': accuracy, 'loss': loss}

    def reset(self):
        """
        Reset accumulated values at the end of a tracking period.
        """

        self.correct = 0
        self.incorrect = 0
        self.loss = 0
        self.loss_updates = 0