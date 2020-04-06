

class MetricTracker:
    """
    Metric Tracker class to store relevant metrics during training or
    validation. Intermediate metrics or network outputs can be accumulated over
    a desired period of time and then turned into a metric representation for
    e.g. tensorboard logging. This can happen e.g. according to a specified
    frequency or at the end of one epoch.
    """
    def __init__(self):
        super(MetricTracker, self).__init__()

    def update(self):
        """
        Update tracker state with new values.
        """
        pass

    def get_metrics(self):
        """
        Turn tracker state into metric representation for e.g. logging purposes
        or result visualization.
        """
        pass

    def reset(self):
        """
        Reset accumulated values at the end of a tracking period.
        """
