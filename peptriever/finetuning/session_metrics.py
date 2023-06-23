import torch

from peptriever.metrics import (
    angular_batch_accuracy,
    angular_margin,
    angular_mean_margin,
    classification_accuracy,
    euclidean_accuracy,
    euclidean_margin,
    euclidean_mean_margin,
)


class SessionMetrics:
    def __init__(self):
        self._inner_metrics = {}
        self._batch_counter = 0

    def __setitem__(self, key, value):
        self._inner_metrics[key] = value

    def __getitem__(self, item):
        return self._inner_metrics[item]

    def __contains__(self, item):
        return item in self._inner_metrics

    def items(self):
        return self._inner_metrics.items()

    def keys(self):
        return self._inner_metrics.keys()

    def values(self):
        return self._inner_metrics.values()

    def reset(self):
        self._inner_metrics = {}
        self._batch_counter = 0

    def update(self, key, value):
        batch_i = self._batch_counter
        prev_value = self._inner_metrics.get(key)
        if prev_value is None:
            next_value = value
        else:
            next_value = (prev_value * batch_i + value) / (batch_i + 1)
        self[key] = next_value
        self._batch_counter += 1


class FinetuningSessionMetrics(SessionMetrics):
    def __init__(self, dist_metric: str):
        super().__init__()
        self.dist_metric = dist_metric
        if dist_metric == "angular":
            self.accuracy = angular_batch_accuracy
            self.margin = angular_margin
            self.mean_margin = angular_mean_margin
        elif dist_metric == "euclidean":
            self.accuracy = euclidean_accuracy
            self.margin = euclidean_margin
            self.mean_margin = euclidean_mean_margin
        else:
            raise NotImplementedError(dist_metric)

    def calc(self, outputs, labels):
        y = outputs["y1"], outputs["y2"]
        batch_accruacy = _to_float(self.accuracy(y))
        self.update("batch_accuracy", value=batch_accruacy)
        margin = _to_float(self.margin(y))
        self.update("margin", value=margin)
        mean_margin = _to_float(self.mean_margin(y))
        self.update("mean_margin", value=mean_margin)

        logits1, logits2 = outputs["scores1"], outputs["scores2"]
        labels1, labels2 = labels
        mlm_accuracy1 = classification_accuracy(labels1, logits1)
        mlm_accuracy2 = classification_accuracy(labels2, logits2)
        self.update("mlm_accuracy1", value=_to_float(mlm_accuracy1))
        self.update("mlm_accuracy2", value=_to_float(mlm_accuracy2))


def _to_float(tensor: torch.Tensor):
    return tensor.detach().cpu().item()
