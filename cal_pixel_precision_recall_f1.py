import numpy as np

class cal_pixel_precision_recall_f1:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.iou = 0.0
        self.iou_count = 0
        self.count = 0

    def update(self, label_trues_mask, label_preds_mask):
        assert label_preds_mask.shape == label_trues_mask.shape, 'mask 尺寸需一致'
        label_trues_mask[label_trues_mask > 0] = 1
        label_preds_mask[label_preds_mask > 0] = 1
        precision, recall, f1, iou, iou_count = self.pixel_cal(label_trues_mask, label_preds_mask)
        self.precision += precision
        self.recall += recall
        self.f1 += f1
        self.iou += iou
        self.iou_count += iou_count
        self.count += 1

    def pixel_cal(self, label_trues_mask, label_preds_mask):
        tp = np.sum(label_trues_mask * label_preds_mask)
        tn = np.sum((1 - label_trues_mask) * (1 - label_preds_mask))
        fp = np.sum(label_trues_mask * (1 - label_preds_mask))
        fn = np.sum((1 - label_trues_mask) * label_preds_mask)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        intersecion = tp
        union = np.sum(label_trues_mask) + np.sum(label_preds_mask) - intersecion
        iou = np.mean((intersecion+1e-5)/(union+1e-5))

        iou_count = 0
        if iou > self.threshold:
            iou_count = 1

        return precision, recall, f1, iou, iou_count

    def get_scores(self):
        if self.count == 0:
            self.count = 1
        self.precision /= self.count
        self.recall /= self.count
        self.f1 /= self.count
        self.iou /= self.count
        self.iou_count /= self.count
        return self.precision, self.recall, self.f1, self.iou, self.iou_count