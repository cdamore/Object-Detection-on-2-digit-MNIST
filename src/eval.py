import numpy as np
from skimage.draw import polygon

# function that computes accuracy given predicted classes and true classes
def compute_classification_acc(pred, gt):
  assert pred.shape == gt.shape
  return (pred == gt).astype(int).sum() / gt.size

# function that computes iou, given predicted bounding boxes and true bounding boxes
def compute_iou(b_pred,b_gt):
  # b_pred: predicted bounding boxes, shape=(n,2,4)
  # b_gt: ground truth bounding boxes, shape=(n,2,4)
  n = np.shape(b_gt)[0]
  L_pred = np.zeros((64,64))
  L_gt = np.zeros((64,64))
  iou = 0.0
  for i in range(n):
    for b in range(2):
      rr, cc = polygon([b_pred[i,b,0],b_pred[i,b,0],b_pred[i,b,2],b_pred[i,b,2]],
                   [b_pred[i,b,1],b_pred[i,b,3],b_pred[i,b,3],b_pred[i,b,1]],[64,64])
      L_pred[rr,cc] = 1

      rr, cc = polygon([b_gt[i,b,0],b_gt[i,b,0],b_gt[i,b,2],b_gt[i,b,2]],
                      [b_gt[i,b,1],b_gt[i,b,3],b_gt[i,b,3],b_gt[i,b,1]],[64,64])
      L_gt[rr,cc] = 1

      iou += (1.0/(2*n))*(np.sum((L_pred+L_gt)==2)/np.sum((L_pred+L_gt)>=1))

      L_pred[:,:] = 0
      L_gt[:,:] = 0

  return iou

# compute accuracy and iou of results given predicted classes and predicted bounding boxes
def evaluation(pred_class, pred_bboxes):
  gt_class = np.load("../data/test_Y.npy")
  gt_bboxes = np.load("../data/test_bboxes.npy")
  acc = compute_classification_acc(pred_class, gt_class)
  iou = compute_iou(pred_bboxes, gt_bboxes)
  print("Classification Acc: ", acc)
  print("BBoxes IOU: ", iou)
