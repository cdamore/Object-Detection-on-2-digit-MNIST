from net import test
from eval import evaluation

if __name__ == '__main__':
  # get predicted classes and bboxes from test data in data folder
  # if train = true, a new model will be trained
  # if train = false, the saved model from ckpt will be used on the test set
  pred_class, pred_bboxes = test(train=False)
  # evaluate the model
  evaluation(pred_class, pred_bboxes)
