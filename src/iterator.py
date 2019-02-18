import numpy as np

# Convinent class for iterating over double digit dataset found in data folder
class DatasetIterator(object):

  def __init__(self, batch_size=64):
    self.batch_size = batch_size
    self.num_samples = 55000
    self.next_batch_pointer = 0
    self.b1 = None
    self.b2 = None
    self.b3 = None
    self.b4 = None
    # load data
    self.train_bboxes = np.load('../data/train_bboxes.npy')
    self.train_X = np.load('../data/train_X.npy').reshape((55000, 64, 64, 1))
    self.train_Y = np.load('../data/train_Y.npy')
    self.labels_Y = self.convert_to_labels(self.train_Y, self.num_samples)
    # shuffle data
    self.shuffle()

  # shuffle samples
  def shuffle(self):
    image_indices = np.random.permutation(np.arange(self.num_samples))
    self.train_X = self.train_X[image_indices]
    self.train_Y = self.train_Y[image_indices]
    self.labels_Y = self.labels_Y[image_indices]
    self.train_bboxes = self.train_bboxes[image_indices]

  # convert image classification [x,y] to an integer between 0 and 55
  # 55 possible 2 digit combinations
  def convert_to_labels(self, Y, size):
    labels = np.zeros(size)
    for i in range(size):
      start_index = 0
      cur_value = 10
      for j in range(Y[i][0]):
        start_index += cur_value
        cur_value =  cur_value - 1
      labels[i] = start_index + Y[i][1] - Y[i][0]
    return labels

  # get next batch
  def get_next_batch(self):
    num_samples_left = self.num_samples - self.next_batch_pointer
    if num_samples_left >= self.batch_size:
      x_batch = self.train_X[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      y_batch = self.labels_Y[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      z_batch = self.train_bboxes[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
      self.next_batch_pointer += self.batch_size
    else:
      self.next_batch_pointer = 0
      x_batch, y_batch, z_batch = self.get_next_batch()

    return x_batch, y_batch.reshape((self.batch_size, 1)), z_batch

# convert integer between 0 and 55 back to original label [x,y]
# 55 possible 2 digit combinations
def covert_from_labels(labels, size):
    Y = np.zeros((size, 2))
    # all possible 2-digit combinations
    inx = [[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8], [0,9],
           [1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9],
           [2,2], [2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9],
           [3,3], [3,4], [3,5], [3,6], [3,7], [3,8], [3,9],
           [4,4], [4,5], [4,6], [4,7], [4,8], [4,9],
           [5,5], [5,6], [5,7], [5,8], [5,9],
           [6,6], [6,7], [6,8], [6,9],
           [7,7], [7,8], [7,9],
           [8,8], [8,8],
           [9,9]]
    for i in range(size):
      Y[i] = inx[labels[i]]
    return Y
