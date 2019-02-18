import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from iterator import DatasetIterator, covert_from_labels

def net(input, is_training):
  # Hyperparameters
  mu = 0
  sigma = 0.01

  # Layer 1: Convolutional. Input = 64x64x1. Output = 32x32x6.
  conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name='conv1_W')
  conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
  conv1   = tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

  # Relu Activation.
  conv1 = tf.nn.relu(conv1)

  # Densenet-like connection: Convolutional. Input = 64x64x6. Output = 64x64x16.
  convs_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 7, 16), mean = mu, stddev = sigma), name='convs_W')
  convs_b = tf.Variable(tf.zeros(16), name='conv2_s')
  convs   = tf.nn.conv2d(tf.concat([input,conv1],axis=3), convs_W, strides=[1, 1, 1, 1], padding='SAME') + convs_b

  # Activation.
  convs = tf.nn.relu(convs)

  # Pooling. Input = 64x64x16. Output = 32x32x16.
  convs = tf.nn.max_pool(convs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  # Layer 3: Convolutional. Output = 28x28x36.
  conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 36), mean = mu, stddev = sigma), name='conv2_W')
  conv2_b = tf.Variable(tf.zeros(36), name='conv2_b')
  conv2   = tf.nn.conv2d(convs, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

  # Batch normalization Output = 28x28x36.
  conv2 = tf.layers.batch_normalization(conv2,training=is_training)

  # Activation.
  conv2 = tf.nn.relu(conv2)

  # Pooling. Input = 28x28x36. Output = 14x14x36.
  conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  # Layer 4: Convolutional. Input = 14x14x36. Output = 14x14x72.
  conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 36, 92), mean = mu, stddev = sigma), name='conv3_W')
  conv3_b = tf.Variable(tf.zeros(92), name='conv3_b')
  conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b

  # Batch normalization Output = 14x14x72.
  conv3 = tf.layers.batch_normalization(conv3,training=is_training)

  # Activation.
  conv3 = tf.nn.relu(conv3)

  # Pooling. Input = 14x14x72. Output = 7x7x72.
  conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  # Layer 5: Convolutional. Input = 7x7x72. Output = 7x7x148.
  conv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 92, 148), mean = mu, stddev = sigma), name='conv4_W')
  conv4_b = tf.Variable(tf.zeros(148), name='conv4_b')
  conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b

  # Batch normalization Output = 7x7x148.
  conv4 = tf.layers.batch_normalization(conv4,training=is_training)

  # Activation.
  conv4 = tf.nn.relu(conv4)

  # Pooling. Input = 7x7x148. Output = 3x3x148.
  conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  # Flatten. Input =  3x3x148. Output = 1332.
  fc0 = flatten(conv4)

  ######################### LABELS (l1) #############################

  # Layer 4: Fully Connected. Input = 1332. Output = 660.
  fc1_W = tf.Variable(tf.truncated_normal(shape=(1332, 660), mean = mu, stddev = sigma), name='fc1_W')
  fc1_b = tf.Variable(tf.zeros(660), name='fc1_b')
  fc1   = tf.matmul(fc0, fc1_W) + fc1_b

  # Activation.
  fc1 = tf.nn.relu(fc1)

  # Layer 5: Fully Connected. Input = 660. Output = 196.
  fc2_W  = tf.Variable(tf.truncated_normal(shape=(660, 196), mean = mu, stddev = sigma), name='fc2_W')
  fc2_b  = tf.Variable(tf.zeros(196), name='fc2_b')
  fc2    = tf.matmul(fc1, fc2_W) + fc2_b

  # Activation.
  fc2    = tf.nn.relu(fc2)

  # Layer 6: Fully Connected. Input = 196. Output = 55.
  fc3_W  = tf.Variable(tf.truncated_normal(shape=(196, 55), mean = mu, stddev = sigma), name='fc3_W')
  fc3_b  = tf.Variable(tf.zeros(55), name='fc3_b')
  l1 = tf.matmul(fc2, fc3_W) + fc3_b

  ######################### BBOX 1 (l2) #############################

  # Layer 4: Fully Connected. Input = 1332. Output = 660.
  fc1_W = tf.Variable(tf.truncated_normal(shape=(1332, 660), mean = mu, stddev = sigma), name='fc1_W')
  fc1_b = tf.Variable(tf.zeros(660), name='fc1_b')
  fc1   = tf.matmul(fc0, fc1_W) + fc1_b

  # Activation.
  fc1 = tf.nn.relu(fc1)

  # Layer 5: Fully Connected. Input = 660. Output = 196.
  fc2_W  = tf.Variable(tf.truncated_normal(shape=(660, 196), mean = mu, stddev = sigma), name='fc2_W')
  fc2_b  = tf.Variable(tf.zeros(196), name='fc2_b')
  fc2    = tf.matmul(fc1, fc2_W) + fc2_b

  # Activation.
  fc2    = tf.nn.relu(fc2)

  # Layer 6: Fully Connected. Input = 196. Output = 55.
  fc3_W  = tf.Variable(tf.truncated_normal(shape=(196, 37), mean = mu, stddev = sigma), name='fc3_W')
  fc3_b  = tf.Variable(tf.zeros(37), name='fc3_b')
  l2 = tf.matmul(fc2, fc3_W) + fc3_b

  ######################### BBOX 2 (l3) #############################

  # Layer 4: Fully Connected. Input = 1332. Output = 660.
  fc1_W = tf.Variable(tf.truncated_normal(shape=(1332, 660), mean = mu, stddev = sigma), name='fc1_W')
  fc1_b = tf.Variable(tf.zeros(660), name='fc1_b')
  fc1   = tf.matmul(fc0, fc1_W) + fc1_b

  # Activation.
  fc1 = tf.nn.relu(fc1)

  # Layer 5: Fully Connected. Input = 660. Output = 196.
  fc2_W  = tf.Variable(tf.truncated_normal(shape=(660, 196), mean = mu, stddev = sigma), name='fc2_W')
  fc2_b  = tf.Variable(tf.zeros(196), name='fc2_b')
  fc2    = tf.matmul(fc1, fc2_W) + fc2_b

  # Activation.
  fc2    = tf.nn.relu(fc2)

  # Layer 6: Fully Connected. Input = 196. Output = 55.
  fc3_W  = tf.Variable(tf.truncated_normal(shape=(196, 37), mean = mu, stddev = sigma), name='fc3_W')
  fc3_b  = tf.Variable(tf.zeros(37), name='fc3_b')
  l3 = tf.matmul(fc2, fc3_W) + fc3_b

  ######################### BBOX 3 (l4) #############################

  # Layer 4: Fully Connected. Input = 1332. Output = 660.
  fc1_W = tf.Variable(tf.truncated_normal(shape=(1332, 660), mean = mu, stddev = sigma), name='fc1_W')
  fc1_b = tf.Variable(tf.zeros(660), name='fc1_b')
  fc1   = tf.matmul(fc0, fc1_W) + fc1_b

  # Activation.
  fc1 = tf.nn.relu(fc1)

  # Layer 5: Fully Connected. Input = 660. Output = 196.
  fc2_W  = tf.Variable(tf.truncated_normal(shape=(660, 196), mean = mu, stddev = sigma), name='fc2_W')
  fc2_b  = tf.Variable(tf.zeros(196), name='fc2_b')
  fc2    = tf.matmul(fc1, fc2_W) + fc2_b

  # Activation.
  fc2    = tf.nn.relu(fc2)

  # Layer 6: Fully Connected. Input = 196. Output = 55.
  fc3_W  = tf.Variable(tf.truncated_normal(shape=(196, 37), mean = mu, stddev = sigma), name='fc3_W')
  fc3_b  = tf.Variable(tf.zeros(37), name='fc3_b')
  l4 = tf.matmul(fc2, fc3_W) + fc3_b

  ######################### BBOX 4 (l5) #############################

  # Layer 4: Fully Connected. Input = 1332. Output = 660.
  fc1_W = tf.Variable(tf.truncated_normal(shape=(1332, 660), mean = mu, stddev = sigma), name='fc1_W')
  fc1_b = tf.Variable(tf.zeros(660), name='fc1_b')
  fc1   = tf.matmul(fc0, fc1_W) + fc1_b

  # Activation.
  fc1 = tf.nn.relu(fc1)

  # Layer 5: Fully Connected. Input = 660. Output = 196.
  fc2_W  = tf.Variable(tf.truncated_normal(shape=(660, 196), mean = mu, stddev = sigma), name='fc2_W')
  fc2_b  = tf.Variable(tf.zeros(196), name='fc2_b')
  fc2    = tf.matmul(fc1, fc2_W) + fc2_b

  # Activation.
  fc2    = tf.nn.relu(fc2)

  # Layer 6: Fully Connected. Input = 196. Output = 55.
  fc3_W  = tf.Variable(tf.truncated_normal(shape=(196, 37), mean = mu, stddev = sigma), name='fc3_W')
  fc3_b  = tf.Variable(tf.zeros(37), name='fc3_b')
  l5 = tf.matmul(fc2, fc3_W) + fc3_b

  return l1, l2, l3, l4, l5


def test(train=True, prefix="test"):
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()

  EPOCHS = 100
  N_BATCHES = 275 # 55,000 / 200
  BATCH_SIZE = 200

  # placeholders for images (x) and labels (y,b1,b2,b3,b4)
  x = tf.placeholder(tf.float32, (None, 64, 64, 1))
  y = tf.placeholder(tf.int32, (None))
  b1 = tf.placeholder(tf.int32, (None))
  b2 = tf.placeholder(tf.int32, (None))
  b3 = tf.placeholder(tf.int32, (None))
  b4 = tf.placeholder(tf.int32, (None))

  # one-hot conversion
  one_hot_y = tf.one_hot(y, 55)
  one_hot_b1 = tf.one_hot(b1, 37)
  one_hot_b2 = tf.one_hot(b2, 37)
  one_hot_b3 = tf.one_hot(b3, 37)
  one_hot_b4 = tf.one_hot(b4, 37)

  rate = 0.0002

  is_training = True

  l1, l2, l3, l4, l5 = net(x, is_training)

  saver = tf.train.Saver(max_to_keep=0)

  # optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate = rate)

  #L1
  cross_entropy_l1 = tf.nn.softmax_cross_entropy_with_logits(logits=l1, labels=one_hot_y)
  loss_operation_l1 = tf.reduce_mean(cross_entropy_l1)
  grads_and_vars_l1 = optimizer.compute_gradients(loss_operation_l1, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation_l1 = optimizer.apply_gradients(grads_and_vars_l1)

  #L2
  cross_entropy_l2 = tf.nn.softmax_cross_entropy_with_logits(logits=l2, labels=one_hot_b1)
  loss_operation_l2 = tf.reduce_mean(cross_entropy_l2)
  grads_and_vars_l2 = optimizer.compute_gradients(loss_operation_l2, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation_l2 = optimizer.apply_gradients(grads_and_vars_l2)

  #L3
  cross_entropy_l3 = tf.nn.softmax_cross_entropy_with_logits(logits=l3, labels=one_hot_b2)
  loss_operation_l3 = tf.reduce_mean(cross_entropy_l3)
  grads_and_vars_l3 = optimizer.compute_gradients(loss_operation_l3, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation_l3 = optimizer.apply_gradients(grads_and_vars_l3)

  #L4
  cross_entropy_l4 = tf.nn.softmax_cross_entropy_with_logits(logits=l4, labels=one_hot_b3)
  loss_operation_l4 = tf.reduce_mean(cross_entropy_l4)
  grads_and_vars_l4 = optimizer.compute_gradients(loss_operation_l4, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation_l4 = optimizer.apply_gradients(grads_and_vars_l4)

  #L5
  cross_entropy_l5 = tf.nn.softmax_cross_entropy_with_logits(logits=l5, labels=one_hot_b4)
  loss_operation_l5 = tf.reduce_mean(cross_entropy_l5)
  grads_and_vars_l5 = optimizer.compute_gradients(loss_operation_l5, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation_l5 = optimizer.apply_gradients(grads_and_vars_l5)

  training_operation = [training_operation_l1,training_operation_l2,training_operation_l3,training_operation_l4,training_operation_l5]

  # Training

  mnistdd_train = DatasetIterator(batch_size=BATCH_SIZE)

  if train:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      print("Training...")
      global_step = 0
      for i in range(EPOCHS):
          for iteration in range(N_BATCHES):
              batch_x, batch_y, batch_z = mnistdd_train.get_next_batch() # get the next batch
              _ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, b1: batch_z[:,0][:,0], b2: batch_z[:,0][:,1], b3: batch_z[:,1][:,0], b4: batch_z[:,1][:,1]})
              global_step += 1
              print("iteration {} ...".format(iteration))

          print("EPOCH {} ...".format(i+1))

      saver.save(sess, '../ckpt/net', global_step=1)
      print("Model saved")

  # Testing
  print("testing...")
  test_data = np.load("../data/" + prefix + "_X.npy")
  num_images = len(test_data)
  test_data = test_data.reshape((num_images, 64, 64, 1))

  with tf.Session() as sess:
    # load model
    saver.restore(sess, tf.train.latest_checkpoint('../ckpt'))

    # l1 Accuracy - predicts the two digits in the image
    l1_test = sess.run(tf.argmax(l1, axis=1), feed_dict={x:  test_data})
    # l2 Accuracy - predicts x-coord of first digit
    l2_test = sess.run(tf.argmax(l2, axis=1), feed_dict={x:  test_data})
    # l3 Accuracy - predicts y-coord of first digit
    l3_test = sess.run(tf.argmax(l3, axis=1), feed_dict={x:  test_data})
    # l4 Accuracy - predicts x-coord of second digit
    l4_test = sess.run(tf.argmax(l4, axis=1), feed_dict={x:  test_data})
    # l5 Accuracy - predicts y-coord of second digit
    l5_test = sess.run(tf.argmax(l5, axis=1), feed_dict={x:  test_data})

    # get labels back to orginal format
    pred_class = covert_from_labels(l1_test, num_images)

    # init space for predicted bboxes
    pred_bboxes = np.zeros((num_images,2,4))

    # assign coordinates back to original format
    pred_bboxes[:,0][:,0] = l2_test
    pred_bboxes[:,0][:,1] = l3_test
    pred_bboxes[:,1][:,0] = l4_test
    pred_bboxes[:,1][:,1] = l5_test
    pred_bboxes[:,0][:,2] = l2_test + 28
    pred_bboxes[:,0][:,3] = l3_test + 28
    pred_bboxes[:,1][:,2] = l4_test + 28
    pred_bboxes[:,1][:,3] = l5_test + 28

    return np.array(pred_class), np.array(pred_bboxes)
