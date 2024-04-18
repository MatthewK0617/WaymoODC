import glob
import math
import uuid
import time
import os

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

from model import ComplexModel, ComplexModel2, ComplexModelWithConfidence

# Given directory
tfrecord_directory = '/home/mk0617/mk1/waymo_open_dataset/uncompressed/tf_example/training/*'

# Constructing the pattern
file_pattern = os.path.join(tfrecord_directory)

# Use glob to create a list of file paths
FILENAMES = glob.glob(file_pattern)

# Print the number of files matched and the filenames (optional)
# print(f"Number of files matched: {len(FILENAMES)}")
# for filename in sorted(FILENAMES):  # Sorting the filenames to make them easier to read
#     print(filename)



# If you use a custom conversion from Scenario to tf.Example, set the correct
# number of map samples here.
num_map_samples = 30000

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir': tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
    'roadgraph_samples/id': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/type': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/valid': tf.io.FixedLenFeature(
        [num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/xyz': tf.io.FixedLenFeature(
        [num_map_samples, 3], tf.float32, default_value=None
    ),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

def _parse(value):
  decoded_example = tf.io.parse_single_example(value, features_description)

  past_states = tf.stack([
      decoded_example['state/past/x'], decoded_example['state/past/y'],
      decoded_example['state/past/length'], decoded_example['state/past/width'],
      decoded_example['state/past/bbox_yaw'],
      decoded_example['state/past/velocity_x'],
      decoded_example['state/past/velocity_y']
  ], -1)

  cur_states = tf.stack([
      decoded_example['state/current/x'], decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1)

  input_states = tf.concat([past_states, cur_states], 1)[..., :2]

  future_states = tf.stack([
      decoded_example['state/future/x'], decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

  past_is_valid = decoded_example['state/past/valid'] > 0
  current_is_valid = decoded_example['state/current/valid'] > 0
  future_is_valid = decoded_example['state/future/valid'] > 0
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1)

  # If a sample was not seen at all in the past, we declare the sample as
  # invalid.
  sample_is_valid = tf.reduce_any(
      tf.concat([past_is_valid, current_is_valid], 1), 1)

  inputs = {
      'input_states': input_states,
      'gt_future_states': gt_future_states,
      'gt_future_is_valid': gt_future_is_valid,
      'object_type': decoded_example['state/type'],
      'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
      'sample_is_valid': sample_is_valid,
  }
  return inputs


def _default_metrics_config():
  config = motion_metrics_pb2.MotionMetricsConfig()
  config_text = """
  track_steps_per_second: 10
  prediction_steps_per_second: 2
  track_history_samples: 10
  track_future_samples: 80
  speed_lower_bound: 1.4
  speed_upper_bound: 11.0
  speed_scale_lower: 0.5
  speed_scale_upper: 1.0
  step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
  }
  step_configurations {
    measurement_step: 9
    lateral_miss_threshold: 1.8
    longitudinal_miss_threshold: 3.6
  }
  step_configurations {
    measurement_step: 15
    lateral_miss_threshold: 3.0
    longitudinal_miss_threshold: 6.0
  }
  max_predictions: 6
  """
  text_format.Parse(config_text, config)
  return config

  

class MotionMetrics(tf.keras.metrics.Metric):
  """Wrapper for motion metrics computation."""

  def __init__(self, config):
    super().__init__()
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []
    self._metrics_config = config

  def reset_state():
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []

  def update_state(self, prediction_trajectory, prediction_score,
                   ground_truth_trajectory, ground_truth_is_valid,
                   prediction_ground_truth_indices,
                   prediction_ground_truth_indices_mask, object_type):
    self._prediction_trajectory.append(prediction_trajectory)
    self._prediction_score.append(prediction_score)
    self._ground_truth_trajectory.append(ground_truth_trajectory)
    self._ground_truth_is_valid.append(ground_truth_is_valid)
    self._prediction_ground_truth_indices.append(
        prediction_ground_truth_indices)
    self._prediction_ground_truth_indices_mask.append(
        prediction_ground_truth_indices_mask)
    self._object_type.append(object_type)

  def result(self):
    # [batch_size, num_preds, 1, 1, steps, 2].
    # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
    prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
    # [batch_size, num_preds, 1].
    prediction_score = tf.concat(self._prediction_score, 0)
    # [batch_size, num_agents, gt_steps, 7].
    ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
    # [batch_size, num_agents, gt_steps].
    ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices = tf.concat(
        self._prediction_ground_truth_indices, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices_mask = tf.concat(
        self._prediction_ground_truth_indices_mask, 0)
    # [batch_size, num_agents].
    object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

    # We are predicting more steps than needed by the eval code. Subsample.
    interval = (
        self._metrics_config.track_steps_per_second //
        self._metrics_config.prediction_steps_per_second)
    prediction_trajectory = prediction_trajectory[...,
                                                  (interval - 1)::interval, :]

    return py_metrics_ops.motion_metrics(
        config=self._metrics_config.SerializeToString(),
        prediction_trajectory=prediction_trajectory,
        prediction_score=prediction_score,
        ground_truth_trajectory=ground_truth_trajectory,
        ground_truth_is_valid=ground_truth_is_valid,
        prediction_ground_truth_indices=prediction_ground_truth_indices,
        prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
        object_type=object_type)


model = ComplexModelWithConfidence(num_agents_per_scenario=128, num_states_steps=11, num_future_steps=80)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()
metrics_config = _default_metrics_config()
motion_metrics = MotionMetrics(metrics_config)
metric_names = config_util.get_breakdown_names_from_motion_config(
    metrics_config)

# @tf.function
def train_step(inputs):
  with tf.GradientTape() as tape:
    # [batch_size, num_agents, D]
    states = inputs['input_states']

    # Predict. [batch_size, num_agents, steps, 2].
    pred_trajectory, _ = model(states, training=True)

    # Set training target.
    prediction_start = metrics_config.track_history_samples + 1

    # [batch_size, num_agents, steps, 7]
    gt_trajectory = inputs['gt_future_states']
    gt_targets = gt_trajectory[..., prediction_start:, :2]

    # [batch_size, num_agents, steps]
    gt_is_valid = inputs['gt_future_is_valid']
    # [batch_size, num_agents, steps]
    weights = (
        tf.cast(inputs['gt_future_is_valid'][..., prediction_start:],
                tf.float32) *
        tf.cast(inputs['tracks_to_predict'][..., tf.newaxis], tf.float32))
    print("gt_targets shape:", gt_targets.shape)
    print("pred_trajectory shape:", pred_trajectory.shape)

    loss_value = loss_fn(gt_targets, pred_trajectory, sample_weight=weights)
  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))

  # [batch_size, num_agents, steps, 2] ->
  # [batch_size, num_agents, 1, 1, steps, 2].
  # The added dimensions are top_k = 1, num_agents_per_joint_prediction = 1.
  pred_trajectory = pred_trajectory[:, :, tf.newaxis, tf.newaxis]

  # Fake the score since this model does not generate any score per predicted
  # trajectory.
  pred_score = tf.zeros(shape=tf.shape(pred_trajectory)[:3])
  print(pred_score.shape)

  # [batch_size, num_agents].
  object_type = inputs['object_type']

  # [batch_size, num_agents].
  batch_size = tf.shape(inputs['tracks_to_predict'])[0]
  num_samples = tf.shape(inputs['tracks_to_predict'])[1]

  pred_gt_indices = tf.range(num_samples, dtype=tf.int64)
  # [batch_size, num_agents, 1].
  pred_gt_indices = tf.tile(pred_gt_indices[tf.newaxis, :, tf.newaxis],
                            (batch_size, 1, 1))
  # [batch_size, num_agents, 1].
  pred_gt_indices_mask = inputs['tracks_to_predict'][..., tf.newaxis]

  motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory,
                              gt_is_valid, pred_gt_indices,
                              pred_gt_indices_mask, object_type)

  return loss_value


dataset = tf.data.TFRecordDataset(FILENAMES) # currently only taking in one tfrecords
dataset = dataset.map(_parse)
dataset = dataset.batch(4)
print(dataset)


epochs = 256    
num_batches_per_epoch = 2

# for epoch in range(epochs):
#   tf.keras.backend.clear_session()
# #   print('\nStart of epoch %d' % (epoch,))
#   start_time = time.time()

#   # Iterate over the batches of the dataset.
#   for step, batch in enumerate(dataset):
#     loss_value = train_step(batch)

#     # Log at end of every epoch.
#     if step == num_batches_per_epoch:
#       print('Epoch %d - TL (batch avg.) at step %d: %.4f' %
#             (epoch, step, float(loss_value / 16)))
#     #   print('Seen so far: %d samples' % ((step + 1) * 64))

#     if step >= num_batches_per_epoch:
#       break

#   # Display metrics at the end of each epoch.
#   train_metric_values = motion_metrics.result()
# #   for i, m in enumerate(
# #       ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
# #     for j, n in enumerate(metric_names):
# #       print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))