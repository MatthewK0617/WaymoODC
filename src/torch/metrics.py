import torch
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2
import numpy as np


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


class MotionMetrics(torch.nn.Module):
    """Wrapper for motion metrics computation in PyTorch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reset_state()

    def reset_state(self):
        """Resets the state variables."""
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
        """Accumulates batch data."""

        # Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(prediction_trajectory, np.ndarray):
            prediction_trajectory = torch.tensor(prediction_trajectory, dtype=torch.float32)
        if isinstance(prediction_score, np.ndarray):
            prediction_score = torch.tensor(prediction_score, dtype=torch.float32)
        if isinstance(ground_truth_trajectory, np.ndarray):
            ground_truth_trajectory = torch.tensor(ground_truth_trajectory, dtype=torch.float32)
        if isinstance(ground_truth_is_valid, np.ndarray):
            ground_truth_is_valid = torch.tensor(ground_truth_is_valid, dtype=torch.bool)
        if isinstance(prediction_ground_truth_indices, np.ndarray):
            prediction_ground_truth_indices = torch.tensor(prediction_ground_truth_indices, dtype=torch.int64)
        if isinstance(prediction_ground_truth_indices_mask, np.ndarray):
            prediction_ground_truth_indices_mask = torch.tensor(prediction_ground_truth_indices_mask, dtype=torch.bool)
        if isinstance(object_type, np.ndarray):
            object_type = torch.tensor(object_type, dtype=torch.int64)

        self._prediction_trajectory.append(prediction_trajectory)
        self._prediction_score.append(prediction_score)
        self._ground_truth_trajectory.append(ground_truth_trajectory)
        self._ground_truth_is_valid.append(ground_truth_is_valid)
        self._prediction_ground_truth_indices.append(prediction_ground_truth_indices)
        self._prediction_ground_truth_indices_mask.append(prediction_ground_truth_indices_mask)
        self._object_type.append(object_type)

    def result(self):
      """Computes the final metric based on accumulated state."""
      # Concatenate lists of tensors along the batch dimension
      prediction_trajectory = torch.cat(self._prediction_trajectory, dim=0).detach().cpu()
      prediction_score = torch.cat(self._prediction_score, dim=0).detach().cpu()
      ground_truth_trajectory = torch.cat(self._ground_truth_trajectory, dim=0).detach().cpu()
      ground_truth_is_valid = torch.cat(self._ground_truth_is_valid, dim=0).detach().cpu()
      prediction_ground_truth_indices = torch.cat(self._prediction_ground_truth_indices, dim=0).detach().cpu()
      prediction_ground_truth_indices_mask = torch.cat(self._prediction_ground_truth_indices_mask, dim=0).detach().cpu()
      object_type = torch.cat(self._object_type, dim=0).detach().long().cpu()


      # Adjust prediction_trajectory dimensions if necessary
      interval = self.config.track_steps_per_second // self.config.prediction_steps_per_second
      prediction_trajectory = prediction_trajectory[..., (interval - 1)::interval, :]

      return py_metrics_ops.motion_metrics(
          config=self.config.SerializeToString(),
          prediction_trajectory=prediction_trajectory.numpy(),
          prediction_score=prediction_score.numpy(),
          ground_truth_trajectory=ground_truth_trajectory.numpy(),
          ground_truth_is_valid=ground_truth_is_valid.numpy(),
          prediction_ground_truth_indices=prediction_ground_truth_indices.numpy(),
          prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask.numpy(),
          object_type=object_type.numpy())
