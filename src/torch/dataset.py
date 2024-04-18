import os
import torch
from torch.utils.data import Dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np
import glob
from features import roadgraph_features, state_features, traffic_light_features


class WaymoTFRecordDataset(Dataset):
    def __init__(self, file_pattern, transform=None):
        """
        Args:
            file_pattern (string): Path to the TFRecord files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.filenames = glob.glob(file_pattern)
        self.transform = transform
        self.features_description = {
            **roadgraph_features,
            **state_features,
            **traffic_light_features,
        }

    def __len__(self):
        return len(self.filenames)

    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.features_description)

    def __getitem__(self, idx):
        # Tensorflow setup to read the tfrecord file
        raw_dataset = tf.data.TFRecordDataset(self.filenames[idx])
        parsed_example = next(iter(raw_dataset.map(self._parse_function)))

        # Extract and process tensors similar to the provided TensorFlow _parse function
        # Note: We directly convert TensorFlow tensors to PyTorch tensors via numpy
        past_states = np.stack(
            [
                parsed_example["state/past/x"].numpy(),
                parsed_example["state/past/y"].numpy(),
                parsed_example["state/past/length"].numpy(),
                parsed_example["state/past/width"].numpy(),
                parsed_example["state/past/bbox_yaw"].numpy(),
                parsed_example["state/past/velocity_x"].numpy(),
                parsed_example["state/past/velocity_y"].numpy(),
            ],
            -1,
        )

        cur_states = np.stack(
            [
                parsed_example["state/current/x"].numpy(),
                parsed_example["state/current/y"].numpy(),
                parsed_example["state/current/length"].numpy(),
                parsed_example["state/current/width"].numpy(),
                parsed_example["state/current/bbox_yaw"].numpy(),
                parsed_example["state/current/velocity_x"].numpy(),
                parsed_example["state/current/velocity_y"].numpy(),
            ],
            -1,
        )

        future_states = np.stack(
            [
                parsed_example["state/future/x"].numpy(),
                parsed_example["state/future/y"].numpy(),
                parsed_example["state/future/length"].numpy(),
                parsed_example["state/future/width"].numpy(),
                parsed_example["state/future/bbox_yaw"].numpy(),
                parsed_example["state/future/velocity_x"].numpy(),
                parsed_example["state/future/velocity_y"].numpy(),
            ],
            -1,
        )

        # Concatenate past, current, and future states
        input_states = np.concatenate([past_states, cur_states], 1)
        gt_future_states = np.concatenate([past_states, cur_states, future_states], 1)

        # Calculate mean and std dev for normalization
        gt_future_states_means = np.mean(gt_future_states, axis=1)
        gt_future_states_stdev = np.std(gt_future_states, axis=1)
        gt_future_states_means = gt_future_states_means[:, np.newaxis]
        gt_future_states_stdev = gt_future_states_stdev[:, np.newaxis]
        gt_future_states_stdev = np.where(gt_future_states_stdev == 0, 1, gt_future_states_stdev)

        # Normalize
        input_states = (input_states - gt_future_states_means) / gt_future_states_stdev
        gt_future_states = (gt_future_states - gt_future_states_means) / gt_future_states_stdev
        
        # Generate PyTorch tensors from numpy arrays
        input_states = torch.tensor(input_states, dtype=torch.float32)
        gt_future_states = torch.tensor(gt_future_states, dtype=torch.float32)

        past_is_valid = parsed_example["state/past/valid"].numpy() > 0
        current_is_valid = parsed_example["state/current/valid"].numpy() > 0
        future_is_valid = parsed_example["state/future/valid"].numpy() > 0

        gt_future_is_valid = torch.tensor(
            np.concatenate(
                [
                    past_is_valid,
                    current_is_valid,
                    future_is_valid,
                ],
                1,
            ),
            dtype=torch.bool,
        )

        tracks_to_predict = torch.tensor(
            parsed_example["state/tracks_to_predict"].numpy() > 0, dtype=torch.bool
        )
        sample_is_valid = torch.tensor(
            np.concatenate([past_is_valid, current_is_valid], 1), dtype=torch.bool
        ).any(dim=1)

        num_agents = 128
        used_agents = 32

        # filtering irrelevant object types
        # instead of doing this, get an even spread? no
        # take the system with the most agents, and set that to max
        object_type = np.array(parsed_example['state/type'].numpy())
        # valid_object_mask = object_type != -1
        
        indices = np.random.choice(num_agents, used_agents, replace=False)
        valid_object_mask = np.zeros(num_agents, dtype=bool)
        valid_object_mask[indices] = True

        input_states = input_states[valid_object_mask]
        gt_future_states = gt_future_states[valid_object_mask]
        gt_future_is_valid = gt_future_is_valid[valid_object_mask]
        object_type = object_type[valid_object_mask]
        tracks_to_predict = tracks_to_predict[valid_object_mask]
        sample_is_valid = sample_is_valid[valid_object_mask]

        # Return a dict of tensors
        return {
            "input_states": input_states,
            "gt_future_states": gt_future_states,
            "gt_future_is_valid": gt_future_is_valid, 
            'object_type': object_type,
            "tracks_to_predict": tracks_to_predict, 
            "sample_is_valid": sample_is_valid, 
        }