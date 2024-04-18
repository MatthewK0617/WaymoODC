import tensorflow as tf


class ComplexModel(tf.keras.Model):
    """A more complex regressor model."""

    def __init__(self, num_agents_per_scenario, num_states_steps, num_future_steps):
        super(ComplexModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_states_steps = num_states_steps
        self._num_future_steps = num_future_steps

        # Define the layers
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.dense3 = tf.keras.layers.Dense(128, activation="relu")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.5)

        # Final regressor layer
        self.regressor = tf.keras.layers.Dense(num_future_steps * 2)

    def call(self, states, training=False):
        states = tf.reshape(states, (-1, self._num_states_steps * 2))

        x = self.dense1(states)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.batch_norm3(x, training=training)
        x = self.dropout3(x, training=training)

        pred = self.regressor(x)
        pred = tf.reshape(
            pred, [-1, self._num_agents_per_scenario, self._num_future_steps, 2]
        )
        return pred


class ComplexModel2(tf.keras.Model):
    """A regressor model using LSTM layers for sequence data."""

    def __init__(self, num_agents_per_scenario, num_states_steps, num_future_steps):
        super(ComplexModel2, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_states_steps = num_states_steps
        self._num_future_steps = num_future_steps
        
        # Define LSTM layers
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        
        self.lstm2 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        
        self.lstm3 = tf.keras.layers.LSTM(128)
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        
        # Final regressor layer
        self.regressor = tf.keras.layers.Dense(num_future_steps * 2)

    def call(self, states, training=False):
    # Assuming states shape is (batch_size, num_agents, timesteps, features)
    # Reshape to combine batch and agents dimensions: new shape will be (batch_size*num_agents, timesteps, features)
        batch_size, num_agents, timesteps, features = states.shape
        states_reshaped = tf.reshape(states, (-1, timesteps, features))
        
        x = self.lstm1(states_reshaped)
        x = self.dropout1(x, training=training)
        
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        
        x = self.lstm3(x)
        x = self.dropout3(x, training=training)
        
        pred = self.regressor(x)
        # After processing, if needed, reshape the output back to the original batch and agent dimensions
        # Assuming your regressor's output is designed to handle the combined batch and agent dimension
        return pred


class ComplexModelWithConfidence(tf.keras.Model):
    """A complex model for trajectory prediction with confidence scores."""

    def __init__(self, num_agents_per_scenario, num_states_steps, num_future_steps):
        super(ComplexModelWithConfidence, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_states_steps = num_states_steps
        self._num_future_steps = num_future_steps

        # Shared layers for feature extraction
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.dense3 = tf.keras.layers.Dense(128, activation="relu")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.5)

        # Trajectory prediction layers
        self.regressor = tf.keras.layers.Dense(num_future_steps * 2, name="trajectory_output")

        # Confidence score prediction layers
        self.conf_dense = tf.keras.layers.Dense(64, activation="relu")
        self.conf_batch_norm = tf.keras.layers.BatchNormalization()
        self.conf_dropout = tf.keras.layers.Dropout(0.5)
        self.conf_regressor = tf.keras.layers.Dense(num_future_steps, activation="sigmoid", name="confidence_output")

    def call(self, states, training=False):
        states = tf.reshape(states, (-1, self._num_states_steps * 2))
        
        # Shared feature extraction
        x = self.dense1(states)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.batch_norm3(x, training=training)
        x = self.dropout3(x, training=training)

        # Trajectory prediction pathway
        trajectory_pred = self.regressor(x)
        trajectory_pred = tf.reshape(trajectory_pred, [-1, self._num_agents_per_scenario, self._num_future_steps, 2])

        # Confidence score prediction pathway
        conf_x = self.conf_dense(x)
        conf_x = self.conf_batch_norm(conf_x, training=training)
        conf_x = self.conf_dropout(conf_x, training=training)
        confidence_score = self.conf_regressor(conf_x)
        confidence_score = tf.reshape(confidence_score, [-1, self._num_agents_per_scenario, 1])

        return trajectory_pred, confidence_score