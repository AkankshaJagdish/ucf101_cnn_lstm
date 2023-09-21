import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import load_img, img_to_array


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_data_from_frames(dataset_path, fraction=0.03, batch_size=5, max_frames_per_sequence=10): 
    """
    Load video data from frames in batches.
    """
    categories = os.listdir(dataset_path)
    
    # Select a fraction of categories
    num_categories = int(len(categories) * fraction)
    selected_categories = np.random.choice(categories, num_categories, replace=False)
    
    for idx, category in enumerate(selected_categories):
        print(f"Processing category {idx + 1}/{num_categories}: {category}")
        sequences = os.listdir(os.path.join(dataset_path, category))
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            X_batch, Y_batch = [], []
            for sequence in batch_sequences:
                frames = glob.glob(os.path.join(dataset_path, category, sequence, '*.jpg'))
                frames = frames[:max_frames_per_sequence]  # Limit number of frames
                
                sequence_data = []
                for frame in frames:
                    # Convert each frame to a normalized image array of size 128x128.
                    img = load_img(frame, target_size=(128, 128))
                    img_array = img_to_array(img) / 255.0
                    sequence_data.append(img_array)
                X_batch.append(np.array(sequence_data))
                Y_batch.append(category)
            
            yield X_batch, Y_batch

dataset_path_frames = "data/UCF101-frames"

import h5py

# Define paths to save the processed data
data_save_path = "./data/data.h5"

# Check if processed data is already saved
if os.path.exists(data_save_path):
    # Load from disk using h5py
    with h5py.File(data_save_path, 'r') as hf:
        X = list(hf['X'][:])
        Y_encoded = list(hf['Y_encoded'][:])
else:
    # Load a small fraction of the categories for demonstration purposes
    X, Y = [], []
    for X_batch, Y_batch in load_data_from_frames(dataset_path_frames):
        X.extend(X_batch)
        Y.extend(Y_batch)

    # Convert category labels to integer encoding for training.
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # Save to disk using h5py
    with h5py.File(data_save_path, 'w') as hf:
        hf.create_dataset("X", data=X, compression="gzip", compression_opts=9)
        hf.create_dataset("Y_encoded", data=Y_encoded, compression="gzip", compression_opts=9)

# Split data into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

import gc

# 2. Create a Federated Dataset

NUM_CLIENTS = len(X_train)
BATCH_SIZE = 16  # Reduced batch size

def create_federated_dataset(X, Y, batch_size=BATCH_SIZE):
    """
    Create a federated dataset where each client's data is a video sequence.
    
    Args:
    - X (list): List of video sequences.
    - Y (list): List of labels.
    - batch_size (int): Batch size for tf.data.Dataset.
    
    Returns:
    - federated_data (list): List of tf.data.Dataset objects for federated training.
    """
    def dataset_from_tensor_slices(X, Y):
        # Create a tf.data.Dataset from the given video sequence and its label.
        # Note: Ensure that X retains its sequence_length dimension.
        return tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(Y)).batch(batch_size)

    federated_data = []
    for x, y in zip(X, Y):
        # Here, [x] is used to ensure the sequence dimension is retained.
        client_data = dataset_from_tensor_slices([x], [y])
        federated_data.append(client_data)
        del x, y
        gc.collect()  # Manually trigger garbage collection
    return federated_data

# Use a subset of clients to reduce memory consumption.
subset_size = min(NUM_CLIENTS, 10)  # Further reduced for demonstration
federated_train_data = create_federated_dataset(X_train[:subset_size], Y_train[:subset_size])

import gc

# 3. Define and Compile the Simplified Model

def create_simplified_cnn_lstm_model():
    """
    Create a simplified CNN-LSTM model for video action recognition.
    """
    input_layer = tf.keras.layers.Input(shape=(None, 128, 128, 3))

    
    # Simplified CNN part
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))(input_layer)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    
    # Simplified LSTM
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(len(np.unique(Y_encoded)), activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def model_fn():
    keras_model = create_simplified_cnn_lstm_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# 4. Federated Training

# Set up federated training using the weighted federated averaging algorithm.
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1)  # Adjust as necessary
)

state = trainer.initialize()

NUM_ROUNDS = 49
NUM_CLIENTS = 6

# Pre-create federated dataset for a larger subset of clients
# We'll take a size that's larger than the number of rounds times the clients per round
PRECREATED_CLIENTS = NUM_ROUNDS * NUM_CLIENTS * 2  # adjust as needed

precreated_federated_data = create_federated_dataset(X_train[:PRECREATED_CLIENTS], Y_train[:PRECREATED_CLIENTS])

# Federated Training

# Set up federated training using the weighted federated averaging algorithm.
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1)  # Adjust as necessary
)

state = trainer.initialize()

for round_num in range(NUM_ROUNDS):
    # Sample client datasets from the precreated federated dataset
    sampled_indices = np.random.choice(len(precreated_federated_data), size=NUM_CLIENTS, replace=False)
    federated_data_sample = [precreated_federated_data[i] for i in sampled_indices]
    
    # Train
    result = trainer.next(state, federated_data_sample)
    state = result.state
    metrics = result.metrics
    print('Round {:2d}, metrics={}'.format(round_num, metrics))


"""
Round 48, metrics=OrderedDict([('distributor', ()), ('client_work', OrderedDict([('train', OrderedDict([('sparse_categorical_accuracy', 1.0), ('loss', 0.80085117), ('num_examples', 6), ('num_batches', 6)]))])), ('aggregator', OrderedDict([('mean_value', ()), ('mean_weight', ())])), ('finalizer', OrderedDict([('update_non_finite', 0)]))])
"""