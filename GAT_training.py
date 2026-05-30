import os

# disable GPU because training is faster on CPU
# for small dataset and lightweight model with batchsize = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from sklearn.model_selection import KFold

from MultiHeadGraphAttention import MultiHeadGraphAttention_v2

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)

def prepare_cross_validation_dataset(k, random_seed):
    """
    Prepares k-fold cross-validation dataset from pre-processed data samples
    given k and random_seed value

    Parameters
    ----------
    k : int
        Number of cross-validation folds
    random_seed : int
        Random set for data shuffling
    """
    # Process PH domain data:
    # LOAD NODE DATA
    Y_pickle_file = open(
        "ph_domain_data/preprocessed_data/Y_values_processed_PROBABILITIES_dim100xNone_06Oct23.pkl",
        "rb",
    )
    X_pickle_file = open(
        "ph_domain_data/preprocessed_data/X_values_processed_dim100xNonex25_06Oct23_onehot_DSSP_shakerupley_0.8chargeneighbourhood.pkl",
        "rb",
    )
    X_loaded = pickle.load(X_pickle_file)
    Y_loaded = pickle.load(Y_pickle_file)

    # LOAD EDGE DATA
    distance_matrix = pickle.load(
        open(
            "ph_domain_data/preprocessed_data/distance_matrices_dim100xNonexNone_06Oct23.pkl",
            "rb",
        )
    )
    inter_residue_unit_vectors = pickle.load(
        open(
            "ph_domain_data/preprocessed_data/inter_residue_unit_vectors_dim100xNonexNonex3_06Oct23.pkl",
            "rb",
        )
    )
    global_neighbourhood_list = []

    distance_cutoff = 200

    for protein in range(0, np.shape(distance_matrix)[0]):
        protein_neighbourhood_list = []
        for AA1 in range(0, np.shape(distance_matrix[protein])[0]):
            for AA2 in range(0, np.shape(distance_matrix[protein])[1]):
                if distance_matrix[protein][AA1][AA2] > 0:
                    if distance_matrix[protein][AA1][AA2] <= distance_cutoff:
                        protein_neighbourhood_list.append([int(AA1), int(AA2)])
        global_neighbourhood_list.append(protein_neighbourhood_list)

    edges_loaded = global_neighbourhood_list

    edge_features = []
    for protein in range(0, 100):
        AA1_edge_features_list = []
        for AA1 in range(0, len(distance_matrix[protein])):
            AA2_edge_features_list = []
            for AA2 in range(0, len(distance_matrix[protein])):
                AA2_edge_features_list.append(
                    [
                        distance_matrix[protein][AA1][AA2],
                        inter_residue_unit_vectors[protein][AA1][AA2][0],
                        inter_residue_unit_vectors[protein][AA1][AA2][1],
                        inter_residue_unit_vectors[protein][AA1][AA2][2],
                    ]
                )
            AA1_edge_features_list.append(AA2_edge_features_list)
        edge_features.append(AA1_edge_features_list)
    print(np.shape(edge_features[1]))

    CV_set_data = {}

    for CV_set in range(0, k):
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        training_split_indices = []
        testing_split_indices = []

        for train, test in kfold.split(X_loaded):
            training_split_indices.append(train)
            testing_split_indices.append(test)

        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        edges_train = []
        edges_test = []
        distance_matrix_train = []
        distance_matrix_test = []
        edge_features_train = []
        edge_features_test = []

        for index in training_split_indices[CV_set]:
            X_train.append(X_loaded[index])
            Y_train.append(Y_loaded[index])
            edges_train.append(edges_loaded[index])
            distance_matrix_train.append(distance_matrix[index])
            edge_features_train.append(edge_features[index])

        for index in testing_split_indices[CV_set]:
            X_test.append(X_loaded[index])
            Y_test.append(Y_loaded[index])
            edges_test.append(edges_loaded[index])
            distance_matrix_test.append(distance_matrix[index])
            edge_features_test.append(edge_features[index])

        X_train = tf.ragged.constant(X_train)

        Y_train = tf.ragged.constant(Y_train)
        X_test = tf.ragged.constant(X_test)
        Y_test = tf.ragged.constant(Y_test)
        edges_train = tf.ragged.constant(edges_train)
        edges_test = tf.ragged.constant(edges_test)
        distance_matrix_train = tf.expand_dims(
            tf.ragged.constant(distance_matrix_train), -1
        )
        distance_matrix_test = tf.expand_dims(
            tf.ragged.constant(distance_matrix_test), -1
        )
        edge_features_train = tf.ragged.constant(edge_features_train)
        edge_features_test = tf.ragged.constant(edge_features_test)
        CV_set_data[CV_set] = [
            X_train,
            X_test,
            Y_train,
            Y_test,
            edges_train,
            edges_test,
            edge_features_train,
            edge_features_test,
        ]
    return CV_set_data


save_models = True
model_name_prefix = "GATv2model_2023-06-10_01"
models_directory = "ph_domain_data/models/"
number_of_features = 25

for k in [5, 10, 20]:
    # Iterate over 5-fold, 10-fold and 20-fold crossvalidation
    for random_seed in [907, 7635]:
        # Repeat for multiple random seeds
        # Prepare dataset for choice of k and seed
        CV_set_data = prepare_cross_validation_dataset(k, random_seed)
        for CV_set in range(0, k):
            # Train model for each fold
            model_name = f"{model_name_prefix}_{str(k)}foldCV_seed{str(random_seed)}_fold{str(CV_set)}"  # noqa: E501

            # Define model using keras functional API
            node_inputs = tf.keras.Input(shape=(None, number_of_features), batch_size=1)
            edges_list = tf.keras.Input(shape=(None, 2), batch_size=1, dtype=tf.int64)
            edge_features = tf.keras.Input(shape=(None, None, 4), batch_size=1)

            MHA = MultiHeadGraphAttention_v2(
                units=12, num_heads=3, merge_type="concat"
            )([node_inputs, edges_list, edge_features])
            skip2 = tf.concat((node_inputs, MHA), axis=-1)
            MHA2 = MultiHeadGraphAttention_v2(
                units=12, num_heads=3, merge_type="concat"
            )([skip2, edges_list, edge_features])
            output = tf.keras.layers.Dense(1, activation="sigmoid")(MHA2)
            prediction = output
            (
                X_train,
                X_test,
                Y_train,
                Y_test,
                edges_train,
                edges_test,
                edge_features_train,
                edge_features_test,
            ) = CV_set_data[CV_set]

            # Build model
            model = tf.keras.models.Model(
                inputs=[node_inputs, edges_list, edge_features], outputs=prediction
            )
            model.summary()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0071),
                loss=tf.keras.losses.MeanSquaredError(),
            )

            # Define early stopping callback
            class CustomEarlyStoppingCallback(tf.keras.callbacks.Callback):
                def __init__(
                    self,
                    validation_data,
                    patience=8,
                    restore_best_weights=True,
                    start_from_epoch=4,
                ):
                    super().__init__()
                    self.validation_data = validation_data
                    self.patience = patience
                    self.restore_best_weights = restore_best_weights
                    self.start_from_epoch = start_from_epoch

                    self.wait = 0
                    self.best_metric = float(0)
                    self.best_weights = None
                    self.stopped_epoch = 0

                def monitor_metric(self):
                    """
                    Function for monitoring MSE, Wasserstein Distance, sensitivity,
                      specificity, precision and F1 score
                    """
                    true_positives = 0
                    false_positives = 0
                    true_negatives = 0
                    false_negatives = 0
                    ws_distances = []
                    mse = []
                    for index in range(0, np.shape(Y_test)[0]):
                        predict = model.predict(
                            [
                                tf.expand_dims(X_test[index], axis=0),
                                tf.expand_dims(edges_test[index], axis=0),
                                tf.expand_dims(edge_features_test[index], axis=0),
                            ],
                            verbose=0,
                        )
                        predict = tf.squeeze(predict, axis=[-1])
                        predict = tf.squeeze(predict, axis=[0])
                        # normalize prediction and ground truth
                        #  to obtain normalized contacts frequency
                        predict = tf.divide(predict, tf.reduce_max(predict))
                        true_y = tf.divide(Y_test[index], tf.reduce_max(Y_test[index]))
                        # additional normalization for calculating WS distance
                        sum_y_val = sum(true_y)
                        normalized_y = [item / sum_y_val for item in true_y]
                        sum_predict = sum(predict)
                        normalized_predict = [item / sum_predict for item in predict]
                        # WS_distance
                        ws_distance = scipy.stats.wasserstein_distance(
                            np.arange(np.shape(normalized_y)[0]),
                            np.arange(np.shape(normalized_predict)[0]),
                            normalized_y,
                            normalized_predict,
                        )
                        ws_distances.append(ws_distance)
                        # MSE
                        mse.append(
                            tf.keras.metrics.mean_squared_error(true_y, predict).numpy()
                        )
                        # accuracy sensitivity specificity precision F1 score
                        threshold = 0.8
                        for u in range(0, len(predict)):
                            if predict[u] >= threshold:
                                if true_y[u] >= threshold:
                                    true_positives = true_positives + 1
                                else:
                                    false_positives = false_positives + 1
                            elif predict[u] < threshold:
                                if true_y[u] < threshold:
                                    true_negatives = true_negatives + 1
                                else:
                                    false_negatives = false_negatives + 1
                    mean_ws_distance = np.mean(ws_distances)
                    mean_mse = np.mean(mse)
                    accuracy = (true_positives + true_negatives) / (
                        true_positives
                        + true_negatives
                        + false_positives
                        + false_negatives
                    )
                    sensitivity = true_positives / (true_positives + false_negatives)
                    specificity = true_negatives / (true_negatives + false_positives)
                    precision = true_positives / (true_positives + false_positives)
                    f1_score = 2 * sensitivity * precision / (sensitivity + precision)
                    print(
                        f"\n sensitivity: {sensitivity} specificity: {specificity} sum: {sensitivity + specificity} precision: {precision} F1: {f1_score} MSE: {mean_mse}"  # noqa: E501
                    )
                    # Use F1 score or MSE as stopping metric
                    return f1_score

                def on_epoch_end(self, epoch, logs=None):
                    current_metric = self.monitor_metric()
                    if epoch < self.start_from_epoch:
                        return

                    if current_metric is None:
                        raise ValueError(f"Metric '{self.monitor}' not found in logs.")

                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.best_weights = self.model.get_weights()
                        self.wait = 0
                    else:
                        self.wait += 1
                        print(
                            f"Early stopping patience: {self.wait}/{str(self.patience)}"
                        )
                        if self.wait >= self.patience:
                            self.stopped_epoch = epoch
                            if (
                                self.restore_best_weights
                                and self.best_weights is not None
                            ):
                                self.model.set_weights(self.best_weights)
                                print(
                                    f"Restoring model weights from epoch {epoch - self.patience + 1}."  # noqa: E501
                                )
                            self.model.stop_training = True

            # Create an instance of the custom callback
            EarlyStopping = CustomEarlyStoppingCallback(
                validation_data=(X_test, Y_test)
            )

            training = model.fit(
                x=[X_train, edges_train, edge_features_train],
                y=Y_train,
                batch_size=1,
                epochs=40,
                validation_data=([X_test, edges_test, edge_features_test], Y_test),
                validation_freq=40,
                shuffle=True,
                callbacks=[EarlyStopping],
            )
            if save_models:
                model.save(models_directory + model_name + ".h5")
