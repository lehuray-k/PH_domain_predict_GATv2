import os
# disable GPU because training is faster on CPU for small dataset and lightweight model with batchsize = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from protein_ML_utils import *
import pickle

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)

# Define custom graph attention architecture
# BUILD THE MODEL
class GraphAttention_v2(layers.Layer):
    """
    Tensorflow Keras layer implementation of graph attention head using the GATv2 mechanism from the paper:
    
    'How Attentive are Graph Attention Networks?', Shaked Brody, Uri Alon, Eran Yahav, arXiv preprint arXiv:2105.14491 (2021)

    This implementation has been adapted to include edge features 

    -------
    Mathematical summary

    The output for the GATv2 layer at node_i (Output_i) is calculated as follows:

        Let h_a be node features of the node_a

        Let e_a,b be features of the edge from node_a to node_b

        Let W_* represent parameters of the network
        
        Calculate the attention coefficient c_(i,j) of a node j connected to node_i
            c_(i,j) = W_attention•LeakyReLu([W_nodes•h_i || W_neighbour-nodes•h_j || W_edges•e_(i,j)])
                where • denotes matric multiplication and || denotes concatenation

        Calculate the attention score for a node j connected to node_i
            a_(i,j) = SOFTMAX_over_all_j( c_(i,j) )

        Calculate Output at node_i
            Output_i = SIGMOID( SUM_over_all_j[a_(i,j) * (W_neighbour•h_j)] )
                where * denotes element-wise multiplication and • denotes matrix multiplication
                Note that the final SIGMOID non-linearity is not implemented in this class, so that the output can be concatenated and/or averaged in a multi-head strategy and then the non-linearity applied
    -------
    """
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        units : int
            Number of hidden units for node parameter matrices (edge and attention parameter matrices are scale accordingly)
        kernel_regularizer :  tf.keras.Regularizer , optional
            Optional regularizer, by default None
        """
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        # Initialise parameters
        # Node parameter matrix
        self.kernel_left = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_left",
        )
        # Node neighbour parameter matrix
        self.kernel_right = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_right",
        )
        # Attention parameter matrix
        self.kernel_attention = self.add_weight(
            shape=((self.units * 2) + input_shape[2][-1], 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        # Edge parameter matrix
        self.kernel_edge_features_attention = self.add_weight(
            shape=(input_shape[2][-1],input_shape[2][-1]),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_edge_features_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges, edge_features = inputs
        node_states = tf.squeeze(node_states,axis=0)
        # If edges is a ragged tensor then convert to normal tensor. We need this because we can't index into the ragged dimension of ragged tensors
        if isinstance(edges,tf.RaggedTensor) == True:
            edges = edges.to_tensor()
        edges = tf.squeeze(edges,axis=0)
        if isinstance(edge_features,tf.RaggedTensor) == True:
            edge_features = edge_features.to_tensor()
        edge_features = tf.squeeze(edge_features,axis=0)
        # Linearly transform node states
        node_states_transformed_left = tf.matmul(node_states, self.kernel_left)
        node_states_transformed_right = tf.matmul(node_states, self.kernel_right)
        # (1) Compute pair-wise attention co-efficients
        node_states_expanded = tf.concat((tf.gather(node_states_transformed_left, edges[:,0]),tf.gather(node_states_transformed_right, edges[:,1])),axis=-1)
        node_states_expanded = tf.nn.leaky_relu(node_states_expanded)
        edge_features_gathered = tf.gather_nd(edge_features,edges)
        edge_features_transformed = tf.matmul(edge_features_gathered, self.kernel_edge_features_attention)
        node_states_expanded = tf.concat((node_states_expanded,edge_features_transformed),axis = -1)
        attention_scores = tf.matmul(node_states_expanded, self.kernel_attention)
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Softmax and normalize to get attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate to calculate output
        node_states_neighbors = tf.gather(node_states_transformed_right, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        out = tf.expand_dims(out, 0)
        return out

class MultiHeadGraphAttention_v2(layers.Layer):
    """
    TF Keras layer which aggregates multiple graph attention heads, either by concatenation or averaging.
    Performs non-linearity (ReLU) on the aggregated attention head outputs
    """
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        """
        Parameters
        ----------
        units : int
            Hidden units dimension size for node parameters (other parameters are scaled accordingly)
        num_heads : int, optional
            Number of graph attention heads, by default 8
        merge_type : str["concat" or "average"], optional
            Optionally specificy method of aggregating graph attention head outputs, either "concat" or "average", by default "concat"
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention_v2(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices, edge_features = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices, edge_features])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

def prepare_cross_validation_dataset(k,random_seed):
    """Prepares k-fold cross-validation dataset from pre-processed data sampples given k and random_seed value

    Parameters
    ----------
    k : int
        Number of cross-validation folds
    random_seed : int
        Random set for data shuffling
    """
    # Process PH domain data:
    # LOAD NODE DATA
    Y_pickle_file = open("ph_domain_data/preprocessed_data/Y_values_processed_PROBABILITIES_dim100xNone_06Oct23.pkl", 'rb')
    X_pickle_file = open("ph_domain_data/preprocessed_data/X_values_processed_dim100xNonex25_06Oct23_onehot_DSSP_shakerupley_0.8chargeneighbourhood.pkl",'rb')
    X_loaded = pickle.load(X_pickle_file)
    Y_loaded = pickle.load(Y_pickle_file)

    # LOAD EDGE DATA
    distance_matrix = pickle.load(open("ph_domain_data/preprocessed_data/distance_matrices_dim100xNonexNone_06Oct23.pkl", 'rb'))
    inter_residue_unit_vectors = pickle.load(open("ph_domain_data/preprocessed_data/inter_residue_unit_vectors_dim100xNonexNonex3_06Oct23.pkl",'rb'))
    global_neighbourghood_list = []

    distance_cutoff = 200

    for protein in range(0,np.shape(distance_matrix)[0]):
        protein_neighbourhood_list = []
        for AA1 in range(0,np.shape(distance_matrix[protein])[0]):
                for AA2 in range(0,np.shape(distance_matrix[protein])[1]):
                    if distance_matrix[protein][AA1][AA2] > 0:
                            if distance_matrix[protein][AA1][AA2] <= distance_cutoff:
                                protein_neighbourhood_list.append([int(AA1),int(AA2)])
        global_neighbourghood_list.append(protein_neighbourhood_list)

    edges_loaded = global_neighbourghood_list

    edge_features = []
    for protein in range(0,100):
        AA1_edge_features_list = []
        for AA1 in range(0,len(distance_matrix[protein])):
                AA2_edge_features_list = []
                for AA2 in range(0,len(distance_matrix[protein])):
                                AA2_edge_features_list.append([distance_matrix[protein][AA1][AA2],inter_residue_unit_vectors[protein][AA1][AA2][0],inter_residue_unit_vectors[protein][AA1][AA2][1],inter_residue_unit_vectors[protein][AA1][AA2][2]])
                AA1_edge_features_list.append(AA2_edge_features_list)
        edge_features.append(AA1_edge_features_list)
    print(np.shape(edge_features[1]))

    CV_set_data = {}

    for CV_set in range(0,k):
        kfold = KFold(n_splits=k,shuffle=True,random_state=random_seed)
        training_split_indices = []
        testing_split_indices = []

        for train,test in kfold.split(X_loaded):
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
        distance_matrix_train = tf.expand_dims(tf.ragged.constant(distance_matrix_train),-1)
        distance_matrix_test = tf.expand_dims(tf.ragged.constant(distance_matrix_test),-1)
        edge_features_train = tf.ragged.constant(edge_features_train)
        edge_features_test = tf.ragged.constant(edge_features_test)
        CV_set_data[CV_set] = [X_train,X_test,Y_train,Y_test,edges_train,edges_test,edge_features_train,edge_features_test]
    return(CV_set_data)

save_models = True
model_name_prefix = "GATv2model_2023-06-10_01"
models_directory = "ph_domain_data/models/"
number_of_features = 25

for k in [5,10,20]:
    # Iterate over 5-fold, 10-fold and 20-fold crossvalidation
    for random_seed in [907,7635]:
        # Repeat for multiple random seeds
        # Prepare dataset for choice of k and seed
        CV_set_data = prepare_cross_validation_dataset(k,random_seed)
        for CV_set in range(0,k):
            # Train model for each fold
            model_name = "%s_%sfoldCV_seed%s_fold%s"%(model_name_prefix,str(k),str(random_seed),str(CV_set))

            # Define model using keras functional API
            node_inputs = tf.keras.Input(shape=(None,number_of_features),batch_size=1)
            edges_list = tf.keras.Input(shape=(None,2),batch_size=1,dtype=tf.int64)
            edge_features = tf.keras.Input(shape=(None,None,4),batch_size=1)

            MHA = MultiHeadGraphAttention_v2(units=12,num_heads=3,merge_type="concat")([node_inputs,edges_list,edge_features])
            skip2 = tf.concat((node_inputs,MHA),axis=-1)
            MHA2 = MultiHeadGraphAttention_v2(units=12,num_heads=3,merge_type="concat")([skip2,edges_list,edge_features])
            output = tf.keras.layers.Dense(1,activation='sigmoid')(MHA2)
            prediction = output
            X_train,X_test,Y_train,Y_test,edges_train,edges_test,edge_features_train,edge_features_test = CV_set_data[CV_set]
            
            # Build model
            model = tf.keras.models.Model(inputs=[node_inputs,edges_list,edge_features], outputs=prediction)
            model.summary()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0071),loss=tf.keras.losses.MeanSquaredError())

            # Define early stopping callback
            class CustomEarlyStoppingCallback(tf.keras.callbacks.Callback):
                def __init__(self, validation_data, patience=8, restore_best_weights=True, start_from_epoch=4):
                    super(CustomEarlyStoppingCallback, self).__init__()
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
                    Function for monitoring MSE, Wasserstein Distance, sensitivity, specificity, precision and F1 score
                    """
                    true_positives = 0
                    false_positives = 0
                    true_negatives = 0
                    false_negatives = 0
                    ws_distances = []
                    mse = []
                    for index in range(0,np.shape(Y_test)[0]): 
                            predict = model.predict([tf.expand_dims(X_test[index],axis=0),tf.expand_dims(edges_test[index],axis=0),tf.expand_dims(edge_features_test[index],axis=0)],verbose=0)
                            predict = tf.squeeze(predict,axis=[-1])
                            predict = tf.squeeze(predict,axis=[0])
                            # normalize prediction and ground truth to obtain normalized contacts frequency
                            predict = tf.divide(predict,tf.reduce_max(predict))
                            true_y = tf.divide(Y_test[index],tf.reduce_max(Y_test[index]))
                            # additional normalization for calculating WS distance
                            sum_y_val = sum(true_y)
                            normalized_y = [item/sum_y_val for item in true_y]
                            sum_predict = sum(predict)
                            normalized_predict = [item/sum_predict for item in predict]
                            # WS_distance
                            ws_distance = scipy.stats.wasserstein_distance(np.arange(np.shape(normalized_y)[0]),np.arange(np.shape(normalized_predict)[0]), normalized_y, normalized_predict)
                            ws_distances.append(ws_distance)
                            # MSE
                            mse.append(tf.keras.metrics.mean_squared_error(true_y, predict).numpy())
                            # accuracy sensitivity specificity precision F1 score
                            threshold = 0.8
                            for u in range(0,len(predict)):
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
                    accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
                    sensitivity = true_positives/(true_positives+false_negatives)
                    specificity = true_negatives/(true_negatives+false_positives)
                    precision = true_positives/(true_positives+false_positives)
                    f1_score = 2*sensitivity*precision/(sensitivity+precision)
                    print('\n sensitivity: %s specificity: %s sum: %s precision: %s F1: %s MSE: %s'%(sensitivity,specificity,sensitivity+specificity,precision,f1_score,mean_mse))
                    # Use F1 score or MSE as stopping metric
                    return(f1_score)

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
                        print("Early stopping patience: %s/%s"%(self.wait,str(self.patience)))
                        if self.wait >= self.patience:
                            self.stopped_epoch = epoch
                            if self.restore_best_weights and self.best_weights is not None:
                                self.model.set_weights(self.best_weights)
                                print(f"Restoring model weights from epoch {epoch - self.patience + 1}.")
                            self.model.stop_training = True

            # Create an instance of the custom callback
            EarlyStopping = CustomEarlyStoppingCallback(validation_data=(X_test, Y_test))

            training = model.fit(
                x=[X_train,edges_train,edge_features_train],
                y=Y_train,
                batch_size=1,
                epochs=40,
                validation_data=([X_test,edges_test,edge_features_test],Y_test),
                validation_freq = 40,
                shuffle=True,
                callbacks=[EarlyStopping])
            if save_models == True:
                model.save(models_directory+model_name+".h5")
