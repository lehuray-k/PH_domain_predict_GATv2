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
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel_left = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_left",
        )
        self.kernel_right = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_right",
        )
        self.kernel_attention = self.add_weight(
            shape=((self.units * 2) + input_shape[2][-1], 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
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
        # if edges is a ragged tensor then convert to normal tensor. We need this because we can't index into the ragged dimension of ragged tensors
        if isinstance(edges,tf.RaggedTensor) == True:
            edges = edges.to_tensor()
        edges = tf.squeeze(edges,axis=0)
        if isinstance(edge_features,tf.RaggedTensor) == True:
            edge_features = edge_features.to_tensor()
        edge_features = tf.squeeze(edge_features,axis=0)
        # Linearly transform node states
        node_states_transformed_left = tf.matmul(node_states, self.kernel_left)
        node_states_transformed_right = tf.matmul(node_states, self.kernel_right)
        # (1) Compute pair-wise attention scores using GATv2_attention mechanism
        node_states_expanded = tf.concat((tf.gather(node_states_transformed_left, edges[:,0]),tf.gather(node_states_transformed_right, edges[:,1])),axis=-1)
        node_states_expanded = tf.nn.leaky_relu(node_states_expanded)
        edge_features_gathered = tf.gather_nd(edge_features,edges)
        edge_features_transformed = tf.matmul(edge_features_gathered, self.kernel_edge_features_attention)
        node_states_expanded = tf.concat((node_states_expanded,edge_features_transformed),axis = -1)
        attention_scores = tf.matmul(node_states_expanded, self.kernel_attention)
        attention_scores = tf.squeeze(attention_scores, -1)
        # (2) Normalize attention scores
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

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed_right, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        #out = tf.math.add(node_states_transformed_left,out)
        out = tf.expand_dims(out, 0)
        return out

class MultiHeadGraphAttention_v2(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
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
    # process PH domain data

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

    # load names of PH domain files so we can track them during cross validation
    ph_domain_names = pickle.load(open("ph_domain_data/preprocessed_data/PH-domain-names_processed_dim100xNonex29_06Oct23_onehot_DSSP_shakerupley_0.8chargeneighbourhood.pkl",'rb'))
    # trim file names down to just the gene name and pdb id
    for file_index in range(0,len(ph_domain_names)):
        if "input" in ph_domain_names[file_index]:
             ph_domain_names[file_index] = ph_domain_names[file_index].replace("OUT_protein_ALL_PIP-headgroups_final_200ns_","").replace("_input.pdb","")
        else:
             ph_domain_names[file_index] = ph_domain_names[file_index].replace("OUT_protein_ALL_PIP-headgroups_final_200ns_","").replace("_PM_modeller.B99990001.pdb","")

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
        ph_domain_names_train = []
        ph_domain_names_test = []

        for index in training_split_indices[CV_set]:
                X_train.append(X_loaded[index])
                Y_train.append(Y_loaded[index])
                edges_train.append(edges_loaded[index])
                distance_matrix_train.append(distance_matrix[index])
                edge_features_train.append(edge_features[index])
                ph_domain_names_train.append(ph_domain_names[index])

        for index in testing_split_indices[CV_set]:
                X_test.append(X_loaded[index])
                Y_test.append(Y_loaded[index])
                edges_test.append(edges_loaded[index])
                distance_matrix_test.append(distance_matrix[index])
                edge_features_test.append(edge_features[index]) 
                ph_domain_names_test.append(ph_domain_names[index])
        
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
        CV_set_data[CV_set] = [X_train,X_test,Y_train,Y_test,edges_train,edges_test,edge_features_train,edge_features_test,ph_domain_names_train,ph_domain_names_test]
    return(CV_set_data)

model_name_prefix = "GATv2model_2023-06-10_01"
models_directory = "ph_domain_data/models/"
number_of_features = 25

CV_metrics = []
per_PH_domain_metrics = []
for k in [20,10,5]:
    # iterate over 5-fold, 10-fold and 20-fold crossvalidation
    for random_seed in [907,7635]:
        # repeat for multiple random seeds
        # prepare dataset for choice of k and seed
        CV_set_data = prepare_cross_validation_dataset(k,random_seed)
        for CV_set in range(0,k):
            # load model for each fold
            model_name = "%s_%sfoldCV_seed%s_fold%s"%(model_name_prefix,str(k),str(random_seed),str(CV_set))
            X_train,X_test,Y_train,Y_test,edges_train,edges_test,distance_matrix_train,distance_matrix_test,ph_domain_names_train,ph_domain_names_test = CV_set_data[CV_set]
            model = keras.models.load_model(models_directory+model_name+".h5",custom_objects={'MultiHeadGraphAttention_v2': MultiHeadGraphAttention_v2})
            MSEs = []
            ws_distances = []
            ws_distances_weights = []
            # evaluate positivies and negatives for different thresholds (0.6 and 0.8). The positive class is predicted when prediction >= threshold
            true_positives = {0.6: 0, 0.8: 0}
            false_positives = {0.6: 0, 0.8: 0}
            true_negatives = {0.6: 0, 0.8: 0}
            false_negatives = {0.6: 0, 0.8: 0}
            false_negative_values = {0.6: [], 0.8: []}
            for index in range(0,np.shape(X_test)[0]):
                # make prediction and max normalize
                predict = model.predict([tf.expand_dims(X_test[index],axis=0),tf.expand_dims(edges_test[index],axis=0),tf.expand_dims(distance_matrix_test[index],axis=0)])
                predict = tf.squeeze(predict,axis=[-1])
                predict = tf.squeeze(predict,axis=[0])
                predict = tf.divide(predict,tf.reduce_max(predict))
                ground_truth = Y_test[index]
                ground_truth = tf.divide(Y_test[index],tf.reduce_max(Y_test[index]))
                # calculate MSE
                mse = tf.keras.metrics.mean_squared_error(ground_truth, predict).numpy()
                MSEs.append(mse)
                # count true/false positives and negatives
                for threshold in [0.6,0.8]:
                    for u in range(0,np.shape(ground_truth)[0]):
                        if predict[u] >= threshold:
                            if ground_truth[u] >= threshold:
                                true_positives[threshold] = true_positives[threshold] + 1
                            else:
                                false_positives[threshold] = false_positives[threshold] + 1
                        elif predict[u] < threshold:
                            if ground_truth[u] < threshold:
                                true_negatives[threshold] = true_negatives[threshold] + 1
                            else:
                                false_negatives[threshold] = false_negatives[threshold] + 1
                                false_negative_values[threshold].append(predict[u])
                # calculate WS distance
                sum_y_val = sum(ground_truth)
                normalized_y = [item/sum_y_val for item in ground_truth]
                sum_predict = sum(predict)
                normalized_predict = [item/sum_predict for item in predict]
                ws_distance = scipy.stats.wasserstein_distance(np.arange(np.shape(normalized_y)[0]),np.arange(np.shape(normalized_predict)[0]),normalized_y, normalized_predict)
                ws_distances.append(ws_distance)
                print("WS: "+str(ws_distance))
                print(ph_domain_names_test[index])
                # store metrics for this PH domain
                per_PH_domain_metrics.append({'ph domain': ph_domain_names_test[index],
                                              'k': k,'seed': random_seed,
                                              'CV fold number': CV_set,
                                              'MSE': mse,
                                              'Wasserstein Distance': ws_distance,
                                              'Ground truth': ground_truth,
                                              'Prediction': predict})
            # calculate mean values across dataset
            mean_ws_distance = np.mean(ws_distances)
            mean_ws_distances_weights = np.mean(ws_distances_weights)
            sensitivity_point8 = true_positives[0.8]/(true_positives[0.8]+false_negatives[0.8])
            specificity_point8 = true_negatives[0.8]/(true_negatives[0.8]+false_positives[0.8])
            sensitivity_point6 = true_positives[0.6]/(true_positives[0.6]+false_negatives[0.6])
            specificity_point6 = true_negatives[0.6]/(true_negatives[0.6]+false_positives[0.6])
            precision_point8 = true_positives[0.8]/(true_positives[0.8]+false_positives[0.8])
            f1_score_point8 = 2*sensitivity_point8*precision_point8/(sensitivity_point8+precision_point8)
            accuracy_point8 = (true_positives[0.8]+true_negatives[0.8])/(true_positives[0.8]+true_negatives[0.8]+false_positives[0.8]+false_negatives[0.8])
            percentile_false_negatives_point_8 = scipy.stats.percentileofscore(predict[predict <= 0.8], np.mean(false_negative_values[0.8]))
            CV_metrics.append({'k': k,
                               'seed': random_seed,
                               'CV fold number': CV_set,
                               'MSE (no padding)': np.mean(MSEs),
                               'Mean Wasserstein Distance': mean_ws_distance,
                               'Accuracy (0.8)': accuracy_point8,
                               'Sensitivity (0.8)': sensitivity_point8,
                               'Specificity (0.8)': specificity_point8,
                               'Precision (0.8)': precision_point8,
                               'F1 score (0.8)': f1_score_point8,
                               'False negative predictions (0.8)': np.mean(false_negative_values[0.8]),
                               'False negative percentile (0.8)': percentile_false_negatives_point_8,
                               'PH domains in test set': ph_domain_names_test})
            print("Sensitivity (0.8): %s Specificity (0.8): %s"%(str(sensitivity_point8),str(specificity_point8)))
            print("Sensitivity (0.6): %s Specificity (0.6): %s"%(str(sensitivity_point6),str(specificity_point6)))
cv_metrics_df = pd.DataFrame.from_dict(CV_metrics)
cv_metrics_df.to_csv("ph_domain_data/cross_validation/cross_validation_metrics_03.csv")
cv_metrics_df.to_pickle("ph_domain_data/cross_validation/cross_validation_metrics_03.pkl")
print(cv_metrics_df.to_string())
print(cv_metrics_df.mean().to_string())
per_ph_domain_metrics_df = pd.DataFrame.from_dict(per_PH_domain_metrics)
per_ph_domain_metrics_df.to_csv("ph_domain_data/cross_validation/per_ph_domain_cross_validation_metrics_03.csv")
per_ph_domain_metrics_df.to_pickle("ph_domain_data/cross_validation/per_ph_domain_cross_validation_metrics_03.pkl")
