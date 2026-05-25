import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadGraphAttention_v2(layers.Layer):
    """
    TF Keras layer which aggregates multiple graph attention heads,
      either by concatenation or averaging.
    Performs non-linearity (ReLU) on the aggregated attention head outputs
    """

    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        """
        Parameters
        ----------
        units : int
            Hidden units dimension size for node parameters
             (other parameters are scaled accordingly)
        num_heads : int, optional
            Number of graph attention heads, by default 8
        merge_type : str["concat" or "average"], optional
            Optionally specificy method of aggregating graph attention head outputs,
              either "concat" or "average", by default "concat"
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

class GraphAttention_v2(layers.Layer):
    """
    Tensorflow Keras layer implementation of graph attention head
    using the GATv2 mechanism from the paper:

    'How Attentive are Graph Attention Networks?', Shaked Brody, Uri Alon, Eran Yahav,
    arXiv preprint arXiv:2105.14491 (2021)

    This implementation has been adapted to include edge features

    -------
    Mathematical summary

    The output for the GATv2 layer at node_i (Output_i) is calculated as follows:

        Let h_a be node features of the node_a

        Let e_a,b be features of the edge from node_a to node_b

        Let W_* represent parameters of the network

        Calculate the attention coefficient c_(i,j) of a node j connected to node_i
            c_(i,j) = W_attention•LeakyReLu(
                            [W_nodes•h_i || W_neighbour-nodes•h_j || W_edges•e_(i,j)]
                            )
                • denotes matric multiplication
                || denotes concatenation

        Calculate the attention score for a node j connected to node_i
            a_(i,j) = SOFTMAX_over_all_j( c_(i,j) )

        Calculate Output at node_i
            Output_i = SIGMOID( SUM_over_all_j[a_(i,j) * (W_neighbour•h_j)] )
                * denotes element-wise multiplication
                • denotes matrix multiplication
                Note that the final SIGMOID non-linearity is not implemented here,
                so that the output can be concatenated and/or averaged in a multi-head
                strategy and then the non-linearity applied
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
            Number of hidden units for node parameter matrices
            (edge and attention parameter matrices are scale accordingly)
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
            shape=(input_shape[2][-1], input_shape[2][-1]),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_edge_features_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges, edge_features = inputs
        node_states = tf.squeeze(node_states, axis=0)
        # If edges is a ragged tensor then convert to normal tensor. We need this
        #  because we can't index into the ragged dimension of ragged tensors
        if isinstance(edges, tf.RaggedTensor):
            edges = edges.to_tensor()
        edges = tf.squeeze(edges, axis=0)
        if isinstance(edge_features, tf.RaggedTensor):
            edge_features = edge_features.to_tensor()
        edge_features = tf.squeeze(edge_features, axis=0)
        # Linearly transform node states
        node_states_transformed_left = tf.matmul(node_states, self.kernel_left)
        node_states_transformed_right = tf.matmul(node_states, self.kernel_right)
        # (1) Compute pair-wise attention co-efficients
        node_states_expanded = tf.concat(
            (
                tf.gather(node_states_transformed_left, edges[:, 0]),
                tf.gather(node_states_transformed_right, edges[:, 1]),
            ),
            axis=-1,
        )
        node_states_expanded = tf.nn.leaky_relu(node_states_expanded)
        edge_features_gathered = tf.gather_nd(edge_features, edges)
        edge_features_transformed = tf.matmul(
            edge_features_gathered, self.kernel_edge_features_attention
        )
        node_states_expanded = tf.concat(
            (node_states_expanded, edge_features_transformed), axis=-1
        )
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

        # (3) Gather node states of neighbors, apply attention scores
        #  and aggregate to calculate output
        node_states_neighbors = tf.gather(node_states_transformed_right, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        out = tf.expand_dims(out, 0)
        return out