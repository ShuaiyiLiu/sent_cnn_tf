import tensorflow as tf
import numpy as np

class SentCNN(object):
    """
    A CNN for utterance and relation pair matching regression.
    Uses an embedding layer, convolutional layer, max-pooling layer,
    and a logistic regression layer.
    """
    
    def __init__(self, 
                 sequence_length, 
                 num_classes, 
                 init_embeddings, 
                 filter_sizes, 
                 num_filters,
                 batch_size, # only need this for dropout layer
                 embeddings_trainable=False):
        """
        :param sequence_length: The length of our sentences. Here we always pad
        our sentences to have the same length (depending on the longest sentences
        in our dataset).
        :param num_classes: Number of classes in the output layer.
        :param init_embeddings: Pre-trained word embeddings or initialied values.
        :filter_sizes: The number of words we want our convolutional filters to cover. 
        We will have num_filters for each size specified here. For example, [3, 4, 5] 
        means that we will have filters that slide over 3, 4 and 5 words respectively, 
        for a total of 3 * num_filters filters.
        :num_filters: The number of filters per filter size (see above).
        :embeddings_trainable: Train embeddings or not.
        """
        # Placeholders for input, output and dropout
        
        # input_x_u: batch_size x sequence_length
        self.input_x_u = tf.placeholder(tf.int32, 
                                        [None, sequence_length],
                                        name="input_x_u")
        # input_x_r: batch_size x num_classes x sequence_length
        self.input_x_r = tf.placeholder(tf.int32, 
                                        [None, num_classes, sequence_length],
                                        name="input_x_r")
        # input_y: batch_size, 
        self.input_y = tf.placeholder(tf.int64, 
                                      [None],
                                      name="input_y")
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.embedding_size = np.shape(init_embeddings)[1]
        
        # Embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(init_embeddings,
                            trainable=embeddings_trainable,
                            dtype=tf.float32,
                            name='W')
            # batch_size x sequence_length x embedding_size
            self.embedded_u = tf.nn.embedding_lookup(W, self.input_x_u)
            print "DEBUG: embedded_u -> %s" % self.embedded_u
            # batch_size x num_classes x sequence_length x embedding_size
            self.embedded_r = tf.nn.embedding_lookup(W, self.input_x_r)
            print "DEBUG: embedded_r -> %s" % self.embedded_r
            # batch_size x sequence_length x embedding_size x 1
            self.embedded_u_expanded = tf.expand_dims(self.embedded_u, -1)
            print "DEBUG: embedded_u_expanded -> %s" % self.embedded_u_expanded
            # batch_size x num_classes x sequence_length x embedding_size x 1
            self.embedded_r_expanded = tf.expand_dims(self.embedded_r, -1)
            print "DEBUG: embedded_r_expanded -> %s" % self.embedded_r_expanded
        
        # Create a convolution + maxpooling layer for each filter size
        pooled_outputs_u = []
        pooled_outputs_r = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s-u" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), 
                                name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), 
                                name='b')
                conv_u = tf.nn.conv2d(
                    self.embedded_u_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-u")             
                # Apply nonlinearity
                h_u = tf.nn.relu(tf.nn.bias_add(conv_u, b), name="relu-u")

                # Maxpooling over outputs
                pooled_u = tf.nn.max_pool(
                    h_u,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool-u")
                pooled_outputs_u.append(pooled_u)
                
                # Pass each element in x_r through the same layer
                pooled_outputs_r_wclasses = []
                for j in range(num_classes):
                    embedded_r = self.embedded_r_expanded[:, j, :, :, :]
                    conv_r_j = tf.nn.conv2d(
                        embedded_r,
                        W, 
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv-r-%s" % j)
                    
                    h_r_j = tf.nn.relu(tf.nn.bias_add(conv_r_j, b), name="relu-r-%s" % j)
                    
                    pooled_r_j = tf.nn.max_pool(
                        h_r_j,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool-r-%s" % j)
                    pooled_outputs_r_wclasses.append(pooled_r_j)
                # out_tensor: batch_size x 1 x num_class x num_filters
                out_tensor = tf.concat(2, pooled_outputs_r_wclasses)
                pooled_outputs_r.append(out_tensor)
                    
        
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print "DEBUG: pooled_outputs_u -> %s" % pooled_outputs_u
        self.h_pool_u = tf.concat(3, pooled_outputs_u)
        print "DEBUG: h_pool_u -> %s" % self.h_pool_u
        # batch_size x 1 x num_filters_total
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, 1, num_filters_total])
        print "DEBUG: h_pool_flat_u -> %s" % self.h_pool_flat_u
        
        
        print "DEBUG: pooled_outputs_r -> %s" % pooled_outputs_r
        self.h_pool_r = tf.concat(3, pooled_outputs_r)
        print "DEBUG: h_pool_r -> %s" % self.h_pool_r
        # h_pool_flat_r: batch_size x num_classes X num_filters_total
        self.h_pool_flat_r = tf.reshape(self.h_pool_r, [-1, num_classes, num_filters_total])
        print "DEBUG: h_pool_flat_r -> %s" % self.h_pool_flat_r
        
        # Add dropout layer to avoid overfitting
        with tf.name_scope("dropout"):
            self.h_features = tf.concat(1, [self.h_pool_flat_u, self.h_pool_flat_r])
            print "DEBUG: h_features -> %s" % self.h_features
            self.h_features_dropped = tf.nn.dropout(self.h_features, 
                                                        self.dropout_keep_prob, 
                                                        noise_shape=[batch_size, 1, num_filters_total])

            self.h_dropped_u = self.h_features_dropped[:, :1, :] + 0.05
            self.h_dropped_r = self.h_features_dropped[:, 1:, :] + 0.05
        
        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            dot =  tf.reduce_sum(tf.mul(self.h_dropped_u, 
                                        self.h_dropped_r), 2)
            print "DEBUG: dot -> %s" % dot
            sqrt_u = tf.sqrt(tf.reduce_sum(self.h_dropped_u**2, 2))
            print "DEBUG: sqrt_u -> %s" % sqrt_u
            sqrt_r = tf.sqrt(tf.reduce_sum(self.h_dropped_r**2, 2))
            print "DEBUG: sqrt_r -> %s" % sqrt_r
            self.cosine = dot / (sqrt_u * sqrt_r + 0.05)
            print "DEBUG: cosine -> %s" % self.cosine
            self.predictions = tf.argmax(self.cosine, 1, name="predictions")
            print "DEBUG: predictions -> %s" % self.predictions
        
        # softmax regression - loss and prediction
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(100 * self.cosine, self.input_y)
            self.loss = tf.reduce_mean(losses)
            
        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")        
        