import numpy as np
import tensorflow as tf
import datetime
import data_helpers as dh
from sent_cnn import SentCNN

def load_data(config):
    """
    Load training examples and pretrained word embeddings from disk.
    Return training inputs, labels and pretrianed embeddings.
    """
    # Load raw data
    wq_file = config["webquestions_examples_file"]
    x_u, x_r, y, max_len = dh.get_training_examples_for_softmax(wq_file)
    # Pad sentences
    pad = lambda x: dh.pad_sentences(x, max_len)
    pad_lst = lambda x: map(pad, x)
    x_u = map(pad, x_u)
    x_r = map(pad_lst, x_r)
    # Load tokens and pretrained embeddings
    we_file = config["word_embeddings_file"]
    voc_size = config["vocabulary_size"]
    embedding_size = config["embedding_size"]
    tokens, U = dh.get_pretrained_wordvec_from_file(we_file, (voc_size, embedding_size))
    # Represent sentences as list(nparray) of ints
    dctize = lambda word: tokens[word] if tokens.has_key(word) else tokens["pad"]
    dctizes = lambda words: map(dctize, words)
    dctizess = lambda wordss: map(dctizes, wordss)
    x_u_i = np.array(map(dctizes, x_u))
    x_r_i = np.array(map(dctizess, x_r))
    y = np.array(y)
    
    return (x_u_i, x_r_i, y, max_len, U)

def train_cnn(x_u_i, x_r_i, y, max_len, U, config):
    
    cnn = SentCNN(sequence_length=max_len, 
                  num_classes=config["num_classes"], 
                  init_embeddings=U, 
                  filter_sizes=config["filter_sizes"], 
                  num_filters=config["num_filters"],
                  embeddings_trainable=config["embeddings_trainable"])
    
    total_iter = config["total_iter"]
    batch_size = config["batch_size"]
    global_step = tf.Variable(0, name="global_step", trainable=True)
    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    val_size = config["val_size"]
    
    x_u_val = x_u_i[:val_size]
    x_u_train = x_u_i[val_size+1:]
    x_r_val = x_r_i[:val_size]
    x_r_train = x_r_i[val_size+1:]
    y_val = y[:val_size]
    y_train = y[val_size+1:]
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for _ in range(total_iter):
            indices = np.random.choice(len(x_u_train), batch_size)
            x_u_batch = x_u_train[indices]
            x_r_batch = x_r_train[indices]
            y_batch = y_train[indices]
            
            feed_dict = {
                cnn.input_x_u: x_u_batch, 
                cnn.input_x_r: x_r_batch,
                cnn.input_y: y_batch
            }
            _, step, loss, accuracy, cosine, pred = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy, cnn.cosine, cnn.predictions], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 50 == 0:
                feed_dict = {
                    cnn.input_x_u: x_u_val,
                    cnn.input_x_r: x_r_val,
                    cnn.input_y: y_val
                }
                dev_loss, dev_accuracy = sess.run(
                    [cnn.loss, cnn.accuracy], feed_dict)
                print("{}: step {}, train loss {:g}, train acc {:g}, dev loss {:g}, dev acc {:g}".format(
                        time_str, step, loss, accuracy, dev_loss, dev_accuracy))  
                print pred[:5], y_batch[:5]