import numpy as np
import tensorflow as tf
import datetime
import time
import os
import data_helpers as dh
from sent_cnn import SentCNN

def load_data(config):
    """
    Load training examples and pretrained word embeddings from disk.
    Return training inputs, labels and pretrianed embeddings.
    """
    # Load raw data
    wq_file = config["webquestions_examples_file"]
    n_neg_sample = config["num_classes"] - 1
    x_u, x_r, y, max_len = dh.get_training_examples_for_softmax(wq_file, n_neg_sample)
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

def train_cnn(x_u_i, x_r_i, y, max_len, U, config, debug=True):
    
    cnn = SentCNN(sequence_length=max_len, 
                  num_classes=config["num_classes"], 
                  init_embeddings=U, 
                  filter_sizes=config["filter_sizes"], 
                  num_filters=config["num_filters"],
                  batch_size=config["batch_size"],
                  embeddings_trainable=config["embeddings_trainable"],
                  l2_reg_lambda=config["l2_reg_lambda"])
    
    total_iter = config["total_iter"]
    batch_size = config["batch_size"]
    global_step = tf.Variable(0, name="global_step", trainable=True)
    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
     
    capped_gvs = grads_and_vars#[(tf.clip_by_value(gv[0], -55.0, 55.0), gv[1]) if gv[0] is not None else 
                   #gv for gv in grads_and_vars]
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
        
        # summary
        #grad_summaries = []
        #for g, v in grads_and_vars:
        #    if g is not None:
        #        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
        #        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #        grad_summaries.append(grad_hist_summary)
        #        grad_summaries.append(sparsity_summary)
        #grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        if debug==False:
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())
        
        for _ in range(total_iter):
            indices = np.random.choice(len(x_u_train), batch_size)
            x_u_batch = x_u_train[indices]
            x_r_batch = x_r_train[indices]
            y_batch = y_train[indices]
            
            # Training procedures
            feed_dict = {
                cnn.input_x_u: x_u_batch, 
                cnn.input_x_r: x_r_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob:config["dropout_keep_prob"]
                
            }
            if debug == False:
                _, step, summaries, loss, accuracy, u, r, dot, su, sr, cosine = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.h_dropped_u, 
                    cnn.h_dropped_r, cnn.dot, cnn.sqrt_u, cnn.sqrt_r, cnn.cosine], feed_dict)
            else:
                _, step, loss, accuracy, u, r, dot, su, sr, cosine, undropped = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.h_dropped_u, 
                    cnn.h_dropped_r, cnn.dot, cnn.sqrt_u, cnn.sqrt_r, cnn.cosine, cnn.h_features], feed_dict)
                gvs = sess.run([gv[0] if gv[0] is not None else cnn.loss for gv in capped_gvs], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if debug == False:
                train_summary_writer.add_summary(summaries, step)
            if step % 50 == 0:
                feed_dict = {
                    cnn.input_x_u: x_u_val,
                    cnn.input_x_r: x_r_val,
                    cnn.input_y: y_val,
                    cnn.dropout_keep_prob:1
                }
                if debug == False:
                    dev_loss, dev_accuracy, summaries = sess.run(
                        [cnn.loss, cnn.accuracy, train_summary_op], feed_dict)
                else:
                    dev_loss, dev_accuracy = sess.run(
                        [cnn.loss, cnn.accuracy], feed_dict)
                print("{}: step {}, train loss {:g}, train acc {:g}, dev loss {:g}, dev acc {:g}".format(
                        time_str, step, loss, accuracy, dev_loss, dev_accuracy)) 
                
                #print gvs
                '''print undropped
                print "----------------------------------"
                print u
                print "----------------------------------"
                print r
                print "----------------------------------"
                print dot
                print "----------------------------------"
                print su
                print "----------------------------------"
                print sr
                print "----------------------------------"
                print cosine'''
                if debug == False:
                    if dev_summary_writer:
                        dev_summary_writer.add_summary(summaries, step)