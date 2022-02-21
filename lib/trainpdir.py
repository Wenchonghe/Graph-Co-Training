import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
import sklearn
import scipy.sparse
import numpy as np
import json
import graph
import random
import os, time, collections, shutil
import scipy.sparse as sp
from scipy import sparse
from sklearn import preprocessing
import matplotlib.pyplot as plt


class gnn():
  
    def __init__(self,classnum,  F, K , F_0=1,F_1=1, M_0 = 1000, filter='chebyshev5', brelu='b1relu',  num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=1, eval_frequency=200,
                dir_name=''):
        
        super(gnn, self).__init__()
        self.regularizers = []
        
        print('NN architecture')
    
        # Store attributes and bind operations.
        self.classnum, self.F, self.K, self.F_0,self.F_1, self.M_0 = classnum,  F, K, F_0,F_1, M_0
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        
        #self.loss_weights = weight_tensor
        # Build the computational graph.
        self.build_graph(M_0, F_0,F_1)



    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = [0]*size
        sess = self._get_session(sess)
        #sess = tf.Session(graph=self.graph)
        #sess.run(self.op_init)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, labels.shape[1], labels.shape[2]))
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
        
        predictions = np.array(predictions)    
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.clock(), time.time()
        #print(labels.shape, data.shape)
        predictions, loss = self.predict(data, labels, sess)
        #print(predictions)
        #ncorrects = sum(predictions == labels)
        #accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        #f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        #string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
        #        accuracy, ncorrects, len(labels), f1, loss)
        string = 'loss: {:.2e}'.format(loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.clock()-t_process, time.time()-t_wall)
        return string, loss

    def fit(self, train_data, infer_labels,  labels, train_idx, val_idx, L1,L2):
        t_process, t_wall = time.clock(), time.time()
        sess = tf.Session(graph=self.graph)
        #shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        #writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        #shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        #os.makedirs(self._get_path('checkpoints'))
        #path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        acc_train = []
        loss_train = []
        loss_average2 = []
        accuracy_average2 = []
        
        acc_val = []
        loss_val= []
        
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
                
            idx = indices.popleft()
            #print(idx)
            batch_data, batch_labels = train_data[idx], infer_labels[idx]
            batch_data = np.reshape(batch_data, [self.batch_size, batch_data.shape[0], batch_data.shape[1]])
            batch_labels = np.reshape(batch_labels, [self.batch_size, batch_labels.shape[0],batch_labels.shape[1] ] )
            if step==1:
                print(batch_data.shape, batch_labels.shape)
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
                
                
                
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout,self.is_training: 1,
                        self.L1sp0_indices: [[i, j] for i, j in zip(*L1[idx].nonzero())],  self.L1sp0_val: L1[idx].data, self.L1sp0_shape: L1[idx].shape,
                        self.L2sp0_indices: [[i, j] for i, j in zip(*L2[idx].nonzero())],  self.L2sp0_val: L2[idx].data, self.L2sp0_shape: L2[idx].shape}
            learning_rate, loss_average,accuracy_average = sess.run([self.op_train, self.op_loss_average, self.accuracy], feed_dict)
            #print("loss_average is %f", loss_average)
            #print("average accuracy is ", accuracy_average)
            accuracy_average2.append(accuracy_average)
            loss_average2.append(loss_average)
            
            
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, np.mean(loss_average2)))
                print('  accuracy_average = {:.2e}'.format(np.mean(accuracy_average2)))
                acc_train.append(np.mean(accuracy_average2))
                loss_train.append(np.mean(loss_average2))
                loss_average2 = []
                accuracy_average2 = []
                
#             # Periodical evaluation of the model.
#             if  step == num_steps:
#                 epoch = step * self.batch_size / train_data.shape[0]
#                 print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
#                 print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                
                val_data = train_data[val_idx]
                val_labels = infer_labels[val_idx]
                val_true_labels = labels[val_idx]
                
                loss_val_batch = []
                acc_val_batch = []
                
                val_predict = []
                for i in range(val_data.shape[0]):
                    val_L1 = L1[val_idx[i]]
                    val_L2 = L2[val_idx[i]]
                    batch_valdata = val_data[i]
                    batch_vallabel = val_labels[i]
                    batch_valtruelabel = val_true_labels[i]
                    batch_valdata = np.reshape(batch_valdata, [self.batch_size, batch_valdata.shape[0], batch_valdata.shape[1]])
                    batch_vallabel = np.reshape(batch_vallabel, [self.batch_size, batch_vallabel.shape[0], batch_vallabel.shape[1]])
                    batch_valtruelabel = np.reshape(batch_valtruelabel, [self.batch_size, batch_valtruelabel.shape[0], batch_valtruelabel.shape[1]])
                    feed_dict = {self.ph_data:batch_valdata ,self.ph_labels:batch_vallabel , self.is_training: 0,
                                self.val_truelabels: batch_valtruelabel, self.ph_dropout: self.dropout,
                                self.L1sp0_indices: [[i, j] for i, j in zip(*val_L1.nonzero())],  self.L1sp0_val: val_L1.data, self.L1sp0_shape: val_L1.shape,
                                self.L2sp0_indices: [[i, j] for i, j in zip(*val_L2.nonzero())],  self.L2sp0_val: val_L2.data, self.L2sp0_shape: val_L2.shape}
                    batch_pred, batch_loss, accuracy_all = sess.run([self.op_prediction, self.op_loss, self.val_accuracy], feed_dict)
#                     correct_prediction = tf.equal(tf.argmax(batch_pred, 2), tf.argmax(batch_valtruelabel,2))
#                     accuracy_all = tf.cast(correct_prediction, tf.float32)
#                     accuracy_all = tf.reduce_mean(accuracy_all)
                    loss_val_batch.append(batch_loss)
                    acc_val_batch.append(accuracy_all)
                    
                    val_predict.append(batch_pred)
                    
                acc_val_batch = np.array(acc_val_batch)
                acc_val.append(np.mean(acc_val_batch))
                loss_val.append(np.mean(loss_val_batch))
                print('  learning_rate = {:.2e}, Validation_loss_average = {:.2e}'.format(learning_rate, np.mean(loss_val_batch)))
                print('  accuracy_average = {:.2e}'.format(np.mean(acc_val_batch)))
                
                plt.plot(acc_train)
                plt.plot(acc_val)
                plt.ylabel('accuracy')
                plt.show()
                plt.plot(loss_train)
                plt.plot(loss_val)
                plt.ylabel('loss')
                plt.show()
                
                
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.clock()-t_process, time.time()-t_wall))


#                 summary = tf.Summary()
#                 summary.ParseFromString(sess.run(self.op_summary, feed_dict))

            #summary.value.add(tag='validation/loss', simple_value=loss)
            #writer.add_summary(summary, step)
                

            #self.op_saver.save(sess, path, global_step=step)

        #writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return val_predict

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def build_graph(self, M_0, F_0,F_1):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, None, F_0), 'data')
                self.ph_labels = tf.placeholder(tf.int64, (self.batch_size, None, F_1), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.is_training = tf.placeholder(tf.float32, (), 'is_training')
                
                self.L1sp0_indices = tf.placeholder(tf.int64)
                self.L1sp0_shape = tf.placeholder(tf.int64)
                self.L1sp0_val = tf.placeholder(tf.float32)
                self.L1 = tf.SparseTensor(self.L1sp0_indices, self.L1sp0_val, self.L1sp0_shape)
                
                self.L2sp0_indices = tf.placeholder(tf.int64)
                self.L2sp0_shape = tf.placeholder(tf.int64)
                self.L2sp0_val = tf.placeholder(tf.float32)
                self.L2 = tf.SparseTensor(self.L2sp0_indices, self.L2sp0_val, self.L2sp0_shape)
                
                self.val_truelabels = tf.placeholder(tf.int64, (self.batch_size, None, F_1), 'val_truelabels')

            # Model.
            op_outputs = self.inference(self.ph_data, self.ph_dropout, self.L1,self.L2, self.is_training)
            self.op_loss, self.op_loss_average = self.loss(op_outputs, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.softmaxlay(op_outputs)
            self.accuracy  = self.compu_accuracy(self.ph_labels, self.op_prediction)
            self.val_accuracy = self.compu_accuracy(self.val_truelabels, self.op_prediction)
    
            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=200)
        
        self.graph.finalize()
        
    def softmaxlay(self, outputs):
        outputs  = tf.nn.softmax(outputs, axis = -1)
        return outputs
    def compu_accuracy(self, labels, preds):
        correct_prediction = tf.equal(tf.argmax(preds, 2), tf.argmax(labels,2))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)
        
    def loss(self, outputs, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('data_loss'):
                #labels = tf.one_hot(labels, self.classnum, axis=-1)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = outputs)
                data_loss  = tf.reduce_mean(cross_entropy)
                #cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = data_loss + regularization
            
            #print(loss, l2_loss, regularization)
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/data_loss', data_loss)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([data_loss, regularization, loss])
                tf.summary.scalar('loss/avg/data_loss', averages.average(data_loss))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var


    def chebyshev5(self, x, L1,L2, Fout, K):
        N, M, Fin = x.get_shape()
        #N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        #L = scipy.sparse.csr_matrix(L)
        #L = graph.rescale_L(L, lmax=2)
        #L = L.tocoo()
        #indices = np.column_stack((L.row, L.col))
        #L = tf.SparseTensor(indices, L.data, L.shape)
        L1 = tf.sparse_reorder(L1)
        L2 = tf.sparse_reorder(L2)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [-1, Fin*N])  # M x Fin*N
        x3 = x0
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        x = tf.concat([x,x], axis = 0)
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L1, x0)
            x = concat(x, x1)
            
            x4 = tf.sparse_tensor_dense_matmul(L2, x3)
            x = concat(x, x4)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L1, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
            
            x5 = 2 * tf.sparse_tensor_dense_matmul(L2, x4) - x3
            x = concat(x, x5)
            x3, x4 = x4, x5
            
        x = tf.reshape(x, [2*K, -1, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*2*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*2*K, Fout], regularization=True)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, -1, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=True)
        return (x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)
        
    def batch_norm_wrapper(self, inputs, is_training, decay = 0.999):
        epsilon = 1e-3
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training==1:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
            train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)


    def call(self, x,dropout, L1,L2, is_training, reuse=False):
        rate = 0.3
        with tf.variable_scope('gnn1', reuse=reuse):
            N, Min, Fin = x.get_shape()
            with tf.name_scope('filter1'):
                x = self.filter(x, L1,L2, self.F, self.K)
                print(L1, self.F, self.K)
            with tf.name_scope('bias_relu1'):
                x = self.brelu(x)
                x = self.batch_norm_wrapper(x, is_training)
                x = tf.nn.relu(x)
                if dropout==1 and is_training==1:
                    x = tf.nn.dropout(x, rate)
                    
        with tf.variable_scope('gnn2', reuse=reuse):
            N, Min, Fin = x.get_shape()
            with tf.name_scope('filter2'):
                x = self.filter(x, L1,L2, self.F, self.K)
                print(L1, self.F, self.K)
            with tf.name_scope('bias_relu2'):
                x = self.brelu(x)
                x = self.batch_norm_wrapper(x, is_training)
                x = tf.nn.relu(x)
                if dropout==1 and is_training==1:
                    x = tf.nn.dropout(x, rate)
        with tf.variable_scope('gnn4', reuse=reuse):
            N, Min, Fin = x.get_shape()
            with tf.name_scope('filter4'):
                x = self.filter(x, L1,L2, self.F, self.K)
                print(L1, self.F, self.K)
            with tf.name_scope('bias_relu4'):
                x = self.brelu(x)
                x = self.batch_norm_wrapper(x, is_training)
                x = tf.nn.relu(x)
                if dropout==1 and is_training==1:
                    x = tf.nn.dropout(x, rate)
                    
        with tf.variable_scope('gnn3', reuse=reuse):
            with tf.name_scope('filter3'):
                x = self.filter(x, L1,L2, self.classnum, self.K)
                print(L1, self.classnum, self.K)
            with tf.name_scope('bias_relu3'):
                x = self.brelu(x)
                    
        return x


    def inference(self, x, dropout, L1,L2, is_training ):
        x = self.call(x,dropout, L1,L2, is_training)

        return x
    
#def laplacian(W):
    #print("W.shape = ",W.shape)
    #d = W.sum(axis = 0)
    #d += np.spacing(np.array(0, W.dtype))
    #d = 1 / np.sqrt(d)
    #D = sp.diags(d.A.squeeze(), 0)
    #print("D.shape = ",D.shape)
    #I = sp.identity(d.size, dtype = W.dtype)
    #L = I - D*W*D
    #L = I - (D.dot(W)).dot(D)
    #return L 
