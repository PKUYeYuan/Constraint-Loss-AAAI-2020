# _*_coding=utf-8_*_
import tensorflow as tf
from ConstraintLoss import ConstraintLoss

class ACNN:
    def __init__(self, is_training, DataLoader, train_mode='base', CL_rate=0.0):
        # the hyper-parameter for ACNN model
        self.HyperP = {
            'pos_feat_dim': 5, # position feature (relative to subject and object) embedding dim
            'POS_dim': 50, # POS feature embedding dim
            'pos_max_distance': 60, # the max distance between sentence word and marked entities
            'cnn_feat_dim': 256,
           }
        self.DataLoader = DataLoader
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # use pre-trained word vector to initial word embedding
        self.word_embedding = tf.get_variable(initializer=DataLoader.word_vec, dtype=tf.float32, name='word_embedding')
        # Basic Model placeholder information
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, DataLoader.sent_len], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, DataLoader.sent_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, DataLoader.sent_len], name='input_pos2')
        self.input_POS = tf.placeholder(dtype=tf.int32, shape=[None, DataLoader.sent_len], name='input_POS')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, DataLoader.class_num], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[DataLoader.batchsize + 1], name='total_shape')

        self.bag_repre = self.__bag_repre__()
        self.prob = tf.nn.softmax(self.bag_repre, -1, name='get_prob')
        self.predictions = self.__predictions__(self.prob)
        self.accuracy = self.__accuracy__(self.predictions, self.input_y)
        if is_training:
            self.entropy_loss = self.__entropy_loss__(self.bag_repre, self.input_y)
            if train_mode=='Base':
                self.constraint_loss = tf.constant(0.0)
            else:
                CLC = ConstraintLoss(DataLoader, CL_rate, self)
                if train_mode == 'Sem':
                    self.constraint_loss = CLC.Semantic()
                else:
                    self.constraint_loss = CLC.Coherent()
            self.total_loss = self.entropy_loss + self.constraint_loss
            self.loss_dict = {'entropy_loss': self.entropy_loss, 'constraint_loss': self.constraint_loss, 'total_loss': self.total_loss}

    def CNNEncoder(self, input_data):
        # Convolution
        with tf.variable_scope("Layer_0"):
            conv_output = tf.layers.conv1d(inputs=input_data, filters=self.HyperP['cnn_feat_dim'], kernel_size=3, padding='same', activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            pooling_output = tf.reduce_max(conv_output, reduction_indices=[1], keep_dims=False)
        return tf.nn.relu(pooling_output)

    def __bag_repre__(self):
        self.pos_num = self.HyperP['pos_max_distance'] * 2 + 3
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.HyperP['pos_feat_dim']], dtype=tf.float32)
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.HyperP['pos_feat_dim']], dtype=tf.float32)
        POS_embedding = tf.get_variable('POS_embedding', [self.DataLoader.POS_num, self.HyperP['POS_dim']], dtype=tf.float32)
        sen_r = tf.get_variable('query_r', [self.HyperP['cnn_feat_dim'], 1], dtype=tf.float32)
        relation_embedding = tf.get_variable('relation_embedding', [self.HyperP['cnn_feat_dim'], self.DataLoader.class_num], dtype=tf.float32)
        sen_d = tf.get_variable('bias_d', [self.DataLoader.class_num], dtype=tf.float32)
        # embedding layer
        inputs_embeddings = tf.concat(axis=2, values=[tf.nn.embedding_lookup(self.word_embedding, self.input_word), tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                      tf.nn.embedding_lookup(pos2_embedding, self.input_pos2), tf.nn.embedding_lookup(POS_embedding, self.input_POS)])
        _embedding = self.CNNEncoder(input_data=inputs_embeddings)
        sen_bag, sen_alpha, sen_s = [], [], []
        # sentence-level attention layer
        for i in range(self.DataLoader.batchsize):
            sen_bag = _embedding[self.total_shape[i]:self.total_shape[i + 1]] # 将每一个bag包含的sentence分离
            sen_bag_size = self.total_shape[i + 1] - self.total_shape[i]
            sen_alpha = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(sen_bag, sen_r), [sen_bag_size])), [1, sen_bag_size])
            sen_s.append(tf.reshape(tf.matmul(sen_alpha, sen_bag), [1, self.HyperP['cnn_feat_dim']]))
        sen_s = tf.concat(sen_s, axis=0)
        sen_out = tf.matmul(tf.nn.relu(sen_s), relation_embedding) + sen_d
        return sen_out

    def __predictions__(self, prob):
        with tf.name_scope("output"):
            predictions = (tf.argmax(prob, 1, name="predictions"))
        return predictions

    def __entropy_loss__(self, bag_repre, label):
        with tf.name_scope("entropy_loss"):
            entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=bag_repre, labels=label))
        return entropy_loss

    def __accuracy__(self, predictions, input_y):
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.argmax(input_y, 1)), "float"))
        return accuracy

