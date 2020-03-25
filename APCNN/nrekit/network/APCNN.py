import nrekit
import tensorflow as tf
class APCNN:
    def __init__(self, data_loader, train_mode='Base', constraint_path=None, CL_rate=0.01, is_training=True):
        batch_size, max_length, word_embedding_dim = data_loader.batch_size, data_loader.max_length, data_loader.word_embedding_dim
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        # self.data_loader = data_loader
        self.rel_tot = data_loader.rel_tot
        self.word_vec_mat = data_loader.word_vec_mat
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
        # Embedding
        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2,
                                                             word_embedding_dim=word_embedding_dim, max_length=max_length)
        # Encoder
        x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
        x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
        # Selector
        self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
        self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
        self._test_logit = tf.add(self._test_logit, tf.constant(0.0), name='get_prob')
        if is_training:
            self.prob = tf.nn.softmax(self._train_logit, axis=-1)
            self.label_onehot = tf.one_hot(indices=self.label, depth=self.rel_tot, dtype=tf.int32)
            self._entropy_loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot)
            if train_mode == 'Base':
                self.constraint_loss = tf.constant(0.0)
            else:
                CLC = nrekit.network.ConstraintLoss.ConstraintLoss(batch_size, data_loader.rel_tot, data_loader.rel2id, constraint_path, CL_rate, self)
                if train_mode == 'Sem':
                    self.constraint_loss = CLC.Semantic()
                else:
                    self.constraint_loss = CLC.Coherent()
            # self.constraint_loss = nrekit.network.SL_loss.__constraint_loss__(self)

    def loss(self):
        self.total_loss = self._entropy_loss + self.constraint_loss
        return self.total_loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit