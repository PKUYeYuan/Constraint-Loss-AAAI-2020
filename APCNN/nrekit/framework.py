import os, sklearn.metrics, sys, time, json, matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.python.framework import graph_util
matplotlib.use('Agg')
# from nrekit.network.Conf import Conf_Para
# from nrekit.network.SL_loss import __SL_attention__
# from nrekit.network.SL_loss import __new_attention__
from nrekit.network.APCNN import APCNN
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class re_framework:
    MODE_BAG = 0 # Train and test the model at bag level.
    MODE_INS = 1 # Train and test the model at instance level

    def __init__(self, train_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def train_one_step(self, sess, model, batch_data_gen, run_array, train_mode='Base'):
        feed_dict = {}
        batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size)
        feed_dict.update({
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        })
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask'],})
        if train_mode != 'Base':
            feed_dict.update({
                model.IndicatorInput['SubjRel']: batch_data['subj_rel_set'],
                model.IndicatorInput['RelObj']: batch_data['rel_obj_set'],
                model.IndicatorInput['SubjRelObj']: batch_data['subj_rel_obj_set'],
                model.IndicatorInput['SubjMulti']: batch_data['subj_multi_set'],
                model.IndicatorInput['ObjMulti']: batch_data['obj_multi_set'],
            })
        result = sess.run(run_array, feed_dict)
        return result



    def train(self, args, test_result_dir='./test_result'):
        print("Start training...")
        constraint_path = os.path.join(args.data_dir, 'Constraints')
        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            with sess.as_default():
                with tf.name_scope("model"):
                    model = APCNN(self.train_data_loader, train_mode=args.train_mode, constraint_path=constraint_path, CL_rate=args.CL_rate, is_training=True)
                optimizer = tf.train.GradientDescentOptimizer(0.5)
                train_op = optimizer.minimize(model.loss())
                sess.run(tf.global_variables_initializer())
                # Training
                best_metric, best_prec, best_recall= 0, None, None
                not_best_count = 0 # Stop training after several epochs without improvement.
                for epoch in range(args.max_epoch):
                    print('###### Epoch %d ######' %(epoch))
                    step = 0
                    while True:
                        RunList = [model.total_loss, model._entropy_loss, model.constraint_loss, model._train_logit, train_op]
                        try:
                            TotalL, EntropyL, ConstraintL, iter_logit, _ = self.train_one_step(sess, model, self.train_data_loader, RunList, train_mode=args.train_mode)
                        except StopIteration:
                            break
                        if step % 100 == 0:
                            print("Step%d/Epoch%d TotalLoss: %.4f, EntropyLoss: %.4f, ConstraintLoss: %.4f"  % (step, epoch, TotalL, EntropyL, ConstraintL))
                        step += 1
                    if not os.path.isdir("./Model/"):
                        os.mkdir("./Model/")
                    graph = tf.get_default_graph()
                    input_graph_def = graph.as_graph_def()
                    constant_graph = graph_util.convert_variables_to_constants(sess, input_graph_def=input_graph_def, output_node_names=['model/get_prob'])
                    with tf.gfile.GFile("./Model/%s-%s-tmp.pb" % (args.dataset, args.train_mode), mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
                    metric, p, r = self.test_withpb(restore_path="./Model/%s-%s-tmp.pb" %(args.dataset, args.train_mode), args=args)
                    if metric > best_metric:
                        best_metric, best_prec, best_recall = metric, p, r
                        print("Best model, storing...")
                        with tf.gfile.GFile("./Model/%s-%s-best.pb" % (args.dataset, args.train_mode), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        print("Finish storing")
                        not_best_count = 0
                    else:
                        not_best_count += 1

                    if not_best_count >= 20:
                        break

                print("######")
                print("Finish training ")
                print("Best epoch auc = %f" % (best_metric))
                write_file = open('../../result.txt', 'a+')
                write_file.write("%.5f %d :Best epoch auc = %f \n" % (args.CL_rate, 1, best_metric))
                write_file.close()
                if (not best_prec is None) and (not best_recall is None):
                    if not os.path.isdir(test_result_dir):
                        os.mkdir(test_result_dir)
                    np.save(os.path.join(test_result_dir, "recall.npy"), best_recall)
                    np.save(os.path.join(test_result_dir, "precision.npy"), best_prec)

    def test(self, ckpt=None, output_path=None):
        print("Testing...")
        data_loader = self.test_data_loader
        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            with sess.as_default():
                with tf.name_scope("model"):
                    model = APCNN(data_loader, is_training=False)
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
                graph = tf.get_default_graph()
                input_graph_def = graph.as_graph_def()
                constant_graph = graph_util.convert_variables_to_constants(sess, input_graph_def=input_graph_def, output_node_names=['model/get_prob'])
                with tf.gfile.GFile(output_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                logit_list = []
                rel_label_list = []
                for i, batch_data in enumerate(data_loader):
                    iter_logit = self.one_step(sess, model, batch_data, [model.test_logit()])[0]
                    logit_list.append(iter_logit)
                    rel_label_list.append(batch_data['multi_rel'])

                # 根据label, logit计算auc
                rel_label_list = np.reshape(rel_label_list, newshape=[-1, len(data_loader.rel2id)])[:len(data_loader.entpair2scope)]
                logit_list = np.reshape(logit_list, newshape=[-1, len(data_loader.rel2id)])[:len(data_loader.entpair2scope)]
                y_true = np.reshape(rel_label_list[:, 1:], -1)
                y_predict = np.reshape(logit_list[:, 1:], -1)
                p, r, _ = sklearn.metrics.precision_recall_curve(y_true, y_predict)
                auc = sklearn.metrics.auc(x=r, y=p)
                print("[TEST] auc: {}".format(auc))
                print("Finish testing")
                return auc, p, r
        
    def test_withpb(self, restore_path, args):
        print("Testing...")
        data_loader = self.test_data_loader
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with tf.gfile.GFile(restore_path, mode='rb') as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            with sess.as_default():
                sess.run(tf.global_variables_initializer())
                logit_list = []
                rel_label_list = []
                for i, batch_data in enumerate(data_loader):
                    iter_logit = self.pb_one_step(sess, batch_data)
                    logit_list.append(iter_logit)
                    rel_label_list.append(batch_data['multi_rel'])
                # 根据label, logit计算auc
                rel_label_list = np.reshape(rel_label_list, newshape=[-1, len(data_loader.rel2id)])[:len(data_loader.entpair2scope)]
                logit_list = np.reshape(logit_list, newshape=[-1, len(data_loader.rel2id)])[:len(data_loader.entpair2scope)]
                y_true = np.reshape(rel_label_list[:, 1:], -1)
                y_predict = np.reshape(logit_list[:, 1:], -1)
                p, r, _ = sklearn.metrics.precision_recall_curve(y_true, y_predict)
                auc = sklearn.metrics.auc(x=r, y=p)
                self.MetricsEvaluation(p, r, auc, args)
                print("[TEST] auc: {}".format(auc))
                print("Finish testing")
                return auc, p, r

    def one_step(self, sess, model, batch_data, run_array):
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        }
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})
        result = sess.run(run_array, feed_dict)
        return result

    def pb_one_step(self, sess, batch_data):
        logit_op = sess.graph.get_tensor_by_name('model/get_prob:0')
        input_word = sess.graph.get_tensor_by_name('model/word:0')
        input_pos1 = sess.graph.get_tensor_by_name('model/pos1:0')
        input_pos2 = sess.graph.get_tensor_by_name('model/pos2:0')
        input_scope = sess.graph.get_tensor_by_name('model/scope:0')
        input_mask = sess.graph.get_tensor_by_name('model/mask:0')
        feed_dict = {
            input_word: batch_data['word'],
            input_pos1: batch_data['pos1'],
            input_pos2: batch_data['pos2'],
            input_scope: batch_data['scope'],
            input_mask: batch_data['mask'],
        }
        result = sess.run(logit_op, feed_dict)
        return result

    def MetricsEvaluation(self, p, r, auc_score, args):
        f1 = (2 * r * p / (r + p + 1e-20)).max()
        plt.plot(r, p, lw=2, label=str(args.train_mode) + str(auc_score))
        print("result: auc = %.4f, max F1 = %.4f" % (auc_score, f1))
        I = [99, 199, 299]
        ordered_p = sorted(p, reverse=True)
        print("P@100: %.4f, P@200: %.4f, P@300: %.4f, Mean: %.4f"
              % (ordered_p[I[0]], ordered_p[I[1]], ordered_p[I[2]],
                 (ordered_p[I[0]] + ordered_p[I[1]] + ordered_p[I[2]]) / 3))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.3, 1.0])
        plt.xlim([0.0, 0.4])
        plt.title(args.dataset + '-' + args.train_mode)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig('./test_result/%s-%s-pr_curve.png' % (args.dataset, args.train_mode))