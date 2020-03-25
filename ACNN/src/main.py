# _*_coding=utf-8_*_
import json, datetime, argparse, sys, os, matplotlib, pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from ACNN import ACNN
from sklearn.metrics import precision_recall_curve, auc
from DataHandler import DataLoader, DataPreprocessor
from tensorflow.python.framework import graph_util
matplotlib.use('Agg')

def train_step(sess, run_list, feed_dict, loss_sum, step_count):
    _, step, accuracy, loss_dict_out = sess.run(run_list, feed_dict)
    for key in loss_sum:
        loss_sum[key] += loss_dict_out[key]
    if step % 100 == 0:
        print("Step %d, OriginalLoss %.4f, ConstraintLoss %.4f, TotalLoss %.4f, Acc %.4f \r"
                         % (step, loss_sum['entropy_loss'] / step_count, loss_sum['constraint_loss'] / step_count, loss_sum['total_loss'] / step_count, accuracy))
    return loss_sum

def train(args):
    TrainDataLoader = DataLoader(args, state='Train')
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        with sess.as_default():
            print('Constructing ACNN model with train mode %s ......' %(args.train_mode))
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
                model = ACNN(is_training=True, DataLoader=TrainDataLoader, train_mode=args.train_mode, CL_rate=args.CL_rate)
            print('ACNN model constructed.')
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # use the default learning rate for AdamOptimizer
            train_op = optimizer.minimize(model.total_loss, global_step=model.global_step)
            sess.run(tf.global_variables_initializer())
            max_auc = 0.0
            for epoch_iter in range(args.total_epoch):
                print('epoch %d/%d' %(epoch_iter, args.total_epoch))
                loss_dict = {'entropy_loss': 0.0, 'constraint_loss': 0.0, 'total_loss': 0.0}
                TrainDataLoader.EpochInit()
                for batch_iter in range(int(TrainDataLoader.datasize / args.batch_size)):
                    feed_dict = TrainDataLoader.next_batch(model, args.train_mode)
                    loss_dict = train_step(sess, [train_op, model.global_step, model.accuracy, model.loss_dict], feed_dict, loss_dict, batch_iter+1)
                graph = tf.get_default_graph()
                input_graph_def = graph.as_graph_def()
                constant_graph = graph_util.convert_variables_to_constants(sess, input_graph_def=input_graph_def, output_node_names=['model/get_prob'])
                with tf.gfile.GFile("../Model/%s-%s-tmp.pb" %(args.dataset, args.train_mode), mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                auc = test('tmp', max_auc)
                if max_auc < auc:
                    max_auc = auc
                    with tf.gfile.GFile("../Model/%s-%s-best.pb" % (args.dataset, args.train_mode), mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
            return max_auc

def test(model_name, max_auc, use_pretrained='no'):
    '''
    :param model_name: the name of loading model
    :param max_auc: current max auc score
    :param use_pretrained: boolen variable, indicating whether using pretrained model
    :return: auc score
    '''
    TestDataLoader = DataLoader(args, state='Test')
    TestDataLoader.PaddingData()
    print('Test instance num:', len(TestDataLoader.y))
    if use_pretrained == 'yes':
        restore_path = os.path.join("../SavedModel/%s-%s-%s.pb" % (args.dataset, args.train_mode, model_name))
    else:
        restore_path = "../Model/%s-%s-%s.pb" % (args.dataset, args.train_mode, model_name)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with tf.gfile.GFile(restore_path, mode='rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            prob_op = sess.graph.get_tensor_by_name('model/get_prob:0')
            test_PR_area_file = open('../prediction_out/PR_result.txt', 'a+')
            allprob = []
            for i in range(int(TestDataLoader.datasize / TestDataLoader.batchsize)):
                feed_dict = TestDataLoader.next_batch(None, 'Base', sess=sess)
                prob = sess.run(prob_op, feed_dict)
                prob = np.reshape(np.array(prob), (TestDataLoader.batchsize, TestDataLoader.class_num))
                allprob += [single_prob for single_prob in prob]
            print('Saving test result...')
            p, r, auc_score = get_auc(allprob[:TestDataLoader.real_datasize], TestDataLoader.y[:TestDataLoader.real_datasize])
            if auc_score > max_auc:
                np.save('../prediction_out/test_precision.npy', p)
                np.save('../prediction_out/test_recall.npy', r)
                np.save('../prediction_out/test_multi_prediction.npy', allprob)
                MetricsEvaluation(p, r, auc_score, test_PR_area_file)
            print('PR curve area:' + str(auc_score))
            test_PR_area_file.write('test_multi_label '+str(model_name) + ' ' + str(auc_score) + '\n')
            return auc_score

def MetricsEvaluation(p, r, auc_score, test_PR_area_file):
    '''
    Obtaining the main metrics (PR curves and Precision@N) according to precision, recall and auc scores.
    :param p: precision
    :param r: recall
    :param auc_score:
    :param test_PR_area_file: the file object to save metrics
    :return: None
    '''
    f1 = (2 * r * p / (r + p + 1e-20)).max()
    plt.plot(r, p, lw=2, label=str(args.train_mode) + str(auc_score))
    print("result: auc = %.4f, max F1 = %.4f" % (auc_score, f1))
    test_PR_area_file.write("result: auc = %.4f, max F1 = %.4f\n" % (auc_score, f1))
    I = [99, 199, 299] # The indexes of P@100, P@200, P@300 score are 99, 199, 299 respectively
    ordered_p = sorted(p, reverse=True)
    print("P@100: %.4f, P@200: %.4f, P@300: %.4f, Mean: %.4f"
          % (ordered_p[I[0]], ordered_p[I[1]], ordered_p[I[2]], (ordered_p[I[0]] + ordered_p[I[1]] + ordered_p[I[2]]) / 3))
    test_PR_area_file.write("P@100: %.4f, P@200: %.4f, P@300: %.4f, Mean: %.4f\n"
          % (ordered_p[I[0]], ordered_p[I[1]], ordered_p[I[2]], (ordered_p[I[0]] + ordered_p[I[1]] + ordered_p[I[2]]) / 3))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title(args.dataset + '-' + args.train_mode)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig('../prediction_out/%s-%s-pr_curve.png' %(args.dataset, args.train_mode))

def get_auc(predict, y):
    '''
    Calculating precision, recall and AUC metrics according to the outputs of neural model: predict and  golden label: y
    :param predict: the predicted probability outputs of neural model
    :param y: the golden relationship label
    :return: precision, recall and auc scores
    '''
    allprob = np.reshape([prob[1:] for prob in predict], newshape=[-1])
    allans = np.reshape([instance[1:] for instance in y], newshape=[-1])
    p, r, _ = precision_recall_curve(allans, allprob)
    auc_score = auc(r, p)
    return p, r, auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='Train', type=str)         # Decide the running mode. Having two values: "Train" and "Test"
    parser.add_argument('--train_mode', default='Base', type=str)   # Assign the training mode. Including three modes: "Base", "Coh", "Sem"
    parser.add_argument('--dataset', default='English', type=str)   # Decide the dataset. "English" or "Chinese"
    parser.add_argument('--CL_rate', default=0.1, type=float)        # Decide the \labmda coefficient of Constraint Loss
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--sent_len', default=70, type=int)         # The final sentence length after padding
    parser.add_argument('--max_distance', default=60, type=int)     # The max distance for the position feature
    args = parser.parse_args()
    DataPreprocessor(args)
    if args.mode == 'Train':
        train(args)
    else:
        test('best', 0.0, use_pretrained='yes')
