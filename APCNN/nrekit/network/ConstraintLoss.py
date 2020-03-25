# _*_coding=utf-8_*_
import numpy as np
import tensorflow as tf
import os, json

class ConstraintLoss:
    def __init__(self, batchsize, class_num, rel2id, constraint_path, loss_rate, model):
        self.model = model
        self.batch_size = batchsize
        self.class_num = class_num
        self.loss_rate = loss_rate
        self.SemConstraints = self.ConstraintLoader(rel2id, constraint_path, 'Sem')
        self.CohConstraints = self.ConstraintLoader(rel2id, constraint_path, 'Coh')
        self.model.IndicatorInput = {
            'SubjRel':     tf.placeholder(dtype=tf.int32, shape=[None], name='SubjRelIndicator'),
            'RelObj':      tf.placeholder(dtype=tf.int32, shape=[None], name='RelObjIndicator'),
            'SubjRelObj':  tf.placeholder(dtype=tf.int32, shape=[None], name='SubjRelObjIndicator'),
            'SubjMulti':   tf.placeholder(dtype=tf.int32, shape=[None], name='SubjMultiIndicator'),
            'ObjMulti':    tf.placeholder(dtype=tf.int32, shape=[None], name='ObjMultiIndicator'),
        }

    def ConstraintLoader(self, rel2id, constraint_path, CLCMode='Sem'):
        Constraints = {'SubjRel': [],'RelObj': [], 'SubjRelObj': [], 'SubjMulti': [], 'ObjMulti': []}
        def merge_type(path):
            rel_constraint = json.load(open(path, 'r', encoding='utf-8'))
            rel_cons_list = [(min(rel2id[cons[0]], rel2id[cons[1]]), max(rel2id[cons[0]], rel2id[cons[1]])) for cons in rel_constraint]
            # add NA invovled constraints and two same relation constraints into constraint list.
            for i in range(len(rel2id)):
                rel_cons_list.append((0, i))
                rel_cons_list.append((i, i))
            return list(set(rel_cons_list))

        def merge_card(path):
            rel_constraint = json.load(open(path, 'r', encoding='utf-8'))
            rel_cons_list = [rel2id[rel] for rel in rel_constraint]
            return list(set(rel_cons_list))

        subj_rel_cons_list = merge_type(os.path.join(constraint_path, 'subj_rel.json'))
        rel_obj_cons_list = merge_type(os.path.join(constraint_path, 'rel_obj.json'))
        subj_rel_obj_cons_list = merge_type(os.path.join(constraint_path, 'subj_rel_obj.json'))
        subj_multi_cons_list = merge_card(os.path.join(constraint_path, 'subj_multi.json'))
        obj_multi_cons_list = merge_card(os.path.join(constraint_path, 'obj_multi.json'))

        if CLCMode == 'Coh':
            for i in range(len(rel2id)):
                for j in range(len(rel2id)):
                    rel_pair = (min(i, j), max(i, j))
                    Constraints['SubjRel'].append(1) if rel_pair in subj_rel_cons_list else Constraints['SubjRel'].append(0)
                    Constraints['RelObj'].append(1) if rel_pair in rel_obj_cons_list else Constraints['RelObj'].append(0)
                    Constraints['SubjRelObj'].append(1) if rel_pair in subj_rel_obj_cons_list else Constraints['SubjRelObj'].append(0)
            for i in range(len(rel2id)):
                Constraints['SubjMulti'].append(1) if i in subj_multi_cons_list else Constraints['SubjMulti'].append(0)
                Constraints['ObjMulti'].append(1) if i in obj_multi_cons_list else Constraints['ObjMulti'].append(0)

        elif CLCMode == 'Sem':
            for subj_rel in subj_rel_cons_list:
                tmp_vector = [0 for _ in range(len(rel2id))]
                tmp_vector[subj_rel[0]] = 1
                tmp_vector[subj_rel[1]] = 1
                Constraints['SubjRel'].append(tmp_vector)
            for rel_obj in rel_obj_cons_list:
                tmp_vector = [0 for _ in range(len(rel2id))]
                tmp_vector[rel_obj[0]] = 1
                tmp_vector[rel_obj[1]] = 1
                Constraints['RelObj'].append(tmp_vector)
            for subj_rel_obj in subj_rel_obj_cons_list:
                tmp_vector = [0 for _ in range(len(rel2id))]
                tmp_vector[subj_rel_obj[0]] = 1
                tmp_vector[subj_rel_obj[1]] = 1
                Constraints['SubjRelObj'].append(tmp_vector)

            for subj_multi in subj_multi_cons_list:
                tmp_vector = [0 for _ in range(len(rel2id))]
                tmp_vector[subj_multi] = 1
                Constraints['SubjMulti'].append(tmp_vector)
            for obj_multi in obj_multi_cons_list:
                tmp_vector = [0 for _ in range(len(rel2id))]
                tmp_vector[obj_multi] = 1
                Constraints['ObjMulti'].append(tmp_vector)

        Constraints['SubjRel'] = np.asarray(Constraints['SubjRel'], dtype=np.float32)
        Constraints['RelObj'] = np.asarray(Constraints['RelObj'], dtype=np.float32)
        Constraints['SubjRelObj'] = np.asarray(Constraints['SubjRelObj'], dtype=np.float32)
        Constraints['SubjMulti'] = np.asarray(Constraints['SubjMulti'], dtype=np.float32)
        Constraints['ObjMulti'] = np.asarray(Constraints['ObjMulti'], dtype=np.float32)
        return Constraints

    def Semantic(self):
        type_SL_loss = self.SemanticType()
        cardinality_SL_loss = self.SemanticCardinality()
        return self.loss_rate * (type_SL_loss + cardinality_SL_loss)

    def SemanticType(self):
        # Get the q vector for each instance pair, q_i = p^m_i + p^n_i - p^m_i * p^n_i, the detail explanation can be find in our paper, as equation (5).
        prob_tile_row = tf.reshape(tf.tile(self.model.prob, [1, self.batch_size]), shape=[-1, self.class_num])
        prob_tile_column = tf.tile(self.model.prob, [self.batch_size, 1])
        pair_prob = tf.add(prob_tile_row, prob_tile_column) - tf.multiply(prob_tile_row, prob_tile_column)
        pair_prob = tf.reshape(tf.concat(pair_prob, 0), shape=[-1, self.class_num])

        # calculate the semantic loss for each sub-category relation constraints
        subj_rel_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['SubjRel'])
        subj_rel_SL = self.calculate_single_SL(subj_rel_vector, self.SemConstraints['SubjRel'])
        rel_obj_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['RelObj'])
        rel_obj_SL = self.calculate_single_SL(rel_obj_vector, self.SemConstraints['RelObj'])
        subj_rel_obj_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['SubjRelObj'])
        subj_rel_obj_SL = self.calculate_single_SL(subj_rel_obj_vector, self.SemConstraints['SubjRelObj'])
        return subj_rel_SL + rel_obj_SL + subj_rel_obj_SL

    def calculate_single_SL(self, vector, satisfy):
        '''
        for each q in vector list, and each constraint in satisfy, use equation(5, 6) in our paper to calculate the constraint loss
        :param vector: the list of q vector
        :param satisfy: the vector set representation of a specific sub-category relation constraints set.
        :return: the Constraint Loss
        '''
        vector_tile = tf.reshape(tf.tile(vector, [1, tf.shape(satisfy)[0]]), shape=[-1, tf.shape(satisfy)[0], self.class_num])
        satisfy_tile = tf.reshape(tf.tile(satisfy, [tf.shape(vector)[0], 1]), shape=[-1, tf.shape(satisfy)[0], self.class_num])
        vector_tile_minus = 1 - vector_tile
        satisfy_tile_minus = 1 - satisfy_tile
        result = tf.add(tf.multiply(vector_tile, satisfy_tile), tf.multiply(vector_tile_minus, satisfy_tile_minus))
        result = tf.reduce_prod(result, 2)
        result = -tf.log(tf.reduce_sum(result, 1))
        result = tf.reduce_sum(result, 0)
        return result

    def SemanticCardinality(self):
        # Get the q vector for each instance pair, q_i = p^m_i * p^n_i, the detail explanation can be find in our paper, as equation (6).
        prob_tile_row = tf.reshape(tf.tile(self.model.prob, [1, self.batch_size]), shape=[-1, self.class_num])
        prob_tile_column = tf.tile(self.model.prob, [self.batch_size, 1])
        pair_prob = tf.multiply(prob_tile_row, prob_tile_column)
        pair_prob = tf.reshape(tf.concat(pair_prob, 0), shape=[-1, self.class_num])

        # calculate the semantic loss for each sub-category relation constraints
        subj_multi_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['SubjMulti'])
        subj_multi_SL = self.calculate_single_SL(subj_multi_vector, self.SemConstraints['SubjMulti'])
        obj_multi_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['ObjMulti'])
        obj_multi_SL = self.calculate_single_SL(obj_multi_vector, self.SemConstraints['ObjMulti'])
        return subj_multi_SL + obj_multi_SL

    def Coherent(self):
        type_SL_loss = self.CoherentType()
        cardinality_SL_loss = self.CoherentCardinality()
        return self.loss_rate * (type_SL_loss + cardinality_SL_loss)

    def CoherentType(self):
        prob_tile_row = tf.reshape(tf.tile(self.model.prob, [1, self.batch_size]), shape=[-1, self.class_num])
        prob_tile_column = tf.tile(self.model.prob, [self.batch_size, 1])
        pair_prob = tf.reshape(tf.matmul(tf.reshape(prob_tile_row, shape=[-1, self.class_num, 1]), tf.ones(shape=[self.batch_size*self.batch_size, 1, self.class_num])) *
                               tf.matmul(tf.ones(shape=[self.batch_size*self.batch_size, self.class_num, 1]), tf.reshape(prob_tile_column, shape=[-1, 1, self.class_num])),
                               shape=[-1, self.class_num * self.class_num])
        # calculate the semantic loss for each sub-category relation constraints
        subj_rel_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['SubjRel'])
        subj_rel_SL = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.multiply(tf.reshape(tf.tile(self.CohConstraints['SubjRel'], [tf.shape(subj_rel_vector)[0]]),
                                                                                 shape=[-1, self.class_num * self.class_num]), subj_rel_vector), axis=-1)))
        rel_obj_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['RelObj'])
        rel_obj_SL = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.multiply(tf.reshape(tf.tile(self.CohConstraints['RelObj'], [tf.shape(rel_obj_vector)[0]]),
                                                                                shape=[-1, self.class_num * self.class_num]), rel_obj_vector), axis=-1)))
        subj_rel_obj_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['SubjRelObj'])
        subj_rel_obj_SL = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.multiply(tf.reshape(tf.tile(self.CohConstraints['SubjRelObj'], [tf.shape(subj_rel_obj_vector)[0]]),
                                                                                     shape=[-1, self.class_num * self.class_num]), subj_rel_obj_vector), axis=-1)))
        return subj_rel_SL + rel_obj_SL + subj_rel_obj_SL

    def CoherentCardinality(self):
        prob_tile_row = tf.reshape(tf.tile(self.model.prob, [1, self.batch_size]), shape=[-1, self.class_num])
        prob_tile_column = tf.tile(self.model.prob, [self.batch_size, 1])
        pair_prob = tf.reshape(tf.multiply(prob_tile_row, prob_tile_column), shape=[-1, self.class_num])
        # calculate the semantic loss for each sub-category relation constraints
        subj_multi_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['SubjMulti'])
        subj_multi_SL = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.multiply(tf.reshape(tf.tile(self.CohConstraints['SubjMulti'], [tf.shape(subj_multi_vector)[0]]),
                                                                                   shape=[-1, self.class_num]), subj_multi_vector), axis=-1)))
        obj_multi_vector = tf.nn.embedding_lookup(pair_prob, self.model.IndicatorInput['ObjMulti'])
        obj_multi_SL = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.multiply(tf.reshape(tf.tile(self.CohConstraints['ObjMulti'], [tf.shape(obj_multi_vector)[0]]),
                                                                                  shape=[-1, self.class_num]), obj_multi_vector), axis=-1)))
        return subj_multi_SL + obj_multi_SL

