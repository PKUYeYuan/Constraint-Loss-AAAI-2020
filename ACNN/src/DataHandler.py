# _*_coding=utf-8_*_
import numpy as np
import json, nltk, os, pathlib

class DataLoader:
    def __init__(self, args, state='Train'):
        self.DataDir = "../../%sData" %(args.dataset)
        self.tmp_data_dir = os.path.join(self.DataDir, 'ACNN_tmpdata')
        if state=='Train':
            self.data_path = os.path.join(self.DataDir, 'ACNN_tmpdata/train')
        else:
            self.data_path = os.path.join(self.DataDir, 'ACNN_tmpdata/test')

        self.POS2id = json.load(open(os.path.join(self.tmp_data_dir, 'POS2id.json'), 'r', encoding='utf-8'))
        self.rel2id = json.load(open(os.path.join(self.tmp_data_dir, 'relation2id.json'), 'r', encoding='utf-8'))
        self.id2rel = dict(zip(self.rel2id.values(), self.rel2id.keys()))
        self.word2id = json.load(open(os.path.join(self.tmp_data_dir, 'word2id.json'), 'r', encoding='utf-8'))
        self.word_vec = np.load(os.path.join(self.tmp_data_dir, 'word_vec.npy'), allow_pickle=True)

        self.word = np.load(os.path.join(self.data_path, 'word.npy'), allow_pickle=True)
        self.POS = np.load(os.path.join(self.data_path, 'POS.npy'), allow_pickle=True)
        self.pos1 = np.load(os.path.join(self.data_path, 'en1pos_feature.npy'), allow_pickle=True)
        self.pos2 = np.load(os.path.join(self.data_path, 'en2pos_feature.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(self.data_path, 'y.npy'), allow_pickle=True)
        self.instance_list = json.load(open(os.path.join(self.data_path, 'instance_list.json'), 'r', encoding='utf-8'))
        self.entity_pair_list = self.GetEntityPairList()

        self.batchsize = args.batch_size
        self.real_datasize = len(self.word)
        self.class_num = len(self.rel2id)
        self.POS_num = len(self.POS2id)
        self.sent_len = args.sent_len
        self.constraint_path = os.path.join(self.DataDir, 'Constraints')

        self.EpochInit()

    def PaddingData(self):
        self.word, self.POS, self.pos1, self.pos2, self.y = list(self.word), list(self.POS), list(self.pos1), list(self.pos2), list(self.y)
        pad_len = 0 if self.datasize % self.batchsize == 0 else self.batchsize - (self.datasize % self.batchsize)
        for _i in range(pad_len):
            self.word.append([[0 for _ in range(self.sent_len)]])
            self.POS.append([[0 for _ in range(self.sent_len)]])
            self.pos1.append([[0 for _ in range(self.sent_len)]])
            self.pos2.append([[0 for _ in range(self.sent_len)]])
            self.y.append([0 for _ in range(self.class_num)])
        self.word, self.POS, self.pos1, self.pos2, self.y = np.asarray(self.word), np.asarray(self.POS), np.asarray(self.pos1), np.asarray(self.pos2), np.asarray(self.y)
        self.EpochInit(shuffle=False)

    def EpochInit(self, shuffle=True):
        self.datasize = len(self.word)
        self.order = list(range(self.datasize))
        if shuffle:
            np.random.shuffle(self.order)
        self.batch_iter = 0

    def GetEntityPairList(self):
        entity_pair_list = []
        for instance in self.instance_list:
            raw_text = instance.strip().split('#')[0].split('@|@')
            entity_pair_list.append((raw_text[0], raw_text[1]))
        return entity_pair_list

    def next_batch(self, model, feed_mode, sess=None):
        # prepare the feed dict input for the model (the placeholders of tensorflow)

        feed_dict = {}
        shape_batch, word_batch, pos1_batch, pos2_batch, POS_batch, total_num = [], [], [], [], [], 0
        temp_input = self.order[self.batch_iter * self.batchsize:(self.batch_iter + 1) * self.batchsize]
        self.batch_iter += 1
        for tmp_i in temp_input:
            shape_batch.append(total_num)
            total_num += len(self.word[tmp_i])
            word_batch += self.word[tmp_i]
            pos1_batch += self.pos1[tmp_i]
            pos2_batch += self.pos2[tmp_i]
            POS_batch += self.POS[tmp_i]
        shape_batch.append(total_num)
        y_batch = self.y[temp_input]
        if sess != None:
            # test phase when loading model from .pd file
            input_total_shape = sess.graph.get_tensor_by_name('model/total_shape:0')
            input_word = sess.graph.get_tensor_by_name('model/input_word:0')
            input_pos1 = sess.graph.get_tensor_by_name('model/input_pos1:0')
            input_pos2 = sess.graph.get_tensor_by_name('model/input_pos2:0')
            input_POS = sess.graph.get_tensor_by_name('model/input_POS:0')
            feed_dict[input_total_shape] = np.array(shape_batch)
            feed_dict[input_word] = np.array(word_batch)
            feed_dict[input_pos1] = np.array(pos1_batch)
            feed_dict[input_pos2] = np.array(pos2_batch)
            feed_dict[input_POS] = np.array(POS_batch)
            return feed_dict
        feed_dict[model.total_shape] = np.array(shape_batch)
        feed_dict[model.input_word] = np.array(word_batch)
        feed_dict[model.input_pos1] = np.array(pos1_batch)
        feed_dict[model.input_pos2] = np.array(pos2_batch)
        feed_dict[model.input_POS] = np.array(POS_batch)
        feed_dict[model.input_y] = y_batch

        if feed_mode == 'Base':
            return feed_dict
        else:
            subj_rel_set, rel_obj_set, subj_rel_obj_set, subj_multi_set, obj_multi_set = [], [], [], [], []
            first_tuple = 0
            index_count = 0
            while (first_tuple < len(temp_input)):
                subj1 = self.entity_pair_list[temp_input[first_tuple]][0]
                obj1 = self.entity_pair_list[temp_input[first_tuple]][1]
                second_tuple = 0
                while (second_tuple < len(temp_input)):
                    if first_tuple >= second_tuple:
                        second_tuple += 1
                        index_count += 1
                        continue
                    subj2 = self.entity_pair_list[temp_input[second_tuple]][0]
                    obj2 = self.entity_pair_list[temp_input[second_tuple]][1]
                    if subj1 == subj2:
                        subj_rel_set.append(index_count)
                        subj_multi_set.append(index_count)
                    if obj1 == obj2:
                        rel_obj_set.append(index_count)
                        obj_multi_set.append(index_count)
                    if subj1 == obj2 or obj1 == subj2:
                        subj_rel_obj_set.append(index_count)
                    second_tuple += 1
                    index_count += 1
                first_tuple += 1
            feed_dict[model.IndicatorInput['SubjRel']] = subj_rel_set
            feed_dict[model.IndicatorInput['RelObj']] = rel_obj_set
            feed_dict[model.IndicatorInput['SubjRelObj']] = subj_rel_obj_set
            feed_dict[model.IndicatorInput['SubjMulti']] = subj_multi_set
            feed_dict[model.IndicatorInput['ObjMulti']] = obj_multi_set
            return feed_dict

class DataPreprocessor:
    def __init__(self, args):
        self.DataDir = "../../%sData" %(args.dataset)
        self.RawDataPath = os.path.join(self.DataDir, 'raw_text')
        self.TmpDataPath = os.path.join(self.DataDir, 'ACNN_tmpdata')
        if pathlib.Path(self.TmpDataPath).exists():
            print("Preprocessed files exist, loading them ......")
        else:
            os.mkdir(self.TmpDataPath)
            self.TrainOutPath = os.path.join(self.TmpDataPath, 'train')
            if not pathlib.Path(self.TrainOutPath).exists():
                os.mkdir(self.TrainOutPath)
            self.TestOutPath = os.path.join(self.TmpDataPath, 'test')
            if not pathlib.Path(self.TestOutPath).exists():
                os.makedirs(self.TestOutPath)
            self.sent_len = args.sent_len
            self.max_distance = args.max_distance
            self.WordVecHandler()
            self.feature_extractor()
            self.handle_train_data()
            self.handle_test_data()

    def GetWordList(self, TextFileName):
        print(TextFileName)
        file_data = open(TextFileName, 'r', encoding='utf-8')
        word_list = []
        for line in file_data.readlines():
            raw = line.strip().split('\t')[5].split(' ')
            word_list += raw
        word_list = list(set(word_list))
        return word_list

    def re_organize_data(self, input_filename, out_filename):
        # re-organize data (entity pair: label: instance bag)
        data = {}
        print('reading data from ' + input_filename)
        data_file = open(input_filename, 'r', encoding='utf-8')
        data_lines = data_file.readlines()
        for line in data_lines:
            raw = line.strip().split('\t')
            entity_pair = raw[2] + '@|@' + raw[3]
            relation = raw[4]
            sentence = raw[5]
            if entity_pair in data:
                if relation not in data[entity_pair]:
                    data[entity_pair][relation] = []
                data[entity_pair][relation].append(sentence)
            else:
                data[entity_pair] = {}
                if relation not in data[entity_pair]:
                    data[entity_pair][relation] = []
                data[entity_pair][relation].append(sentence)
        json.dump(data, open(out_filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        return data

    def WordVecHandler(self):
        if pathlib.Path(os.path.join(self.TmpDataPath, 'word2id.json')).exists():
            pass
        else:
            print("Word vector handling process begin...")
            # Obtain the word list and POS list according to train and test raw text
            TrainWordList = self.GetWordList(os.path.join(self.RawDataPath, 'train_cleaned.txt'))
            TestWordList = self.GetWordList(os.path.join(self.RawDataPath, 'test_cleaned.txt'))
            WordList = list(set(TestWordList + TrainWordList))
            POS2id = json.load(open(os.path.join(self.RawDataPath, 'POS2id.json'), 'r', encoding='utf-8'))
            print('Data word list length: ', len(WordList))
            print('Data POS list length: ', len(POS2id))
            vec_file = open(os.path.join(self.RawDataPath, 'vec.txt'), 'r', encoding='utf-8')
            lines = vec_file.readlines()[1:]
            word_embedding_dim = len(lines[0].strip().split())-1
            # adding <PADDING>
            WordDict = {'<PADDING>': 0}
            WordIdCount = 1
            WordVec = [[0.0 for _ in range(word_embedding_dim)]]

            for line in lines:
                raw = line.strip().split()
                word = raw[0]
                vec = list(map(float, raw[1:]))
                if word in WordList:
                    WordDict[word] = WordIdCount
                    WordIdCount += 1
                    # if WordIdCount % 1000 == 0:
                    #     print(WordIdCount)
                    WordVec.append(vec)
            # add <UNK>
            WordDict['<UNK>'] = WordIdCount
            WordVec.append(np.random.randn(word_embedding_dim))
            # saving the results
            json.dump(WordDict, open(os.path.join(self.TmpDataPath, 'word2id.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
            json.dump(POS2id, open(os.path.join(self.TmpDataPath, 'POS2id.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
            np.save(os.path.join(self.TmpDataPath, 'word_vec.npy'), np.array(WordVec, dtype=np.float32))
            print('word vector handle over!', 'word list size is ', WordIdCount)

    def feature_extractor(self):
        word2id = json.load(open(os.path.join(self.TmpDataPath, 'word2id.json'), 'r', encoding='utf-8'))
        pos2id = json.load(open(os.path.join(self.TmpDataPath, 'POS2id.json'), 'r', encoding='utf-8'))
        # Extracting POS and entity position feature and padding
        # processing training data
        if pathlib.Path(os.path.join(self.TrainOutPath, 'out_feature.json')).exists():
            pass
        else:
            if pathlib.Path(os.path.join(self.TrainOutPath, 'out.json')).exists():
                train_data = json.load(open(os.path.join(self.TrainOutPath, 'out.json'), 'r', encoding='utf-8'))
            else:
                train_data = self.re_organize_data(os.path.join(self.RawDataPath, 'train_cleaned.txt'), os.path.join(self.TrainOutPath, 'out.json'))
            train_out_feature = {}
            train_data_length = len(train_data)
            for entity_pair in train_data:
                train_out_feature[entity_pair] = {}
                for relation in train_data[entity_pair]:
                    train_out_feature[entity_pair][relation] = []
                    for sentence in train_data[entity_pair][relation]:
                        words = sentence.split()
                        word_pos = nltk.pos_tag(words)
                        POS = [tmp[1] for tmp in word_pos]
                        # POS_list += POS
                        # entity position
                        entity_pair_tmp = entity_pair.split('@|@')
                        en1pos, en2pos = get_entity_position(entity_pair_tmp[0], entity_pair_tmp[1], words)
                        # extracting the features and padding
                        word_padding = []
                        POS_padding = []
                        mask_feature = []
                        en1pos_feature = []
                        en2pos_feature = []
                        mask_indicator = 1
                        for tmp_i in range(self.sent_len):
                            mask_feature.append(mask_indicator)
                            word_padding.append(word2id['<PADDING>'])
                            POS_padding.append(pos2id['<PADDING>'])
                            en1pos_feature.append(pos_embed(tmp_i - en1pos, self.max_distance))
                            en2pos_feature.append(pos_embed(tmp_i - en2pos, self.max_distance))
                            if tmp_i == en1pos or tmp_i == en2pos:
                                mask_indicator += 1
                        for tmp_i in range(min(self.sent_len, len(words))):
                            if words[tmp_i] not in word2id:
                                tmp_word = word2id['<UNK>']
                            else:
                                tmp_word = word2id[words[tmp_i]]
                            if POS[tmp_i] not in pos2id:
                                tmp_POS = pos2id['<UNK>']
                            else:
                                tmp_POS = pos2id[POS[tmp_i]]
                            word_padding[tmp_i] = tmp_word
                            POS_padding[tmp_i] = tmp_POS
                        if len(words) < self.sent_len:
                            for tmp_i in range(len(words), self.sent_len):
                                mask_feature[tmp_i] = 0

                        train_out_feature[entity_pair][relation].append([word_padding, POS_padding, en1pos_feature, en2pos_feature, mask_feature])
            # pickle.dump(train_out_feature, codecs.open(Para_Conf.train_out_feature_filename, 'wb'))
            json.dump(train_out_feature, open(os.path.join(self.TrainOutPath, 'out_feature.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        # test multi-label
        if pathlib.Path(os.path.join(self.TestOutPath, 'out_feature.json')).exists():
            pass
        else:
            if pathlib.Path(os.path.join(self.TestOutPath, 'out.json')).exists():
                test_data = json.load(open(os.path.join(self.TestOutPath, 'out.json'), 'r', encoding='utf-8'))
            else:
                test_data = self.re_organize_data(os.path.join(self.RawDataPath, 'test_cleaned.txt'), os.path.join(self.TestOutPath, 'out.json'))
            test_out_feature = {}
            test_data_length = len(test_data)
            for entity_pair in test_data:
                test_out_feature[entity_pair] = {}
                for relation in test_data[entity_pair]:
                    test_out_feature[entity_pair][relation] = []
                    for sentence in test_data[entity_pair][relation]:
                        words = sentence.split()
                        word_pos = nltk.pos_tag(words)
                        POS = [tmp[1] for tmp in word_pos]
                        # POS_list += POS
                        # entity position
                        entity_pair_tmp = entity_pair.split('@|@')
                        en1pos, en2pos = get_entity_position(entity_pair_tmp[0], entity_pair_tmp[1], words)
                        # feature extract and padding
                        word_padding = []
                        POS_padding = []
                        en1pos_feature = []
                        en2pos_feature = []
                        mask_feature = []
                        mask_indicator = 1
                        for tmp_i in range(self.sent_len):
                            mask_feature.append(mask_indicator)
                            word_padding.append(word2id['<PADDING>'])
                            POS_padding.append(pos2id['<PADDING>'])
                            en1pos_feature.append(pos_embed(tmp_i - en1pos, self.max_distance))
                            en2pos_feature.append(pos_embed(tmp_i - en2pos, self.max_distance))
                            if tmp_i == en1pos or tmp_i == en2pos:
                                mask_indicator += 1
                        for tmp_i in range(min(self.sent_len, len(words))):
                            if words[tmp_i] not in word2id:
                                tmp_word = word2id['<UNK>']
                            else:
                                tmp_word = word2id[words[tmp_i]]
                            if POS[tmp_i] not in pos2id:
                                tmp_POS = pos2id['<UNK>']
                            else:
                                tmp_POS = pos2id[POS[tmp_i]]
                            word_padding[tmp_i] = tmp_word
                            POS_padding[tmp_i] = tmp_POS
                        if len(words) < self.sent_len:
                            for tmp_i in range(len(words), self.sent_len):
                                mask_feature[tmp_i] = 0
                        test_out_feature[entity_pair][relation].append([word_padding, POS_padding, en1pos_feature, en2pos_feature, mask_feature])
            json.dump(test_out_feature, open(os.path.join(self.TestOutPath, 'out_feature.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    def save_data(self, file_dir, word, POS, en1pos, en2pos, mask, y, instance_list):
        np.save(os.path.join(file_dir, 'word.npy'), np.array(word))
        np.save(os.path.join(file_dir, 'POS.npy'), np.array(POS))
        np.save(os.path.join(file_dir, 'en1pos_feature.npy'), np.array(en1pos))
        np.save(os.path.join(file_dir, 'en2pos_feature.npy'), np.array(en2pos))
        np.save(os.path.join(file_dir, 'mask_feature.npy'), np.array(mask))
        np.save(os.path.join(file_dir, 'y.npy'), np.array(y))
        json.dump(instance_list, open(os.path.join(file_dir, 'instance_list.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    def handle_train_data(self):
        # process training data，each train[entity_pair][relation] is an instance bag, and has a gold relation tag
        train_instance_list = []
        relation2id = json.load(open(os.path.join(self.RawDataPath, 'relation2id.json'), 'r', encoding='utf-8'))
        json.dump(relation2id, open(os.path.join(self.TmpDataPath, 'relation2id.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        if not pathlib.Path(os.path.join(self.TrainOutPath, 'out_feature.json')).exists():
            self.feature_extractor()
        train_out_feature = json.load(open(os.path.join(self.TrainOutPath, 'out_feature.json'), 'r', encoding='utf-8'))
        train_x_word = []
        train_x_POS = []
        train_x_en1pos_feature = []
        train_x_en2pos_feature = []
        train_x_mask_feature = []
        train_y = []
        # First, order the keys of dict
        entity_pair_keys = sorted(train_out_feature.keys())
        for entity_pair in entity_pair_keys:
            relation_keys = sorted(train_out_feature[entity_pair].keys())
            for relation in relation_keys:
                train_instance_list.append(entity_pair + '@|@' + relation)
                relation_vec = [0 for i in range(len(relation2id))]
                relation_vec[relation2id[relation]] = 1
                train_y.append(relation_vec)
                bag_info = train_out_feature[entity_pair][relation]
                bag_word = []
                bag_POS = []
                bag_en1pos_feature = []
                bag_en2pos_feature = []
                bag_mask_feature = []
                for sentence in bag_info:
                    bag_word.append(sentence[0])
                    bag_POS.append(sentence[1])
                    bag_en1pos_feature.append(sentence[2])
                    bag_en2pos_feature.append(sentence[3])
                    bag_mask_feature.append(sentence[4])
                train_x_word.append(bag_word)
                train_x_POS.append(bag_POS)
                train_x_en1pos_feature.append(bag_en1pos_feature)
                train_x_en2pos_feature.append(bag_en2pos_feature)
                train_x_mask_feature.append(bag_mask_feature)
        self.save_data(self.TrainOutPath, train_x_word, train_x_POS, train_x_en1pos_feature,
                  train_x_en2pos_feature, train_x_mask_feature, train_y, train_instance_list)

    def handle_test_data(self):
        test_instance_list = []
        relation2id = json.load(open(os.path.join(self.RawDataPath, 'relation2id.json'), 'r', encoding='utf-8'))
        if not pathlib.Path(os.path.join(self.TestOutPath, 'out_feature.json')).exists():
            self.feature_extractor()
        test_out_feature = json.load(open(os.path.join(self.TestOutPath, 'out_feature.json'), 'r', encoding='utf-8'))
        test_x_word = []
        test_x_POS = []
        test_x_en1pos_feature = []
        test_x_en2pos_feature = []
        test_x_mask_feature = []
        test_y = []
        entity_pair_keys = sorted(test_out_feature.keys())
        for entity_pair in entity_pair_keys:
            relation_keys = sorted(test_out_feature[entity_pair].keys())
            relation_vec = [0 for i in range(len(relation2id))]
            bag_word = []
            bag_POS = []
            bag_en1pos_feature = []
            bag_en2pos_feature = []
            bag_mask_feature = []
            instance_name_string = entity_pair + '@|@'
            for relation in relation_keys:
                instance_name_string += relation + '@|@'
                relation_vec[relation2id[relation]] = 1
                bag_info = test_out_feature[entity_pair][relation]
                for sentence in bag_info:
                    bag_word.append(sentence[0])
                    bag_POS.append(sentence[1])
                    bag_en1pos_feature.append(sentence[2])
                    bag_en2pos_feature.append(sentence[3])
                    bag_mask_feature.append(sentence[4])
            test_instance_list.append(instance_name_string)
            test_x_word.append(bag_word)
            test_x_POS.append(bag_POS)
            test_x_en1pos_feature.append(bag_en1pos_feature)
            test_x_en2pos_feature.append(bag_en2pos_feature)
            test_x_mask_feature.append(bag_mask_feature)
            test_y.append(relation_vec)
        self.save_data(self.TestOutPath, test_x_word, test_x_POS, test_x_en1pos_feature,
                  test_x_en2pos_feature, test_x_mask_feature, test_y, test_instance_list)


# embedding the position
def pos_embed(x, max_distance):
    if x < - max_distance:
        return int(0)
    if -max_distance <= x and x <= max_distance:
        return int(x + max_distance + 1)
    if x > max_distance:
        return int(max_distance * 2 + 2)

# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def levenshtein(first_str, second_str):
    if len(first_str) > len(second_str):
        first_str, second_str = second_str, first_str
    if len(first_str) == 0:
        return len(second_str)
    if len(second_str) == 0:
        return len(first_str)
    first_length = len(first_str) + 1
    second_length = len(second_str) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first_str[i-1] != second_str[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]

def get_entity_position_not_completely_match(en1, en2, sentence):
    en1_sim = []
    en2_sim = []
    for i in range(len(sentence)):
        en1_sim.append(levenshtein(en1, sentence[i]))
        en2_sim.append(levenshtein(en2, sentence[i]))
    en1_pos = int(np.argmin(np.array(en1_sim)))
    en2_pos = int(np.argmin(np.array(en2_sim)))
    return en1_pos, en2_pos

def get_entity_position(en1, en2, sentence):
    en1pos = -1
    en2pos = -1
    for i in range(len(sentence)):
        if sentence[i] == en1:
            en1pos = i
        if sentence[i] == en2:
            en2pos = i
    if en1pos == -1 or en2pos == -1:
        en1pos, en2pos = get_entity_position_not_completely_match(en1, en2, sentence)
        if en1pos == -1 or en2pos == -1:
            if en1pos == -1:
                en1pos = 0
            else:
                en2pos = 0
    return en1pos, en2pos
