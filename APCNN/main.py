import os, json, pathlib, argparse
import nrekit
from nrekit.data_loader import json_file_data_loader

def ConvertTxt2Json(RawDir):
    '''
    covert the original txt data to following json format
    file_name: Json file storing the data in the following format
        [
            {
                'sentence': 'Bill Gates is the founder of Microsoft .',
                'head': {'word': 'Bill Gates', ...(other information)},
                'tail': {'word': 'Microsoft', ...(other information)},
                'relation': 'founder'
            },
            ...
        ]
    word_vec_file_name: Json file storing word vectors in the following format
        [
            {'word': 'the', 'vec': [0.418, 0.24968, ...]},
            {'word': ',', 'vec': [0.013441, 0.23682, ...]},
            ...
        ]
    rel2id_file_name: Json file storing relation-to-id diction in the following format
        {
            'NA': 0
            'founder': 1
            ...
        }
    '''
    if pathlib.Path(os.path.join(RawDir, 'train.json')).exists():
    # if False:
        print('JSON format files for APCNN exist.')
    else:
        print('Convert raw text to JSON format begin.')
        # change train data
        train_data_file = open(os.path.join(RawDir, 'train_cleaned.txt'), 'r', encoding='utf-8')
        train_out_list = []
        train_lines = train_data_file.readlines()
        for line in train_lines:
            train_instance = {}
            raw = line.strip().split('\t')
            train_instance['sentence'] = raw[5]
            train_instance['head'] = {'word': raw[2]}
            train_instance['tail'] = {'word': raw[3]}
            train_instance['relation'] = raw[4]
            train_out_list.append(train_instance)
        json.dump(train_out_list, open(os.path.join(RawDir, 'train.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        # change test data
        test_data_file = open(os.path.join(RawDir, 'test_cleaned.txt'), 'r', encoding='utf-8')
        test_out_list = []
        test_lines = test_data_file.readlines()
        for line in test_lines:
            test_instance = {}
            raw = line.strip().split('\t')
            test_instance['sentence'] = raw[5]
            test_instance['head'] = {'word': raw[2]}
            test_instance['tail'] = {'word': raw[3]}
            test_instance['relation'] = raw[4]
            test_out_list.append(test_instance)
        json.dump(test_out_list, open(os.path.join(RawDir, 'test.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        # change word vector
        word_vec_file = open(os.path.join(RawDir, 'vec.txt'), 'r', encoding='utf-8')
        lines = word_vec_file.readlines()
        word_vec_list = []
        for line in lines[1:]:
            raw = line.strip().split()
            word = raw[0]
            vec = list(map(float, raw[1:]))
            tmp_dict = {}
            tmp_dict['word'] = word
            tmp_dict['vec'] = vec
            word_vec_list.append(tmp_dict)
        json.dump(word_vec_list, open(os.path.join(RawDir, 'word_vec.json'), 'w', encoding='utf-8'), indent=4)
        print('Convert raw text file to json format over!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="Test")
    parser.add_argument("--dataset", type=str, default="Chinese")
    parser.add_argument("--data_dir", type=str, default="../ChineseData/")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, default=60)
    parser.add_argument("--train_mode", type=str, default='Sem', help='Base, Sem, Coh')
    parser.add_argument("--CL_rate", type=float, default=0.001)

    args = parser.parse_args()
    args.data_dir = "../%sData/" %(args.dataset)
    # print(args.data_dir)
    ConvertTxt2Json(os.path.join(args.data_dir, 'raw_text'))
    word_embedding_dim = 50 if args.dataset == 'English' else 300
    train_loader = json_file_data_loader(os.path.join(args.data_dir, 'APCNN_tmpdata'),
                                         os.path.join(args.data_dir, 'raw_text/train.json'),
                                         os.path.join(args.data_dir, 'raw_text/word_vec.json'),
                                         os.path.join(args.data_dir, 'raw_text/relation2id.json'),
                                         mode=json_file_data_loader.MODE_RELFACT_BAG,
                                         shuffle=True,
                                         max_length=args.max_length,
                                         word_embedding_dim = word_embedding_dim,
                                         batch_size=args.batch_size)
    test_loader = json_file_data_loader(os.path.join(args.data_dir, 'APCNN_tmpdata'),
                                        os.path.join(args.data_dir, 'raw_text/test.json'),
                                        os.path.join(args.data_dir, 'raw_text/word_vec.json'),
                                        os.path.join(args.data_dir, 'raw_text/relation2id.json'),
                                        mode=json_file_data_loader.MODE_ENTPAIR_BAG,
                                        shuffle=False,
                                        max_length=args.max_length,
                                        word_embedding_dim=word_embedding_dim,
                                        batch_size=args.batch_size)
    framework = nrekit.framework.re_framework(train_loader, test_loader)
    if args.mode=='Train':
        framework.train(args)
    if args.mode=='Test':
        framework.test_withpb(restore_path="./SavedModel/%s-%s-best.pb" %(args.dataset, args.train_mode), args=args)
