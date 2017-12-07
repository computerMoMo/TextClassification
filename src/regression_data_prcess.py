# -*- coding:utf-8 -*-
import codecs
import json
import os
import random
import numpy as np
import pickle
import jieba

ConfDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "conf")
DataDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
UNKNOWN = "<OOV>"


def _save_vocab(dict, path):
    # save utf-8 code dictionary
    outfile = codecs.open(path, "w", encoding='utf-8')
    for k, v in dict.items():
        # k is unicode, v is int
        line = k + "\t" + str(v) + "\n"  # unicode
        outfile.write(line)
    outfile.close()


def load_vector_file(vector_file_path):
    vector_dicts = dict()
    with codecs.open(vector_file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            vector_dicts[line[0]] = np.asarray(list(map(float, line[1:])), dtype=np.float32)
    return vector_dicts


def load_data(configs):
    corpus_dir = os.path.join(DataDirPath, configs["corpus_id"])
    text_file_list = os.listdir(corpus_dir)

    Data_lists = []
    for file_name in text_file_list:
        text_reader = codecs.open(os.path.join(corpus_dir, file_name), mode='r', encoding="utf-8")
        for text_line in text_reader.readlines():
            text_line_list = text_line.strip().split("\t")
            text_str = text_line_list[0]
            text_score = float(text_line_list[1])
            if configs["language"] == "ch":
                item = (jieba.lcut(text_str.strip()), [ch for ch in text_str.strip()], text_score)
            elif configs["language"] == "en":
                item = (text_str.split(), [ch for ch in text_str.replace(" ", "")], text_score)
            else:
                raise ValueError(configs["language"] + " language not support yet")
            Data_lists.append(item)

    print("total data nums:", len(Data_lists))
    random.shuffle(Data_lists)
    return Data_lists


def data_vectorization(configs, data_lists):
    if configs["word_embedding_file"] != "None":
        pre_train_word_vector_dicts = load_vector_file(os.path.join(DataDirPath, configs["word_embedding_file"]))
    else:
        pre_train_word_vector_dicts = None
    if configs["char_embedding_file"] != "None":
        pre_train_char_vector_dicts = load_vector_file(os.path.join(DataDirPath, configs["char_embedding_file"]))
    else:
        pre_train_char_vector_dicts = None
    word_vector_dim = int(configs["word_embedding_dim"])
    char_vector_dim = int(configs["char_embedding_dim"])

    word_vectors = dict()
    char_vectors = dict()
    exist_word_nums = 0
    exist_char_nums = 0
    for (text_list, char_list, _) in data_lists:
        # word vectors
        for word in text_list:
            if pre_train_word_vector_dicts:
                if word not in word_vectors:
                    if word in pre_train_word_vector_dicts:
                        word_vectors[word] = pre_train_word_vector_dicts[word]
                        exist_word_nums += 1
                    else:
                        word_vectors[word] = np.random.uniform(low=-0.5, high=0.5, size=word_vector_dim)
            else:
                if word not in word_vectors:
                    word_vectors[word] = np.random.uniform(low=-0.5, high=0.5, size=word_vector_dim)
        # char vectors
        for char in char_list:
            if pre_train_char_vector_dicts:
                if char not in char_vectors:
                    if char in pre_train_char_vector_dicts:
                        char_vectors[char] = pre_train_char_vector_dicts[char]
                        exist_char_nums += 1
                    else:
                        char_vectors[char] = np.random.uniform(low=-0.5, high=0.5, size=char_vector_dim)
            else:
                if char not in char_vectors:
                    char_vectors[char] = np.random.uniform(low=-0.5, high=0.5, size=char_vector_dim)
    word_vectors[UNKNOWN] = np.random.uniform(low=-0.5, high=0.5, size=word_vector_dim)
    char_vectors[UNKNOWN] = np.random.uniform(low=-0.5, high=0.5, size=char_vector_dim)
    print(exist_word_nums, "/", len(word_vectors), " words find in pre trained word vectors")
    print(exist_char_nums, "/", len(char_vectors), " chars find in pre trained char vectors")
    print("word vectors dict size:", len(word_vectors))
    print("char vectors dict size:", len(char_vectors))

    word_to_id_dicts = dict()
    char_to_id_dicts = dict()
    word_vector_values = []
    char_vector_values = []
    for idx, word in zip(range(0, len(word_vectors)), word_vectors.keys()):
        word_to_id_dicts[word] = idx
        word_vector_values.append(word_vectors[word])
    for idx, char in zip(range(0, len(char_vectors)), char_vectors.keys()):
        char_to_id_dicts[char] = idx
        char_vector_values.append(char_vectors[char])
    return word_vector_values, char_vector_values, word_to_id_dicts, char_to_id_dicts


def generate_train_valid_data(configs, data_lists, word_to_id_dicts, char_to_id_dicts):
    x_data = []
    y_data = []
    word_max_len = int(configs["word_max_len"])
    char_max_len = int(configs["char_max_len"])
    unknown_word_id = word_to_id_dicts[UNKNOWN]
    unknown_char_id = char_to_id_dicts[UNKNOWN]
    for (text_list, char_list, score) in data_lists:
        x_item = []
        for word in text_list:
            if word in word_to_id_dicts:
                x_item.append(word_to_id_dicts[word])
            else:
                x_item.append(unknown_word_id)
        if len(x_item) > word_max_len:
            x_item = x_item[:word_max_len]
        else:
            while len(x_item) < word_max_len:
                x_item.append(unknown_word_id)
        for char in char_list:
            if char in char_to_id_dicts:
                x_item.append(char_to_id_dicts[char])
            else:
                x_item.append(unknown_char_id)
        if len(x_item) > word_max_len+char_max_len:
            x_item = x_item[:word_max_len+char_max_len]
        else:
            while len(x_item) < word_max_len+char_max_len:
                x_item.append(unknown_char_id)
        x_data.append(x_item)
        # y_score = [0.0]*num_classes
        # y_score[tag] = 1.0
        y_data.append([score])
    valid_len = int(len(x_data)*configs["split"])
    x_valid_data = x_data[:valid_len]
    y_valid_data = y_data[:valid_len]
    x_train_data = x_data[valid_len:]
    y_train_data = y_data[valid_len:]
    print("train data nums:", len(x_train_data))
    print("valid data nums:", len(x_valid_data))
    return x_train_data, y_train_data, x_valid_data, y_valid_data


if __name__ == "__main__":
    # 解析配置json
    configs = json.load(open(os.path.join(ConfDirPath, "regression_data_process.json")))
    data_serialization_dir = os.path.join(DataDirPath, configs["serialization_dir"])
    if not os.path.exists(data_serialization_dir):
        os.mkdir(data_serialization_dir)
    # 读取数据集
    data_lists = load_data(configs)
    # 数据向量化
    word_vector_values, char_vector_values, word_to_id_dicts, char_to_id_dicts = data_vectorization(configs, data_lists)
    pickle.dump(np.asarray(word_vector_values, dtype=np.float32), open(os.path.join(data_serialization_dir, "word_vectors"), "wb"))
    pickle.dump(np.asarray(char_vector_values, dtype=np.float32), open(os.path.join(data_serialization_dir, "char_vectors"), "wb"))
    pickle.dump(word_to_id_dicts, open(os.path.join(data_serialization_dir, "word_to_id_dicts"), "wb"))
    pickle.dump(char_to_id_dicts, open(os.path.join(data_serialization_dir, "char_to_id_dicts"), "wb"))
    # 生成train和valid data
    x_train_data, y_train_data, x_valid_data, y_valid_data = generate_train_valid_data(configs, data_lists, word_to_id_dicts, char_to_id_dicts)
    pickle.dump(x_train_data, open(os.path.join(data_serialization_dir, "x_train_data"), "wb"))
    pickle.dump(y_train_data, open(os.path.join(data_serialization_dir, "y_train_data"), "wb"))
    pickle.dump(x_valid_data, open(os.path.join(data_serialization_dir, "x_valid_data"), "wb"))
    pickle.dump(y_valid_data, open(os.path.join(data_serialization_dir, "y_valid_data"), "wb"))

    print("x train data shape:", np.asarray(x_train_data).shape)
    print("y train data shape:", np.asarray(y_train_data).shape)
