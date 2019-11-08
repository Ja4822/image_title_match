import tensorflow as tf
from bert import modeling
import os
import pandas as pd
import load_data
import tokenization
import numpy as np

# bert配置文件
BERT_CONFIG = modeling.BertConfig.from_json_file('../model/uncased_L-12_H-768_A-12/bert_config.json')
VOCAB_FILE = '../model/uncased_L-12_H-768_A-12/vocab.txt'
BATCH_SIZE = 128
NUM_LABLES = 2
IS_TRAINING = False
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 5e-5

# titles文件路径
DATA_PATH  = '../data/sorted_titles.csv'

def create_input():
    data = pd.read_csv(DATA_PATH)
    texts=[]
    m, n = data.shape
    for i in range(m):
        row = str(data.iloc[i,:])
        texts.append(row)

    # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE)
    input_idsList=[]
    input_masksList=[]
    segment_idsList=[]
    for t in texts:
        single_input_id, single_input_mask, single_segment_id=load_data.convert_single_example(MAX_SEQ_LENGTH,tokenizer,t)
        input_idsList.append(single_input_id)
        input_masksList.append(single_input_mask)
        segment_idsList.append(single_segment_id)

    input_idsList = np.asarray(input_idsList,dtype=np.int32)
    input_masksList = np.asarray(input_masksList,dtype=np.int32)
    segment_idsList = np.asarray(segment_idsList,dtype=np.int32)
    # lables = np.asarray(lables,dtype=np.int32)
    return input_idsList, input_masksList, segment_idsList# , lables

def save_results(matrix, i_idx, j_idx, file_name):
    print('=============== SAVING RESULTS ===============')
    with open('../data/%s.csv'%(file_name), 'a') as f:
        row = ''
        for i in range(i_idx):
            row += str(i)+','
        row = row[:-1]
        f.write(row+'\n')
        for i in range(i_idx):
            row = ''
            for j in range(j_idx):
                row += str(matrix[i][j])+','
            row = row[:-1]
            f.write(row+'\n')
        print('[INFO] Finish saving %s.csv'%(file_name))
    print('==============================================')

def get_cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

if __name__ == "__main__":

    print('============= Loading data ==============')
    # input_idsList, input_masksList, segment_idsList, lables = create_input()
    input_idsList, input_masksList, segment_idsList = create_input()
    num_data = len(input_idsList)
    print('[INFO] Data num = %d'%(num_data))

    print('============ Restoring model ============')
    # 创建bert的输入
    input_ids    = tf.placeholder(shape=[None, MAX_SEQ_LENGTH], dtype=tf.int32,name="input_ids")
    input_mask   = tf.placeholder(shape=[None, MAX_SEQ_LENGTH], dtype=tf.int32,name="input_mask")
    segment_ids  = tf.placeholder(shape=[None, MAX_SEQ_LENGTH], dtype=tf.int32,name="segment_ids")
    # input_lables = tf.placeholder(shape=None, dtype=tf.int32, name="lable_ids")
    # 创建bert模型
    model = modeling.BertModel(
        config=BERT_CONFIG,
        is_training=IS_TRAINING,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )

    # 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size]
    layer_indexes = [-1, -2, -3, -4]
    hidden_size = model.get_sequence_output().shape[-1].value
    all_layers = [model.all_encoder_layers[i] for i in layer_indexes]
    tmp_layer = all_layers[0]
    for layer in all_layers:
        tmp_layer += layer
    concat_layer = (tmp_layer - all_layers[0])/len(layer_indexes)
    output_layer = tf.slice(concat_layer, [0,0,0], [-1,1,hidden_size])

    ### final bert output, sentence vector
    output_layer = tf.squeeze(output_layer, axis=1)

    init_checkpoint = "../model/uncased_L-12_H-768_A-12/bert_model.ckpt"
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                        init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    
    '''
    print("========== Trainable Variables ==========")
    # 打印加载模型的参数
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s"%(var.name, var.shape,
                        init_string))
    '''

    print('========== Calculate text vector ==========')
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        num_iteration = num_data // BATCH_SIZE + 1
        text_vectors_list = []
        print('===========================================')
        for i in range(num_iteration):
            print('[INFO] processing NO.%d batch'%(i))
            if (i+1)*BATCH_SIZE < num_data:
                # batch_lables = lables[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_input_idsList = input_idsList[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_input_masksList = input_masksList[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_segment_idsList = segment_idsList[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            else:
                # batch_lables = lables[i*BATCH_SIZE:]
                batch_input_idsList = input_idsList[i*BATCH_SIZE:]
                batch_input_masksList = input_masksList[i*BATCH_SIZE:]
                batch_segment_idsList = segment_idsList[i*BATCH_SIZE:]

            text_vectors = sess.run(output_layer, feed_dict={
                input_ids:batch_input_idsList, 
                input_mask:batch_input_masksList,
                segment_ids:batch_segment_idsList})

            for text_vector in text_vectors:
                text_vectors_list.append(text_vector)
    
    print('========== CALCULATE COS SIMILARITY ==========')
    cos_sim_mat = np.zeros((num_data, num_data))
    for i in range(len(text_vectors_list)):
        print('calculate NO.%d text'%(i))
        for j in range(len(text_vectors_list)):
            if j > i:
                cos_sim = get_cos_sim(text_vectors_list[i], text_vectors_list[j])
                cos_sim = round(cos_sim, 3)
                cos_sim_mat[i][j] = cos_sim
            elif j == i:
                cos_sim_mat[i][j] = 1.0
            else:
                cos_sim_mat[i][j] = cos_sim_mat[j][i]
        break
    print('[INFO] Get cos sim matrix = \n', cos_sim_mat)
    
    print('============== Saveing results ==============')
    # save_results(cos_sim_mat, len(text_vectors_list), len(text_vectors_list), 'text_vectors')



