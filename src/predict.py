import os
import cv2
import numpy as np
import pandas as pd
import load_data
import tokenization
from bert import modeling
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# bert配置文件
BERT_CONFIG = modeling.BertConfig.from_json_file('../model/uncased_L-12_H-768_A-12/bert_config.json')
VOCAB_FILE = '../model/uncased_L-12_H-768_A-12/vocab.txt'
BERT_CKPT = '../model/uncased_L-12_H-768_A-12/bert_model.ckpt'
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 5e-5

# resnet
MODEL_PATH = '../model/fine_tuned_model/'
RESNET_CKPT = '../model/resnet/resnet_v1_50.ckpt'
CHECKPOINT_EXCLUDE_SCOPES = 'Logits'
IMG_SIZE = 32

# data
DATA_PATH  = '../data/sorted_titles.csv'
IMG_PATH = '../data/sorted/'

# 训练参数
EPOCH = 10
IS_TRAINING = True
BATCH_SIZE = 32
NUM_LABELS = 2
FC_SIZE = 2000
TEST_PERCENTAGE = 0.4

def create_text_input():
    data = pd.read_csv(DATA_PATH)
    texts=[]
    # lables=[]
    m, n = data.shape
    for i in range(m):
        # 数据集中第952张图片不能用
        if i == 952:
            continue
        row = str(data.iloc[i,:])
        texts.append(row)
        # lables.append(0)

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

def create_image_input():
    image_list = []
    img_names = os.listdir(IMG_PATH)
    for i in range(len(img_names)):
        img_names[i] = int(img_names[i][:-4])
    img_names = np.array(img_names)
    img_names = np.sort(img_names)
    img_names = list(img_names)
    for img_name in img_names:
        img_path = os.path.join(IMG_PATH, str(img_name)+'.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img = img / 255.0
        image_list.append(img)
    label_list = [1]*len(image_list)
    with tf.Session() as sess:
        label_list = sess.run(tf.one_hot(label_list, NUM_LABELS, on_value=1, off_value=0))

    return image_list, label_list

if __name__ == "__main__":

    print('======================= Loading texts =======================')
    input_idsList, input_masksList, segment_idsList = create_text_input()
    print('[INFO] texts  num = %d'%(len(input_idsList)))

    print('======================= Loading images ======================')
    input_imagesList, input_labelsList = create_image_input()
    print('[INFO] images num = %d, shape = %s'%(len(input_imagesList), np.array(input_imagesList).shape))
    print('[INFO] labels num = %d, shape = %s'%(len(input_labelsList), np.array(input_labelsList).shape))

    print('=================================================')
    with tf.Session() as sess:
        # 初始化
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # 加载模型
        saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH,  'model.ckpt-10.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        graph = tf.get_default_graph()

        # 获得输入输出的op
        input_ids = graph.get_operation_by_name('bert_model/input_ids').outputs[0]
        input_masks = graph.get_operation_by_name('bert_model/input_mask').outputs[0]
        input_segments = graph.get_operation_by_name('bert_model/segment_ids').outputs[0]
        input_images = graph.get_operation_by_name('resnet_model/input_images').outputs[0]
        input_labels = graph.get_operation_by_name('resnet_model/input_labels').outputs[0]
        prediction = tf.get_collection('predict')[0]

        print(input_ids)
        print(input_masks)
        print(input_segments)
        print(input_images)
        print(input_labels)
        print(prediction)
        
        batch_data = tf.data.Dataset.from_tensor_slices(
                    (input_ids, input_masks, input_segments, input_images, input_labels))
        batch_data = batch_data.shuffle(20).batch(BATCH_SIZE).repeat()
        batch_data_iterator = batch_data.make_initializable_iterator()
        batch_data = batch_data_iterator.get_next()

        sess.run(batch_data_iterator.initializer, 
                feed_dict={input_ids:input_idsList, input_masks:input_masksList, input_segments:segment_idsList,
                    input_images:input_imagesList, input_labels:input_labelsList})

        acc_list = []
        total_num = np.array(input_idsList).shape[0]
        num_iteration = total_num // BATCH_SIZE + 1
        for i in range(num_iteration):
            print('%d/%d'%((i+1), num_iteration))

            batch_ids, batch_masks, batch_segments, batch_images, batch_labels = sess.run(batch_data)

            predictions = sess.run(prediction,
                    feed_dict={input_ids:batch_ids, input_masks:batch_masks, input_segments:batch_segments,
                        input_images:batch_images, input_labels:batch_labels})
            print(predictions[:10])
            print(batch_labels[:10])
            result_idx = sess.run(tf.argmax(predictions, 1))
            print(result_idx)
            ground_truth_idx = sess.run(tf.argmax(batch_labels, 1))
            print(ground_truth_idx)
            accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(result_idx, ground_truth_idx), dtype=tf.float32)))
            print(accuracy)
            acc_list.append(accuracy)
        acc_mean = np.mean(np.array(acc_list))

        print('[INFO] Prediction is ', result_idx)
        print('[INFO] Ground truth is ', ground_truth_idx)
        print('[INFO] Accuracy is ', acc_mean)