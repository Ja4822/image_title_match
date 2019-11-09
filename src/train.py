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
BATCH_SIZE = 16
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
    return image_list

def create_train_test_samples(ids, masks, segmensts, images):
    # 将imgs分为正负样本，正样本不打乱，负样本打乱顺序
    pos_imgs, neg_imgs = train_test_split(images, train_size=0.5, test_size=0.5, shuffle=False)
    print('[INFO] pos samples, neg samples = (%d, %d)'%(len(pos_imgs), len(neg_imgs)))
    # 生成正负样本的labels
    pos_labels = [1]*len(pos_imgs)
    neg_labels = [0]*len(neg_imgs)
    pos_labels.extend(neg_labels)
    labels = pos_labels
    # 保证打乱的负样本确实是都打乱了
    index = []
    for i in range(len(neg_imgs)):
        while True:
            n = np.random.randint(0, len(neg_imgs))
            if i == n:
                continue
            else:
                index.append(n)
                break
    neg_imgs = np.array(neg_imgs)[index]
    neg_imgs = neg_imgs.tolist()
    pos_imgs.extend(neg_imgs)
    # 打乱后的iamges
    images = pos_imgs
    with tf.Session() as sess:
        labels = sess.run(tf.one_hot(labels, NUM_LABELS, on_value=1, off_value=0))

    train_ids, test_ids           = train_test_split(ids, train_size=1-TEST_PERCENTAGE, 
                                                    test_size=TEST_PERCENTAGE, random_state=0)
    train_masks, test_masks       = train_test_split(masks, train_size=1-TEST_PERCENTAGE, 
                                                    test_size=TEST_PERCENTAGE, random_state=0)
    train_segments, test_segments = train_test_split(segmensts, train_size=1-TEST_PERCENTAGE, 
                                                    test_size=TEST_PERCENTAGE, random_state=0)
    train_images, test_images     = train_test_split(images, train_size=1-TEST_PERCENTAGE, 
                                                    test_size=TEST_PERCENTAGE, random_state=0)
    train_labels, test_labels     = train_test_split(labels, train_size=1-TEST_PERCENTAGE, 
                                                    test_size=TEST_PERCENTAGE, random_state=0)

    return train_ids, test_ids, train_masks, test_masks, train_segments, test_segments,\
        train_images, test_images, train_labels, test_labels

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

if __name__ == "__main__":

    print('======================= Loading texts =======================')
    input_idsList, input_masksList, segment_idsList = create_text_input()
    print('[INFO] texts  num = %d'%(len(input_idsList)))

    print('======================= Loading images ======================')
    input_images = create_image_input()
    print('[INFO] images num = %d, shape = %s'%(len(input_images), np.array(input_images).shape))
    # print('[INFO] labels num = %d, shape = %s'%(len(input_labels), np.array(input_labels).shape))

    print('================ Seperate train and test data ===============')
    train_ids, test_ids, train_masks, test_masks, train_segments, test_segments, \
    train_images, test_images, train_labels, test_labels = create_train_test_samples(
        input_idsList, input_masksList, segment_idsList, input_images)

    print('[INFO] train_ids, test_ids           = (%d, %d)'%(len(train_ids), len(test_ids)))
    print('[INFO] train_masks, test_masks       = (%d, %d)'%(len(train_masks), len(test_masks)))
    print('[INFO] train_segments, test_segments = (%d, %d)'%(len(train_segments), len(test_segments)))
    print('[INFO] train_images, test_images     = (%s, %s)'%(np.array(train_images).shape, np.array(test_images).shape))
    print('[INFO] train_labels, test_labels     = (%s, %s)'%(np.array(train_labels).shape, np.array(test_labels).shape))
    num_train_data = len(train_ids)

    print('====================== Restoring model ======================')
    print('====================== Restoring bert =======================')
    with tf.variable_scope('bert_model'):
        # bert的输入
        input_ids    = tf.placeholder(shape=[None, MAX_SEQ_LENGTH], dtype=tf.int32,name="input_ids")
        input_mask   = tf.placeholder(shape=[None, MAX_SEQ_LENGTH], dtype=tf.int32,name="input_mask")
        segment_ids  = tf.placeholder(shape=[None, MAX_SEQ_LENGTH], dtype=tf.int32,name="segment_ids")
        # BERTMODEL
        model = modeling.BertModel(
            config=BERT_CONFIG,
            is_training=IS_TRAINING,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )

        # 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size]
        bert_layer_indexes = [-1, -2, -3, -4]
        bert_hidden_size = model.get_sequence_output().shape[-1].value
        bert_all_layers = [model.all_encoder_layers[i] for i in bert_layer_indexes]
        bert_tmp_layer = bert_all_layers[0]
        for layer in bert_all_layers:
            bert_tmp_layer += layer
        bert_concat_layer = (bert_tmp_layer - bert_all_layers[0])/len(bert_layer_indexes)
        bert_output_layer = tf.slice(bert_concat_layer, [0,0,0], [-1,1,bert_hidden_size])

        ### final bert output, sentence vector
        bert_output_layer = tf.squeeze(bert_output_layer, axis=1, name='concated_bert_output')

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, BERT_CKPT)

        tf.train.init_from_checkpoint(BERT_CKPT, assignment_map)
    print('===================== Restoring resnet ======================')
    with tf.variable_scope('resnet_model'):
        ### RESNETMODEL
        input_images = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='input_images')
        input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS], name='input_labels')

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, _ = resnet_v1.resnet_v1_50(input_images, num_classes=None, is_training=IS_TRAINING)
        # 加载pre-trained model
        tvars = tf.trainable_variables()
        tvar_list = []
        for tvar in tvars:
            if 'resnet' in tvar.name:
                tvar_list.append(tvar)
        tvars = tvar_list
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, RESNET_CKPT)
        tf.train.init_from_checkpoint(RESNET_CKPT, assignment_map)


        logits = tf.squeeze(logits, axis=[1, 2])
        # resnet_fc = slim.fully_connected(logits, num_outputs=bert_hidden_size, activation_fn=None,
        #             weights_initializer=tf.initializers.variance_scaling(), scope='resnet_fc')

    print('================== Building mixed model =====================')
    with tf.variable_scope('mixed_model'):
        ### resnet输出的向量维度和bert的维度不同，先过一层dense，再拼接
        ### model: bert_output_layer + resnet_fc --> model_concat_layer
        ### bn --> dense --> dropout
        model_concat_layer = tf.concat([bert_output_layer, logits], -1)
        model_bn_layer = slim.batch_norm(model_concat_layer, decay=0.9, 
                    zero_debias_moving_mean=True, is_training=IS_TRAINING, scope='model_bn1')
        model_fc_layer = slim.fully_connected(model_bn_layer, num_outputs=FC_SIZE,
                    weights_initializer=tf.initializers.variance_scaling(), scope='model_fc1')
        model_dropout = slim.dropout(model_fc_layer, is_training=IS_TRAINING, scope='model_dropout1')
        # ### bn --> dense --> dropout
        # model_bn_layer = slim.batch_norm(model_dropout, decay=0.9, 
        #             zero_debias_moving_mean=True, is_training=IS_TRAINING, scope='model_bn2')
        # model_fc_layer = slim.fully_connected(model_bn_layer, num_outputs=FC_SIZE,
        #             weights_initializer=tf.initializers.variance_scaling(), scope='model_fc2')
        # model_dropout = slim.dropout(model_fc_layer, is_training=IS_TRAINING, scope='model_dropout2')
        ### dense --> softmax
        model_fc_layer = slim.fully_connected(model_dropout, num_outputs=NUM_LABELS, activation_fn=None, 
                    weights_initializer=tf.initializers.variance_scaling(), scope='model_fc3')
        model_logits = slim.softmax(model_fc_layer, scope='softmax')
        tf.add_to_collection("predict", model_logits)

    with tf.variable_scope('loss'):
        
        loss = tf.losses.softmax_cross_entropy(input_labels, model_logits, weights=1.0)
        # loss = tf.losses.get_total_loss()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # 参考AdamOptimizer源码确定参数
            train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    with tf.variable_scope('accuracy'):
        predict = tf.argmax(model_logits, 1)
        correct_prediction = tf.equal(tf.argmax(model_logits, 1), tf.argmax(input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('summaty'):
        saver = tf.train.Saver()

    print('=================== Trainable Variables =====================')
    # tvars = tf.trainable_variables()
    # for var in tvars:
    #     init_string = ""
    #     if var.name in initialized_variable_names:
    #         init_string = ", *INIT_FROM_CKPT*"
    #     print("name = %s, shape = %s%s"%(var.name, var.shape,
    #                     init_string))

    print('=================== Setting input data ======================')
    # 设置batch
    with tf.variable_scope('input_data'):
        train_batch = tf.data.Dataset.from_tensor_slices(
                    (input_ids, input_mask, segment_ids, input_images, input_labels))
        train_batch = train_batch.shuffle(20).batch(BATCH_SIZE).repeat()
        train_batch_iterator = train_batch.make_initializable_iterator()
        train_batch_data = train_batch_iterator.get_next()

        test_batch = tf.data.Dataset.from_tensor_slices(
                    (input_ids, input_mask, segment_ids, input_images, input_labels))
        test_batch = test_batch.shuffle(20).batch(BATCH_SIZE).repeat()
        test_batch_iterator = test_batch.make_initializable_iterator()
        test_batch_data = test_batch_iterator.get_next()


    with tf.Session() as sess:
        # 初始化
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        print('==================== Starting training ======================')
        num_iteration_train = np.array(train_ids).shape[0] // BATCH_SIZE + 1
        num_iteration_test = np.array(test_ids).shape[0] // BATCH_SIZE + 1
        # 初始化迭代器
        sess.run(train_batch_iterator.initializer, 
                feed_dict={input_ids:train_ids, input_mask:train_masks, segment_ids:train_segments,
                    input_images:train_images, input_labels:train_labels})
        sess.run(test_batch_iterator.initializer, 
                feed_dict={input_ids:test_ids, input_mask:test_masks, segment_ids:test_segments,
                    input_images:test_images, input_labels:test_labels})

        for epoch in range(EPOCH):
            train_loss_list = []
            train_acc_list = []
            for i in range(num_iteration_train):
                batch_ids, batch_masks, batch_segments, batch_images, batch_labels = sess.run(train_batch_data)
                # print(batch_labels[0])
                train_loss, train_acc, _ , prediction = sess.run([loss, accuracy, train_op, model_logits],
                    feed_dict={input_ids:batch_ids, input_mask:batch_masks, segment_ids:batch_segments,
                        input_images:batch_images, input_labels:batch_labels})
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                if i % 20 == 0 or i+1 == num_iteration_train:
                    # print(batch_labels[:10])
                    # print(prediction[:10])
                    print('Epoch %d/%d, batch %d/%d, tr_loss = %.3f, tr_acc = %.3f'\
                        %(epoch+1, EPOCH, i+1, num_iteration_train, train_loss, train_acc))
            # 每次训练对整个test进行测试
            test_loss_list = []
            test_acc_list = []
            for j in range(num_iteration_test):
                batch_test_ids, batch_test_masks, batch_test_segments, batch_test_images, batch_test_labels = sess.run(test_batch_data)
                test_loss, test_acc = sess.run([loss, accuracy],
                    feed_dict={input_ids:batch_test_ids, input_mask:batch_test_masks, segment_ids:batch_test_segments,
                        input_images:batch_test_images, input_labels:batch_test_labels})
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            
            # 每个epoch平均的loss, acc
            train_loss = np.mean(train_loss_list)
            train_acc = np.mean(train_acc_list)
            test_loss = np.mean(test_loss_list)
            test_acc = np.mean(test_acc_list)
            print('Epoch %d/%d, tr_loss = %.3f, tr_acc = %.3f, te_loss = %.3f, te_acc = %.3f'\
                %(epoch+1, EPOCH, train_loss, train_acc, test_loss, test_acc))
                
            if epoch % 50 == 0 or epoch + 1 == EPOCH:
                saver.save(sess, MODEL_PATH+'model.ckpt', epoch+1)
                print('[INFO] save model in %s'%(MODEL_PATH+'model.ckpt'))