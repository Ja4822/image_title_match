'''
model = ResNet50(weights='imagenet')
img_path = '../data/test_0.JPEG'
img = image.load_img(img_path, target_size=(224,224))
# print(np.array(img).shape)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
print(img.shape)
predict = model.predict(img)
result = decode_predictions(predict)
print(result)
'''
'''
tokenizer = Tokenizer(token_dict)
text = '语言模型'
token = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=512)
#predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]
#print(np.array(predicts).shape)
#print(predicts)
'''

# import common
import os
import codecs
import numpy as np
# import inceptionv3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import decode_predictions, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input
from keras import layers
from keras.optimizers import RMSprop
from keras import backend as K
# import bert
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Lambda

def load_data():
    pass
def squeeze(x):
    return tf.squeeze(x, axis=[1,2])

if __name__ == '__main__':
    # 加载inceptionv3模型
    print('====== LOAD INCEPTIONV3 ======')
    cv_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(64,64,3))
    #　添加dense层
    cv_input = Input((64,64,3))
    cv_output = cv_model(cv_input)
    img_feature = Dense(768)(cv_output)
    img_feature = Lambda(squeeze, argument={'x':img_feature})(img_feature)
    #img_feature = tf.squeeze(img_feature, axis=[1,2])
    # 设置dense层之前不可训练
    for layer in cv_model.layers:
        layer.trainable = False

    # 加载bert模型
    pretrained_path = '../model/uncased_L-12_H-768_A-12'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    os.environ['TF_KERAS'] = '1'

    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    print('====== LOAD BERT ======')
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for layer in bert_model.layers:
        layer.trainable = False

    # 获取bert的输入输出
    bert_input = bert_model.input
    bert_input_token = bert_input[0]
    bert_input_segment = bert_input[1]
    bert_output = bert_model.output

    bert_out_merged = bert_output
    bert_out_merged = tf.squeeze(bert_out_merged[:, 0:1, :], axis=1)

    print('====== KEY LAYERS ======')
    print('cv_input: ', cv_input)
    print('cv_output: ', cv_output)
    print('img_feature: ', img_feature)
    print('bert_input_token: ', bert_input_token)
    print('bert_input_segment: ', bert_input_segment)
    print('bert_output: ', bert_out_merged)

    conc_layer = layers.concatenate([bert_out_merged, img_feature])
    print('conc layer: ', conc_layer)
    #x = keras.layers.concatenate([bert_out_merged, img_feature])
    x = Dense(1000)(conc_layer)
    x = BatchNormalization()(x)
    x = Dense(1000)(x)
    x = BatchNormalization()(x)
    main_output = Dense(2, activation='softmax')(x)
    print('main_output: ', main_output)

    merged_model = Model([bert_input_token, bert_input_segment, cv_input], [main_output])
    merged_model.summary()
    op = RMSprop(lr=0.001, decay=1e-6)

    merged_model.compile(loss='binary_cross_entropy', optimizer=op, metrics=['accuracy'])
