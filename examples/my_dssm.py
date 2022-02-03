# coding: utf-8

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 功能: BATCH负采样模型，通过将内部矛盾转移为敌我矛盾，解决east sample问题
# 时间: 2021/12/9 11:00
# 模型结构：简单一层双塔

import tensorflow as tf
import numpy as np
import time
import random

# 输入
USER_SLOT_SIZE = 50
ITEM_SLOT_SIZE = 74
SLOT_SIZE = USER_SLOT_SIZE + ITEM_SLOT_SIZE
EMB_DIM = 16

# embedding用户物料拆分
USER_INPUT_SIZE = USER_SLOT_SIZE * EMB_DIM
ITEM_INPUT_SIZE = ITEM_SLOT_SIZE * EMB_DIM

# 模型结构
DEEP_LAYERS_1 = 64  # 所有实验的输出都是64
BATCH_SIZE = 2048
NEG_NUM = 2
HARD_INSTANCE_NUM = 1

# 超参数
TOP = 0.05
BOTTOM = 0.1
T = 10

# 本地调试
IS_DEBUG = 0
if IS_DEBUG:
    BATCH_SIZE = 24
    # np.set_printoptions(threshold=np.inf)
    print(tf.__version__)

"""input"""
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SLOT_SIZE * EMB_DIM], name="embeddings")
    y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1], name="label")

    sample_weight = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1], name="sample_weight")

    x_user, x_item = tf.split(x, [USER_INPUT_SIZE, ITEM_INPUT_SIZE], 1)

"""user deep part"""
with tf.name_scope("user_tower"):
    with tf.name_scope("deep_layer_1"):
        w01 = tf.Variable(tf.random_normal([USER_INPUT_SIZE, DEEP_LAYERS_1], 0.0, 10), name="w01")
        b01 = tf.Variable(tf.random_normal([1, DEEP_LAYERS_1], 0.0, 10), name="b01")

        y_deep_01 = tf.add(tf.matmul(x_user, w01), b01)

    user_tower_output_ = y_deep_01
    user_tower_output = user_tower_output_ / tf.norm(user_tower_output_, axis=1, ord=2, keep_dims=True)

"""item deep part"""
with tf.name_scope("item_tower"):
    with tf.name_scope("deep_layer_1"):
        w11 = tf.Variable(tf.random_normal([ITEM_INPUT_SIZE, DEEP_LAYERS_1], 0.0, 10), name="w11")
        b11 = tf.Variable(tf.random_normal([1, DEEP_LAYERS_1], 0.0, 10), name="b11")

        y_deep_11 = tf.add(tf.matmul(x_item, w11), b11)

    item_tower_ouput_ = y_deep_11
    item_tower_ouput = item_tower_ouput_ / tf.norm(item_tower_ouput_, axis=1, ord=2, keep_dims=True)

"""easy neg instance"""
# 构造N批负样本，axis=0concat
item_neg = item_tower_ouput
for i in range(NEG_NUM):
    random_seed = int((random.random() + i) * BATCH_SIZE)
    item_shuffle = tf.gather(item_tower_ouput, tf.random_shuffle(tf.range(tf.shape(item_tower_ouput)[0]), seed=random_seed))
    item_neg = tf.concat([item_neg, item_shuffle], 0)
#[NEG_NUMM+1, 64]
a = tf.multiply(tf.tile(user_tower_output, [NEG_NUM + 1, 1]), item_neg)

# [(NEG_NUM + 1) * B, 1],user和负样本的内积算好了
prod_raw = tf.reduce_sum(a, 1, True)

# [BATCH_SIZE, NEG_NUM + 1] @TODO!!!
prod_inner = tf.transpose(tf.reshape(tf.transpose(prod_raw), [NEG_NUM + 1, BATCH_SIZE]))
# prod_inner = tf.transpose(tf.reshape(prod_raw, [NEG_NUM + 1, BATCH_SIZE])) 与上等价

"""hard neg instance"""
all_ip = tf.matmul(user_tower_output, tf.transpose(item_tower_ouput)) #[2048,2048]，一行里是某一个user对该batch所有item的
# 找出每个用户的前102个item
# shape都是[2048,102]
top_k_output, top_k_index = tf.nn.top_k(all_ip, int(BATCH_SIZE * TOP))
# 找出每个用户的后205个item
# shape都是[2048,204]
bottom_k_output, bottom_k_index = tf.nn.top_k(-all_ip, int(BATCH_SIZE * BOTTOM))

#axis=1拼接
# shape都是[2048,306]
black_index = tf.concat([top_k_index, bottom_k_index], -1)

#【1，2048】
aa = tf.expand_dims(tf.constant(np.arange(BATCH_SIZE)), 0)

# 【2048，2048】
a = tf.tile(aa, [BATCH_SIZE, 1])
all_index = tf.cast(a, tf.int32)

# [2048,1741]，这是中间的index？？？
while_index = tf.sparse_tensor_to_dense(tf.sets.set_difference(all_index, black_index))
# [2048,1741]
my_num = BATCH_SIZE - int(BATCH_SIZE * TOP) - int(BATCH_SIZE * BOTTOM) - 1
# 是while_index整个matrix，只是为了确定列数
while_index_fix = tf.slice(while_index, [0, 0], [-1, my_num])

# [2048,1741]
# 按行shuffle一下
while_index_shuffle = tf.transpose(tf.random.shuffle(tf.transpose(while_index_fix), seed=BATCH_SIZE))
# [2048,1]
while_index_result = tf.slice(while_index_shuffle, [0, 0], [-1, HARD_INSTANCE_NUM])
# [2048,1]
hard_neg = tf.batch_gather(all_ip, while_index_result)

"""cal loss"""
# prod_inner：【2048，3】==>[2048, 4]
prod_inner = tf.concat([prod_inner, hard_neg], -1)
prod_inner_t = prod_inner * T

prob = tf.nn.softmax(prod_inner_t, name="prob")
hit_prob = tf.slice(prob, [0, 0], [-1, 1])  # 只取第一列正样本

# 求均值
loss = -tf.reduce_mean(tf.log(hit_prob + 1e-24))

uid_emb = tf.identity(user_tower_output, name="uid_emb")
mid_emb = tf.identity(item_tower_ouput, name="mid_emb")

"""log 函数"""
def count_most_val(tensor_value):
    hash_val = tf.strings.to_hash_bucket_strong(tf.as_string(tensor_value), BATCH_SIZE * 10, [14, 2])
    one_hot_val = tf.one_hot(indices=hash_val, depth=BATCH_SIZE * 10)

    one_hot_val_sum = tf.reduce_sum(one_hot_val, axis=0)
    count, index = tf.nn.top_k(one_hot_val_sum, 10)

    return count

"""log 输出节点"""
sample_weight_node = tf.identity(sample_weight, "sample_weight_node")
product = tf.reduce_sum(user_tower_output * item_tower_ouput, axis=1, keepdims=True)
predict_out = tf.sigmoid(product, name="predict_out")
predict_auc = tf.metrics.auc(labels=y, predictions=predict_out, num_thresholds=2000)
auc_out = tf.identity(predict_auc, name="auc_out")
loss = tf.identity(loss, name="loss")
# id = tf.identity(y, name="id")
# most_count_uid = tf.identity(count_most_val(id), name="most_count_uid")
log1 = prod_inner
log2 = prob

# 用于计算各物料类型LOSS
mblog_type_pre_list = []
for i in range(0, 9):
    # 内容类型：1&视频、2&文章、3&gif、4&长图、5&全景、6&图片、7&链接、8&投票
    is_mblog_type = tf.squeeze(tf.equal(sample_weight, i))
    mblog_type_pre = tf.boolean_mask(hit_prob, is_mblog_type, axis=0, name="mblog_type_" + str(i) + "_pre")
    mblog_type_pre_list.append(mblog_type_pre)

    mblog_type_num = tf.identity(tf.shape(mblog_type_pre)[0], "mblog_type_" + str(i) + "_num")
    mblog_type_loss = -tf.reduce_mean(tf.log(mblog_type_pre + 1e-24), name="mblog_type_" + str(i) + "_loss")

"""train"""
with tf.name_scope("train"):
    LEARNING_RATE = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_step = optimizer.minimize(loss)

init = tf.group([tf.local_variables_initializer(), tf.global_variables_initializer()], name="weidl_init")

"""loss 部分"""
dx_ = tf.gradients(loss, x)

# 偏导数
dw01_ = tf.gradients(loss, w01)
db01_ = tf.gradients(loss, b01)

dw11_ = tf.gradients(loss, w11)
db11_ = tf.gradients(loss, b11)

# 重命名

dx = tf.identity(dx_[0], name="dx")

dw01 = tf.identity(dw01_[0], name="dw01")
db01 = tf.identity(db01_[0], name="db01")

dw11 = tf.identity(dw11_[0], name="dw11")
db11 = tf.identity(db11_[0], name="db11")

if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run(init)

        tf.train.write_graph(sess.graph_def, "./", "yuanbo5_hotmblog_recall_dssm_ctr_sprouts.pb", as_text=False)
        # tf.train.write_graph(sess.graph_def, "./", "deepfm_model_ref.txt", as_text=True)
        # writer = tf.summary.FileWriter('./deepfm_model_ref', sess.graph)

        # 打印所有需要训练的参数
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            tensor_name = str(k).strip(":0")
            # print('"%s",' % tensor_name)
            # print("Shape: ", v.shape)
            print('"%s" shape is %s,' % (tensor_name, str(v.shape)))
            # if '/w' in  tensor_name:
            #     print('"%s",' % tensor_name)

        for i in range(IS_DEBUG):
            feed_dict = {

                # w01: np.random.normal(loc=0.0, scale=0.1, size=[USER_INPUT_SIZE, DEEP_LAYERS_1]),
                # b01: np.random.normal(loc=0.0, scale=0.1, size=[1, DEEP_LAYERS_1]),
                # w11: np.random.normal(loc=0.0, scale=0.1, size=[ITEM_INPUT_SIZE, DEEP_LAYERS_1]),
                # b11: np.random.normal(loc=0.0, scale=0.1, size=[1, DEEP_LAYERS_1]),
                x: np.random.rand(BATCH_SIZE, SLOT_SIZE * EMB_DIM),
                y: np.reshape(np.random.randint(2, size=BATCH_SIZE), [BATCH_SIZE, 1]),
                sample_weight: np.array([[6], [6], [1], [0], [0], [1], [6], [6], [1], [0], [0], [1], [6], [6], [1], [0], [0], [1], [6], [6], [1], [0], [0], [1]])
            }

            # 初始化
            run_list = [init]
            sess.run(run_list)

            # test1
            run_list = [w01, b01, w11, b11]
            init_ = sess.run(run_list)
            # print "w1:", init_[0]
            # print "b1:", init_[1]

            # test2
            start = time.time()

            run_list = [predict_out, loss, prob]
            predictions_, loss_, prob_ = sess.run(run_list, feed_dict=feed_dict)
            # print "loss:", loss_
            # print "predictions:", predictions_
            # print "prob_:", prob_

            run_list = [sample_weight_node, uid_emb, mid_emb, log1, log2]
            result = sess.run(run_list, feed_dict=feed_dict)
            for i in result:
                print (i)
                print ("-------")

            end = time.time()
            # print "time one epoch:", str(end - start)

            # test3
            start = time.time()

            run_list = [dx, dw01, dw11, db01, db11]
            grad_list = sess.run(run_list, feed_dict=feed_dict)

            end = time.time()
            # print "time one epoch:", str(end - start)
            #
            # print "grad_x:", grad_list[0]
            # print "grad_dw:", grad_list[1]