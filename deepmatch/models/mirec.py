"""
Author:
    Mingxing liu, lmxhappy@sina.com
"""
from deepctr.feature_column import build_model_input
from deepctr.layers import DNN
from deepctr.layers.utils import NoMask, combined_dnn_input
from tensorflow.python.keras.models import Model

from deepmatch.layers import PoolingLayer
from deepmatch.utils import get_item_embedding
from ..inputs import input_from_feature_columns, create_embedding_matrix
from ..layers.core import SampledSoftmaxLayer, EmbeddingIndex


def user_embed_layer(user_features, user_feature_columns, seed, embedding_matrix_dict, l2_reg_embedding):
    '''
    user input的embed layer
    '''
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)

    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    return user_dnn_input

# def item_embed_layer(item_feature_columns, item_features, embedding_matrix_dict):
#     '''
#     item input的embed layer
#     '''
#     # movie_id的embed matrix
#     item_feature_name = item_feature_columns[0].name
#     item_vocabulary_size = item_feature_columns[0].vocabulary_size
#
#     movie_id_embed_matrix = item_features[item_feature_name]
#     item_index = list(range(item_vocabulary_size))
#     item_index = EmbeddingIndex(item_index)(movie_id_embed_matrix)
#
#     # movie_id的embed matrix
#     item_embedding_matrix = embedding_matrix_dict[item_feature_name]
#     tmp = item_embedding_matrix(item_index)  # [209, 16]
#
#     # 这里是恒等变换
#     item_embedding_weight = NoMask()(tmp)  # 【B,16]
#
#     return item_embedding_weight

def MIRec(user_feature_columns, item_feature_columns, num_sampled=5,
               user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, output_activation='linear', seed=1024, ):
    """Instantiates the YoutubeDNN Model architecture. 包括定义网络的input、output和网络本身

    :param user_feature_columns: An iterable containing user's features used by  the model. 是user特征的meta信息。
    :param item_feature_columns: An iterable containing item's features used by  the model. 是item特征的meta信息。
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param output_activation: Activation function to use in output layer
    :return: A Keras model instance.

    """

    # 给用户和item建立一个embed matrix的dict
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed)
    # 1 user侧
    # 1.1 user侧input定义
    user_features = build_model_input(user_feature_columns)
    user_inputs_list = list(user_features.values())

    # 1.2 经过embed layer层
    user_dnn_input = user_embed_layer(user_features, user_feature_columns, seed, embedding_matrix_dict, l2_reg_embedding)

    # 1.3 经过MLP网络
    #[?, 16]
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation=output_activation, seed=seed)(user_dnn_input)

    # 2 item侧
    # 2.1 item侧input定义
    item_features = build_model_input(item_feature_columns)
    item_inputs_list = list(item_features.values())

    # item_embedding_weight = item_embed_layer(item_feature_columns, item_features, embedding_matrix_dict)

    # 2.2 经过embed layer层
    item_dnn_input = user_embed_layer(item_features, item_feature_columns, seed, embedding_matrix_dict, l2_reg_embedding)

    # 2.3 经过MLP网络
    #[?, 16]
    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation=output_activation, seed=seed)(item_dnn_input)

    # 3 关联做softmax
    # 这里也是恒等变换

    item_feature_name = item_feature_columns[0].name
    movie_id_embed_matrix = item_features[item_feature_name]

    softmax_layer = SampledSoftmaxLayer(num_sampled=num_sampled)
    output = softmax_layer([item_dnn_out, user_dnn_out, movie_id_embed_matrix]) #item_features[item_feature_name]:[?,1]
    print(type(user_inputs_list[0]))

    # 前面的是user侧的input，最后一个是item侧的input
    inputs = user_inputs_list + item_inputs_list
    model = Model(inputs=inputs, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding", item_dnn_out)

    return model
