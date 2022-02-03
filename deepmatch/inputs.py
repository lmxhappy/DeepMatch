from itertools import chain

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, create_embedding_matrix, embedding_lookup, \
    get_dense_input, varlen_embedding_lookup, get_varlen_pooling_list, mergeDict


def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False, embedding_matrix_dict=None):
    '''
    特征经过embed layer层。分成两类，分别经过embed layer，并最后输出output。

    返回，两个list：稀疏特征的embed layer的tensor；空list
    '''
    # 分成两类
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if embedding_matrix_dict is None:
        embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                        seq_mask_zero=seq_mask_zero)

    # 1 第一类-sparse特征
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    # 2 第二类——变长特征
    # fetures:dict,key是特征名字，value是一个tensor，shape是[B,1]，特别的，hist_movie_id的value的shape是[B, 50]
    # sequence_embed_dict：[B, 50, 16]
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)

    # group_embedding_dict是一个dict
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        # 从一个dict里将所有value掏出来，放到一个list里
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))

    return group_embedding_dict, dense_value_list
