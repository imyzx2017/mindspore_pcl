# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Architecture"""
import mindspore.nn as nn
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer


class MeanConv(nn.Cell):
    """
    BGCF mean aggregate layer.

    Args:
        feature_in_dim (int): The input feature dimension.
        feature_out_dim (int): The output feature dimension.
        activation (str): Activation function applied to the output of the layer, eg. 'relu'. Default: 'tanh'.
        dropout (float): Dropout ratio for the dropout layer. Default: 0.2.

    Inputs:
        - self_feature (Tensor) - Tensor of shape :math:`(batch_size, feature_dim)`.
        - neigh_feature (Tensor) - Tensor of shape :math:`(batch_size, neighbour_num, feature_dim)`.

    Outputs:
        Tensor, output tensor.
    """

    def __init__(self,
                 name,
                 feature_in_dim,
                 feature_out_dim,
                 activation,
                 dropout=0.2):
        super(MeanConv, self).__init__()

        self.out_weight = Parameter(
            initializer("XavierUniform", [feature_in_dim * 2, feature_out_dim], dtype=mstype.float32),
            name=name + 'out_weight')

        if activation == "tanh":
            self.act = P.Tanh()
        elif activation == "relu":
            self.act = P.ReLU()
        else:
            raise ValueError("activation should be tanh or relu")

        self.cast = P.Cast()
        self.matmul = P.MatMul()
        self.concat = P.Concat(axis=1)
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)

    def construct(self, self_feature, neigh_feature):
        neigh_matrix = self.reduce_mean(neigh_feature, 1)
        neigh_matrix = self.dropout(neigh_matrix)

        output = self.concat((self_feature, neigh_matrix))
        output = self.act(self.matmul(output, self.out_weight))
        return output


class AttenConv(nn.Cell):
    """
    BGCF attention aggregate layer.

    Args:
        feature_in_dim (int): The input feature dimension.
        feature_out_dim (int): The output feature dimension.
        dropout (float): Dropout ratio for the dropout layer. Default: 0.2.

    Inputs:
        - self_feature (Tensor) - Tensor of shape :math:`(batch_size, feature_dim)`.
        - neigh_feature (Tensor) - Tensor of shape :math:`(batch_size, neighbour_num, feature_dim)`.

    Outputs:
        Tensor, output tensor.
    """

    def __init__(self,
                 name,
                 feature_in_dim,
                 feature_out_dim,
                 dropout=0.2):
        super(AttenConv, self).__init__()

        self.out_weight = Parameter(
            initializer("XavierUniform", [feature_in_dim * 2, feature_out_dim], dtype=mstype.float32),
            name=name + 'out_weight')
        self.cast = P.Cast()
        self.squeeze = P.Squeeze(1)
        self.concat = P.Concat(axis=1)
        self.expanddims = P.ExpandDims()
        self.softmax = P.Softmax(axis=-1)
        self.matmul = P.MatMul()
        self.matmul_3 = P.BatchMatMul()
        self.matmul_t = P.BatchMatMul(transpose_b=True)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)

    def construct(self, self_feature, neigh_feature):
        """Attention aggregation"""
        query = self.expanddims(self_feature, 1)
        neigh_matrix = self.dropout(neigh_feature)

        score = self.matmul_t(query, neigh_matrix)
        score = self.softmax(score)
        atten_agg = self.matmul_3(score, neigh_matrix)
        atten_agg = self.squeeze(atten_agg)

        output = self.matmul(self.concat((atten_agg, self_feature)), self.out_weight)
        return output


class BGCF(nn.Cell):
    """
    BGCF architecture.

    Args:
        dataset_argv (list[int]): A list of the dataset argv.
        architect_argv (list[int]): A list of the model layer argv.
        activation (str): Activation function applied to the output of the layer, eg. 'relu'. Default: 'tanh'.
        neigh_drop_rate (list[float]): A list of the dropout ratio.
        num_user (int): The num of user.
        num_item (int): The num of item.
        input_dim (int): The feature dim.
    """

    def __init__(self,
                 dataset_argv,
                 architect_argv,
                 activation,
                 neigh_drop_rate,
                 num_user,
                 num_item,
                 input_dim):
        super(BGCF, self).__init__()

        self.user_embeddings = Parameter(initializer("XavierUniform", [num_user, input_dim], dtype=mstype.float32),
                                         name='user_embed')
        self.item_embeddings = Parameter(initializer("XavierUniform", [num_item, input_dim], dtype=mstype.float32),
                                         name='item_embed')
        self.cast = P.Cast()
        self.tanh = P.Tanh()
        self.shape = P.Shape()
        self.split = P.Split(0, 2)
        self.gather = P.GatherV2()
        self.reshape = P.Reshape()
        self.concat_0 = P.Concat(0)
        self.concat_1 = P.Concat(1)

        (self.input_dim, self.num_user, self.num_item) = dataset_argv
        self.layer_dim = architect_argv

        self.gnew_agg_mean = MeanConv('gnew_agg_mean', self.input_dim, self.layer_dim,
                                      activation=activation, dropout=neigh_drop_rate[1])
        self.gnew_agg_mean.to_float(mstype.float16)

        self.gnew_agg_user = AttenConv('gnew_agg_att_user', self.input_dim,
                                       self.layer_dim, dropout=neigh_drop_rate[2])
        self.gnew_agg_user.to_float(mstype.float16)

        self.gnew_agg_item = AttenConv('gnew_agg_att_item', self.input_dim,
                                       self.layer_dim, dropout=neigh_drop_rate[2])
        self.gnew_agg_item.to_float(mstype.float16)

        self.user_feature_dim = self.input_dim
        self.item_feature_dim = self.input_dim

        self.final_weight = Parameter(
            initializer("XavierUniform", [self.input_dim * 3, self.input_dim * 3], dtype=mstype.float32),
            name='final_weight')

        self.raw_agg_funcs_user = MeanConv('raw_agg_user', self.input_dim, self.layer_dim,
                                           activation=activation, dropout=neigh_drop_rate[0])
        self.raw_agg_funcs_user.to_float(mstype.float16)

        self.raw_agg_funcs_item = MeanConv('raw_agg_item', self.input_dim, self.layer_dim,
                                           activation=activation, dropout=neigh_drop_rate[0])
        self.raw_agg_funcs_item.to_float(mstype.float16)

    def construct(self,
                  u_id,
                  pos_item_id,
                  neg_item_id,
                  pos_users,
                  pos_items,
                  u_group_nodes,
                  u_neighs,
                  u_gnew_neighs,
                  i_group_nodes,
                  i_neighs,
                  i_gnew_neighs,
                  neg_group_nodes,
                  neg_neighs,
                  neg_gnew_neighs,
                  neg_item_num):
        """Aggregate user and item embeddings"""
        all_user_embed = self.gather(self.user_embeddings, self.concat_0((u_id, pos_users)), 0)

        u_self_matrix_at_layers = self.gather(self.user_embeddings, u_group_nodes, 0)
        u_neigh_matrix_at_layers = self.gather(self.item_embeddings, u_neighs, 0)

        u_output_mean = self.raw_agg_funcs_user(u_self_matrix_at_layers, u_neigh_matrix_at_layers)

        u_gnew_neighs_matrix = self.gather(self.item_embeddings, u_gnew_neighs, 0)
        u_output_from_gnew_mean = self.gnew_agg_mean(u_self_matrix_at_layers, u_gnew_neighs_matrix)

        u_output_from_gnew_att = self.gnew_agg_user(u_self_matrix_at_layers,
                                                    self.concat_1((u_neigh_matrix_at_layers, u_gnew_neighs_matrix)))

        u_output = self.concat_1((u_output_mean, u_output_from_gnew_mean, u_output_from_gnew_att))
        all_user_rep = self.tanh(u_output)

        all_pos_item_embed = self.gather(self.item_embeddings, self.concat_0((pos_item_id, pos_items)), 0)

        i_self_matrix_at_layers = self.gather(self.item_embeddings, i_group_nodes, 0)
        i_neigh_matrix_at_layers = self.gather(self.user_embeddings, i_neighs, 0)

        i_output_mean = self.raw_agg_funcs_item(i_self_matrix_at_layers, i_neigh_matrix_at_layers)

        i_gnew_neighs_matrix = self.gather(self.user_embeddings, i_gnew_neighs, 0)
        i_output_from_gnew_mean = self.gnew_agg_mean(i_self_matrix_at_layers, i_gnew_neighs_matrix)

        i_output_from_gnew_att = self.gnew_agg_item(i_self_matrix_at_layers,
                                                    self.concat_1((i_neigh_matrix_at_layers, i_gnew_neighs_matrix)))

        i_output = self.concat_1((i_output_mean, i_output_from_gnew_mean, i_output_from_gnew_att))
        all_pos_item_rep = self.tanh(i_output)

        neg_item_embed = self.gather(self.item_embeddings, neg_item_id, 0)

        neg_self_matrix_at_layers = self.gather(self.item_embeddings, neg_group_nodes, 0)
        neg_neigh_matrix_at_layers = self.gather(self.user_embeddings, neg_neighs, 0)

        neg_output_mean = self.raw_agg_funcs_item(neg_self_matrix_at_layers, neg_neigh_matrix_at_layers)

        neg_gnew_neighs_matrix = self.gather(self.user_embeddings, neg_gnew_neighs, 0)
        neg_output_from_gnew_mean = self.gnew_agg_mean(neg_self_matrix_at_layers, neg_gnew_neighs_matrix)

        neg_output_from_gnew_att = self.gnew_agg_item(neg_self_matrix_at_layers,
                                                      self.concat_1(
                                                          (neg_neigh_matrix_at_layers, neg_gnew_neighs_matrix)))

        neg_output = self.concat_1((neg_output_mean, neg_output_from_gnew_mean, neg_output_from_gnew_att))
        neg_output = self.tanh(neg_output)

        neg_output_shape = self.shape(neg_output)
        neg_item_rep = self.reshape(neg_output,
                                    (self.shape(neg_item_embed)[0], neg_item_num, neg_output_shape[-1]))

        return all_user_embed, all_user_rep, all_pos_item_embed, all_pos_item_rep, neg_item_embed, neg_item_rep
