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

"""train bert network without lossscale"""

import os

import numpy as np
from src.bert_for_pre_training import BertNetworkWithLoss, BertTrainOneStepWithLossScaleCell
from src.bert_model import BertConfig

import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.nn import learning_rate_schedule as lr_schedules
from mindspore.nn.optim import Lamb
from mindspore.train.callback import Callback
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.model import Model

DATA_DIR = ["/home/workspace/mindspore_dataset/bert/example/examples.tfrecord"]
SCHEMA_DIR = "/home/workspace/mindspore_dataset/bert/example/datasetSchema.json"

def get_config(version='base', batch_size=1):
    """get config"""
    if version == 'base':
        bert_config = BertConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=21136,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            input_mask_from_dataset=True,
            token_type_ids_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float32)
    elif version == 'large':
        bert_config = BertConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=30522,
            hidden_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            input_mask_from_dataset=True,
            token_type_ids_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float16,
            enable_fused_layernorm=True)
    else:
        bert_config = BertConfig(batch_size=batch_size)
    return bert_config


def me_de_train_dataset():
    """test me de train dataset"""
    # apply repeat operations
    repeat_count = 1
    ds = de.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["input_ids", "input_mask", "segment_ids",
                                                                "next_sentence_labels", "masked_lm_positions",
                                                                "masked_lm_ids", "masked_lm_weights"], shuffle=False)
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="masked_lm_ids")
    ds = ds.map(operations=type_cast_op, input_columns="masked_lm_positions")
    ds = ds.map(operations=type_cast_op, input_columns="next_sentence_labels")
    ds = ds.map(operations=type_cast_op, input_columns="segment_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    batch_size = int(os.getenv('BATCH_SIZE', '16'))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)
    return ds


def weight_variable(shape):
    """weight variable"""
    np.random.seed(1)
    ones = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
    return Tensor(ones)


class BertLearningRate(lr_schedules.LearningRateSchedule):
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_lr = lr_schedules.WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
        warmup_lr = self.warmup_lr(global_step)
        decay_lr = self.decay_lr(global_step)
        lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        return lr


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []
        self.overflow_list = []
        self.lossscale_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs[0].asnumpy()[0])
        self.overflow_list.append(cb_params.net_outputs[1].asnumpy())
        self.lossscale_list.append(cb_params.net_outputs[2].asnumpy())
        print("epoch: {}, outputs are: {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))


def test_bert_tdt():
    """test bert tdt"""
    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", reserve_class_name_in_scope=False)
    context.set_context(enable_graph_kernel=True)
    ds = me_de_train_dataset()
    config = get_config(version='large', batch_size=16)
    netwithloss = BertNetworkWithLoss(config, True)
    lr = BertLearningRate(decay_steps=ds.get_dataset_size()*ds.get_repeat_count(), learning_rate=5e-5,
                          end_learning_rate=1e-9, power=10.0, warmup_steps=0)
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower()
    no_decay_filter = lambda x: 'layernorm' in x.name.lower() or 'bias' in x.name.lower()
    decay_params = list(filter(decay_filter, net_with_loss.trainable_params()))
    other_params = list(filter(no_decay_filter, net_with_loss.trainable_params()))
    group_params = [{'params': decay_params, 'weight_decay': 0.01},
                    {'params': other_params}]
    optimizer = Lamb(group_params, lr)
    scale_window = 3
    scale_manager = DynamicLossScaleManager(262144, 2, scale_window)
    netwithgrads = BertTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,
                                                     scale_update_cell=scale_manager.get_update_cell())
    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    callback = ModelCallback()
    netwithloss.init_parameters_data()
    params = netwithloss.trainable_params()
    for param in params:
        value = param.data
        name = param.name
        if isinstance(value, Tensor):
            if name.split('.')[-1] in ['weight']:
                if name.split('.')[-3] in ['cls2']:
                    logger.info("***************** BERT param name is 1 {}".format(name))
                    param.set_data(weight_variable(value.asnumpy().shape))
                else:
                    logger.info("***************** BERT param name is 2 {}".format(name))
                    tempshape = value.asnumpy().shape
                    shape = (tempshape[1], tempshape[0])
                    weight_value = weight_variable(shape).asnumpy()
                    param.set_data(Tensor(np.transpose(weight_value, [1, 0])))
            else:
                logger.info("***************** BERT param name is 3 {}".format(name))
                param.set_data(weight_variable(value.asnumpy().shape))
    model.train(1, ds, callbacks=callback, dataset_sink_mode=False)

    # assertion occurs while the loss value, overflow state or loss_scale value is wrong
    loss_value = np.array(callback.loss_list)
    expect_loss_value = [12.559319, 12.333815, 12.339806, 12.350235, 12.343947, 12.830965, 12.375336, 12.973715,
                         12.57929, 12.7766905]
    error = loss_value - expect_loss_value
    print("loss value: {}".format(loss_value))
    print("error value: {}".format(error))
    assert np.allclose(loss_value, expect_loss_value, 0, 0.0005)

    overflow = np.array(callback.overflow_list)
    expect_overflow = [True, True, True, True, False, False, False, True, False, False]
    print("overflow: {}".format(overflow))
    assert (overflow == expect_overflow).all()

    loss_scale = np.array(callback.lossscale_list)
    expect_loss_scale = [131072.0, 65536.0, 32768.0, 16384.0, 16384.0, 16384.0, 32768.0, 16384.0, 16384.0, 16384.0]
    print("loss scale: {}".format(loss_scale))
    assert np.allclose(loss_scale, expect_loss_scale, 0, 0)


if __name__ == '__main__':
    test_bert_tdt()
