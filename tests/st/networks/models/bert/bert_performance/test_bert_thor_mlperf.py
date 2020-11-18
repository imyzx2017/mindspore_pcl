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

"""test bert thor performance with 8p on mlperf dataset"""

import os
import time
from multiprocessing import Process, Queue
import pytest
import numpy as np
import mindspore.dataset as dataset
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
from mindspore import log as logger
from mindspore.train.callback import Callback
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C
from model_zoo.official.nlp.bert_thor.src.bert_for_pre_training import BertNetworkWithLoss, BertTrainOneStepCell
from model_zoo.official.nlp.bert_thor.src.bert_net_config import bert_net_cfg
from model_zoo.official.nlp.bert_thor.src.config import cfg
from model_zoo.official.nlp.bert_thor.src.lr_generator import get_bert_lr, get_bert_damping
from model_zoo.official.nlp.bert_thor.src.model_thor import Model
from model_zoo.official.nlp.bert_thor.src.thor_for_bert_arg import THOR

MINDSPORE_HCCL_CONFIG_PATH = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"
DATASET_PATH = "/home/workspace/mindspore_dataset/bert/thor/en-wiki-512_test_first1wan"
load_checkpoint_path = ""
data_sink_steps = 100
train_steps = 200
batch_size = 12

np.random.seed(1)
dataset.config.set_seed(1)
os.environ['GLOG_v'] = str(2)

class TimeMonitor(Callback):
    """Time Monitor."""
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_mseconds_list = []
        self.per_step_mseconds_list = []

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds_list.append(epoch_mseconds)
        per_step_mseconds = epoch_mseconds / self.data_size
        self.per_step_mseconds_list.append(per_step_mseconds)
        print("epoch: {}, per_step_mseconds are {}".format(cb_params.cur_epoch_num, str(per_step_mseconds)), flush=True)

class LossCallback(Callback):
    def __init__(self):
        super(LossCallback, self).__init__()
        self.loss_list = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs.asnumpy())
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)), flush=True)

def create_bert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None):
    """create train dataset"""
    # apply repeat operations
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if "tfrecord" in file_name:
            data_files.append(os.path.join(data_dir, file_name))
    data_files = sorted(data_files)
    ds = de.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                            columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                          "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                            shuffle=de.Shuffle.FILES if do_shuffle == "true" else False,
                            num_shards=device_num, shard_id=rank, shard_equal_rows=True)
    ori_dataset_size = ds.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="masked_lm_ids")
    ds = ds.map(operations=type_cast_op, input_columns="masked_lm_positions")
    ds = ds.map(operations=type_cast_op, input_columns="next_sentence_labels")
    ds = ds.map(operations=type_cast_op, input_columns="segment_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    logger.info("data size: {}".format(ds.get_dataset_size()))
    logger.info("repeat count: {}".format(ds.get_repeat_count()))
    return ds


def _set_bert_all_reduce_split():
    """set bert all_reduce fusion split, support num_hidden_layers is 12 and 24."""
    from mindspore.parallel._auto_parallel_context import auto_parallel_context
    if bert_net_cfg.num_hidden_layers == 12:
        if bert_net_cfg.use_relative_positions:
            auto_parallel_context().set_all_reduce_fusion_split_indices([29, 58, 87, 116, 145, 174, 203, 217],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([29, 58, 87, 116, 145, 174, 203, 217],
                                                                        "hccl_world_groupsum3")
        else:
            auto_parallel_context().set_all_reduce_fusion_split_indices([28, 55, 82, 109, 136, 163, 190, 205],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([28, 55, 82, 109, 136, 163, 190, 205],
                                                                        "hccl_world_groupsum3")
    elif bert_net_cfg.num_hidden_layers == 24:
        if bert_net_cfg.use_relative_positions:
            auto_parallel_context().set_all_reduce_fusion_split_indices([30, 90, 150, 210, 270, 330, 390, 421],
                                                                        "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([30, 90, 150, 210, 270, 330, 390, 421],
                                                                        "hccl_world_groupsum3")
        else:
            auto_parallel_context().set_all_reduce_fusion_split_indices([38, 77], "hccl_world_groupsum1")
            auto_parallel_context().set_all_reduce_fusion_split_indices([38, 77], "hccl_world_groupsum3")


def train_process_bert_thor(q, device_id, epoch_size, device_num):
    os.system("mkdir " + str(device_id))
    os.chdir(str(device_id))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id, save_graphs=False)
    context.set_context(reserve_class_name_in_scope=False)
    context.set_context(max_call_depth=3000)
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = MINDSPORE_HCCL_CONFIG_PATH
    os.environ['RANK_ID'] = str(device_id)
    os.environ['RANK_SIZE'] = str(device_num)

    D.init()
    rank = device_id % device_num
    context.reset_auto_parallel_context()
    _set_bert_all_reduce_split()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=device_num)

    bert_net_cfg.num_hidden_layers = 4
    ds = create_bert_dataset(device_num=device_num, rank=rank, do_shuffle=False, data_dir=DATASET_PATH, schema_dir=None)
    net_with_loss = BertNetworkWithLoss(bert_net_cfg, True)

    new_repeat_count = epoch_size * ds.get_dataset_size() // data_sink_steps
    new_repeat_count = min(new_repeat_count, train_steps // data_sink_steps)


    lr = get_bert_lr()
    damping = get_bert_damping()
    optimizer = THOR(filter(lambda x: x.requires_grad, net_with_loss.get_parameters()), lr, cfg.Thor.momentum,
                     filter(lambda x: 'matrix_A' in x.name, net_with_loss.get_parameters()),
                     filter(lambda x: 'matrix_G' in x.name, net_with_loss.get_parameters()),
                     cfg.Thor.weight_decay, cfg.Thor.loss_scale, bert_net_cfg.num_hidden_layers,
                     bert_net_cfg.batch_size, damping)
    time_monitor_callback = TimeMonitor(data_sink_steps)
    loss_callback = LossCallback()
    callback = [time_monitor_callback, loss_callback]

    if load_checkpoint_path:
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(net_with_loss, param_dict)

    net_with_grads = BertTrainOneStepCell(net_with_loss, optimizer=optimizer)
    model = Model(net_with_grads, frequency=cfg.Thor.frequency)
    model.train(new_repeat_count, ds, callbacks=callback, dataset_sink_mode=True, sink_size=data_sink_steps)

    loss_list = loss_callback.loss_list
    per_step_mseconds = time_monitor_callback.per_step_mseconds_list
    q.put({'loss': loss_list, 'cost': per_step_mseconds})


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_bert_thor_mlperf_8p():
    """test bert thor mlperf 8p"""
    q = Queue()
    device_num = 8
    epoch_size = 2
    process = []
    for i in range(device_num):
        device_id = i
        process.append(Process(target=train_process_bert_thor, args=(q, device_id, epoch_size, device_num)))

    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    sum_loss_list = []
    sum_cost_list = []
    for _ in range(train_steps // data_sink_steps):
        sum_loss_list.append(0.0)
        sum_cost_list.append(0.0)

    for _ in range(device_num):
        output = q.get()
        loss_list = output['loss']
        cost_list = output['cost']
        sum_loss_list = np.sum([loss_list, sum_loss_list], axis=0)
        sum_cost_list = np.sum([cost_list, sum_cost_list], axis=0)

    for j in range(train_steps // data_sink_steps):
        print("epoch: ", j, "sum_loss: ", sum_loss_list[j], "sum_cost: ", sum_cost_list[j])

    mean_loss = sum_loss_list[-1] / device_num
    mean_cost = sum_cost_list[-1] / device_num

    for i in range(device_num):
        os.system("rm -rf " + str(i))

    print("End training...")
    assert mean_cost < 64.2
    assert mean_loss < 7.9

if __name__ == '__main__':
    test_bert_thor_mlperf_8p()
