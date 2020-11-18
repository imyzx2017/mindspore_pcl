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
"""Train api."""
import os
import argparse
import pickle

import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn import Momentum
from mindspore.nn.optim import Adam, Lamb
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore import context, Parameter
from mindspore.context import ParallelMode
from mindspore.communication import management as MultiAscend
from mindspore.train.serialization import load_checkpoint
from mindspore.common import set_seed

from config import TransformerConfig
from src.dataset import load_dataset
from src.transformer import TransformerNetworkWithLoss, TransformerTrainOneStepWithLossScaleCell
from src.transformer.infer_mass import infer
from src.utils import LossCallBack
from src.utils import one_weight, zero_weight, weight_variable
from src.utils import square_root_schedule
from src.utils.lr_scheduler import polynomial_decay_scheduler, BertLearningRate

parser = argparse.ArgumentParser(description='MASS train entry point.')
parser.add_argument("--config", type=str, required=True, help="model config json file path.")
parser.add_argument("--platform", type=str, required=True, help="model working platform.")

def get_config(config):
    config = TransformerConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config


def _train(model, config: TransformerConfig,
           pre_training_dataset=None, fine_tune_dataset=None, test_dataset=None,
           callbacks: list = None):
    """
    Train model.

    Args:
        model (Model): MindSpore model instance.
        config (TransformerConfig): Config of mass model.
        pre_training_dataset (Dataset): Pre-training dataset.
        fine_tune_dataset (Dataset): Fine-tune dataset.
        test_dataset (Dataset): Test dataset.
        callbacks (list): A list of callbacks.
    """
    callbacks = callbacks if callbacks else []

    if pre_training_dataset is not None:
        print(" | Start pre-training job.")

        if os.getenv("RANK_SIZE") is not None and int(os.getenv("RANK_SIZE")) > 1:
            print(f" | Rank {MultiAscend.get_rank()} Call model train.")

        model.train(config.epochs, pre_training_dataset,
                    callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode,
                    sink_size=config.dataset_sink_step)

        # Test the accuracy of the model.
        if test_dataset is not None:
            print(" | Start test job.")
            result = infer(_config)
            with open("validation_res_after_pre_training.bin", "wb") as f:
                pickle.dump(result, f, 1)

    if fine_tune_dataset is not None:
        print(" | Start fine-tuning job.")

        model.train(config.epochs, fine_tune_dataset,
                    callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode,
                    sink_size=config.dataset_sink_step)

        # Test the accuracy of the model.
        if test_dataset is not None:
            print(" | Start test job.")
            result = infer(_config)
            with open("validation_res_after_pre_training.bin", "wb") as f:
                pickle.dump(result, f, 1)


def _load_checkpoint_to_net(config, network):
    """load parameters to network from checkpoint."""
    if config.existed_ckpt:
        if config.existed_ckpt.endswith(".npz"):
            weights = np.load(config.existed_ckpt)
        else:
            weights = load_checkpoint(config.existed_ckpt)
        for param in network.trainable_params():
            weights_name = param.name
            if weights_name not in weights:
                raise ValueError(f"Param {weights_name} is not found in ckpt file.")

            if isinstance(weights[weights_name], Parameter):
                param.set_data(weights[weights_name].data)
            elif isinstance(weights[weights_name], Tensor):
                param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
            elif isinstance(weights[weights_name], np.ndarray):
                param.set_data(Tensor(weights[weights_name], config.dtype))
            else:
                param.set_data(weights[weights_name])
    else:
        for param in network.trainable_params():
            name = param.name
            value = param.data
            if isinstance(value, Tensor):
                if name.endswith(".gamma"):
                    param.set_data(one_weight(value.asnumpy().shape))
                elif name.endswith(".beta") or name.endswith(".bias"):
                    param.set_data(zero_weight(value.asnumpy().shape))
                else:
                    param.set_data(weight_variable(value.asnumpy().shape))


def _get_lr(config, update_steps):
    """generate learning rate."""
    if config.lr_scheduler == "isr":
        lr = Tensor(square_root_schedule(lr=config.lr,
                                         update_num=update_steps,
                                         decay_start_step=config.decay_start_step,
                                         warmup_steps=config.warmup_steps,
                                         min_lr=config.min_lr), dtype=mstype.float32)
    elif config.lr_scheduler == "poly":
        lr = Tensor(polynomial_decay_scheduler(lr=config.lr,
                                               min_lr=config.min_lr,
                                               decay_steps=config.decay_steps,
                                               total_update_num=update_steps,
                                               warmup_steps=config.warmup_steps,
                                               power=config.poly_lr_scheduler_power), dtype=mstype.float32)
    else:
        lr = config.lr
    return lr


def _get_optimizer(config, network, lr):
    """get mass optimizer, support Adam, Lamb, Momentum."""
    if config.optimizer.lower() == "adam":
        optimizer = Adam(network.trainable_params(), lr, beta1=0.9, beta2=0.98)
    elif config.optimizer.lower() == "lamb":
        lr = BertLearningRate(decay_steps=12000, learning_rate=config.lr, end_learning_rate=config.min_lr,
                              power=10.0, warmup_steps=config.warmup_steps)
        decay_params = list(filter(lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
                                   network.trainable_params()))
        other_params = list(filter(lambda x: 'layernorm' in x.name.lower() or 'bias' in x.name.lower(),
                                   network.trainable_params()))
        group_params = [{'params': decay_params, 'weight_decay': 0.01},
                        {'params': other_params}]

        optimizer = Lamb(group_params, lr, eps=1e-6)
    elif config.optimizer.lower() == "momentum":
        optimizer = Momentum(network.trainable_params(), lr, momentum=0.9)
    else:
        raise ValueError(f"optimizer only support `adam` and `momentum` now.")
    return optimizer


def _build_training_pipeline(config: TransformerConfig,
                             pre_training_dataset=None,
                             fine_tune_dataset=None,
                             test_dataset=None,
                             platform="Ascend"):
    """
    Build training pipeline.

    Args:
        config (TransformerConfig): Config of mass model.
        pre_training_dataset (Dataset): Pre-training dataset.
        fine_tune_dataset (Dataset): Fine-tune dataset.
        test_dataset (Dataset): Test dataset.
    """
    net_with_loss = TransformerNetworkWithLoss(config, is_training=True)
    net_with_loss.init_parameters_data()
    _load_checkpoint_to_net(config, net_with_loss)

    dataset = pre_training_dataset if pre_training_dataset is not None \
        else fine_tune_dataset

    if dataset is None:
        raise ValueError("pre-training dataset or fine-tuning dataset must be provided one.")

    update_steps = config.epochs * dataset.get_dataset_size()

    lr = _get_lr(config, update_steps)

    optimizer = _get_optimizer(config, net_with_loss, lr)

    # loss scale.
    if config.loss_scale_mode == "dynamic":
        scale_manager = DynamicLossScaleManager(init_loss_scale=config.init_loss_scale,
                                                scale_factor=config.loss_scale_factor,
                                                scale_window=config.scale_window)
    else:
        scale_manager = FixedLossScaleManager(loss_scale=config.init_loss_scale, drop_overflow_update=True)
    net_with_grads = TransformerTrainOneStepWithLossScaleCell(network=net_with_loss, optimizer=optimizer,
                                                              scale_update_cell=scale_manager.get_update_cell())
    net_with_grads.set_train(True)
    model = Model(net_with_grads)
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_steps,
                                   keep_checkpoint_max=config.keep_ckpt_max)

    rank_size = os.getenv('RANK_SIZE')
    callbacks = []
    callbacks.append(time_cb)
    if rank_size is not None and int(rank_size) > 1:
        loss_monitor = LossCallBack(config, rank_id=MultiAscend.get_rank())
        callbacks.append(loss_monitor)
        if MultiAscend.get_rank() % 8 == 0:
            ckpt_callback = ModelCheckpoint(
                prefix=config.ckpt_prefix,
                directory=os.path.join(config.ckpt_path, 'ckpt_{}'.format(MultiAscend.get_rank())),
                config=ckpt_config)
            callbacks.append(ckpt_callback)

    if rank_size is None or int(rank_size) == 1:
        ckpt_callback = ModelCheckpoint(
            prefix=config.ckpt_prefix,
            directory=os.path.join(config.ckpt_path, 'ckpt_{}'.format(os.getenv('DEVICE_ID'))),
            config=ckpt_config)
        loss_monitor = LossCallBack(config, rank_id=os.getenv('DEVICE_ID'))
        callbacks.append(loss_monitor)
        callbacks.append(ckpt_callback)

    print(f" | ALL SET, PREPARE TO TRAIN.")
    _train(model=model, config=config,
           pre_training_dataset=pre_training_dataset,
           fine_tune_dataset=fine_tune_dataset,
           test_dataset=test_dataset,
           callbacks=callbacks)


def _setup_parallel_env(platform):
    context.reset_auto_parallel_context()
    MultiAscend.init()
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,
        device_num=MultiAscend.get_group_size(),
        gradients_mean=True
    )


def train_parallel(config: TransformerConfig, platform: "Ascend"):
    """
    Train model with multi ascend chips.

    Args:
        config (TransformerConfig): Config for MASS model.
    """
    _setup_parallel_env(platform)

    print(f" | Starting training on {os.getenv('RANK_SIZE', None)} devices.")

    pre_train_dataset = load_dataset(
        data_files=config.pre_train_dataset,
        batch_size=config.batch_size, epoch_count=1,
        sink_mode=config.dataset_sink_mode,
        sink_step=config.dataset_sink_step,
        rank_size=MultiAscend.get_group_size(),
        rank_id=MultiAscend.get_rank()
    ) if config.pre_train_dataset else None
    fine_tune_dataset = load_dataset(
        data_files=config.fine_tune_dataset,
        batch_size=config.batch_size, epoch_count=1,
        sink_mode=config.dataset_sink_mode,
        sink_step=config.dataset_sink_step,
        rank_size=MultiAscend.get_group_size(),
        rank_id=MultiAscend.get_rank()
    ) if config.fine_tune_dataset else None
    test_dataset = load_dataset(
        data_files=config.test_dataset,
        batch_size=config.batch_size, epoch_count=1,
        sink_mode=config.dataset_sink_mode,
        sink_step=config.dataset_sink_step,
        rank_size=MultiAscend.get_group_size(),
        rank_id=MultiAscend.get_rank()
    ) if config.test_dataset else None

    _build_training_pipeline(config=config,
                             pre_training_dataset=pre_train_dataset,
                             fine_tune_dataset=fine_tune_dataset,
                             test_dataset=test_dataset,
                             platform=platform)


def train_single(config: TransformerConfig, platform: "Ascend"):
    """
    Train model on single device.

    Args:
        config (TransformerConfig): Config for model.
    """
    print(" | Starting training on single device.")
    pre_train_dataset = load_dataset(data_files=config.pre_train_dataset,
                                     batch_size=config.batch_size,
                                     epoch_count=1,
                                     sink_mode=config.dataset_sink_mode,
                                     sink_step=config.dataset_sink_step) if config.pre_train_dataset else None
    fine_tune_dataset = load_dataset(data_files=config.fine_tune_dataset,
                                     batch_size=config.batch_size,
                                     epoch_count=1,
                                     sink_mode=config.dataset_sink_mode,
                                     sink_step=config.dataset_sink_step) if config.fine_tune_dataset else None
    test_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                epoch_count=1,
                                sink_mode=config.dataset_sink_mode,
                                sink_step=config.dataset_sink_step) if config.test_dataset else None

    _build_training_pipeline(config=config,
                             pre_training_dataset=pre_train_dataset,
                             fine_tune_dataset=fine_tune_dataset,
                             test_dataset=test_dataset,
                             platform=platform)


def _check_args(config):
    if not os.path.exists(config):
        raise FileNotFoundError("`config` is not existed.")
    if not isinstance(config, str):
        raise ValueError("`config` must be type of str.")


if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    device_id = os.getenv('DEVICE_ID', None)
    if device_id is None:
        device_id = 0
    device_id = int(device_id)
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.platform,
        reserve_class_name_in_scope=False,
        device_id=device_id,
        max_call_depth=2000)

    _rank_size = os.getenv('RANK_SIZE')

    _check_args(args.config)
    _config = get_config(args.config)

    set_seed(_config.random_seed)
    context.set_context(save_graphs=_config.save_graphs)

    if _rank_size is not None and int(_rank_size) > 1:
        train_parallel(_config, args.platform)
    else:
        train_single(_config, args.platform)
