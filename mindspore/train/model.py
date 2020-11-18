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
"""Model."""
from collections.abc import Iterable

import os
import math
import numpy as np

from mindspore import log as logger
from ..common.tensor import Tensor
from ..nn.metrics import get_metrics
from .._checkparam import check_input_data, check_output_data, Validator
from .callback import _InternalCallbackParam, RunContext, _CallbackManager
from .. import context
from ..parallel._utils import _get_parallel_mode, _get_device_num, _get_global_rank, \
    _get_parameter_broadcast, _device_number_check, _parameter_broadcast_check
from ..parallel._ps_context import _is_role_pserver, _is_role_sched
from ..nn.metrics import Loss
from .. import nn
from ..nn.wrap.cell_wrapper import _VirtualDatasetCell
from ..context import ParallelMode
from ..parallel._cost_model_context import _set_multi_subgraphs
from .dataset_helper import DatasetHelper, connect_network_with_dataset
from . import amp


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class Model:
    """
    High-Level API for Training or Testing.

    `Model` groups layers into an object with training and inference features.

    Args:
        network (Cell): A training or testing network.
        loss_fn (Cell): Objective function, if loss_fn is None, the
                             network should contain the logic of loss and grads calculation, and the logic
                             of parallel if needed. Default: None.
        optimizer (Cell): Optimizer for updating the weights. Default: None.
        metrics (Union[dict, set]): A Dictionary or a set of metrics to be evaluated by the model during
                        training and testing. eg: {'accuracy', 'recall'}. Default: None.
        eval_network (Cell): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
                             `eval_network`. Default: None.
        eval_indexes (list): When defining the `eval_network`, if `eval_indexes` is None, all outputs of the
                             `eval_network` would be passed to metrics, otherwise `eval_indexes` must contain three
                             elements, including the positions of loss value, predicted value and label. The loss
                             value would be passed to the `Loss` metric, the predicted value and label would be passed
                             to other metric. Default: None.
        amp_level (str): Option for argument `level` in `mindspore.amp.build_train_network`, level for mixed
            precision training. Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

            - O0: Do not change.
            - O2: Cast network to float16, keep batchnorm run in float32, using dynamic loss scale.
            - O3: Cast network to float16, with additional property 'keep_batchnorm_fp32=False'.
            - auto: Set to level to recommended level in different devices. Set level to O2 on GPU, Set
              level to O3 Ascend. The recommended level is choose by the export experience, cannot
              always generalize. User should specify the level for special network.

            O2 is recommended on GPU, O3 is recommended on Ascend.

        loss_scale_manager (Union[None, LossScaleManager]): If it is None, the loss would not be scaled. Otherwise,
            scale the loss by LossScaleManager and optimizer can not be None.It is a key argument.
            e.g. Use `loss_scale_manager=None` to set the value.
        keep_batchnorm_fp32 (bool): Keep Batchnorm running in `float32`. If it is set to true, the level setting before
            will be overwritten. Default: True.

    Examples:
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        >>>         self.bn = nn.BatchNorm2d(64)
        >>>         self.relu = nn.ReLU()
        >>>         self.flatten = nn.Flatten()
        >>>         self.fc = nn.Dense(64*224*224, 12) # padding=0
        >>>
        >>>     def construct(self, x):
        >>>         x = self.conv(x)
        >>>         x = self.bn(x)
        >>>         x = self.relu(x)
        >>>         x = self.flatten(x)
        >>>         out = self.fc(x)
        >>>         return out
        >>>
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
        >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
        >>> dataset = get_dataset()
        >>> model.train(2, dataset)
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        self._network = network
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._loss_scale_manager = None
        self._loss_scale_manager_set = False
        self._keep_bn_fp32 = True
        self._check_kwargs(kwargs)
        self._amp_level = amp_level
        self._process_amp_args(kwargs)
        self._parallel_mode = _get_parallel_mode()
        self._device_number = _get_device_num()
        self._global_rank = _get_global_rank()
        self._parameter_broadcast = _get_parameter_broadcast()

        self._train_network = self._build_train_network()
        self._build_eval_network(metrics, eval_network, eval_indexes)
        self._build_predict_network()

    def _process_amp_args(self, kwargs):
        if self._amp_level in ["O0", "O3"]:
            self._keep_bn_fp32 = False
        if 'keep_batchnorm_fp32' in kwargs:
            self._keep_bn_fp32 = kwargs['keep_batchnorm_fp32']
        if 'loss_scale_manager' in kwargs:
            self._loss_scale_manager = kwargs['loss_scale_manager']
            self._loss_scale_manager_set = True

    def _check_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in ['loss_scale_manager', 'keep_batchnorm_fp32']:
                raise ValueError(f"Unsupported arg '{arg}'")

    def _build_train_network(self):
        """Build train network"""
        network = self._network
        if self._loss_scale_manager is not None and self._optimizer is None:
            raise ValueError("Optimizer can not be None when set loss_scale_manager.")
        if self._optimizer:
            if self._loss_scale_manager_set:
                network = amp.build_train_network(network,
                                                  self._optimizer,
                                                  self._loss_fn,
                                                  level=self._amp_level,
                                                  loss_scale_manager=self._loss_scale_manager,
                                                  keep_batchnorm_fp32=self._keep_bn_fp32)
            else:
                network = amp.build_train_network(network,
                                                  self._optimizer,
                                                  self._loss_fn,
                                                  level=self._amp_level,
                                                  keep_batchnorm_fp32=self._keep_bn_fp32)
        elif self._loss_fn:
            network = nn.WithLossCell(network, self._loss_fn)
        # If need to check if loss_fn is not None, but optimizer is None

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()
            if self._optimizer is None:
                # In this case, multiple optimizer(s) is supposed to be included in 'self._network'
                _set_multi_subgraphs()
        return network

    def _build_eval_network(self, metrics, eval_network, eval_indexes):
        """Build the network for evaluation."""
        self._metric_fns = get_metrics(metrics)
        if not self._metric_fns:
            return

        if eval_network is not None:
            if eval_indexes is not None and not (isinstance(eval_indexes, list) and len(eval_indexes) == 3):
                raise ValueError("Eval_indexes must be a list or None. If eval_indexes is a list, length of it \
                                 must be three. But got {}".format(eval_indexes))

            self._eval_network = eval_network
            self._eval_indexes = eval_indexes
        else:
            if self._loss_fn is None:
                raise ValueError("loss_fn can not be None.")
            self._eval_network = nn.WithEvalCell(self._network, self._loss_fn, self._amp_level in ["O2", "O3", "auto"])
            self._eval_indexes = [0, 1, 2]

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            if self._optimizer:
                self._eval_network = _VirtualDatasetCell(self._eval_network)
            if self._optimizer is None:
                # In this case, multiple optimizer(s) is supposed to be included in 'self._network'
                _set_multi_subgraphs()
            self._eval_network.set_auto_parallel()

    def _build_predict_network(self):
        """Build the network for prediction."""
        self._predict_network = self._network
        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self._predict_network = _VirtualDatasetCell(self._network)
            # Unlike the cases in build_train_network() and build_eval_network(), 'multi_subgraphs' is not set
            self._predict_network.set_auto_parallel()

    def _clear_metrics(self):
        """Clear metrics local values."""
        for metric in self._metric_fns.values():
            metric.clear()

    def _update_metrics(self, outputs):
        """Update metrics local values."""
        if not isinstance(outputs, tuple):
            raise ValueError("The `outputs` is not tuple.")

        if self._eval_indexes is not None and len(outputs) < 3:
            raise ValueError("The length of `outputs` must be greater than or equal to 3, \
                             but got {}".format(len(outputs)))

        for metric in self._metric_fns.values():
            if self._eval_indexes is None:
                metric.update(*outputs)
            else:
                if isinstance(metric, Loss):
                    metric.update(outputs[self._eval_indexes[0]])
                else:
                    metric.update(outputs[self._eval_indexes[1]], outputs[self._eval_indexes[2]])

    def _get_metrics(self):
        """Get metrics local values."""
        metrics = dict()
        for key, value in self._metric_fns.items():
            metrics[key] = value.eval()
        return metrics

    def _get_scaling_sens(self):
        """get the scaling sens"""
        scaling_sens = 1
        if self._loss_scale_manager is not None:
            scaling_sens = self._loss_scale_manager.get_loss_scale()
        if self._parallel_mode == ParallelMode.DATA_PARALLEL:
            scaling_sens /= self._device_number
        return scaling_sens

    def _exec_preprocess(self, network, is_train, phase, dataset, dataset_sink_mode, sink_size=-1, epoch_num=1):
        """Initializes dataset."""
        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1
        dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)

        if dataset_sink_mode:
            network = connect_network_with_dataset(network, dataset_helper)

        network.set_train(is_train)
        network.phase = phase

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        return dataset_helper, network

    def _init(self, train_dataset=None, valid_dataset=None, sink_size=-1):
        """
        Initialize compute graphs and data graphs with the sink mode.

        Note:
            Pre-init process only supports `GRAPH_MODE` and `Ascend` target currently.

        Args:
            train_dataset (Dataset): A training dataset iterator. If `train_dataset` is defined, training graphs will be
                                     initialized. Default: None.
            valid_dataset (Dataset): A evaluating dataset iterator. If `valid_dataset` is defined, evaluation graphs
                                     will be initialized, and `metrics` in `Model` can not be None. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        if context.get_context("mode") != context.GRAPH_MODE or context.get_context("device_target") != "Ascend":
            raise RuntimeError('Pre-init process only supports GRAPH MODE and Ascend target currently.')

        if not train_dataset and not valid_dataset:
            raise ValueError('Both train_dataset and valid_dataset can not be None or empty.')

        _device_number_check(self._parallel_mode, self._device_number)

        if train_dataset:
            _parameter_broadcast_check(self._parallel_mode, self._parameter_broadcast)
            if self._parameter_broadcast:
                self._train_network.set_broadcast_flag()

            train_dataset.__no_send__ = True
            train_dataset_helper, train_network = self._exec_preprocess(self._train_network,
                                                                        is_train=True,
                                                                        phase='train',
                                                                        dataset=train_dataset,
                                                                        dataset_sink_mode=True,
                                                                        sink_size=sink_size)
            self._train_network = train_network
            for inputs in train_dataset_helper:
                self._train_network.compile(*inputs)
                break

        if valid_dataset:
            if not self._metric_fns:
                raise RuntimeError('If define `valid_dataset`, metric fn can not be None or empty.')

            valid_dataset.__no_send__ = True
            valid_dataset_helper, eval_network = self._exec_preprocess(self._eval_network,
                                                                       is_train=False,
                                                                       phase='eval',
                                                                       dataset=valid_dataset,
                                                                       dataset_sink_mode=True)
            self._eval_network = eval_network
            for inputs in valid_dataset_helper:
                self._eval_network.compile(*inputs)
                break

    def _train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1):
        """
        Training.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) will be
                                     returned and passed to the network. Otherwise, a tuple (data, label) will
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            callbacks (list): List of callback objects which should be executed while training. Default: None.
            dataset_sink_mode (bool): Determine whether the data should be passed through the dataset channel.
                                      Default: True.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        epoch = Validator.check_positive_int(epoch)
        if self._parameter_broadcast:
            self._train_network.set_broadcast_flag()

        cb_params = _InternalCallbackParam()
        cb_params.train_network = self._train_network
        cb_params.epoch_num = epoch
        if dataset_sink_mode and sink_size > 0:
            cb_params.batch_num = sink_size
        else:
            cb_params.batch_num = train_dataset.get_dataset_size()
        cb_params.mode = "train"
        cb_params.loss_fn = self._loss_fn
        cb_params.optimizer = self._optimizer
        cb_params.parallel_mode = self._parallel_mode
        cb_params.device_number = self._device_number
        cb_params.train_dataset = train_dataset
        cb_params.list_callback = self._transform_callbacks(callbacks)
        cb_params.train_dataset_element = None
        cb_params.network = self._network
        if _is_role_pserver() or _is_role_sched():
            epoch = 1

        # build callback list
        with _CallbackManager(callbacks) as list_callback:
            if not dataset_sink_mode:
                self._train_process(epoch, train_dataset, list_callback, cb_params)
            elif context.get_context("device_target") == "CPU" or context.get_context("mode") == context.PYNATIVE_MODE:
                logger.warning("The CPU or PyNative mode cannot support dataset sink mode currently."
                               "So the training process will be performed with dataset not sink.")
                self._train_process(epoch, train_dataset, list_callback, cb_params)
            else:
                self._train_dataset_sink_process(epoch, train_dataset, list_callback, cb_params, sink_size)

    @staticmethod
    def _transform_callbacks(callbacks):
        """Transform callback to a list."""
        if callbacks is None:
            return []

        if isinstance(callbacks, Iterable):
            return list(callbacks)

        return [callbacks]

    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None, sink_size=-1):
        """
        Training process. The data would be passed to network through dataset channel.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        if sink_size == -1:
            epoch_num = epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size())
            train_dataset.__total_batch__ = epoch * sink_size

        dataset_helper, train_network = self._exec_preprocess(self._train_network,
                                                              is_train=True,
                                                              phase='train',
                                                              dataset=train_dataset,
                                                              dataset_sink_mode=True,
                                                              sink_size=sink_size,
                                                              epoch_num=epoch_num)
        self._train_network = train_network
        cb_params.train_network = self._train_network
        cb_params.cur_step_num = 0

        run_context = RunContext(cb_params)
        list_callback.begin(run_context)

        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1
            list_callback.epoch_begin(run_context)

            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                cb_params.train_dataset_element = inputs
                list_callback.step_begin(run_context)
                outputs = self._train_network(*inputs)
                cb_params.cur_step_num += dataset_helper.sink_size()
                cb_params.net_outputs = outputs
                list_callback.step_end(run_context)

            dataset_helper.continue_send()
            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break
        dataset_helper.stop_send()

        list_callback.end(run_context)

    def _train_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
        """
        Training process. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        dataset_helper, _ = self._exec_preprocess(self._train_network,
                                                  is_train=True,
                                                  phase='train',
                                                  dataset=train_dataset,
                                                  dataset_sink_mode=False,
                                                  epoch_num=epoch)
        cb_params.cur_step_num = 0
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False

        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1

            list_callback.epoch_begin(run_context)

            for next_element in dataset_helper:
                len_element = len(next_element)
                next_element = _transfer_tensor_to_tuple(next_element)
                if self._loss_fn and len_element != 2:
                    raise ValueError("when loss_fn is not None, train_dataset should "
                                     "return two elements, but got {}".format(len_element))
                cb_params.cur_step_num += 1

                cb_params.train_dataset_element = next_element
                list_callback.step_begin(run_context)
                outputs = self._train_network(*next_element)
                cb_params.net_outputs = outputs
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    _, overflow, _ = outputs
                    overflow = np.all(overflow.asnumpy())
                    self._loss_scale_manager.update_loss_scale(overflow)

                list_callback.step_end(run_context)
                if _is_role_pserver():
                    os._exit(0)
                should_stop = should_stop or run_context.get_stop_requested()
                if should_stop:
                    break

            train_dataset.reset()

            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        list_callback.end(run_context)

    def train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1):
        """
        Training API where the iteration is controlled by python front-end.

        When setting pynative mode or CPU, the training process will be performed with dataset not sink.

        Note:
            If dataset_sink_mode is True, epoch of training should be equal to the count of repeat
            operation in dataset processing. Otherwise, errors could occur since the amount of data
            is not equal to the required amount of training .
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.
            If sink_size > 0, each epoch the dataset can be traversed unlimited times until you get sink_size
            elements of the dataset. Next epoch continues to traverse from the end position of the previous traversal.

        Args:
            epoch (int): Generally, total number of iterations on the data per epoch.
                         When dataset_sink_mode is set to true and sink_size>0, each epoch sink sink_size
                         steps on the data instead of total number of iterations.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            callbacks (list): List of callback objects which should be executed while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel. Default: True.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink.
            sink_size (int): Control the amount of data in each sink.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             If dataset_sink_mode is False, set sink_size as invalid. Default: -1.

        Examples:
            >>> dataset = get_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
            >>> loss_scale_manager = FixedLossScaleManager()
            >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
            >>> model.train(2, dataset)
        """
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)
        if sink_size == -1:
            sink_size = train_dataset.get_dataset_size()
        Validator.check_is_int(sink_size)
        if sink_size < -1 or sink_size == 0:
            raise ValueError("The sink_size must be -1 or positive, but got sink_size {}.".format(sink_size))

        _device_number_check(self._parallel_mode, self._device_number)
        _parameter_broadcast_check(self._parallel_mode, self._parameter_broadcast)

        self._train(epoch,
                    train_dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=dataset_sink_mode,
                    sink_size=sink_size)

    def _eval_dataset_sink_process(self, valid_dataset, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network through dataset channel.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.

        Returns:
            Dict, which returns the loss value and metrics values for the model in the test mode.
        """
        run_context = RunContext(cb_params)

        dataset_helper, eval_network = self._exec_preprocess(self._eval_network,
                                                             is_train=False,
                                                             phase='eval',
                                                             dataset=valid_dataset,
                                                             dataset_sink_mode=True)
        self._eval_network = eval_network
        cb_params.eval_network = self._eval_network
        list_callback.begin(run_context)

        for inputs in dataset_helper:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)

            outputs = self._eval_network(*inputs)

            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)

        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)

        return metrics

    def _eval_process(self, valid_dataset, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network directly.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.

        Returns:
            Dict, which returns the loss value and metrics values for the model in the test mode.
        """
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)

        dataset_helper, _ = self._exec_preprocess(self._eval_network,
                                                  is_train=False,
                                                  phase='eval',
                                                  dataset=valid_dataset,
                                                  dataset_sink_mode=False)
        for next_element in dataset_helper:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)
            next_element = _transfer_tensor_to_tuple(next_element)
            outputs = self._eval_network(*next_element)
            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)

        valid_dataset.reset()

        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)
        return metrics

    def eval(self, valid_dataset, callbacks=None, dataset_sink_mode=True):
        """
        Evaluation API where the iteration is controlled by python front-end.

        Configure to pynative mode or CPU, the evaluating process will be performed with dataset non-sink mode.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            callbacks (list): List of callback objects which should be executed while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel. Default: True.

        Returns:
            Dict, which returns the loss value and metrics values for the model in the test mode.

        Examples:
            >>> dataset = get_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
            >>> model = Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
            >>> model.eval(dataset)
        """
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)
        _device_number_check(self._parallel_mode, self._device_number)
        if not self._metric_fns:
            raise ValueError("metric fn can not be None or empty.")

        cb_params = _InternalCallbackParam()
        cb_params.eval_network = self._eval_network
        cb_params.valid_dataset = valid_dataset
        cb_params.batch_num = valid_dataset.get_dataset_size()
        cb_params.mode = "eval"
        cb_params.cur_step_num = 0
        cb_params.list_callback = self._transform_callbacks(callbacks)
        cb_params.network = self._network

        self._clear_metrics()

        if context.get_context("device_target") == "CPU" and dataset_sink_mode:
            dataset_sink_mode = False
            logger.warning("CPU cannot support dataset sink mode currently."
                           "So the evaluating process will be performed with dataset non-sink mode.")

        with _CallbackManager(callbacks) as list_callback:
            if dataset_sink_mode:
                return self._eval_dataset_sink_process(valid_dataset, list_callback, cb_params)
            return self._eval_process(valid_dataset, list_callback, cb_params)

    def predict(self, *predict_data):
        """
        Generate output predictions for the input samples.

        Data could be a single tensor, a list of tensor, or a tuple of tensor.

        Note:
            Batch data should be put together in one tensor.

        Args:
           predict_data (Tensor): Tensor of predict data. can be array, list or tuple.

        Returns:
            Tensor, array(s) of predictions.

        Examples:
            >>> input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), mindspore.float32)
            >>> model = Model(Net())
            >>> model.predict(input_data)
        """
        self._predict_network.set_train(False)
        check_input_data(*predict_data, data_class=Tensor)
        result = self._predict_network(*predict_data)

        check_output_data(result)
        return result


__all__ = ["Model"]
