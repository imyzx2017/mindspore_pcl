# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
The configuration module provides various functions to set and get the supported
configuration parameters, and read a configuration file.
"""
import random
import numpy
import mindspore._c_dataengine as cde

__all__ = ['set_seed', 'get_seed', 'set_prefetch_size', 'get_prefetch_size', 'set_num_parallel_workers',
           'get_num_parallel_workers', 'set_monitor_sampling_interval', 'get_monitor_sampling_interval', 'load']

INT32_MAX = 2147483647
UINT32_MAX = 4294967295

_config = cde.GlobalContext.config_manager()


def set_seed(seed):
    """
    Set the seed to be used in any random generator. This is used to produce deterministic results.

    Note:
        This set_seed function sets the seed in the Python random library and numpy.random library
        for deterministic Python augmentations using randomness. This set_seed function should
        be called with every iterator created to reset the random seed. In the pipeline, this
        does not guarantee deterministic results with num_parallel_workers > 1.

    Args:
        seed(int): Seed to be set.

    Raises:
        ValueError: If seed is invalid (< 0 or > MAX_UINT_32).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> # Set a new global configuration value for the seed value.
        >>> # Operations with randomness will use the seed value to generate random values.
        >>> ds.config.set_seed(1000)
    """
    if seed < 0 or seed > UINT32_MAX:
        raise ValueError("Seed given is not within the required range.")
    _config.set_seed(seed)
    random.seed(seed)
    # numpy.random isn't thread safe
    numpy.random.seed(seed)


def get_seed():
    """
    Get the seed.

    Returns:
        Int, seed.
    """
    return _config.get_seed()


def set_prefetch_size(size):
    """
    Set the number of rows to be prefetched.

    Args:
        size (int): Total number of rows to be prefetched.

    Raises:
        ValueError: If prefetch_size is invalid (<= 0 or > MAX_INT_32).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> # Set a new global configuration value for the prefetch size.
        >>> ds.config.set_prefetch_size(1000)
    """
    if size <= 0 or size > INT32_MAX:
        raise ValueError("Prefetch size given is not within the required range.")
    _config.set_op_connector_size(size)


def get_prefetch_size():
    """
    Get the prefetch size in number of rows.

    Returns:
        Size, total number of rows to be prefetched.
    """
    return _config.get_op_connector_size()


def set_num_parallel_workers(num):
    """
    Set the default number of parallel workers.

    Args:
        num (int): Number of parallel workers to be used as a default for each operation.

    Raises:
        ValueError: If num_parallel_workers is invalid (<= 0 or > MAX_INT_32).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> # Set a new global configuration value for the number of parallel workers.
        >>> # Now parallel dataset operators will run with 8 workers.
        >>> ds.config.set_num_parallel_workers(8)
    """
    if num <= 0 or num > INT32_MAX:
        raise ValueError("Number of parallel workers given is not within the required range.")
    _config.set_num_parallel_workers(num)


def get_num_parallel_workers():
    """
    Get the default number of parallel workers.

    Returns:
        Int, number of parallel workers to be used as a default for each operation
    """
    return _config.get_num_parallel_workers()


def set_monitor_sampling_interval(interval):
    """
    Set the default interval (in milliseconds) for monitor sampling.

    Args:
        interval (int): Interval (in milliseconds) to be used for performance monitor sampling.

    Raises:
        ValueError: If interval is invalid (<= 0 or > MAX_INT_32).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> # Set a new global configuration value for the monitor sampling interval.
        >>> ds.config.set_monitor_sampling_interval(100)
    """
    if interval <= 0 or interval > INT32_MAX:
        raise ValueError("Interval given is not within the required range.")
    _config.set_monitor_sampling_interval(interval)


def get_monitor_sampling_interval():
    """
    Get the default interval of performance monitor sampling.

    Returns:
        Int, interval (in milliseconds) for performance monitor sampling.
    """
    return _config.get_monitor_sampling_interval()


def set_callback_timeout(timeout):
    """
    Set the default timeout (in seconds) for DSWaitedCallback.
    In case of a deadlock, the wait function will exit after the timeout period.

    Args:
        timeout (int): Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.

    Raises:
        ValueError: If timeout is invalid (<= 0 or > MAX_INT_32).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> # Set a new global configuration value for the timeout value.
        >>> ds.config.set_callback_timeout(100)
    """
    if timeout <= 0 or timeout > INT32_MAX:
        raise ValueError("Timeout given is not within the required range.")
    _config.set_callback_timeout(timeout)


def get_callback_timeout():
    """
    Get the default timeout for DSWaitedCallback.
    In case of a deadlock, the wait function will exit after the timeout period.

    Returns:
        Int, the duration in seconds
    """
    return _config.get_callback_timeout()


def __str__():
    """
    String representation of the configurations.

    Returns:
        Str, configurations.
    """
    return str(_config)


def load(file):
    """
    Load configurations from a file.

    Args:
        file (str): Path of the configuration file to be loaded.

    Raises:
        RuntimeError: If file is invalid and parsing fails.

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> # Set new default configuration values according to values in the configuration file.
        >>> ds.config.load("path/to/config/file")
        >>> # example config file:
        >>> # {
        >>> #     "logFilePath": "/tmp",
        >>> #     "numParallelWorkers": 4,
        >>> #     "seed": 5489,
        >>> #     "monitorSamplingInterval": 30
        >>> # }
    """
    _config.load(file)
