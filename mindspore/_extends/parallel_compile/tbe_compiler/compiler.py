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
"""tbe compiler"""
import json
import os
import sys
from te.platform.cce_conf import te_set_version
from te.platform.fusion_util import fusion_op
import te
from common import check_kernel_info, get_args, get_build_in_impl_path

build_in_impl_path = get_build_in_impl_path()

# op function list
op_build = "compile"
fusion_pattern_start_flag = "fusion_pattern_start"
fusion_pattern_end_flag = "fusion_pattern_end"

def _initialize(impl_path):
    """Initialize"""
    if impl_path == "":
        op_module_name = build_in_impl_path
    else:
        op_module_name = impl_path
    if not op_module_name:
        raise ValueError("Can not find the env TBE_IMPL_PATH")

    sys.path.insert(0, op_module_name)

def _replace_range(args):
    for arg in args:
        if not arg.__contains__('range'):
            continue
        shape_range = arg["range"]
        for range_item in shape_range:
            for index, value in enumerate(range_item):
                if value < 0:
                    range_item[index] = None

def build_op(build_type, json_str):
    """
    call op functions with function name and input args json_str

    Args:
        build_type : op function name
        json_str (str): op function input args

    Raises:
        Exception: If specific keyword is not found.
    """
    kernel_info = json.loads(json_str)
    check_kernel_info(kernel_info)
    te_set_version(kernel_info["op_info"]["socVersion"])
    op_name = kernel_info['op_info']['name']

    try:
        custom_flag = False
        if 'impl_path' in kernel_info and kernel_info['impl_path'] is not None:
            impl_path = os.path.realpath(kernel_info['impl_path'])
            if os.path.isfile(impl_path):
                path, file_name = os.path.split(impl_path)
                op_name, _ = os.path.splitext(file_name)
                impl_path = path
                custom_flag = True
            else:
                impl_path = ""
        _initialize(impl_path)

        inputs_args = get_args(kernel_info['op_info'], 'inputs')
        outputs_args = get_args(kernel_info['op_info'], 'outputs')
        attrs_args = get_args(kernel_info['op_info'], 'attrs')
        kernel_name = kernel_info['op_info']['kernel_name']
        is_dynamic_shape = kernel_info['op_info']['is_dynamic_shape']
        if is_dynamic_shape:
            _replace_range(inputs_args)
            _replace_range(outputs_args)

        if custom_flag:
            op_module = __import__(op_name)
        else:
            if is_dynamic_shape:
                op_module = __import__("impl.dynamic."+op_name, globals(), locals(), [op_name], 0)
            else:
                op_module = __import__("impl."+op_name, globals(), locals(), [op_name], 0)
        # get function
        if build_type == op_build:
            if custom_flag:
                py_fn_name = kernel_info['op_info']['name']
            else:
                py_fn_name = op_name
        else:
            raise ValueError("function {} is not supported by Tbe op {}.".format(build_type, op_name))
        op_func = getattr(op_module, py_fn_name, None)
        if op_func is None:
            raise ValueError("Op:{} function {} is not supported by Tbe.".format(op_name, build_type))

        # call function
        if op_name == "bounding_box_encode":
            return op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name_val=kernel_name)

        if is_dynamic_shape:
            with te.op.dynamic():
                op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name=kernel_name)
                return te.op.get_compile_info()
        else:
            return op_func(*inputs_args, *outputs_args, *attrs_args, kernel_name=kernel_name)

    except Exception as e:
        raise RuntimeError(e)


def compile_fusion_op(json_str):
    """
    compile fusion op with input args json_str

    Args:
        json_str (str): op function input args

    Raises:
        Exception: If specific keyword is not found.
    """
    args = json.loads(json_str)
    te_set_version(args['fusion_op']["socVersion"])
    if 'fusion_op' not in args or not args['fusion_op']:
        raise ValueError("Json string Errors, key:fusion_op not found.")
    fusion_op_arg = args['fusion_op']
    return fusion_op(json.dumps(fusion_op_arg))


def compile_with_json(json_str):
    """
    Compile tbe with json.

    Args:
        json_str (str): jason file path.

    """
    json_info = json.loads(json_str)
    if "fusion_op" in json_info:
        ret = compile_fusion_op(json_str)
    else:
        ret = build_op(op_build, json_str)
    return ret

if __name__ == "__main__":
    in_args = sys.stdin.readline()
    result = compile_with_json(in_args)
    sys.stdout.write(fusion_pattern_start_flag + str(result) + fusion_pattern_end_flag)
    sys.stdout.flush()
