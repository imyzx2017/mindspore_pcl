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
"""Validators for TensorOps.
"""
import numbers
from functools import wraps
import numpy as np
from mindspore._c_dataengine import TensorOp

from mindspore.dataset.core.validator_helpers import check_value, check_uint8, FLOAT_MAX_INTEGER, check_pos_float32, \
    check_2tuple, check_range, check_positive, INT32_MAX, parse_user_args, type_check, type_check_list, \
    check_tensor_op, UINT8_MAX, check_value_normalize_std
from .utils import Inter, Border, ImageBatchFormat


def check_crop_size(size):
    """Wrapper method to check the parameters of crop size."""
    type_check(size, (int, list, tuple), "size")
    if isinstance(size, int):
        check_value(size, (1, FLOAT_MAX_INTEGER))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        for value in size:
            check_value(value, (1, FLOAT_MAX_INTEGER))
    else:
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")


def check_cut_mix_batch_c(method):
    """Wrapper method to check the parameters of CutMixBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [image_batch_format, alpha, prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(image_batch_format, (ImageBatchFormat,), "image_batch_format")
        check_pos_float32(alpha)
        check_positive(alpha, "alpha")
        check_value(prob, [0, 1], "prob")
        return method(self, *args, **kwargs)

    return new_method


def check_resize_size(size):
    """Wrapper method to check the parameters of resize."""
    if isinstance(size, int):
        check_value(size, (1, FLOAT_MAX_INTEGER))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        for i, value in enumerate(size):
            check_value(value, (1, INT32_MAX), "size at dim {0}".format(i))
    else:
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")


def check_mix_up_batch_c(method):
    """Wrapper method to check the parameters of MixUpBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [alpha], _ = parse_user_args(method, *args, **kwargs)
        check_positive(alpha, "alpha")
        check_pos_float32(alpha)

        return method(self, *args, **kwargs)

    return new_method


def check_normalize_c_param(mean, std):
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal")
    for mean_value in mean:
        check_pos_float32(mean_value)
    for std_value in std:
        check_pos_float32(std_value)


def check_normalize_py_param(mean, std):
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal")
    for mean_value in mean:
        check_value(mean_value, [0., 1.], "mean_value")
    for std_value in std:
        check_value_normalize_std(std_value, [0., 1.], "std_value")


def check_fill_value(fill_value):
    if isinstance(fill_value, int):
        check_uint8(fill_value)
    elif isinstance(fill_value, tuple) and len(fill_value) == 3:
        for value in fill_value:
            check_uint8(value)
    else:
        raise TypeError("fill_value should be a single integer or a 3-tuple.")


def check_padding(padding):
    """Parsing the padding arguments and check if it is legal."""
    type_check(padding, (tuple, list, numbers.Number), "padding")
    if isinstance(padding, numbers.Number):
        check_value(padding, (0, INT32_MAX), "padding")
    if isinstance(padding, (tuple, list)):
        if len(padding) not in (2, 4):
            raise ValueError("The size of the padding list or tuple should be 2 or 4.")
        for i, pad_value in enumerate(padding):
            type_check(pad_value, (int,), "padding[{}]".format(i))
            check_value(pad_value, (0, INT32_MAX), "pad_value")


def check_degrees(degrees):
    """Check if the degrees is legal."""
    type_check(degrees, (numbers.Number, list, tuple), "degrees")
    if isinstance(degrees, numbers.Number):
        check_value(degrees, (0, float("inf")), "degrees")
    elif isinstance(degrees, (list, tuple)):
        if len(degrees) == 2:
            type_check_list(degrees, (numbers.Number,), "degrees")
            if degrees[0] > degrees[1]:
                raise ValueError("degrees should be in (min,max) format. Got (max,min).")
        else:
            raise TypeError("If degrees is a sequence, the length must be 2.")


def check_random_color_adjust_param(value, input_name, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
    """Check the parameters in random color adjust operation."""
    type_check(value, (numbers.Number, list, tuple), input_name)
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError("The input value of {} cannot be negative.".format(input_name))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        check_range(value, bound)
        if value[0] > value[1]:
            raise ValueError("value should be in (min,max) format. Got (max,min).")


def check_erasing_value(value):
    if not (isinstance(value, (numbers.Number, str, bytes)) or
            (isinstance(value, (tuple, list)) and len(value) == 3)):
        raise ValueError("The value for erasing should be either a single value, "
                         "or a string 'random', or a sequence of 3 elements for RGB respectively.")


def check_crop(method):
    """A wrapper that wraps a parameter checker around the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)

        return method(self, *args, **kwargs)

    return new_method


def check_posterize(method):
    """A wrapper that wraps a parameter checker around the original function(posterize operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [bits], _ = parse_user_args(method, *args, **kwargs)
        if bits is not None:
            type_check(bits, (list, tuple, int), "bits")
        if isinstance(bits, int):
            check_value(bits, [1, 8])
        if isinstance(bits, (list, tuple)):
            if len(bits) != 2:
                raise TypeError("Size of bits should be a single integer or a list/tuple (min, max) of length 2.")
            for item in bits:
                check_uint8(item, "bits")
            # also checks if min <= max
            check_range(bits, [1, 8])
        return method(self, *args, **kwargs)

    return new_method


def check_resize_interpolation(method):
    """A wrapper that wraps a parameter checker around the original function(resize interpolation operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, interpolation], _ = parse_user_args(method, *args, **kwargs)
        check_resize_size(size)
        if interpolation is not None:
            type_check(interpolation, (Inter,), "interpolation")

        return method(self, *args, **kwargs)

    return new_method


def check_resize(method):
    """A wrapper that wraps a parameter checker around the original function(resize operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size], _ = parse_user_args(method, *args, **kwargs)
        check_resize_size(size)

        return method(self, *args, **kwargs)

    return new_method


def check_size_scale_ration_max_attempts_paras(size, scale, ratio, max_attempts):
    """Wrapper method to check the parameters of RandomCropDecodeResize and SoftDvppDecodeRandomCropResizeJpeg."""

    check_crop_size(size)
    if scale is not None:
        type_check(scale, (tuple,), "scale")
        type_check_list(scale, (float, int), "scale")
        check_range(scale, [0, FLOAT_MAX_INTEGER])
        if scale[0] > scale[1]:
            raise ValueError("scale should be in (min,max) format. Got (max,min).")
    if ratio is not None:
        type_check(ratio, (tuple,), "ratio")
        type_check_list(ratio, (float, int), "ratio")
        check_range(ratio, [0, FLOAT_MAX_INTEGER])
        if ratio[0] > ratio[1]:
            raise ValueError("ratio should be in (min,max) format. Got (max,min).")
    if max_attempts is not None:
        check_value(max_attempts, (1, FLOAT_MAX_INTEGER))


def check_random_resize_crop(method):
    """A wrapper that wraps a parameter checker around the original function(random resize crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, scale, ratio, interpolation, max_attempts], _ = parse_user_args(method, *args, **kwargs)
        if interpolation is not None:
            type_check(interpolation, (Inter,), "interpolation")
        check_size_scale_ration_max_attempts_paras(size, scale, ratio, max_attempts)

        return method(self, *args, **kwargs)

    return new_method


def check_prob(method):
    """A wrapper that wraps a parameter checker (to confirm probability) around the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(prob, (float, int,), "prob")
        check_value(prob, [0., 1.], "prob")

        return method(self, *args, **kwargs)

    return new_method


def check_normalize_c(method):
    """A wrapper that wraps a parameter checker around the original function(normalize operation written in C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_c_param(mean, std)

        return method(self, *args, **kwargs)

    return new_method


def check_normalize_py(method):
    """A wrapper that wraps a parameter checker around the original function(normalize operation written in Python)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_py_param(mean, std)

        return method(self, *args, **kwargs)

    return new_method


def check_random_crop(method):
    """Wrapper method to check the parameters of random crop."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, padding, pad_if_needed, fill_value, padding_mode], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)
        type_check(pad_if_needed, (bool,), "pad_if_needed")
        if padding is not None:
            check_padding(padding)
        if fill_value is not None:
            check_fill_value(fill_value)
        if padding_mode is not None:
            type_check(padding_mode, (Border,), "padding_mode")

        return method(self, *args, **kwargs)

    return new_method


def check_random_color_adjust(method):
    """Wrapper method to check the parameters of random color adjust."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [brightness, contrast, saturation, hue], _ = parse_user_args(method, *args, **kwargs)
        check_random_color_adjust_param(brightness, "brightness")
        check_random_color_adjust_param(contrast, "contrast")
        check_random_color_adjust_param(saturation, "saturation")
        check_random_color_adjust_param(hue, 'hue', center=0, bound=(-0.5, 0.5), non_negative=False)

        return method(self, *args, **kwargs)

    return new_method


def check_random_rotation(method):
    """Wrapper method to check the parameters of random rotation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees, resample, expand, center, fill_value], _ = parse_user_args(method, *args, **kwargs)
        check_degrees(degrees)

        if resample is not None:
            type_check(resample, (Inter,), "resample")
        if expand is not None:
            type_check(expand, (bool,), "expand")
        if center is not None:
            check_2tuple(center, "center")
        if fill_value is not None:
            check_fill_value(fill_value)

        return method(self, *args, **kwargs)

    return new_method


def check_ten_crop(method):
    """Wrapper method to check the parameters of crop."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, use_vertical_flip], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)

        if use_vertical_flip is not None:
            type_check(use_vertical_flip, (bool,), "use_vertical_flip")

        return method(self, *args, **kwargs)

    return new_method


def check_num_channels(method):
    """Wrapper method to check the parameters of number of channels."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_output_channels], _ = parse_user_args(method, *args, **kwargs)
        if num_output_channels is not None:
            if num_output_channels not in (1, 3):
                raise ValueError("Number of channels of the output grayscale image"
                                 "should be either 1 or 3. Got {0}".format(num_output_channels))

        return method(self, *args, **kwargs)

    return new_method


def check_pad(method):
    """Wrapper method to check the parameters of random pad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [padding, fill_value, padding_mode], _ = parse_user_args(method, *args, **kwargs)
        check_padding(padding)
        check_fill_value(fill_value)
        type_check(padding_mode, (Border,), "padding_mode")

        return method(self, *args, **kwargs)

    return new_method


def check_random_perspective(method):
    """Wrapper method to check the parameters of random perspective."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [distortion_scale, prob, interpolation], _ = parse_user_args(method, *args, **kwargs)

        check_value(distortion_scale, [0., 1.], "distortion_scale")
        check_value(prob, [0., 1.], "prob")
        type_check(interpolation, (Inter,), "interpolation")

        return method(self, *args, **kwargs)

    return new_method


def check_mix_up(method):
    """Wrapper method to check the parameters of mix up."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [batch_size, alpha, is_single], _ = parse_user_args(method, *args, **kwargs)

        check_value(batch_size, (1, FLOAT_MAX_INTEGER))
        check_positive(alpha, "alpha")
        type_check(is_single, (bool,), "is_single")

        return method(self, *args, **kwargs)

    return new_method


def check_random_erasing(method):
    """Wrapper method to check the parameters of random erasing."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [prob, scale, ratio, value, inplace, max_attempts], _ = parse_user_args(method, *args, **kwargs)

        check_value(prob, [0., 1.], "prob")
        check_range(scale, [0, FLOAT_MAX_INTEGER])
        check_range(ratio, [0, FLOAT_MAX_INTEGER])
        check_erasing_value(value)
        type_check(inplace, (bool,), "inplace")
        check_value(max_attempts, (1, FLOAT_MAX_INTEGER))

        return method(self, *args, **kwargs)

    return new_method


def check_cutout(method):
    """Wrapper method to check the parameters of cutout operation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [length, num_patches], _ = parse_user_args(method, *args, **kwargs)

        check_value(length, (1, FLOAT_MAX_INTEGER))
        check_value(num_patches, (1, FLOAT_MAX_INTEGER))

        return method(self, *args, **kwargs)

    return new_method


def check_linear_transform(method):
    """Wrapper method to check the parameters of linear transform."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transformation_matrix, mean_vector], _ = parse_user_args(method, *args, **kwargs)
        type_check(transformation_matrix, (np.ndarray,), "transformation_matrix")
        type_check(mean_vector, (np.ndarray,), "mean_vector")

        if transformation_matrix.shape[0] != transformation_matrix.shape[1]:
            raise ValueError("transformation_matrix should be a square matrix. "
                             "Got shape {} instead".format(transformation_matrix.shape))
        if mean_vector.shape[0] != transformation_matrix.shape[0]:
            raise ValueError("mean_vector length {0} should match either one dimension of the square"
                             "transformation_matrix {1}.".format(mean_vector.shape[0], transformation_matrix.shape))

        return method(self, *args, **kwargs)

    return new_method


def check_random_affine(method):
    """Wrapper method to check the parameters of random affine."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees, translate, scale, shear, resample, fill_value], _ = parse_user_args(method, *args, **kwargs)
        check_degrees(degrees)

        if translate is not None:
            type_check(translate, (list, tuple), "translate")
            type_check_list(translate, (int, float), "translate")
            if len(translate) != 2 and len(translate) != 4:
                raise TypeError("translate should be a list or tuple of length 2 or 4.")
            for i, t in enumerate(translate):
                check_value(t, [-1.0, 1.0], "translate at {0}".format(i))

        if scale is not None:
            type_check(scale, (tuple, list), "scale")
            type_check_list(scale, (int, float), "scale")
            if len(scale) == 2:
                for i, s in enumerate(scale):
                    check_positive(s, "scale[{}]".format(i))
                if scale[0] > scale[1]:
                    raise ValueError("Input scale[1] must be equal to or greater than scale[0].")
            else:
                raise TypeError("scale should be a list or tuple of length 2.")

        if shear is not None:
            type_check(shear, (numbers.Number, tuple, list), "shear")
            if isinstance(shear, numbers.Number):
                check_positive(shear, "shear")
            else:
                type_check_list(shear, (int, float), "shear")
                if len(shear) not in (2, 4):
                    raise TypeError("shear must be of length 2 or 4.")
                if len(shear) == 2 and shear[0] > shear[1]:
                    raise ValueError("Input shear[1] must be equal to or greater than shear[0]")
                if len(shear) == 4 and (shear[0] > shear[1] or shear[2] > shear[3]):
                    raise ValueError("Input shear[1] must be equal to or greater than shear[0] and "
                                     "shear[3] must be equal to or greater than shear[2].")

        type_check(resample, (Inter,), "resample")

        if fill_value is not None:
            check_fill_value(fill_value)

        return method(self, *args, **kwargs)

    return new_method


def check_rescale(method):
    """Wrapper method to check the parameters of rescale."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [rescale, shift], _ = parse_user_args(method, *args, **kwargs)
        check_pos_float32(rescale)
        type_check(shift, (numbers.Number,), "shift")

        return method(self, *args, **kwargs)

    return new_method


def check_uniform_augment_cpp(method):
    """Wrapper method to check the parameters of UniformAugment C++ op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms, num_ops], _ = parse_user_args(method, *args, **kwargs)
        type_check(num_ops, (int,), "num_ops")
        check_positive(num_ops, "num_ops")

        if num_ops > len(transforms):
            raise ValueError("num_ops is greater than transforms list size")
        type_check_list(transforms, (TensorOp,), "tensor_ops")

        return method(self, *args, **kwargs)

    return new_method


def check_bounding_box_augment_cpp(method):
    """Wrapper method to check the parameters of BoundingBoxAugment C++ op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transform, ratio], _ = parse_user_args(method, *args, **kwargs)
        type_check(ratio, (float, int), "ratio")
        check_value(ratio, [0., 1.], "ratio")
        type_check(transform, (TensorOp,), "transform")
        return method(self, *args, **kwargs)

    return new_method


def check_auto_contrast(method):
    """Wrapper method to check the parameters of AutoContrast ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [cutoff, ignore], _ = parse_user_args(method, *args, **kwargs)
        type_check(cutoff, (int, float), "cutoff")
        check_value(cutoff, [0, 100], "cutoff")
        if ignore is not None:
            type_check(ignore, (list, tuple, int), "ignore")
        if isinstance(ignore, int):
            check_value(ignore, [0, 255], "ignore")
        if isinstance(ignore, (list, tuple)):
            for item in ignore:
                type_check(item, (int,), "item")
                check_value(item, [0, 255], "ignore")
        return method(self, *args, **kwargs)

    return new_method


def check_uniform_augment_py(method):
    """Wrapper method to check the parameters of Python UniformAugment op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms, num_ops], _ = parse_user_args(method, *args, **kwargs)
        type_check(transforms, (list,), "transforms")

        if not transforms:
            raise ValueError("transforms list is empty.")

        for transform in transforms:
            if isinstance(transform, TensorOp):
                raise ValueError("transform list only accepts Python operations.")

        type_check(num_ops, (int,), "num_ops")
        check_positive(num_ops, "num_ops")
        if num_ops > len(transforms):
            raise ValueError("num_ops cannot be greater than the length of transforms list.")

        return method(self, *args, **kwargs)

    return new_method


def check_positive_degrees(method):
    """A wrapper method to check degrees parameter in RandomSharpness and RandomColor ops (Python and C++)"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees], _ = parse_user_args(method, *args, **kwargs)

        if degrees is not None:
            if not isinstance(degrees, (list, tuple)):
                raise TypeError("degrees must be either a tuple or a list.")
            type_check_list(degrees, (int, float), "degrees")
            if len(degrees) != 2:
                raise ValueError("degrees must be a sequence with length 2.")
            for degree in degrees:
                check_value(degree, (0, FLOAT_MAX_INTEGER))
            if degrees[0] > degrees[1]:
                raise ValueError("degrees should be in (min,max) format. Got (max,min).")

        return method(self, *args, **kwargs)

    return new_method


def check_random_select_subpolicy_op(method):
    """Wrapper method to check the parameters of RandomSelectSubpolicyOp."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [policy], _ = parse_user_args(method, *args, **kwargs)
        type_check(policy, (list,), "policy")
        if not policy:
            raise ValueError("policy can not be empty.")
        for sub_ind, sub in enumerate(policy):
            type_check(sub, (list,), "policy[{0}]".format([sub_ind]))
            if not sub:
                raise ValueError("policy[{0}] can not be empty.".format(sub_ind))
            for op_ind, tp in enumerate(sub):
                check_2tuple(tp, "policy[{0}][{1}]".format(sub_ind, op_ind))
                check_tensor_op(tp[0], "op of (op, prob) in policy[{0}][{1}]".format(sub_ind, op_ind))
                check_value(tp[1], (0, 1), "prob of (op, prob) policy[{0}][{1}]".format(sub_ind, op_ind))

        return method(self, *args, **kwargs)

    return new_method


def check_soft_dvpp_decode_random_crop_resize_jpeg(method):
    """Wrapper method to check the parameters of SoftDvppDecodeRandomCropResizeJpeg."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, scale, ratio, max_attempts], _ = parse_user_args(method, *args, **kwargs)
        check_size_scale_ration_max_attempts_paras(size, scale, ratio, max_attempts)

        return method(self, *args, **kwargs)

    return new_method


def check_random_solarize(method):
    """Wrapper method to check the parameters of RandomSolarizeOp."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [threshold], _ = parse_user_args(method, *args, **kwargs)

        type_check(threshold, (tuple,), "threshold")
        type_check_list(threshold, (int,), "threshold")
        if len(threshold) != 2:
            raise ValueError("threshold must be a sequence of two numbers")
        for element in threshold:
            check_value(element, (0, UINT8_MAX))
        if threshold[1] < threshold[0]:
            raise ValueError("threshold must be in min max format numbers")

        return method(self, *args, **kwargs)

    return new_method
