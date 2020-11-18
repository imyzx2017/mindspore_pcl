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

"""Define the grad rules of neural network related operations."""
import numpy as np
from mindspore.ops import _selected_grad_ops as SG
from mindspore.ops.primitive import constexpr
from mindspore.common.tensor import Tensor
from .grad_base import bprop_getters
from .. import functional as F
from .. import operations as P
from ...common import dtype as mstype
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations import _grad_ops as G
from ..operations import _inner_ops as inner
from ... import context


@bprop_getters.register(P.BiasAdd)
def get_bprop_bias_add(self):
    """Grad definition for `BiasAdd` operation."""
    bias_grad = SG.BiasAddGrad(self.data_format)

    def bprop(x, w, out, dout):
        return dout, bias_grad(dout)

    return bprop


@bprop_getters.register(P.Conv2D)
def get_bprop_conv2d(self):
    """Grad definition for `Conv2D` operation."""
    input_grad = P.Conv2DBackpropInput(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    filter_grad = G.Conv2DBackpropFilter(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(dout, w, get_shape(x))
        dw = filter_grad(dout, x, get_shape(w))
        return dx, dw

    return bprop


@bprop_getters.register(inner.ExtractImagePatches)
def get_bprop_extract_image_patches(self):
    """Grad definition for `ExtractImagePatches` operation."""
    get_shape = P.Shape()
    reshape = P.Reshape()
    extract_image_patches = inner.ExtractImagePatches(ksizes=self.ksizes,
                                                      strides=self.strides,
                                                      rates=self.rates,
                                                      padding=self.padding)
    concat = P.Concat(axis=-1)
    expand_dims = P.ExpandDims()
    scatter_nd = P.ScatterNd()
    dtype = P.DType()
    fill = P.Fill()
    slice_op = P.Slice()
    transpose = P.Transpose()
    cast = P.Cast()
    matmul = P.MatMul()

    _, ksizes_row, ksizes_col, _ = self.ksizes

    def bprop(x, out, dout):
        x_shape = get_shape(x)
        x_batch, x_depth, x_row, x_col = x_shape
        x_indices_num = x_row * x_col + 1
        x_idx = cast(F.tuple_to_array(range(1, x_indices_num)), mstype.float32)
        x_idx = reshape(x_idx, (1, 1, x_row, x_col))
        x_idx_patch = cast(extract_image_patches(x_idx), mstype.int32)
        x_idx_patch = transpose(x_idx_patch, (0, 2, 3, 1))

        out_shape = get_shape(out)
        _, _, out_row, out_col = out_shape
        out_indices_num = out_row * out_col * ksizes_row * ksizes_col
        out_idx = F.tuple_to_array(range(out_indices_num))
        out_idx = reshape(out_idx, (1, out_row, out_col, ksizes_row * ksizes_col))

        idx_tensor = concat((expand_dims(x_idx_patch, -1), expand_dims(out_idx, -1)))
        idx_tensor = reshape(idx_tensor, (-1, 2))
        sp_shape = (x_indices_num, out_indices_num)
        sp_tensor = scatter_nd(idx_tensor, fill(dtype(dout), (out_indices_num,), 1), sp_shape)
        sp_tensor = slice_op(sp_tensor, (1, 0), (x_indices_num - 1, out_indices_num))

        grad = transpose(dout, (0, 2, 3, 1))
        grad = reshape(grad, (x_batch, out_row, out_col, ksizes_row, ksizes_col, x_depth))
        grad = transpose(grad, (1, 2, 3, 4, 0, 5))
        grad = reshape(grad, (-1, x_batch * x_depth))

        jac = matmul(sp_tensor, grad)
        dx = reshape(jac, (x_row, x_col, x_batch, x_depth))
        dx = transpose(dx, (2, 3, 0, 1))
        return (dx,)

    def bprop_ge(x, out, dout):
        x_shape = get_shape(x)
        x_batch, x_row, x_col, x_depth = x_shape
        x_indices_num = x_row * x_col + 1
        x_idx = F.tuple_to_array(range(1, x_indices_num))
        x_idx = reshape(x_idx, (1, x_row, x_col, 1))
        x_idx_patch = extract_image_patches(x_idx)

        out_shape = get_shape(out)
        _, out_row, out_col, _ = out_shape
        out_indices_num = out_row * out_col * ksizes_row * ksizes_col
        out_idx = F.tuple_to_array(range(out_indices_num))
        out_idx = reshape(out_idx, (1, out_row, out_col, ksizes_row * ksizes_col))

        idx_tensor = concat((expand_dims(x_idx_patch, -1), expand_dims(out_idx, -1)))
        idx_tensor = reshape(idx_tensor, (-1, 2))
        sp_shape = (x_indices_num, out_indices_num)
        sp_tensor = scatter_nd(idx_tensor, fill(dtype(dout), (out_indices_num,), 1), sp_shape)
        sp_tensor = slice_op(sp_tensor, (1, 0), (x_indices_num - 1, out_indices_num))

        grad = reshape(dout, (x_batch, out_row, out_col, ksizes_row, ksizes_col, x_depth))
        grad = transpose(grad, (1, 2, 3, 4, 0, 5))
        grad = reshape(grad, (-1, x_batch * x_depth))

        jac = matmul(sp_tensor, grad)
        dx = reshape(jac, (x_row, x_col, x_batch, x_depth))
        dx = transpose(dx, (2, 0, 1, 3))

        return (dx,)

    if context.get_context("enable_ge"):
        return bprop_ge

    return bprop


@bprop_getters.register(P.DepthwiseConv2dNative)
def get_bprop_depthwise_conv2d_native(self):
    """Grad definition for `DepthwiseConv2dNative` operation."""
    input_grad = G.DepthwiseConv2dNativeBackpropInput(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pads, self.mode, self.stride,
        self.dilation, self.group
    )
    filter_grad = G.DepthwiseConv2dNativeBackpropFilter(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pads, self.mode, self.stride,
        self.dilation, self.group
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(get_shape(x), w, dout)
        dw = filter_grad(x, get_shape(w), dout)
        return dx, dw

    return bprop


@bprop_getters.register(P.MaxPoolWithArgmax)
def get_bprop_max_pool_with_argmax(self):
    """Grad definition for `MaxPoolWithArgmax` operation."""
    maxpool_grad = G.MaxPoolGradWithArgmax(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding)

    def bprop(x, out, dout):
        dx = maxpool_grad(x, dout[0], out[1])
        return (dx,)

    return bprop


@bprop_getters.register(G.MaxPoolGrad)
def get_bprop_max_pool_grad_grad(self):
    """Grad definition for `MaxPoolGrad` operation."""
    maxpool_grad_grad = G.MaxPoolGradGrad(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding)

    def bprop(x1, x2, grad, out, dout):
        dx1 = zeros_like(x1)
        dx2 = zeros_like(x2)
        dgrad = maxpool_grad_grad(x1, x2, dout)
        return (dx1, dx2, dgrad)

    return bprop


@bprop_getters.register(G.MaxPoolGradGrad)
def get_bprop_max_pool_grad_grad_grad(self):
    """Grad definition for `MaxPoolGradGrad` operation."""
    maxpool_grad = G.MaxPoolGrad(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding)

    def bprop(x1, x2, grad, out, dout):
        dx1 = zeros_like(x1)
        dx2 = zeros_like(x2)
        dgrad = maxpool_grad(x1, x2, dout)
        return (dx1, dx2, dgrad)

    return bprop


@bprop_getters.register(P.MaxPool)
def get_bprop_max_pool_grad(self):
    """Grad definition for `MaxPool` operation."""
    maxpool_grad = G.MaxPoolGrad(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding,
        data_format=self.format)

    def bprop(x, out, dout):
        dx = maxpool_grad(x, out, dout)
        return (dx,)

    return bprop


def _windowed_output_size(input_size, ksize, stride, padding):
    """
    helper func for AvgPoolGrad
    """

    tmp_output = 0
    tmp_pad_need = 0
    tmp_pad_before = 0
    tmp_pad_after = 0
    if padding == 'VALID':
        tmp_output = (input_size - ksize + stride) // stride
        tmp_pad_before = 0
        tmp_pad_after = 0
    elif padding == 'SAME':
        tmp_output = (input_size + stride - 1) // stride
        tmp_pad_need = max(0, (tmp_output - 1) * stride + ksize - input_size)
        tmp_pad_before = tmp_pad_need // 2
        tmp_pad_after = tmp_pad_need - tmp_pad_before
    return tmp_output, tmp_pad_before, tmp_pad_after


@constexpr
def _get_mean_matrix(x_shape, ksize, stride, padding, x_dtype):
    """
    helper func for AvgPoolGrad.

    `assist_input_matrix` is a 2d matrix with input_shape after padding,
    the value of element which is padded is 0, else are 1.
    For each element of output, it is mapped for slide window: `[h*h_stride : h*h_stride + h_ksize,
    w*w_stride : w*w_stride + w_ksize]` of `assist_input_matrix`, so the sum of slide window is the
    number of input that assosiate with output element.
    """

    n_input, c_input, h_input, w_input = x_shape
    h_ksize, w_ksize = ksize[2], ksize[3]
    h_stride, w_stride = stride[2], stride[3]
    n_output = n_input
    c_output = c_input
    h_output, w_output = 0, 0
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    h_output, pad_top, pad_bottom = _windowed_output_size(h_input, h_ksize,
                                                          h_stride, padding)
    w_output, pad_left, pad_right = _windowed_output_size(w_input, w_ksize,
                                                          w_stride, padding)

    output_size = n_output * c_output * h_output * w_output
    output_shape = (n_output, c_output, h_output, w_output)
    output = np.array([0.0] * output_size)
    output = np.reshape(output, output_shape)

    in_shape_after_padding_2d = (h_input + pad_top + pad_bottom, w_input + pad_left + pad_right)
    assist_input_matrix = np.ones(in_shape_after_padding_2d).astype(np.float32)
    if pad_top > 0:
        assist_input_matrix[:pad_top, :] = 0
    if pad_bottom > 0:
        assist_input_matrix[-pad_bottom:, :] = 0
    if pad_left > 0:
        assist_input_matrix[:, :pad_left] = 0
    if pad_right > 0:
        assist_input_matrix[:, -pad_right:] = 0

    for h in range(h_output):
        for w in range(w_output):
            curr_input = assist_input_matrix[h*h_stride : h*h_stride + h_ksize, w*w_stride : w*w_stride + w_ksize]
            curr_sum = np.sum(curr_input)
            if curr_sum > 0:
                output[:, :, h, w] = 1. / curr_sum
    return Tensor(output, x_dtype)


@constexpr
def _get_kernel_matrix(x_shape_nchw, kernel_matrix_shape, padding, x_dtype):
    kernel_matrix = np.ones(kernel_matrix_shape)
    return Tensor(kernel_matrix, x_dtype)


@bprop_getters.register(P.AvgPool)
def get_bprop_avg_pool_grad(self):
    """Grad definition for `AvgPool` operation."""

    # the parameter of AvgPoolGrad in GPU and TBE/CPU is not same
    if self.target == "GPU":
        avgpool_grad_gpu = G.AvgPoolGradGpu(
                        ksize=self.ksize,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.format)

        def bprop_gpu(x, out, dout):
            dx = avgpool_grad_gpu(x, out, dout)
            return (dx,)

        bprop_fn = bprop_gpu

    elif self.target == "GE":
        avgpool_grad_ge = G.AvgPoolGrad(
                        ksize=self.ksize,
                        strides=self.strides,
                        padding=self.padding)
        shape_op = P.Shape()

        def bprop_ge(x, out, dout):
            dx = avgpool_grad_ge(shape_op(x), dout)
            return (dx,)

        bprop_fn = bprop_ge

    else:
        avgpool_grad_vm = G.AvgPoolGradVm(
                        ksize=self.ksize,
                        strides=self.strides,
                        padding=self.padding)
        k_size_nchw = avgpool_grad_vm.ksize
        stride_nchw = avgpool_grad_vm.strides
        padding = self.padding

        def bprop_vm(x, out, dout):
            x_shape_nchw = F.shape(x)
            x_dtype = F.dtype(x)
            kernel_matrix_shape = (1, x_shape_nchw[1],
                                   k_size_nchw[2],
                                   k_size_nchw[3])
            mean_matrix = _get_mean_matrix(x_shape_nchw, k_size_nchw, stride_nchw, padding, x_dtype)
            kernel_matrix = _get_kernel_matrix(x_shape_nchw, kernel_matrix_shape, padding, x_dtype)
            dx = avgpool_grad_vm(x_shape_nchw, dout, mean_matrix, kernel_matrix)
            return (dx,)

        bprop_fn = bprop_vm

    return bprop_fn


@bprop_getters.register(P.DropoutGenMask)
def get_bprop_dropout_gen_mask(self):
    """Grad definition for `DropoutGenMask` operation."""

    def bprop(shape, keep_prob, out, dout):
        return (zeros_like(shape), zeros_like(keep_prob))

    return bprop


@bprop_getters.register(P.DropoutDoMask)
def get_bprop_dropout_do_mask(self):
    """Grad definition for `DropoutDoMask` operation."""
    do_mask = P.DropoutDoMask()

    def bprop(x, y, keep_prob, out, dout):
        return (do_mask(dout, y, keep_prob), zeros_like(y), zeros_like(keep_prob))

    return bprop


@bprop_getters.register(P.ReLU)
def get_bprop_relu(self):
    """Grad definition for `ReLU` operation."""
    input_grad = G.ReluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, out)
        return (dx,)

    return bprop


@bprop_getters.register(P.ReLU6)
def get_bprop_relu6(self):
    """Grad definition for `ReLU6` operation."""
    input_grad = G.ReLU6Grad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.ReLUV2)
def get_bprop_relu_v2(self):
    """Grad definition for `ReLUV2` operation."""
    input_grad = G.ReluGradV2()

    def bprop(x, out, dout):
        mask = out[1]
        dx = input_grad(dout[0], mask)
        return (dx,)

    return bprop


@bprop_getters.register(P.HSwish)
def get_bprop_hswish(self):
    """Grad definition for `HSwish` operation."""
    input_grad = G.HSwishGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.HSigmoid)
def get_bprop_hsigmoid(self):
    """Grad definition for `HSigmoid` operation."""
    input_grad = G.HSigmoidGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.Elu)
def get_bprop_elu(self):
    """Grad definition for `Elu` operation."""
    input_grad = G.EluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, out)
        return (dx,)

    return bprop


@bprop_getters.register(P.Sigmoid)
def get_bprop_sigmoid(self):
    """Grad definition for `Sigmoid` operation."""
    input_grad = G.SigmoidGrad()

    def bprop(x, out, dout):
        dx = input_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Softmax)
def get_bprop_softmax(self):
    """Grad definition for `Softmax` operation."""
    sum_func = P.ReduceSum(keep_dims=True)
    sub = P.Sub()
    mul = P.Mul()
    axis = self.axis

    def bprop(x, out, dout):
        dx = mul(out, sub(dout, sum_func(mul(out, dout), axis)))
        return (dx,)

    return bprop


@bprop_getters.register(P.LogSoftmax)
def get_bprop_log_softmax(self):
    """Grad definition for `LogSoftmax` operation."""
    logsoftmax_grad = G.LogSoftmaxGrad(self.axis)

    def bprop(x, out, dout):
        dx = logsoftmax_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Softplus)
def get_bprop_softplus(self):
    """Grad definition for `Softplus` operation."""
    softplus_grad = G.SoftplusGrad()

    def bprop(x, out, dout):
        dx = softplus_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.Softsign)
def get_bprop_softsign(self):
    """Grad definition for `Softsign` operation."""
    mul = P.Mul()
    absolute = P.Abs()
    div = P.Div()
    square = P.Square()

    def bprop(x, out, dout):
        dx = mul(dout, div(1, square(1 + absolute(x))))
        return (dx,)

    return bprop


@bprop_getters.register(P.Tanh)
def get_bprop_tanh(self):
    """Grad definition for `Tanh` operation."""
    tanh_grad = SG.TanhGrad()

    def bprop(x, out, dout):
        dx = tanh_grad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Gelu)
def get_bprop_gelu(self):
    """Grad definition for `Gelu` operation."""
    input_grad = G.GeluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x, out)
        return (dx,)

    return bprop


@bprop_getters.register(P.FusedBatchNorm)
def get_bprop_fused_batch_norm(self):
    """Grad definition for `FusedBatchNorm` operation."""
    input_grad = G.FusedBatchNormGrad(self.epsilon, self.momentum)
    target_cpu = False
    if self.target == "CPU":
        input_grad = G.FusedBatchNormGradCPU(self.epsilon, self.momentum)
        target_cpu = True
    def bprop(x, scale, b, mean, variance, out, dout):
        saved_mean = out[3]
        saved_variance = out[4]
        if target_cpu:
            out = input_grad(dout[0], x, scale, b, saved_mean, saved_variance)
        else:
            out = input_grad(dout[0], x, scale, saved_mean, saved_variance)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)

    return bprop


@bprop_getters.register(P.FusedBatchNormEx)
def get_bprop_fused_batch_norm_ex(self):
    """Grad definition for `FusedBatchNormEx` operation."""
    input_grad = G.FusedBatchNormGradEx(self.epsilon, self.momentum, self.format)

    def bprop(x, scale, b, mean, variance, out, dout):
        saved_mean = out[3]
        saved_variance = out[4]
        reserve = out[5]
        out = input_grad(dout[0], x, scale, saved_mean, saved_variance, reserve)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)

    return bprop


@bprop_getters.register(P.BatchNorm)
def get_bprop_batch_norm(self):
    """Grad definition for `BatchNorm` operation."""
    is_training = self.is_training
    input_grad = G.BatchNormGrad(is_training, self.epsilon)

    def bprop(x, scale, b, mean, variance, out, dout):
        if is_training:
            saved_reserve_1 = out[3]
            saved_reserve_2 = out[4]
        else:
            saved_reserve_1 = mean
            saved_reserve_2 = variance
        out = input_grad(dout[0], x, scale, saved_reserve_1, saved_reserve_2)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)

    return bprop


@bprop_getters.register(P.LayerNorm)
def get_bprop_layer_norm(self):
    """Grad definition for `LayerNorm` operation."""
    layer_norm_grad = G.LayerNormGrad(self.begin_norm_axis, self.begin_params_axis)

    def bprop(x, gamma, beta, out, dout):
        dx, d_gamma, d_beta = layer_norm_grad(
            x, dout[0], out[2], out[1], gamma)
        return dx, d_gamma, d_beta

    return bprop


@bprop_getters.register(P.L2Normalize)
def get_bprop_l2normalize(self):
    """Grad definition for `L2Normalize` operation."""
    input_grad = G.L2NormalizeGrad(self.axis, self.epsilon)

    def bprop(x, out, dout):
        dx = input_grad(x, out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.SoftmaxCrossEntropyWithLogits)
def get_bprop_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SoftmaxCrossEntropyWithLogits` operation."""
    expand = P.ExpandDims()

    def bprop(logits, labels, out, dout):
        grad = out[1]
        grad = grad * expand(dout[0], -1)
        return grad, zeros_like(labels)

    return bprop


@bprop_getters.register(P.SparseSoftmaxCrossEntropyWithLogits)
def get_bprop_sparse_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SparseSoftmaxCrossEntropyWithLogits` operation."""
    is_grad = self.is_grad
    grad_op = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=True)

    def bprop(logits, labels, out, dout):
        grad = out[0]
        if not is_grad:
            # if construct use loss
            grad = grad_op(logits, labels)
            grad = F.depend(grad, out)
            grad = grad * dout
        return grad, zeros_like(labels)

    return bprop


@bprop_getters.register(P.ResizeBilinear)
def get_bprop_resize_bilinear(self):
    """Grad definition for `ResizeBilinear` operation."""
    resize_grad = G.ResizeBilinearGrad(self.align_corners)

    def bprop(x, out, dout):
        dx = resize_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.OneHot)
def get_bprop_onehot(self):
    """Grad definition for `OneHot` operation."""

    def bprop(indices, depth, on_value, off_value, out, dout):
        return zeros_like(indices), zeros_like(depth), zeros_like(on_value), zeros_like(off_value)

    return bprop


@constexpr
def _range_op(start, limit, delta, dtype):
    """helper function for Grad TopK"""
    output_tensor = Tensor(list(range(start, limit, delta)), dtype)
    return output_tensor

@constexpr
def _get_1d_shape(in_shape):
    """helper function for Grad TopK"""
    out_shape = 1
    for i in in_shape:
        out_shape *= i
    return (out_shape,)

@bprop_getters.register(P.TopK)
def get_bprop_top_kv2(self):
    """Grad definition for `TopK` operation."""
    scatter = P.ScatterNd()
    expand_dims = P.ExpandDims()
    shape_op = P.Shape()
    reshape_op = P.Reshape()
    dtype = P.DType()

    def bprop(input_x, k, out, dout):

        in_shape = shape_op(input_x)
        in_lastdim = in_shape[-1]

        indices = out[1]
        ind_shape = shape_op(indices)
        ind_lastdim = ind_shape[-1]

        ind_2d = reshape_op(indices, (-1, ind_lastdim))
        outerdim = shape_op(ind_2d)[0]

        # [0, outterdim, 2*outerdim, ..., (k-1)*outerdim]
        indices_dtype = dtype(indices)
        range_flatten_index = _range_op(0, outerdim * in_lastdim, in_lastdim, indices_dtype)

        # expand_dims to (k, 1), then broadcast
        ind = reshape_op(ind_2d + expand_dims(range_flatten_index, -1), (-1,))
        in_shape_1d = _get_1d_shape(in_shape)

        out_grad = reshape_op(
            scatter(
                expand_dims(ind, -1),
                reshape_op(dout[0], (-1,)),
                in_shape_1d),
            in_shape)
        return out_grad, zeros_like(k)

    return bprop


@bprop_getters.register(P.SmoothL1Loss)
def get_bprop_smooth_l1_loss(self):
    """Grad definition for `SmoothL1Loss` operation."""
    grad = G.SmoothL1LossGrad(self.beta)

    def bprop(prediction, target, out, dout):
        dx = grad(prediction, target, dout)
        dy = grad(target, prediction, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.L2Loss)
def get_bprop_l2_loss(self):
    """Grad definition for `L2Loss` operation."""

    def bprop(x, out, dout):
        dx = x * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.RNNTLoss)
def get_bprop_rnnt_loss(self):
    """Grad definition for `RNNTLoss` operation."""

    def bprop(acts, labels, act_lens, label_lens, out, dout):
        grad = out[1]
        return grad, zeros_like(labels), zeros_like(act_lens), zeros_like(label_lens)
    return bprop


@bprop_getters.register(P.PReLU)
def get_bprop_prelu(self):
    """Grad definition for `PReLU` operation."""
    grad = G.PReLUGrad()

    def bprop(x, w, out, dout):
        dx, dw = grad(dout, x, w)
        return dx, dw

    return bprop


@bprop_getters.register(P.LSTM)
def get_bprop_lstm(self):
    """Grad definition for `LSTM` operation."""
    lstm_grad_data = G.LSTMGradData(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    lstm_grad_weight = G.LSTMGradWeight(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )
    lstm_grad = G.LSTMGrad(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    def bprop(x, hx, cx, w, out, dout):
        y, _, _, reserve, state = out
        dy, dhy, dcy, _, _ = dout
        dx, dhx, dcx = lstm_grad_data(y, dy, dhy, dcy, w, hx, cx, reserve, state)
        dw = lstm_grad_weight(F.depend(x, dx), hx, y, reserve, state)
        return dx, dhx, dcx, dw

    #
    def bprop_cpu(x, hx, cx, w, out, dout):
        y, hy, cy, reserve, _ = out
        dy, dhy, dcy, _, _ = dout
        dx, dhx, dcx, dw = lstm_grad(x, hx, cx, w, y, hy, cy, dy, dhy, dcy, reserve)
        return dx, dhx, dcx, dw

    if context.get_context('device_target') == "CPU":
        return bprop_cpu

    return bprop


@bprop_getters.register(P.DynamicRNN)
def get_bprop_dynamic_rnn(self):
    """Grad definition for `DynamicRNN` operation."""
    dynamic_rnn_grad = G.DynamicRNNGrad(cell_type=self.cell_type,
                                        direction=self.direction,
                                        cell_depth=self.cell_depth,
                                        use_peephole=self.use_peephole,
                                        keep_prob=self.keep_prob,
                                        cell_clip=self.cell_clip,
                                        num_proj=self.num_proj,
                                        time_major=self.time_major,
                                        forget_bias=self.forget_bias)
    expand_dims = P.ExpandDims()

    def bprop(x, w, b, seq_length, init_h, init_c, out, dout):
        dy, dh, dc, _, _, _, _, _, = dout
        dh = dh[-1]
        dc = dc[-1]
        y, h, c, i, j, f, o, tanhct = out
        dw, db, dx, dh_prev, dc_prev = dynamic_rnn_grad(x, w, b, y, init_h[0], init_c[0], h,
                                                        c, dy, dh, dc, i, j, f, o, tanhct)
        dh_prev = expand_dims(dh_prev, 0)
        dc_prev = expand_dims(dc_prev, 0)
        return dx, dw, db, (0), dh_prev, dc_prev
    return bprop


@bprop_getters.register(inner.DynamicGRUV2)
def get_bprop_dynamic_gru_v2(self):
    """Grad definition for `DynamicGRUV2` operation."""
    dynamic_gru_v2_grad = G.DynamicGRUV2Grad(self.direction, self.cell_depth, self.keep_prob, self.cell_clip,
                                             self.num_proj, self.time_major, 'double_bias', self.gate_order,
                                             self.reset_after)

    def bprop(x, winput, whidden, binput, bhidden, seq, init_h, out, dout):
        y, out_h, update, reset, new, hidden_new = out
        dy, dout_h, _, _, _, _ = dout

        dw_input, dw_hidden, db_input, db_hidden, dx, dh_prev = dynamic_gru_v2_grad(x, winput, whidden, y, init_h,
                                                                                    out_h, dy, dout_h[-1], update,
                                                                                    reset, new, hidden_new, None, None)
        return dx, dw_input, dw_hidden, db_input, db_hidden, (0), dh_prev
    return bprop


@bprop_getters.register(P.SigmoidCrossEntropyWithLogits)
def get_bprop_sigmoid_crossentropy_with_logits(self):
    """Grad definition for `SigmoidCrossEntropyWithLogits` operation."""
    op = G.SigmoidCrossEntropyWithLogitsGrad()

    def bprop(x, y, out, dout):
        dx = op(x, y, dout)
        return (dx, zeros_like(y))

    return bprop


@bprop_getters.register(P.Pad)
def get_bprop_pad(self):
    """Grad definition for `Pad` operation."""
    shape_op = P.Shape()
    paddings = self.paddings

    def bprop(x, out, dout):
        begin = ()
        for item in paddings:
            begin += (item[0],)
        shp = shape_op(x)
        dx = P.Slice()(dout, begin, shp)
        return (dx,)

    return bprop


@bprop_getters.register(P.MirrorPad)
def get_bprop_mirror_pad(self):
    """Grad definition for `MirrorPad` operation."""
    mirror_pad_grad = G.MirrorPadGrad(self.mode)

    def bprop(x, paddings, out, dout):
        dx = mirror_pad_grad(dout, paddings)
        return (dx, zeros_like(paddings))

    return bprop


@bprop_getters.register(P.ROIAlign)
def get_bprop_roi_align(self):
    """Grad definition for `ROIAlign` operation."""
    shape_op = P.Shape()
    pooled_height = self.pooled_height
    pooled_width = self.pooled_width
    spatial_scale = self.spatial_scale
    sample_num = self.sample_num

    def bprop(inputs, rois, out, dout):
        inputs_shape = shape_op(inputs)
        dx = G.ROIAlignGrad(inputs_shape,
                            pooled_height,
                            pooled_width,
                            spatial_scale,
                            sample_num,
                            )(dout, rois)
        return dx, zeros_like(rois)

    return bprop


@bprop_getters.register(P.Conv2DBackpropInput)
def get_bprop_conv2d_backprop_input(self):
    """Grad definition for `Conv2DBackpropInput` operation."""
    filter_grad = G.Conv2DBackpropFilter(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )
    input_grad = P.Conv2D(
        self.out_channel, self.kernel_size, pad_mode=self.pad_mode.lower(), pad=self.pad,
        dilation=self.dilation, stride=self.stride, group=self.group, data_format=self.format
    )

    def bprop(x, w, f_sizes, out, dout):
        dx = input_grad(dout, w)
        dw = filter_grad(x, dout, F.shape(w))
        return dx, dw, zeros_like(f_sizes)

    return bprop


@bprop_getters.register(P.BinaryCrossEntropy)
def get_bprop_binary_cross_entropy(self):
    """Grad definition for `BinaryCrossEntropy` operation."""
    grad = G.BinaryCrossEntropyGrad(self.reduction)

    def bprop(x, y, weight, out, dout):
        dx = grad(x, y, dout, weight)
        return dx, zeros_like(y), zeros_like(weight)

    return bprop

@bprop_getters.register(P.KLDivLoss)
def get_bprop_kl_div_loss(self):
    """Grad definition for `KLDivLoss` operation."""
    grad = G.KLDivLossGrad(self.reduction)

    def bprop(x, y, out, dout):
        dx, dy = grad(x, y, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.Dropout)
def get_bprop_dropout(self):
    """Grad definition for `Dropout` operation."""
    grad = G.DropoutGrad(self.keep_prob)

    def bprop(x, out, dout):
        _, mask = out
        dy, _ = dout
        dx = grad(dy, mask)
        return (dx,)

    return bprop


@bprop_getters.register(P.CTCLoss)
def get_bprop_ctc_loss(self):
    """Grad definition for `CTCLoss` operation"""
    expand = P.ExpandDims()

    def bprop(inputs, labels_indices, labels_values, sequence_length, out, dout):
        grad_loss = out[1]
        grad = grad_loss * expand(dout[0], -1)
        return grad, zeros_like(labels_indices), zeros_like(labels_values), zeros_like(sequence_length)

    return bprop


@bprop_getters.register(P.BasicLSTMCell)
def get_bprop_basic_lstm_cell(self):
    """Grad definition for `BasicLSTMCell` operation."""
    basic_lstm_cell_cstate_grad = G.BasicLSTMCellCStateGrad(
        forget_bias=self.forget_bias,
        activation=self.activation
    )

    basic_lstm_cell_weight_grad = G.BasicLSTMCellWeightGrad()

    basic_lstm_cell_input_grad = G.BasicLSTMCellInputGrad(keep_prob=self.keep_prob)

    def bprop(x, h, c, w, b, out, dout):
        _, _, it, jt, ft, ot, tanhct = out
        dct, dht, _, _, _, _, _ = dout
        dgate, dct_1 = basic_lstm_cell_cstate_grad(c, dht, dct, it, jt, ft, ot, tanhct)
        dxt, dht = basic_lstm_cell_input_grad(dgate, w)
        dw, db = basic_lstm_cell_weight_grad(F.depend(x, dxt), h, dgate)
        return dxt, dht, dct_1, dw, db
    return bprop


@bprop_getters.register(P.LRN)
def get_bprop_lrn(self):
    """Grad definition for `LRN` operation."""
    grad = G.LRNGrad(self.depth_radius, self.bias, self.alpha, self.beta)

    def bprop(x, out, dout):
        dx = grad(dout, x, out)
        return (dx,)

    return bprop
