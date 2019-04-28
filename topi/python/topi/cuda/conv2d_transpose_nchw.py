# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Conv2d transpose template for cuda backend"""

import tvm
from tvm import autotvm

from .injective import _schedule_injective
from .tensor_intrin import dp4a
from .. import nn, generic
from ..util import equal_const_int, get_const_tuple, traverse_inline

@autotvm.task.register_topi_compute(nn.conv2d_transpose_nchw, ['cuda', 'gpu'], "direct")
def conv2d_transpose_nchw_cuda(cfg, Input, Filter, strides, padding, out_dtype):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]
    strides : tuple of two ints
        The spatial stride along height and width
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    out_dtype: str
        The output type. This is used in mixed precision

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_c, in_h, in_w = get_const_tuple(Input.shape)
    _, out_c, filter_h, filter_w = get_const_tuple(Filter.shape)
    stride_h, stride_w = strides

    # attach stride info to config, this is used in schedule space definition
    cfg.stride = strides

    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = nn.get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    FirstPad = nn.pad(Input,
                      [0, 0, (bpad_top + stride_h - 1) // stride_h,
                       (bpad_left + stride_w - 1) // stride_w],
                      [0, 0, (bpad_bottom + stride_h - 1) // stride_h,
                       (bpad_right + stride_w - 1) // stride_w], name='FirstPad')

    # remove extra padding introduced by dilatation
    border_h = (stride_h - bpad_top % stride_h) % stride_h
    border_w = (stride_w - bpad_left % stride_w) % stride_w

    # dilation stage
    data = FirstPad
    strides = [1, 1, stride_h, stride_w]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            _dilate(b, dc, h + dh + border_h, w + dw + border_w).astype(out_dtype) *
            Filter[dc, c, filter_h - 1 - dh, filter_w - 1 - dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output

@autotvm.task.register_topi_schedule(generic.schedule_conv2d_transpose_nchw,
                                     ['cuda', 'gpu'], 'direct')
def schedule_conv2d_transpose_nchw_cuda(cfg, outs):
    """TOPI Schedule callback for conv2d transpose operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The parameters for this template

    outs: Array of Tensor
        The computation graph description of conv2d transpose
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d transpose.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_transpose_nchw':
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])
            ##### space definition end #####

            if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, 'local')
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope('local')
                OL = conv

            # create cache stage
            s[pad_data].set_scope('shared')
            AA = pad_data
            WW = s.cache_read(kernel, 'shared', [OL])

            # tile and bind spatial axes
            n, f, y, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            bf = s[output].fuse(n, bf)
            s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
            s[output].bind(by, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vf, tvm.thread_axis("vthread"))
            s[output].bind(vy, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))
            s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
            s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
            s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
            s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
            s[OL].compute_at(s[output], tx)

            # tile reduction axes
            n, f, y, x = s[OL].op.axis
            rc, ry, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, ry, rx, rci, n, f, y, x)

            s[AA].compute_at(s[OL], rcm)
            s[WW].compute_at(s[OL], rcm)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, y, x = s[load].op.axis
                fused = s[load].fuse(n, f, y, x)
                tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
                ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
                tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
                s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
                s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.task.register_topi_compute(nn.conv2d_transpose_nchw, ['cuda', 'gpu'], "int8")
def conv2d_transpose_cuda_NCHWc_int8(cfg, Input, Filter, strides, padding, out_dtype):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]
    strides : tuple of two ints
        The spatial stride along height and width
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    out_dtype: str
        The output type. This is used in mixed precision

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    ic_block_factor = 4
    oc_block_factor = 4

    pre_computed = len(Filter.shape) == 6

    if not pre_computed:
        batch, in_c, in_h, in_w = get_const_tuple(Input.shape)
        in_c, out_c, filter_h, filter_w = get_const_tuple(Filter.shape)
        packed_data = tvm.compute((batch, in_c // ic_block_factor, in_h, in_w,
                                   ic_block_factor),
                                   lambda n, c, h, w, vc: Input[n, c * ic_block_factor + vc, h, w],
                                   name="packed_data")
        packed_kernel = tvm.compute((in_c // ic_block_factor, out_c // oc_block_factor, filter_h, filter_w, oc_block_factor,
                                     ic_block_factor),
                                    lambda ic_chunk, oc_chunk, kh, kw, oc_block, ic_block:
                                        Filter[ic_chunk * ic_block_factor + ic_block,
                                               oc_chunk * oc_block_factor + oc_block, kh, kw],
                                    name="packed_kernel")
    else:
        packed_data = Input
        packed_kernel = Filter

    stride_h, stride_w = strides
    batch, ic_chunk, in_h, in_w, ic_block = get_const_tuple(packed_data.shape)
    ic_chunk, oc_chunk, filter_h, filter_w, ic_block, oc_block = get_const_tuple(packed_kernel.shape)

    # attach stride info to config, this is used in schedule space definition
    cfg.stride = strides

    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = nn.get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    FirstPad = nn.pad(packed_data,
                      [0, 0, (bpad_top + stride_h - 1) // stride_h,
                       (bpad_left + stride_w - 1) // stride_w, 0],
                      [0, 0, (bpad_bottom + stride_h - 1) // stride_h,
                       (bpad_right + stride_w - 1) // stride_w, 0], name='pad_data')

    # remove extra padding introduced by dilatation
    border_h = (stride_h - bpad_top % stride_h) % stride_h
    border_w = (stride_w - bpad_left % stride_w) % stride_w

    # dilation stage
    data = FirstPad
    strides = [1, 1, stride_h, stride_w, 1]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    dcc = tvm.reduce_axis((0, ic_chunk), name='dc_chunk')
    dcb = tvm.reduce_axis((0, ic_block), name='dc_block')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    oshape = (batch, oc_chunk, out_h, out_w, oc_block)
    Deconv = tvm.compute(
        oshape,
        lambda b, oc_chunk, h, w, oc_block: tvm.sum(
            _dilate(b, dcc, h + dh + border_h, w + dw + border_w, dcb).astype('int32') *
            packed_kernel[dcc, oc_chunk, filter_h - 1 - dh, filter_w - 1 - dw, oc_block, dcb].astype('int32'),
            axis=[dcc, dh, dw, dcb]), name='Deconv')
    Output = tvm.compute(Deconv.shape, lambda *idx: Deconv(*idx).astype(out_dtype), tag='conv2d_transpose_NCHWc_int8', name='OutCast')

    return Output


_dp4a = dp4a('shared', 'shared', 'local')


@autotvm.task.register_topi_schedule(generic.schedule_conv2d_transpose_nchw,
                                     ['cuda', 'gpu'], 'int8')
def schedule_conv2d_transpose_NCHWc_int8_cuda(cfg, outs):
    """TOPI Schedule callback for conv2d transpose operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The parameters for this template

    outs: Array of Tensor
        The computation graph description of conv2d transpose
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d transpose.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_transpose_NCHWc_int8':
            print('Schedule')
            output = op.output(0)
            conv = op.input_tensors[0]
            packed_data, packed_kernel = conv.op.input_tensors

            if isinstance(packed_data.op, tvm.tensor.ComputeOp) and 'pad' in packed_data.op.tag:
                pad_data = packed_data
                packed_data = pad_data.op.input_tensors[0]
            else:
                pad_data = packed_data

            if autotvm.GLOBAL_SCOPE.in_tuning:
                s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
                s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
            else:
                if isinstance(packed_kernel.op, tvm.tensor.ComputeOp) and \
                    packed_kernel.name == 'packed_kernel':
                    # data and kernel are not pre-computed, schedule layout transform here
                    _schedule_injective(packed_data.op, s)
                    _schedule_injective(packed_kernel.op, s)

            ##### space definition begin #####
            n, f, y, x, c = s[conv].op.axis
            rc, ry, rc, rc_block = s[conv].op.reduce_axis
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])
            ##### space definition end #####

            s[conv].set_scope('local')
            OL = conv

            # handle bias
            if output.op not in s.outputs:
                s[output].compute_inline
                output = s.outputs[0].output(0)

            # create cache stage
            s[pad_data].compute_inline()
            # s[pad_data].set_scope('shared')
            AA = s.cache_read(pad_data, 'shared', [conv])
            WW = s.cache_read(packed_kernel, 'shared', [OL])

            # tile and bind spatial axes
            n, f, y, x, c = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            bf = s[output].fuse(n, bf)
            s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
            s[output].bind(by, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vf, tvm.thread_axis("vthread"))
            s[output].bind(vy, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))
            s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
            s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
            s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
            s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)

            s[OL].compute_at(s[output], tx)

            # tile reduction axes
            n, f, y, x, c = s[conv].op.axis
            rc, ry, rx, _ = s[conv].op.reduce_axis
            rco, rci = cfg['tile_rc'].apply(s, OL, rc)
            s[OL].reorder(rco, ry, rx, rci, n, f, y, x, c, rc_block)
            _, rc_block = s[conv].split(rc_block, factor=4)
            #s[conv].tensorize(rc_block, _dp4a)
            #s[conv].vectorize(rc_block)

            s[AA].compute_at(s[conv], rco)
            s[WW].compute_at(s[conv], rco)

            # cooperative fetching
            for load in [AA, WW]:
                fused = s[load].op.axis
                fused = s[load].fuse(*fused)
                tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
                ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
                tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
                s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
                s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(s, outs[0].op, _callback)

    return s
