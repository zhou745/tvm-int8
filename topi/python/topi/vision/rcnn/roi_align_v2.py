# pylint: disable=invalid-name
"""Roi align v2 in
   https://github.com/TuSimple/mxnet/blob/master/src/operator/contrib/roi_align_v2-inl.h"""

import tvm
from ...util import get_const_tuple


@tvm.target.generic_func
def roi_align_v2(data, rois, pooled_size, spatial_scale):
    """ROI align operator in NCHW layout.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    rois : tvm.Tensor
        2-D with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : int or list/tuple of two ints
        output size, or [out_height, out_width]

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [num_roi, channel, pooled_size, pooled_size]
    """
    _, channel, height, width = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)
    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    def _bilinear(i, c, y, x):
        y_low = y.astype('int32')
        x_low = x.astype('int32')
        y_high = tvm.min(tvm.ceil(y).astype('int32'), height - 1)
        x_high = tvm.min(tvm.ceil(x).astype('int32'), width - 1)
        y_lerp = y - y_low
        x_lerp = x - x_low
        bottom = x_lerp * data[i, c, y_high, x_high] + \
            (1-x_lerp) * data[i, c, y_high, x_low]
        top = x_lerp * data[i, c, y_low, x_high] + \
            (1-x_lerp) * data[i, c, y_low, x_low]
        return y_lerp * bottom + (1-y_lerp) * top

    def _sample(i, c, ph, pw):
        roi = rois[i]
        batch_index = roi[0].astype('int32')
        roi_start_w = roi[1] * spatial_scale
        roi_start_h = roi[2] * spatial_scale
        roi_end_w = roi[3] * spatial_scale
        roi_end_h = roi[4] * spatial_scale

        roi_h = roi_end_h - roi_start_h
        roi_w = roi_end_w - roi_start_w
        roi_h = roi_h
        roi_w = roi_w
        bin_h = roi_h / pooled_size_h
        bin_w = roi_w / pooled_size_w

        hstart = ph * bin_h
        wstart = pw * bin_w
        hend = (ph + 1) * bin_h
        wend = (pw + 1) * bin_w
        hstart = tvm.min(tvm.max(hstart + roi_start_h, 0), height-1)
        wstart = tvm.min(tvm.max(wstart + roi_start_w, 0), width-1)
        hend = tvm.min(tvm.max(hend + roi_start_h, 0), height-1)
        wend = tvm.min(tvm.max(wend + roi_start_w, 0), width-1)
        non_empty = tvm.all(hstart < hend, wstart < wend)

        def min_value(dtype):
            return tvm.expr.Select(non_empty, tvm.min_value(dtype), tvm.const(0.0, dtype))

        stride_h = (hend - hstart) / 3.0
        stride_w = (wend - wstart) / 3.0
        hstart += stride_h
        wstart += stride_w
        stride_h = tvm.max(0.01, stride_h)
        stride_w = tvm.max(0.01, stride_w)
        _max = tvm.comm_reducer(lambda x, y: tvm.make._OpMax(x, y), min_value, name='max')
        rh = tvm.reduce_axis((0, tvm.expr.Select(non_empty, 2, 0)), 'rh')
        rw = tvm.reduce_axis((0, tvm.expr.Select(non_empty, 2, 0)), 'rw')
        return _max(_bilinear(batch_index, c, hstart + rh*stride_h, wstart+rw*stride_w),
                    axis=[rh, rw])

    return tvm.compute((num_roi, channel, pooled_size_h, pooled_size_w), _sample,
                       tag='pool,roi_align_v2')
