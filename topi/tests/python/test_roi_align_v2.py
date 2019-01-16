"""Test code for vision package"""
import numpy as np
import tvm
import topi
import math

from topi.vision import ssd, nms


def verify_roi_align(batch_size, in_channel, size, num_roi, pooled_size, spatial_scale):
    data_shape = (batch_size, in_channel, size, size)
    rois_shape = (num_roi, 5)
    data=tvm.placeholder(data_shape)
    rois=tvm.placeholder(rois_shape)
    np_data = np.random.uniform(size=data_shape).reshape(data_shape).astype('float32') * size
    np_rois = np.random.uniform(size=rois_shape).astype('float32') * size
    np_rois[:, 0] = 0

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = topi.vision.rcnn.roi_align_v2(data, rois, pooled_size=pooled_size, spatial_scale=spatial_scale)
            s = topi.generic.schedule_roi_align_v2(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_rois = tvm.nd.array(np_rois, ctx)
        tvm_out = tvm.nd.array(np.zeros((num_roi, in_channel, pooled_size, pooled_size)).astype(out.dtype), ctx=ctx)
        f = tvm.build(s, [data, rois, out], device)
        f(tvm_data, tvm_rois, tvm_out)

        import mxnet
        mx_ctx = mxnet.gpu(0)
        mx_data = mxnet.nd.array(np_data, mx_ctx)
        mx_rois = mxnet.nd.array(np_rois, mx_ctx)
        mx_out = mxnet.nd.contrib.ROIAlign_v2(mx_data, mx_rois, pooled_size=(pooled_size, pooled_size), spatial_scale=spatial_scale)
        mx_out = mx_out.asnumpy()

        tvm_out = tvm_out.asnumpy()

        np.testing.assert_allclose(tvm_out, mx_out, rtol=1e-3)

    for device in ['cuda', 'llvm']:
        check_device(device)


def test_roi_align_v2():
    verify_roi_align(1, 1, 14, 64, 7, 1.)
    verify_roi_align(1, 1, 14, 64, 7, 0.5)


if __name__ == "__main__":
    test_roi_align_v2()
