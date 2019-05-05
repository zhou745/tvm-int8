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
#pylint: disable=unused-argument
"""Automatic quantization toolkit."""
from __future__ import absolute_import
import numpy as np

from . import _quantize
from .. import expr as _expr
from .. import ir_pass as _ir_pass
from .. import build_module as _build
from .. import op as _op
from ... import make as _make, context
from ..base import NodeBase, register_relay_node
from ...contrib import graph_runtime


class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3


def kind2str(kind):
    """Convert a `QAnnotateKind` to string"""
    str_map = {
        QAnnotateKind.INPUT: "input",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.ACTIVATION: "activation",
    }
    assert kind in str_map
    return str_map[kind]


@register_relay_node("relay.quantize.QConfig")
class QConfig(NodeBase):
    """Configure the quantization behavior by setting config variables.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use qconfig instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    _node_defaults = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "global_scale": 8.0,
        "skip_k_conv": 1,
        "round_for_shift": True,
        "store_lowbit_output": True,
        "debug_enabled_ops": None,
        "use_stop_fusion": True,
        "quantize_dense": True
    }

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(QConfig, self).__init__(handle)
        self.handle = handle

    def guard(self, ref_call):
        op_name = ref_call.op.name
        if self.debug_enabled_ops is not None:
            name_list = [x.value for x in self.debug_enabled_ops]
            if op_name not in name_list:
                return False
        return True

    def get_nbit_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, 'nbit_' + name)

    def get_dtype_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, 'dtype_' + name)

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope(self)

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError(
                "'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(QConfig, self).__setattr__(name, value)


def current_qconfig():
    """Get the current quantization configuration."""
    return _quantize._GetCurrentQConfig()


def qconfig(**kwargs):
    """Configure the quantization behavior by setting config variables.

    Parameters
    ---------
    nbit_dict: dict of QAnnotateKind -> int
        Number of bit for every kind of annotate field.

    global_scale: float
        The global scale for calibration.

    skip_k_conv: int
        The number of skipped conv2d.

    round_for_shift: boolean
        Whether to add bias for rounding during shift.

    store_lowbit_output: boolean
        Whether to store low-bit integer back as output before dequantizing.
        Some accelerators need this, e.g. VTA.

    use_stop_fusion: boolean
        Whether add stop_fusion when casting to dtype_activation. stop_fusion forces lowbit
        results to be stored in memory.

    quantize_dense: boolean
        Whether to quantize dense layers.

    Returns
    -------
    config: QConfig
        The quantization configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k]
                 for k, v in QConfig._node_defaults.items()}
    return _make.node("relay.quantize.QConfig", **node_args)


CONV_COUNTER = 0


def _conv_counter():
    """Get the global counter for conv2d."""
    return CONV_COUNTER


def _set_conv_counter(n):
    """Set the value of the global conv2d counter."""
    global CONV_COUNTER
    CONV_COUNTER = n


def annotate(graph):
    """Given a float32 graph, annotate will rewrite the graph
    and return back a graph which simulates the error brought by
    current quantization scheme.

    Parameters
    ---------
    graph: Function
        The original graph

    Returns
    -------
    ret: Function
        The graph after annotation
    """
    _set_conv_counter(0)  # reset counter
    return _quantize.annotate(graph)


def collect_stats(graph, dataset):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    quantized_exprs = []

    def visit_func(expr):
        """Internal visit function"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op and expr.attrs.kind != QAnnotateKind.WEIGHT:
            quantized_exprs.append(expr.args[0])

    _ir_pass.post_order_visit(graph, visit_func)
    graph = _expr.Function(graph.params, _expr.Tuple(quantized_exprs))

    graph_json, lib, params = _build.build(graph, 'cuda')
    module = graph_runtime.create(graph_json, lib, context('cuda', 0))
    module.set_input(**params)

    num_outputs = module.get_num_outputs()
    outputs = [[] for i in range(num_outputs)]

    for batch in dataset:
        module.set_input(**batch)
        module.run()
        for i in range(num_outputs):
            output = module.get_output(i).asnumpy()
            outputs[i].append(output)

    return [np.concatenate(arr) for arr in outputs]


from scipy import stats
def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def _get_optimal_threshold(arr, num_bins=8001, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    assert isinstance(arr, np.ndarray)
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2
    assert np.allclose(hist_edges[zero_bin_idx] + hist_edges[zero_bin_idx + 1],
                       0, rtol=1e-5, atol=1e-7)

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2,
                   num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    return min_val, max_val, min_divergence, opt_th
# pylint: enable=line-too-long


def calibrate(graph, dataset=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max for every `simulated_quantize`
    operator.

    Parameters
    ---------
    graph: Function
        The simulation graph after annotation.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after calibration
    """
    def power2_scale(arr):
        """calculate weight scale with nearest mode-2 scale"""
        if not isinstance(arr, np.ndarray):
            arr = arr.asnumpy()
        val = np.amax(np.abs(arr))
        #return(val if val > 1 else 1.0)
       	return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0

    def kld(arr):
        if not isinstance(arr, np.ndarray):
            arr = arr.asnumpy()
        _, _, _, val = _get_optimal_threshold(arr, num_bins=8001, num_quantized_bins=255)
        return val
        return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0

    #fcalib = power2_scale
    fcalib = kld

    cfg = current_qconfig()
    const_params = {}
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")

    outputs = None
    counter = [0]

    def visit_func(expr):
        """Internal visit function"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            nbit = cfg.get_nbit_by_kind(kind)

            valid_bit = nbit - attrs.sign

            if kind == QAnnotateKind.WEIGHT:
                var = expr.args[0]
                assert isinstance(var, _expr.Constant)
                scale = power2_scale(var.data)
            else:
                if outputs is not None:
                    data = outputs[counter[0]]
                    counter[0] += 1
                    print('{} / {} ...'.format(counter[0], len(outputs)))
                    scale = fcalib(data)
                    print(scale)
                else:
                    scale = cfg.global_scale

            def _make_const(val):
                return _expr.const(val, 'float32')

            valid_range = 2**valid_bit
            const_params[ndom_scale] = _make_const(scale / valid_range)
            const_params[nclip_min] = _make_const(- (valid_range - 1))
            const_params[nclip_max] = _make_const((valid_range - 1))

    _ir_pass.post_order_visit(graph, visit_func)
    original_graph = graph
    graph = _expr.bind(original_graph, const_params)

    if dataset is not None:
        print('Calibrating on dataset')
        outputs = collect_stats(graph, dataset)
        _ir_pass.post_order_visit(original_graph, visit_func)
        assert counter[0] == len(outputs)
        graph = _expr.bind(original_graph, const_params)

    return graph


def realize(graph):
    """The realize pass will transform the simulated quantized
    graph, which computes with float32 actually, to a real low-bit
    integer graph. It will replace the simulated_quantize with
    several fine-grained operators like add, multiply, and shift
    as more as possible for performance (fusion, etc.)

    Parameters
    ---------
    graph: Function
        The simulated graph after calibrating.

    Returns
    -------
    ret: Function
        The graph after realization
    """
    return _quantize.realize(graph)


def quantize(graph, params=None, dataset=None):
    """ The quantization procedure. Before running the three main
    procedure of quantization, "annotate", "calibrate" and "realize"
    , we need to do "SimplifyInference", "FoldScaleAxis", "FoldConstant"
    first for optimizing.

    Parameters
    ---------
    graph: Function
        The original graph.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """
    opt_passes = ["SimplifyInference",
                  "FoldScaleAxis",
                  "FoldConstant",
                  "CanonicalizeOps"]
    with _build.build_config(add_pass=opt_passes):
        graph = _build.optimize(graph, params=params)
    print("optimize finished")
    graph = annotate(graph)
    graph = calibrate(graph, dataset)
    print(graph)
    graph = realize(graph)
    graph = _ir_pass.fold_constant(graph)
    return graph
