# Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import benchmark_op


def ori_sparse_reshape_graph(slice_input, begin, newshape, pack_const):
    slice67_out = tf.strided_slice(
            slice_input,
            begin=begin,
            end=[0, 2],
            strides=[1, 1],
            begin_mask=1,
            end_mask=1,
            shrink_axis_mask=2
        )

    slice67_out = tf.reshape(slice67_out, [-1, 1])
    shape_out = tf.shape(slice67_out)
    slice57_out = tf.strided_slice(
        shape_out, 
        begin=[0],
        end=[1],
        strides=[1],
        shrink_axis_mask=1
    )
    
    const2 = pack_const
    input_shape = tf.stack([slice57_out, const2])
    input_shape = tf.cast(input_shape, tf.int64)

    range_out = tf.range(0, slice57_out, 1)
    range_out = tf.reshape(range_out, [-1, 1])
    range_out_64 = tf.cast(range_out, dtype=tf.int64)
    concat_out = tf.concat([range_out_64, slice67_out], axis=-1)
    
    values = np.arange(slice_input.shape[0], dtype=np.float32)
    
    sparse_tensor = tf.SparseTensor(
        indices=concat_out,
        values=values,
        dense_shape=input_shape
    )
    sparse_tensor_out = tf.sparse.reshape(sparse_tensor, newshape)
    return sparse_tensor_out.indices, sparse_tensor_out.dense_shape, concat_out


def opt_sparse_reshape_graph(slice_input, begin, newshape, pack_const):
    custom_out1, custom_out2 = gen_embedding_fused_ops.KPFusedSparseReshape(
            slice_input=slice_input,
            begin=begin,
            new_shape=newshape,
            pack_const=pack_const,
    )
    return custom_out1, custom_out2


class TestFusedSparseReshape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize"""
        cls.config = tf.compat.v1.ConfigProto()
        cls.config.intra_op_parallelism_threads = 16
        cls.config.inter_op_parallelism_threads = 1

        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()

    @classmethod
    def tearDownClass(cls):
        # cls.sess.close()
        return 

    def _run_kp_reshape_test(self, slice_shape, base_slice_input, base_begin, base_newshape, pack_const, num_runs):
        with tf.Graph().as_default():
            slice_input = tf.compat.v1.placeholder(tf.int64, shape=slice_shape, name="slice_input")
            begin = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="begin")
            newshape = tf.compat.v1.placeholder(tf.int64, shape=(2,), name="newshape")

            feed = {
                slice_input: base_slice_input,
                begin: base_begin,
                newshape: base_newshape
            }

            with tf.name_scope("ori"):
                out_ori1, out_ori2, out_ori3 = ori_sparse_reshape_graph(slice_input, begin, newshape, pack_const)
            with tf.name_scope("opt"):
                out_opt1, out_opt2 = opt_sparse_reshape_graph(slice_input, begin, newshape, pack_const)
            
            with tf.compat.v1.Session(config=self.config) as sess:
                out_ori_val1, out_ori_val2, out_ori_val3 = sess.run(
                    [out_ori1, out_ori2, out_ori3], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_ori
                )
                out_opt_val1, out_opt_val2 = sess.run(
                    [out_opt1,out_opt2], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_opt
                )
                
                # 功能测试
                np.testing.assert_array_equal(
                    out_opt_val1,
                    out_ori_val1,
                    err_msg="Segment count mismatch"
                )
                np.testing.assert_array_equal(
                    out_opt_val2,
                    out_ori_val2,
                    err_msg="Segment count mismatch"
                )
                
                benchmark_op(
                    sess,
                    feed,
                    [out_ori1, out_ori2, out_ori3],
                    [out_opt1, out_opt2],
                    self.run_options,
                    self.run_metadata_ori,
                    self.run_metadata_opt,
                    op_name="KPFusedSparseReshape",
                    start_op="ori/StridedSlice",
                    end_op="ori/SparseReshape",
                    num_runs=num_runs,
                    tag="------TF_origin-----"
                )


    def test_kp_sparse_reshape(self):
        base_slice_input = np.array([[0, 0], [0, 1], [1, 2], [3, 4]], dtype=np.int64)
        base_begin = [0, 1]
        base_newshape = [2, 4]
        pack_const = 2
        self._run_kp_reshape_test((4, 2), base_slice_input, base_begin, base_newshape, pack_const, num_runs=100)
        
    def test_kp_reshape_2(self):
        base_slice_input = np.array([[0, 1]], dtype=np.int64)
        base_begin = [0, 1]
        base_newshape = [-1, 1]
        pack_const = 1
        self._run_kp_reshape_test((1, 2), base_slice_input, base_begin, base_newshape, pack_const, num_runs=100)


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)