import tensorflow as tf
import numpy as np
import unittest

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import perf_run, generate_timeline, wrapper_sess

np.random.seed(140)


def opt_fused_embedding_padding_fast_graph(input0, input1, input2, input3):
        # execute custom op
        _, custom_out = gen_embedding_fused_ops.kp_fused_embedding_padding_fast(input0, input1, input2, input3)
        return custom_out
    
def opt_fused_embedding_padding_graph(input0, input1, input2, input3):
    # execute custom op
    _, custom_out = gen_embedding_fused_ops.kp_fused_embedding_padding(input0, input1, input2, input3)
    return custom_out

def ori_fused_embedding_padding_fast_graph(input0, input1, input2, input3):
    cast = tf.cast(input0, tf.int32)
    begin = tf.constant([0], dtype=tf.int32)
    end = tf.constant([1], dtype=tf.int32)
    strides = tf.constant([1], dtype=tf.int32)
    hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
    sub_out = hash_rows - input2
    const = tf.constant(10, dtype=tf.int32)
    pack = tf.stack([sub_out, const], axis=0)
    fill = tf.fill(pack, tf.constant(0, dtype=tf.float32))
    concat = tf.concat([input1, fill], 0)
    reshape = tf.reshape(concat, input3)
    shape_tensor = tf.shape(reshape)
    output = tf.strided_slice(shape_tensor, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
    return output

def ori_fused_embedding_padding_graph(input0, input1, input2, input3):
    cast = tf.cast(input0, tf.int32)
    begin = tf.constant([0], dtype=tf.int32)
    end = tf.constant([1], dtype=tf.int32)
    strides = tf.constant([1], dtype=tf.int32)
    hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
    sub_out = hash_rows - input2
    const = tf.constant(10, dtype=tf.int32)
    pack = tf.stack([sub_out, const], axis=0)
    fill = tf.fill(pack, tf.constant(0, dtype=tf.float32))
    concat = tf.concat([input1, fill], 0)
    output = tf.reshape(concat, input3)
    return output


class TestFusedEmbeddingPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize config"""
        cls.config = tf.compat.v1.ConfigProto()
        cls.config.intra_op_parallelism_threads = 16
        cls.config.inter_op_parallelism_threads = 16

        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()

    @classmethod
    def tearDownClass(cls):
        return
    
    def test_func_kp_fused_embedding_padding(self):
        # Create Graph
        with tf.Graph().as_default():
            input0 = tf.compat.v1.placeholder(tf.int64, shape=[2], name="input0")
            input1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name="input1")
            input2 = tf.compat.v1.placeholder(tf.int32, shape=[], name="input2")
            input3 = tf.compat.v1.placeholder(tf.int32, shape=[2], name="input3")
            """Initialize test data"""
            feed = {
                input0: np.array([6, 10]).astype(np.int64),
                input1: np.random.rand(4, 10).astype(np.float),
                input2: 4,
                input3: np.array([-1, 20]).astype(np.int32),
            }
            with tf.name_scope("ori"):
                out_ori = ori_fused_embedding_padding_graph(input0, input1, input2, input3)
            with tf.name_scope("opt"):
                out_opt = opt_fused_embedding_padding_graph(input0, input1, input2, input3)
        
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                ori_result = sess.run([out_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori)
                opt_result = sess.run([out_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt)

                np.testing.assert_array_equal(
                    ori_result,
                    opt_result,
                    err_msg="result mismatch"
                )

                from tensorflow.python.client import timeline
                tl_ori = timeline.Timeline(self.run_metadata_ori.step_stats)
                tl_opt = timeline.Timeline(self.run_metadata_opt.step_stats)
                ctf_ori = tl_ori.generate_chrome_trace_format()
                ctf_opt = tl_opt.generate_chrome_trace_format()

                with open("timeline_ori.json", "w") as f:
                    f.write(ctf_ori)
                with open("timeline_opt.json", "w") as f:
                    f.write(ctf_opt)
                
                # perftest
                perf_run(wrapper_sess(sess, [out_ori], feed), wrapper_sess(sess, [out_opt], feed_dict=feed), "KPFusedEmbeddingPadding")

    def test_func_kp_fused_embedding_padding_fast(self):
        # Create Graph
        with tf.Graph().as_default():
            input0 = tf.compat.v1.placeholder(tf.int64, shape=[2], name="input0")
            input1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name="input1")
            input2 = tf.compat.v1.placeholder(tf.int32, shape=[], name="input2")
            input3 = tf.compat.v1.placeholder(tf.int32, shape=[2], name="input3")
            """Initialize test data"""
            feed = {
                input0: np.array([6, 10]).astype(np.int64),
                input1: np.random.rand(4, 10).astype(np.float),
                input2: 4,
                input3: np.array([-1, 20]).astype(np.int32),
            }
            with tf.name_scope("ori"):
                out_ori = ori_fused_embedding_padding_fast_graph(input0, input1, input2, input3)
            with tf.name_scope("opt"):
                out_opt = opt_fused_embedding_padding_fast_graph(input0, input1, input2, input3)
        
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                ori_result = sess.run([out_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori)
                opt_result = sess.run([out_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt)

                np.testing.assert_array_equal(
                    ori_result,
                    opt_result,
                    err_msg="result mismatch"
                )

                generate_timeline(self.run_metadata_ori.step_stats, f"{self._testMethodName}_ori")
                generate_timeline(self.run_metadata_opt.step_stats, f"{self._testMethodName}_opt")

                # perftest
                perf_run(wrapper_sess(sess, [out_ori], feed), wrapper_sess(sess, [out_opt], feed_dict=feed), "KPFusedEmbeddingPaddingFast")

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)