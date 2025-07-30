import tensorflow as tf
import numpy as np
import unittest

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import perf_run, generate_timeline, wrapper_sess


def ori_fused_embedding_action_id_gather_graph(input0, input1, input2, input3):
    gather1 = tf.gather(input1, input0, axis=0)
    gather2 = tf.gather(gather1, input2, axis=0)
    pack1 = tf.stack([input3, 1680], axis=0)
    pack2 = tf.stack([input3, -1], axis=0)
    reshape = tf.reshape(gather2, pack2)
    fill = tf.fill(pack1, tf.constant(0, dtype=tf.float32))
    output = tf.concat([reshape, fill], axis=-1)
    return output


def opt_fused_embedding_action_id_gather_graph(input0, input1, input2, input3):
    output = gen_embedding_fused_ops.KPFusedEmbeddingActionIdGather(
        input0=input0,
        input1=input1,
        input2=input2,
        input3=input3,
    )
    return output


class TestFusedEmbeddingActionIdGather(unittest.TestCase):
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

    def test_kp_fused_embedding_action_id_gather(self):
        # Create Graph
        with tf.Graph().as_default():
            indices1_shape = (8, 10)
            indices2_shape = (5, 6)
            params_shape = (80, 300)
            input0 = tf.compat.v1.placeholder(tf.int64, shape=indices1_shape, name="input_0")
            input1 = tf.compat.v1.placeholder(tf.float32, shape=params_shape, name="input_1")
            input2 = tf.compat.v1.placeholder(tf.int32, shape=indices2_shape, name="input_2")
            input3 = tf.compat.v1.placeholder(tf.int32, shape=[], name="input_3")
            """Initialize test data"""
            feed = {
                input0: np.random.randint(0, params_shape[0], indices1_shape).astype(np.int64),
                input1: np.random.random(params_shape).astype(np.float32),
                input2: np.random.randint(0, indices1_shape[0], indices2_shape).astype(np.int32),
                input3: params_shape[0],
            }
            with tf.name_scope("ori"):
                out_ori = ori_fused_embedding_action_id_gather_graph(input0, input1, input2, input3)
            with tf.name_scope("opt"):
                out_opt = opt_fused_embedding_action_id_gather_graph(input0, input1, input2, input3)
        
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                out_ori_val = sess.run([out_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori)
                out_opt_val = sess.run([out_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt)

                np.testing.assert_array_equal(
                    out_ori_val,
                    out_opt_val,
                    err_msg="result mismatch"
                )

                generate_timeline(self.run_metadata_ori.step_stats, f"{self._testMethodName}_ori")
                generate_timeline(self.run_metadata_opt.step_stats, f"{self._testMethodName}_opt")

                # perftest
                perf_run(wrapper_sess(sess, [out_ori], feed_dict=feed), 
                         wrapper_sess(sess, [out_opt], feed_dict=feed), 
                         "KPFusedEmbeddingActionIdGather")
        

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)