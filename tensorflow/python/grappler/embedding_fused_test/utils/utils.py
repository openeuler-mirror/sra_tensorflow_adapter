import timeit
import json
import os

from tensorflow.python.client import timeline


def extract_op_dur(timeline_file, op_name):
    """从 timeline JSON 文件中提取指定算子(fusedOp)的耗时（μs）"""
    with open(f"timeline/{timeline_file}.json", "r") as f:
        trace_events = json.load(f)["traceEvents"]  # timeline.json的格式
    durations = [e["dur"] for e in trace_events if e.get("name") == op_name and "dur" in e]
    return durations[0]


def extract_op_total_time(timeline_file, start_op, end_op):
    """计算从 start_op 到 end_op 的总耗时（包含调度空隙）"""
    with open(f"timeline/{timeline_file}.json", "r") as f:
        trace_events = json.load(f)["traceEvents"]
    start_event = next(e for e in trace_events if e.get("args", {}).get("name") == start_op)  # 找到 timeline 里第一个 name 等于 start_op 的事件
    end_event   = next(e for e in trace_events if e.get("args", {}).get("name") == end_op)    # 找不到会报错
    start_time = start_event["ts"]
    end_time = end_event["ts"] + end_event["dur"]  # ts 是开始时间，dur是算子的持续时间
    return end_time - start_time


def benchmark_op(
    sess,
    feed,
    out_ori,
    out_opt,
    run_options,
    run_metadata_ori,
    run_metadata_opt,
    op_name,
    start_op,
    end_op,
    num_runs=500,
    tag="--------TF_origin---------"
):
    print("-" * 60)
    print("-" * 60)
    print("new test")

    total_times_ori = 0.0
    total_times_opt = 0.0

    for i in range(num_runs):
        # 执行原始算子
        sess.run(
            out_ori,
            feed_dict=feed,
            options=run_options,
            run_metadata=run_metadata_ori
        )
        # 执行优化后的算子
        sess.run(
            out_opt,
            feed_dict=feed,
            options=run_options,
            run_metadata=run_metadata_opt
        )

        # 生成 timeline 文件
        filename_ori = f"{op_name}_ori"
        filename_opt = f"{op_name}_opt"
        generate_timeline(run_metadata_ori.step_stats, filename_ori)
        generate_timeline(run_metadata_opt.step_stats, filename_opt)

        # 统计时延
        total_times_ori += extract_op_total_time(filename_ori, start_op, end_op)
        total_times_opt += extract_op_dur(filename_opt, op_name)

    # 计算平均值和加速比
    avg_ori = total_times_ori / num_runs
    avg_opt = total_times_opt / num_runs
    speedup = (avg_ori - avg_opt) / avg_ori * 100 if avg_ori > 0 else 0

    # 打印结果
    print(f"{tag}: {avg_ori:.4f} us per run")
    print(f"{op_name}: {avg_opt:.4f} us per run")
    print(f"improve: {speedup:.2f}%")


def perf_run(ori_func, opt_func, name, warmup=5, iters=5):
    print(f"\nWarmup ori: {warmup} iters")
    for _ in range(warmup):
        ori_func()
    
    print(f"Running performance test: ori {iters} iters")
    total_time = timeit.timeit(ori_func, number=iters)
    ori_avg_time = total_time / iters * 1000
    print(f"{name}: {ori_avg_time:.6f} ms per run")

    print(f"\nWarmup opt: {warmup} iters")
    for _ in range(warmup):
        opt_func()

    print(f"Running performance test: opt {iters} iters")
    total_time = timeit.timeit(opt_func, number=iters)
    opt_avg_time = total_time / iters * 1000
    print(f"{name}: {opt_avg_time:.6f} ms per run")

    improvement = (ori_avg_time - opt_avg_time) / ori_avg_time * 100
    print(f"improve: {improvement:.2f}%")


def generate_timeline(step_stats, filename):
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()

    with open(f"timeline/{filename}.json", "w") as f:
        f.write(ctf)


def wrapper_sess(sess, fetches, feed_dict=None, options=None, run_metadata=None):
    return lambda: sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)