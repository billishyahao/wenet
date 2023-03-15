# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os
import copy
import sys
import time

import torch
import yaml
import numpy as np

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model



def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--mode', required=True, help='benchmark mode')
    parser.add_argument('--profile', required=True, type=str, help='profile')
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def print_input_output_info(onnx_model, name, prefix="\t\t"):
    input_names = [node.name for node in onnx_model.graph.input]
    input_shapes = [[d.dim_value for d in node.type.tensor_type.shape.dim]
                    for node in onnx_model.graph.input]
    output_names = [node.name for node in onnx_model.graph.output]
    output_shapes = [[d.dim_value for d in node.type.tensor_type.shape.dim]
                     for node in onnx_model.graph.output]
    print("{}{} inputs : {}".format(prefix, name, input_names))
    print("{}{} input shapes : {}".format(prefix, name, input_shapes))
    print("{}{} outputs: {}".format(prefix, name, output_names))
    print("{}{} output shapes : {}".format(prefix, name, output_shapes))


def do_baseline_measure(model, inputs, loop=300, profile=False):
    # Case #1 baseline
    t1 = time.perf_counter()
    
    for i in range(loop):
        
        output = model(*inputs)
    
    t2 = time.perf_counter()

    ct = (t2-t1) * 1000.0 /loop
    print('Wenet ASR IPEX BF16 Average Inference Latency: %0.4f ms' % ct)
    print("Mean throughput (std dev): %.2f fps" % (1000/ct))

    if profile in ('true', 'y', 'yes'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            output = model(*inputs)
            
        # p.export_chrome_trace("trace.json")
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        print("\t\tbench baseline, done!")


def do_torchscript_measure(model, inputs, loop=300, profile=False):
    
  with torch.no_grad():
    script_module = torch.jit.trace(model, inputs, strict=False)
    freezed_script_module = torch.jit.freeze(script_module)

    print("model has been frozen...")
    print(model)

    # warm up
    for i in range(2):
        out = freezed_script_module(*inputs)
    print(freezed_script_module.graph_for(*inputs))
    exit()

    t1 = time.perf_counter()

    for i in range(loop):
        out = freezed_script_module(*inputs)
    
    t2 = time.perf_counter()

    ct = (t2-t1) * 1000.0 /loop
    print('Wenet ASR IPEX BF16 Average Inference Latency: %0.4f ms' % ct)
    print("Mean throughput (std dev): %.2f fps" % (1000/ct))

    if profile in ('true', 'y', 'yes'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            out = freezed_script_module(*inputs)
        
        # p.export_chrome_trace("trace.json")
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        print("\t\tbench torch script, done!")


def do_ipex_measure(model, inputs, loop=300, profile=False):
    import intel_extension_for_pytorch as ipex
    
    ipex.optimize(model, dtype=torch.float32, inplace=True, auto_kernel_selection=True)
    model = torch.jit.trace(model, inputs, strict=False).eval()
    model = torch.jit.freeze(model)

    print("model has been frozen...")
    print(model)

    # warm up
    for i in range(2):
        out = model(*inputs)

    t1 = time.perf_counter()

    for i in range(loop):
        out = model(*inputs)
    
    t2 = time.perf_counter()

    ct = (t2-t1) * 1000.0 /loop
    print('Wenet ASR IPEX BF16 Average Inference Latency: %0.4f ms' % ct)
    print("Mean throughput (std dev): %.2f fps" % (1000/ct))

    if profile in ('true', 'y', 'yes'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            out = model(*inputs)
        # p.export_chrome_trace("trace.json")
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        print("\t\tbench torch script encoder, done!")


def do_ipex_bf16_measure(model, inputs,  loop=300, profile=False):
    import intel_extension_for_pytorch as ipex

    with torch.cpu.amp.autocast(), torch.no_grad():
        ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        model = torch.jit.trace(model, inputs, strict=False).eval()
        model = torch.jit.freeze(model)

        print("model has been frozen...")
        print(model)

        # warm up
        for i in range(2):
            out = model(*inputs)

        print("hebi-dbg: model graph for")
        print(model.graph_for(*inputs))
        
        exit(1)

        t1 = time.perf_counter()

        for i in range(loop):
            out = model(*inputs)
        
        t2 = time.perf_counter()

        ct = (t2-t1) * 1000.0 /loop
        print('Wenet ASR IPEX BF16 Average Inference Latency: %0.4f ms' % ct)
        print("Mean throughput (std dev): %.2f fps" % (1000/ct))


        if profile in ('true', 'y', 'yes'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
                out = model(*inputs)
            # p.export_chrome_trace("trace.json")
            print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
            print("\t\tbench torch script encoder, done!")


def main():
    torch.manual_seed(777)
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    script_model = torch.jit.script(model)
    speech = torch.randint(1, 100, (1, 71, 80))
    print(speech)
    speech_lengths = torch.tensor([71])
    text = torch.randint(1, 100, (1, 71))
    print(text)
    text_lengths = torch.tensor([1])
    print(text_lengths.dim())
    print(text_lengths.shape)
    inputs = (speech, speech_lengths, text, text_lengths)
    print("before script")
    script_model(*inputs)
    print("after running script model")

    loop = 500

    
    # if args.mode == "baseline":
    #     print("entering do baseline measurement")
    #     do_baseline_measure(model, inputs, loop=loop)
    print(type(args.profile))
    print(args.profile)
    if args.mode == "torchscript":
        print("entering do torchscript measurement")
        do_torchscript_measure(model, inputs, loop=loop, profile=args.profile)
    elif args.mode == "ipex":
        print("entering do ipex measurement")
        do_ipex_measure(model, inputs, loop=loop, profile=args.profile)
    elif args.mode == "ipex_bf16":
        print("entering do ipex_bf16 measurement")
        do_ipex_bf16_measure(model, inputs, loop=loop, profile=args.profile)





if __name__ == '__main__':
    main()
