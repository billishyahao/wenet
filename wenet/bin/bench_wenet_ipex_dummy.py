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

from torch import nn

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
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        model = torch.jit.trace(model, inputs, strict=False).eval()
        model = torch.jit.freeze(model)

        print("model has been frozen...")
        print(model)

        # warm up
        for i in range(5):
            out = model(inputs)

        print("hebi-dbg: model graph for")
        print(model.graph_for(inputs))
        
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

    # model = init_model(configs)
    # load_checkpoint(model, args.checkpoint)
    # model.eval()

    # script_model = torch.jit.script(model)
    # speech = torch.randint(1, 100, (1, 71, 80))
    # print(speech)
    # speech_lengths = torch.tensor([71])
    # text = torch.randint(1, 100, (1, 71))
    # print(text)
    # text_lengths = torch.tensor([1])
    # print(text_lengths.dim())
    # print(text_lengths.shape)
    # inputs = (speech, speech_lengths, text, text_lengths)
    # print("before script")
    # script_model(*inputs)
    # print("after running script model")



    class FakeModel1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(256, 256, 8, stride=1, groups=256)
            self.conv2d = nn.Conv2d(256, 256, 1, stride=1)

        def forward(self, x):
            out = self.conv1d(x) # [1, 256, 17]
            print(out.shape)
            out = out.view(1, 256, 1, -1)
            return self.conv2d(out)

    class FakeModel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(256, 256, 8, groups=2)
            self.conv2d = nn.Conv2d(256, 256, 1)

        def forward(self, x):
            out = self.conv1d(x) # [1, 256, 17]
            print(out.shape)
            out = out.view(1, 256, 1, -1)
            return self.conv2d(out)
    
    class FakeModel3(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(256, 256, 8, groups=256)
            self.conv2d = nn.Conv2d(256, 6, kernel_size=(1,1), groups=1)

        def forward(self, x):
            out = self.conv1d(x) # [1, 256, 17]
            print(out.shape)
            out = out.view(1, 256, 1, -1)
            return self.conv2d(out)

    class ConvolutionModule(nn.Module):
        """ConvolutionModule in Conformer model."""
        def __init__(self,
                    channels: int,
                    kernel_size: int = 15,
                    activation: nn.Module = nn.ReLU(),
                    norm: str = "batch_norm",
                    causal: bool = False,
                    bias: bool = True):
            """Construct an ConvolutionModule object.
            Args:
                channels (int): The number of channels of conv layers.
                kernel_size (int): Kernel size of conv layers.
                causal (int): Whether use causal convolution or not
            """
            super().__init__()

            self.pointwise_conv1 = nn.Conv1d(
                channels,
                2 * channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
            # self.lorder is used to distinguish if it's a causal convolution,
            # if self.lorder > 0: it's a causal convolution, the input will be
            #    padded with self.lorder frames on the left in forward.
            # else: it's a symmetrical convolution
            if causal:
                padding = 0
                self.lorder = kernel_size - 1
            else:
                # kernel_size should be an odd number for none causal convolution
                assert (kernel_size - 1) % 2 == 0
                padding = (kernel_size - 1) // 2
                self.lorder = 0
            self.depthwise_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=padding,
                groups=channels,
                bias=bias,
            )

            assert norm in ['batch_norm', 'layer_norm']
            if norm == "batch_norm":
                self.use_layer_norm = False
                self.norm = nn.BatchNorm1d(channels)
            else:
                self.use_layer_norm = True
                self.norm = nn.LayerNorm(channels)

            self.pointwise_conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
            self.activation = activation

        def forward(
            self,
            x: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            cache: torch.Tensor = torch.zeros((0, 0, 0)),
        ):
            """Compute convolution module.
            Args:
                x (torch.Tensor): Input tensor (#batch, time, channels).
                mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                    (0, 0, 0) means fake mask.
                cache (torch.Tensor): left context cache, it is only
                    used in causal convolution (#batch, channels, cache_t),
                    (0, 0, 0) meas fake cache.
            Returns:
                torch.Tensor: Output tensor (#batch, time, channels).
            """
            # exchange the temporal dimension and the feature dimension
            x = x.transpose(1, 2)  # (#batch, channels, time)

            # mask batch padding
            if mask_pad.size(2) > 0:  # time > 0
                x.masked_fill_(~mask_pad, 0.0)

            if self.lorder > 0:
                if cache.size(2) == 0:  # cache_t == 0
                    x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
                else:
                    assert cache.size(0) == x.size(0)  # equal batch
                    assert cache.size(1) == x.size(1)  # equal channel
                    x = torch.cat((cache, x), dim=2)
                assert (x.size(2) > self.lorder)
                new_cache = x[:, :, -self.lorder:]
            else:
                # It's better we just return None if no cache is required,
                # However, for JIT export, here we just fake one tensor instead of
                # None.
                new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

            # GLU mechanism
            x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
            x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

            # 1D Depthwise Conv
            x = self.depthwise_conv(x)
            if self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.activation(self.norm(x))
            if self.use_layer_norm:
                x = x.transpose(1, 2)
            x = self.pointwise_conv2(x)
            # mask batch padding
            if mask_pad.size(2) > 0:  # time > 0
                x.masked_fill_(~mask_pad, 0.0)

            return x.transpose(1, 2), new_cache

    # model3 = FakeModel3()
    # inp3 = torch.randn(1, 256, 24)


    model4 = ConvolutionModule(channels= 256, kernel_size=1)
    inp4 = torch.randn(1,  24, 256)

    o = model4(inp4)

    # exit(1)

    # inp3 = torch.randn(1, 3, 24, 24)

    # o = model3(inp3)

    # exit(1)

   
    
    loop = 500

    inp = inp4
    model = model4.eval()
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
        do_ipex_bf16_measure(model, inp, loop=loop, profile=args.profile)





if __name__ == '__main__':
    main()
