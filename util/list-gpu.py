r"""List GPUs
"""
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    # Hardware info
    num_gpus = tf.contrib.eager.num_gpus()
    num_cpu_threads = os.cpu_count()
    local_device_protos = device_lib.list_local_devices()
    gpu_list = dict([(x.name, x.physical_device_desc) for x in local_device_protos if x.device_type == 'GPU'])
    print('GPUs: {}, CPU Threads: {}'.format(num_gpus, num_cpu_threads))
    print(gpu_list)
#     GPUs: 1, CPU Threads: 32
# {'/device:GPU:0': 'device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:41:00.0, compute capability: 6.1'}