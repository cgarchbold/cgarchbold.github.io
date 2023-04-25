---
id: 1
title: "Using Neural Style to Create Deep Art"
subtitle: "Quick tutorial on how to use the famous style transfer neural network"
date: "2019.12.08"
tags: "cnn, tensorflow"
---


# Using Neural Style to Create Deep Art

Hello all! In this notebook I will give a quick tutorial on how to use the famous style transfer neural network popularized by Stanford researchers to create your own stylized art. **No GPU needed!**



---


*How we will create these types of images without an expensive GPU?* Using Google's cloud computing service **Google Colab** and Google's cloud storage service **Google Drive**.

![Google Colaboratory](images/colab_favicon_256px.png)
![alt text](images/Google-Drive.ico)


# Sticking the Nueral-Style Github repo inside your Google Drive

https://github.com/anishathalye/neural-style

Here you will find a GitHub repository made by [anishathalye](https://github.com/anishathalye), feel free to check his stuff out! 

Download this GitHub repository (click clone or download), unzip and upload these files to a folder inside of your google drive.

Then hop onto [Colab](https://colab.research.google.com), and open your first Colab Notebook






# Importing Google Drive into Google Colab

Using the command below you can import your entire Google Drive into this Colab Notebook. Login with your google account and copy paste the code and you're ready to go!


```
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    

Then you can cd ("change directory") to make your working directory the directory you put the neural-style repo


```
%cd drive/My\ Drive/Data/NueralStyle/neural-style-master
```

    /content/drive/My Drive/Data/NueralStyle/neural-style-master
    

# Setting the Notebook to use a GPU

With Colab, Google offers a free GPU to use with image processing. This can be enabled in the Runtime tab at the top.


# Taking care of a Package Version

Colab comes prepackaged with all the things we need to run this neural-style algorithm in python. However, due when this project was created, current versions of the scipy package do not work. Therefore , we will install the old version.


```
pip install scipy==1.1.0
```

    Collecting scipy==1.1.0
    [?25l  Downloading https://files.pythonhosted.org/packages/a8/0b/f163da98d3a01b3e0ef1cab8dd2123c34aee2bafbb1c5bffa354cc8a1730/scipy-1.1.0-cp36-cp36m-manylinux1_x86_64.whl (31.2MB)
    [K     |████████████████████████████████| 31.2MB 1.7MB/s 
    [?25hRequirement already satisfied: numpy>=1.8.2 in /usr/local/lib/python3.6/dist-packages (from scipy==1.1.0) (1.17.3)
    [31mERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.[0m
    Installing collected packages: scipy
      Found existing installation: scipy 1.3.1
        Uninstalling scipy-1.3.1:
          Successfully uninstalled scipy-1.3.1
    Successfully installed scipy-1.1.0
    

# Looking at all the arguments you have to play with

The nueral_style.py takes a bunch of various command line arguments. You can add a new command line arguement with the -- modifier. I recommend you set up a checkpoint system with the following command line argument

```
--checkpoint-output output_{:05}.jpg --checkpoint-iterations 100
```
This will save a checkpoint image every hundred interations.

By default the neural-style runs 1000 interations, but you are free to change that as well. Typically good results occur between 500-2000 interations.

You will need to set your image for content you want to style with the command line argument (Here I give it mid.jpg which is a mid-size image of myself):
```
--content ./Nueral_Style_Images/mid.jpg
```
You will need to set your image for the style with the command line argument:
```
--style ./Nueral_Style_Images/starry_night_google.jpg
```
Lastly, you will need to specify an output with the command line arguement
```
--output ./Nueral_Style_Images/new.jpg
```



You can use the --help command line arguement to see a list of all available command line arguments to play with.






```
!python neural_style.py --help
```

    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/vgg.py:9: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/vgg.py:11: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    2019-11-01 18:06:01.480383: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
    2019-11-01 18:06:01.480744: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1bff2c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2019-11-01 18:06:01.480781: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2019-11-01 18:06:01.486252: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2019-11-01 18:06:01.628914: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:01.629894: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5852140 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2019-11-01 18:06:01.629928: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2019-11-01 18:06:01.631110: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:01.631830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:04.0
    2019-11-01 18:06:01.645802: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:06:01.837512: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:06:01.922926: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-11-01 18:06:01.948074: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-11-01 18:06:02.174672: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-11-01 18:06:02.309754: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-11-01 18:06:02.704266: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-11-01 18:06:02.704577: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:02.705415: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:02.706082: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-11-01 18:06:02.709398: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:06:02.710922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-11-01 18:06:02.710959: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-11-01 18:06:02.710988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-11-01 18:06:02.712165: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:02.713000: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:02.713731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10805 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    usage: neural_style.py [-h] --content CONTENT --styles STYLE [STYLE ...]
                           --output OUTPUT [--iterations ITERATIONS]
                           [--print-iterations PRINT_ITERATIONS]
                           [--checkpoint-output OUTPUT]
                           [--checkpoint-iterations CHECKPOINT_ITERATIONS]
                           [--progress-write] [--progress-plot] [--width WIDTH]
                           [--style-scales STYLE_SCALE [STYLE_SCALE ...]]
                           [--network VGG_PATH]
                           [--content-weight-blend CONTENT_WEIGHT_BLEND]
                           [--content-weight CONTENT_WEIGHT]
                           [--style-weight STYLE_WEIGHT]
                           [--style-layer-weight-exp STYLE_LAYER_WEIGHT_EXP]
                           [--style-blend-weights STYLE_BLEND_WEIGHT [STYLE_BLEND_WEIGHT ...]]
                           [--tv-weight TV_WEIGHT] [--learning-rate LEARNING_RATE]
                           [--beta1 BETA1] [--beta2 BETA2] [--eps EPSILON]
                           [--initial INITIAL]
                           [--initial-noiseblend INITIAL_NOISEBLEND]
                           [--preserve-colors] [--pooling POOLING] [--overwrite]
    
    optional arguments:
      -h, --help            show this help message and exit
      --content CONTENT     content image
      --styles STYLE [STYLE ...]
                            one or more style images
      --output OUTPUT       output path
      --iterations ITERATIONS
                            iterations (default 1000)
      --print-iterations PRINT_ITERATIONS
                            statistics printing frequency
      --checkpoint-output OUTPUT
                            checkpoint output format, e.g. output_{:05}.jpg or
                            output_%05d.jpg
      --checkpoint-iterations CHECKPOINT_ITERATIONS
                            checkpoint frequency
      --progress-write      write iteration progess data to OUTPUT's dir
      --progress-plot       plot iteration progess data to OUTPUT's dir
      --width WIDTH         output width
      --style-scales STYLE_SCALE [STYLE_SCALE ...]
                            one or more style scales
      --network VGG_PATH    path to network parameters (default imagenet-vgg-
                            verydeep-19.mat)
      --content-weight-blend CONTENT_WEIGHT_BLEND
                            content weight blend, conv4_2 * blend + conv5_2 *
                            (1-blend) (default 1)
      --content-weight CONTENT_WEIGHT
                            content weight (default 5.0)
      --style-weight STYLE_WEIGHT
                            style weight (default 500.0)
      --style-layer-weight-exp STYLE_LAYER_WEIGHT_EXP
                            style layer weight exponentional increase -
                            weight(layer<n+1>) = weight_exp*weight(layer<n>)
                            (default 1)
      --style-blend-weights STYLE_BLEND_WEIGHT [STYLE_BLEND_WEIGHT ...]
                            style blending weights
      --tv-weight TV_WEIGHT
                            total variation regularization weight (default 100.0)
      --learning-rate LEARNING_RATE
                            learning rate (default 10.0)
      --beta1 BETA1         Adam: beta1 parameter (default 0.9)
      --beta2 BETA2         Adam: beta2 parameter (default 0.999)
      --eps EPSILON         Adam: epsilon parameter (default 1e-08)
      --initial INITIAL     initial image
      --initial-noiseblend INITIAL_NOISEBLEND
                            ratio of blending initial image with normalized noise
                            (if no initial image specified, content image is used)
                            (default None)
      --preserve-colors     style-only transfer (preserving colors) - if color
                            transfer is not needed
      --pooling POOLING     pooling layer configuration: max or avg (default max)
      --overwrite           write file even if there is already a file with that
                            name
    

# Time To Try it Out!




```
!python neural_style.py --content ./Nueral_Style_Images/infest.jpg --styles ./Nueral_Style_Images/wierd.jpg --output ./Nueral_Style_Images/new.jpg --overwrite --checkpoint-output output_{:05}.jpg --checkpoint-iterations 100 --iterations 1000 --content-weight 25
```

    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/vgg.py:9: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/vgg.py:11: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    2019-11-01 18:06:31.021741: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
    2019-11-01 18:06:31.021992: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2d292c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2019-11-01 18:06:31.022035: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2019-11-01 18:06:31.024216: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2019-11-01 18:06:31.098985: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:31.100000: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x6974140 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2019-11-01 18:06:31.100034: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2019-11-01 18:06:31.100288: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:31.101074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:04.0
    2019-11-01 18:06:31.101430: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:06:31.102759: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:06:31.104116: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-11-01 18:06:31.104709: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-11-01 18:06:31.106709: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-11-01 18:06:31.108581: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-11-01 18:06:31.112997: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-11-01 18:06:31.113101: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:31.113911: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:31.114536: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-11-01 18:06:31.114624: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:06:31.116203: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-11-01 18:06:31.116255: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-11-01 18:06:31.116268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-11-01 18:06:31.116417: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:31.117143: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:31.117873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10805 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    2019-11-01 18:06:41.209892: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.210669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:04.0
    2019-11-01 18:06:41.210767: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:06:41.210820: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:06:41.210860: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-11-01 18:06:41.210900: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-11-01 18:06:41.210937: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-11-01 18:06:41.210972: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-11-01 18:06:41.211007: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-11-01 18:06:41.211131: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.211860: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.212600: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-11-01 18:06:41.213475: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.214244: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:04.0
    2019-11-01 18:06:41.214307: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:06:41.214364: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:06:41.214403: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-11-01 18:06:41.214437: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-11-01 18:06:41.214506: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-11-01 18:06:41.214540: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-11-01 18:06:41.214589: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-11-01 18:06:41.214699: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.215533: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.216247: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-11-01 18:06:41.216301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-11-01 18:06:41.216324: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-11-01 18:06:41.216341: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-11-01 18:06:41.216491: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.217228: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:06:41.217918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10805 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/stylize.py:73: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/vgg.py:78: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    tcmalloc: large alloc 3317915648 bytes == 0x66196000 @  0x7fe2436e81e7 0x7fe20309d282 0x7fe206c71e8a 0x7fe207047c82 0x7fe207049594 0x7fe2070ba7be 0x7fe2070bdc4d 0x7fe2070be0bf 0x7fe1fe51b7f6 0x7fe1fe51bb1f 0x7fe1fe5cbf79 0x7fe1fe5c9648 0x7fe241fca66f 0x7fe2430ac6db 0x7fe2433e588f
    tcmalloc: large alloc 3317915648 bytes == 0x6cf74000 @  0x7fe2436e81e7 0x7fe20309d282 0x7fe206c71e8a 0x7fe207047c82 0x7fe207049594 0x7fe2070ba7be 0x7fe2070bdc4d 0x7fe2070be0bf 0x7fe1fe51b7f6 0x7fe1fe50de45 0x7fe1fe5cbf79 0x7fe1fe5c9648 0x7fe241fca66f 0x7fe2430ac6db 0x7fe2433e588f
    2019-11-01 18:07:14.267653: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:14.268487: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:04.0
    2019-11-01 18:07:14.268584: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:07:14.268624: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:07:14.268664: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-11-01 18:07:14.268731: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-11-01 18:07:14.268784: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-11-01 18:07:14.268855: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-11-01 18:07:14.268894: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-11-01 18:07:14.269037: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:14.269977: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:14.270689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-11-01 18:07:14.270752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-11-01 18:07:14.270774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-11-01 18:07:14.270793: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-11-01 18:07:14.270938: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:14.271750: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:14.272453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10805 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    tcmalloc: large alloc 2026749952 bytes == 0x6ba98000 @  0x7fe2436e81e7 0x7fe20309d282 0x7fe206c71e8a 0x7fe207047c82 0x7fe207049594 0x7fe2070ba7be 0x7fe2070bdc4d 0x7fe2070be0bf 0x7fe1fe51b7f6 0x7fe1fe50de45 0x7fe1fe5cbf79 0x7fe1fe5c9648 0x7fe241fca66f 0x7fe2430ac6db 0x7fe2433e588f
    tcmalloc: large alloc 2026749952 bytes == 0x68522000 @  0x7fe2436e81e7 0x7fe20309d282 0x7fe206c71e8a 0x7fe207047c82 0x7fe207049594 0x7fe2070ba7be 0x7fe2070bdc4d 0x7fe2070be0bf 0x7fe1fe51b7f6 0x7fe1fe50de45 0x7fe1fe5cbf79 0x7fe1fe5c9648 0x7fe241fca66f 0x7fe2430ac6db 0x7fe2433e588f
    tcmalloc: large alloc 2026749952 bytes == 0x634b4000 @  0x7fe2436e81e7 0x7fe20309d282 0x7fe206c71e8a 0x7fe207047c82 0x7fe207049594 0x7fe2070ba7be 0x7fe2070bdc4d 0x7fe2070be0bf 0x7fe1fe51b7f6 0x7fe1fe50de45 0x7fe1fe5cbf79 0x7fe1fe5c9648 0x7fe241fca66f 0x7fe2430ac6db 0x7fe2433e588f
    tcmalloc: large alloc 2026749952 bytes == 0x64fa8000 @  0x7fe2436e81e7 0x7fe20309d282 0x7fe206c71e8a 0x7fe207047c82 0x7fe207049594 0x7fe2070ba7be 0x7fe2070bdc4d 0x7fe2070be0bf 0x7fe1fe51b7f6 0x7fe1fe51bb1f 0x7fe1fe5cbf79 0x7fe1fe5c9648 0x7fe241fca66f 0x7fe2430ac6db 0x7fe2433e588f
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/stylize.py:98: The name tf.random_normal is deprecated. Please use tf.random.normal instead.
    
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/stylize.py:165: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
    
    2019-11-01 18:07:41.152548: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:41.153333: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:04.0
    2019-11-01 18:07:41.153437: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-11-01 18:07:41.153473: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:07:41.153495: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-11-01 18:07:41.153518: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-11-01 18:07:41.153539: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-11-01 18:07:41.153559: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-11-01 18:07:41.153580: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-11-01 18:07:41.153661: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:41.154519: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:41.155145: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-11-01 18:07:41.155190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-11-01 18:07:41.155205: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-11-01 18:07:41.155216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-11-01 18:07:41.155379: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:41.156179: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-11-01 18:07:41.156883: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10805 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
    WARNING:tensorflow:From /content/drive/My Drive/Data/NueralStyle/neural-style-master/stylize.py:171: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    Optimization started...
    Iteration    1/1000
    2019-11-01 18:07:45.880749: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-11-01 18:07:46.487587: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    Iteration    2/1000 (22 sec elapsed, 6 hr 20 min remaining)
    Iteration    3/1000 (24 sec elapsed, 3 hr 20 min remaining)
    Iteration    4/1000 (25 sec elapsed, 2 hr 20 min remaining)
    Iteration    5/1000 (26 sec elapsed, 1 hr 50 min remaining)
    Iteration    6/1000 (27 sec elapsed, 1 hr 32 min remaining)
    Iteration    7/1000 (29 sec elapsed, 1 hr 20 min remaining)
    Iteration    8/1000 (30 sec elapsed, 1 hr 11 min remaining)
    Iteration    9/1000 (31 sec elapsed, 1 hr 5 min remaining)
    Iteration   10/1000 (32 sec elapsed, 1 hr 0 min remaining)
    Iteration   11/1000 (34 sec elapsed, 56 min 29 sec remaining)
    Iteration   12/1000 (35 sec elapsed, 20 min 50 sec remaining)
    Iteration   13/1000 (36 sec elapsed, 20 min 48 sec remaining)
    Iteration   14/1000 (38 sec elapsed, 20 min 48 sec remaining)
    Iteration   15/1000 (39 sec elapsed, 20 min 48 sec remaining)
    Iteration   16/1000 (40 sec elapsed, 20 min 47 sec remaining)
    Iteration   17/1000 (41 sec elapsed, 20 min 46 sec remaining)
    Iteration   18/1000 (43 sec elapsed, 20 min 45 sec remaining)
    Iteration   19/1000 (44 sec elapsed, 20 min 43 sec remaining)
    Iteration   20/1000 (45 sec elapsed, 20 min 43 sec remaining)
    Iteration   21/1000 (46 sec elapsed, 20 min 42 sec remaining)
    Iteration   22/1000 (48 sec elapsed, 20 min 41 sec remaining)
    Iteration   23/1000 (49 sec elapsed, 20 min 41 sec remaining)
    Iteration   24/1000 (50 sec elapsed, 20 min 40 sec remaining)
    Iteration   25/1000 (52 sec elapsed, 20 min 39 sec remaining)
    Iteration   26/1000 (53 sec elapsed, 20 min 38 sec remaining)
    Iteration   27/1000 (54 sec elapsed, 20 min 37 sec remaining)
    Iteration   28/1000 (55 sec elapsed, 20 min 36 sec remaining)
    Iteration   29/1000 (57 sec elapsed, 20 min 35 sec remaining)
    Iteration   30/1000 (58 sec elapsed, 20 min 34 sec remaining)
    Iteration   31/1000 (59 sec elapsed, 20 min 33 sec remaining)
    Iteration   32/1000 (1 min 0 sec elapsed, 20 min 33 sec remaining)
    Iteration   33/1000 (1 min 2 sec elapsed, 20 min 32 sec remaining)
    Iteration   34/1000 (1 min 3 sec elapsed, 20 min 31 sec remaining)
    Iteration   35/1000 (1 min 4 sec elapsed, 20 min 30 sec remaining)
    Iteration   36/1000 (1 min 6 sec elapsed, 20 min 30 sec remaining)
    Iteration   37/1000 (1 min 7 sec elapsed, 20 min 30 sec remaining)
    Iteration   38/1000 (1 min 8 sec elapsed, 20 min 29 sec remaining)
    Iteration   39/1000 (1 min 9 sec elapsed, 20 min 29 sec remaining)
    Iteration   40/1000 (1 min 11 sec elapsed, 20 min 28 sec remaining)
    Iteration   41/1000 (1 min 12 sec elapsed, 20 min 27 sec remaining)
    Iteration   42/1000 (1 min 13 sec elapsed, 20 min 26 sec remaining)
    Iteration   43/1000 (1 min 14 sec elapsed, 20 min 25 sec remaining)
    Iteration   44/1000 (1 min 16 sec elapsed, 20 min 24 sec remaining)
    Iteration   45/1000 (1 min 17 sec elapsed, 20 min 23 sec remaining)
    Iteration   46/1000 (1 min 18 sec elapsed, 20 min 21 sec remaining)
    Iteration   47/1000 (1 min 20 sec elapsed, 20 min 20 sec remaining)
    Iteration   48/1000 (1 min 21 sec elapsed, 20 min 19 sec remaining)
    Iteration   49/1000 (1 min 22 sec elapsed, 20 min 18 sec remaining)
    Iteration   50/1000 (1 min 23 sec elapsed, 20 min 17 sec remaining)
    Iteration   51/1000 (1 min 25 sec elapsed, 20 min 16 sec remaining)
    Iteration   52/1000 (1 min 26 sec elapsed, 20 min 14 sec remaining)
    Iteration   53/1000 (1 min 27 sec elapsed, 20 min 13 sec remaining)
    Iteration   54/1000 (1 min 29 sec elapsed, 20 min 12 sec remaining)
    Iteration   55/1000 (1 min 30 sec elapsed, 20 min 12 sec remaining)
    Iteration   56/1000 (1 min 31 sec elapsed, 20 min 10 sec remaining)
    Iteration   57/1000 (1 min 32 sec elapsed, 20 min 8 sec remaining)
    Iteration   58/1000 (1 min 34 sec elapsed, 20 min 8 sec remaining)
    Iteration   59/1000 (1 min 35 sec elapsed, 20 min 7 sec remaining)
    Iteration   60/1000 (1 min 36 sec elapsed, 20 min 5 sec remaining)
    Iteration   61/1000 (1 min 38 sec elapsed, 20 min 4 sec remaining)
    Iteration   62/1000 (1 min 39 sec elapsed, 20 min 3 sec remaining)
    Iteration   63/1000 (1 min 40 sec elapsed, 20 min 3 sec remaining)
    Iteration   64/1000 (1 min 41 sec elapsed, 20 min 1 sec remaining)
    Iteration   65/1000 (1 min 43 sec elapsed, 19 min 59 sec remaining)
    Iteration   66/1000 (1 min 44 sec elapsed, 19 min 58 sec remaining)
    Iteration   67/1000 (1 min 45 sec elapsed, 19 min 58 sec remaining)
    Iteration   68/1000 (1 min 47 sec elapsed, 19 min 56 sec remaining)
    Iteration   69/1000 (1 min 48 sec elapsed, 19 min 54 sec remaining)
    Iteration   70/1000 (1 min 49 sec elapsed, 19 min 53 sec remaining)
    Iteration   71/1000 (1 min 50 sec elapsed, 19 min 52 sec remaining)
    Iteration   72/1000 (1 min 52 sec elapsed, 19 min 51 sec remaining)
    Iteration   73/1000 (1 min 53 sec elapsed, 19 min 49 sec remaining)
    Iteration   74/1000 (1 min 54 sec elapsed, 19 min 48 sec remaining)
    Iteration   75/1000 (1 min 55 sec elapsed, 19 min 47 sec remaining)
    Iteration   76/1000 (1 min 57 sec elapsed, 19 min 47 sec remaining)
    Iteration   77/1000 (1 min 58 sec elapsed, 19 min 46 sec remaining)
    Iteration   78/1000 (1 min 59 sec elapsed, 19 min 45 sec remaining)
    Iteration   79/1000 (2 min 1 sec elapsed, 19 min 43 sec remaining)
    Iteration   80/1000 (2 min 2 sec elapsed, 19 min 42 sec remaining)
    Iteration   81/1000 (2 min 3 sec elapsed, 19 min 41 sec remaining)
    Iteration   82/1000 (2 min 4 sec elapsed, 19 min 39 sec remaining)
    Iteration   83/1000 (2 min 6 sec elapsed, 19 min 38 sec remaining)
    Iteration   84/1000 (2 min 7 sec elapsed, 19 min 37 sec remaining)
    Iteration   85/1000 (2 min 8 sec elapsed, 19 min 36 sec remaining)
    Iteration   86/1000 (2 min 10 sec elapsed, 19 min 34 sec remaining)
    Iteration   87/1000 (2 min 11 sec elapsed, 19 min 32 sec remaining)
    Iteration   88/1000 (2 min 12 sec elapsed, 19 min 31 sec remaining)
    Iteration   89/1000 (2 min 13 sec elapsed, 19 min 30 sec remaining)
    Iteration   90/1000 (2 min 15 sec elapsed, 19 min 29 sec remaining)
    Iteration   91/1000 (2 min 16 sec elapsed, 19 min 27 sec remaining)
    Iteration   92/1000 (2 min 17 sec elapsed, 19 min 26 sec remaining)
    Iteration   93/1000 (2 min 19 sec elapsed, 19 min 25 sec remaining)
    Iteration   94/1000 (2 min 20 sec elapsed, 19 min 24 sec remaining)
    Iteration   95/1000 (2 min 21 sec elapsed, 19 min 22 sec remaining)
    Iteration   96/1000 (2 min 22 sec elapsed, 19 min 20 sec remaining)
    Iteration   97/1000 (2 min 24 sec elapsed, 19 min 19 sec remaining)
    Iteration   98/1000 (2 min 25 sec elapsed, 19 min 19 sec remaining)
    Iteration   99/1000 (2 min 26 sec elapsed, 19 min 18 sec remaining)
    Iteration  100/1000 (2 min 28 sec elapsed, 19 min 16 sec remaining)
    Iteration  101/1000 (2 min 29 sec elapsed, 19 min 15 sec remaining)
    Iteration  102/1000 (2 min 31 sec elapsed, 21 min 3 sec remaining)
    Iteration  103/1000 (2 min 33 sec elapsed, 21 min 2 sec remaining)
    Iteration  104/1000 (2 min 34 sec elapsed, 21 min 0 sec remaining)
    Iteration  105/1000 (2 min 35 sec elapsed, 20 min 58 sec remaining)
    Iteration  106/1000 (2 min 37 sec elapsed, 20 min 57 sec remaining)
    Iteration  107/1000 (2 min 38 sec elapsed, 20 min 56 sec remaining)
    Iteration  108/1000 (2 min 39 sec elapsed, 20 min 55 sec remaining)
    Iteration  109/1000 (2 min 40 sec elapsed, 20 min 53 sec remaining)
    Iteration  110/1000 (2 min 42 sec elapsed, 20 min 52 sec remaining)
    Iteration  111/1000 (2 min 43 sec elapsed, 20 min 51 sec remaining)
    Iteration  112/1000 (2 min 44 sec elapsed, 19 min 1 sec remaining)
    Iteration  113/1000 (2 min 45 sec elapsed, 19 min 0 sec remaining)
    Iteration  114/1000 (2 min 47 sec elapsed, 18 min 58 sec remaining)
    Iteration  115/1000 (2 min 48 sec elapsed, 18 min 58 sec remaining)
    Iteration  116/1000 (2 min 49 sec elapsed, 18 min 57 sec remaining)
    Iteration  117/1000 (2 min 51 sec elapsed, 18 min 56 sec remaining)
    Iteration  118/1000 (2 min 52 sec elapsed, 18 min 54 sec remaining)
    Iteration  119/1000 (2 min 53 sec elapsed, 18 min 52 sec remaining)
    Iteration  120/1000 (2 min 54 sec elapsed, 18 min 50 sec remaining)
    Iteration  121/1000 (2 min 56 sec elapsed, 18 min 49 sec remaining)
    Iteration  122/1000 (2 min 57 sec elapsed, 18 min 48 sec remaining)
    Iteration  123/1000 (2 min 58 sec elapsed, 18 min 47 sec remaining)
    Iteration  124/1000 (3 min 0 sec elapsed, 18 min 45 sec remaining)
    Iteration  125/1000 (3 min 1 sec elapsed, 18 min 44 sec remaining)
    Iteration  126/1000 (3 min 2 sec elapsed, 18 min 42 sec remaining)
    Iteration  127/1000 (3 min 3 sec elapsed, 18 min 42 sec remaining)
    Iteration  128/1000 (3 min 5 sec elapsed, 18 min 41 sec remaining)
    Iteration  129/1000 (3 min 6 sec elapsed, 18 min 39 sec remaining)
    Iteration  130/1000 (3 min 7 sec elapsed, 18 min 38 sec remaining)
    Iteration  131/1000 (3 min 9 sec elapsed, 18 min 37 sec remaining)
    Iteration  132/1000 (3 min 10 sec elapsed, 18 min 35 sec remaining)
    Iteration  133/1000 (3 min 11 sec elapsed, 18 min 34 sec remaining)
    Iteration  134/1000 (3 min 12 sec elapsed, 18 min 32 sec remaining)
    Iteration  135/1000 (3 min 14 sec elapsed, 18 min 31 sec remaining)
    Iteration  136/1000 (3 min 15 sec elapsed, 18 min 30 sec remaining)
    Iteration  137/1000 (3 min 16 sec elapsed, 18 min 28 sec remaining)
    Iteration  138/1000 (3 min 18 sec elapsed, 18 min 26 sec remaining)
    Iteration  139/1000 (3 min 19 sec elapsed, 18 min 25 sec remaining)
    Iteration  140/1000 (3 min 20 sec elapsed, 18 min 24 sec remaining)
    Iteration  141/1000 (3 min 21 sec elapsed, 18 min 24 sec remaining)
    Iteration  142/1000 (3 min 23 sec elapsed, 18 min 22 sec remaining)
    Iteration  143/1000 (3 min 24 sec elapsed, 18 min 22 sec remaining)
    Iteration  144/1000 (3 min 25 sec elapsed, 18 min 21 sec remaining)
    Iteration  145/1000 (3 min 27 sec elapsed, 18 min 20 sec remaining)
    Iteration  146/1000 (3 min 28 sec elapsed, 18 min 18 sec remaining)
    Iteration  147/1000 (3 min 29 sec elapsed, 18 min 17 sec remaining)
    Iteration  148/1000 (3 min 30 sec elapsed, 18 min 15 sec remaining)
    Iteration  149/1000 (3 min 32 sec elapsed, 18 min 15 sec remaining)
    Iteration  150/1000 (3 min 33 sec elapsed, 18 min 13 sec remaining)
    Iteration  151/1000 (3 min 34 sec elapsed, 18 min 11 sec remaining)
    Iteration  152/1000 (3 min 36 sec elapsed, 18 min 9 sec remaining)
    Iteration  153/1000 (3 min 37 sec elapsed, 18 min 8 sec remaining)
    Iteration  154/1000 (3 min 38 sec elapsed, 18 min 6 sec remaining)
    Iteration  155/1000 (3 min 39 sec elapsed, 18 min 6 sec remaining)
    Iteration  156/1000 (3 min 41 sec elapsed, 18 min 4 sec remaining)
    Iteration  157/1000 (3 min 42 sec elapsed, 18 min 3 sec remaining)
    Iteration  158/1000 (3 min 43 sec elapsed, 18 min 1 sec remaining)
    Iteration  159/1000 (3 min 45 sec elapsed, 18 min 0 sec remaining)
    Iteration  160/1000 (3 min 46 sec elapsed, 17 min 58 sec remaining)
    Iteration  161/1000 (3 min 47 sec elapsed, 17 min 57 sec remaining)
    Iteration  162/1000 (3 min 48 sec elapsed, 17 min 56 sec remaining)
    Iteration  163/1000 (3 min 50 sec elapsed, 17 min 55 sec remaining)
    Iteration  164/1000 (3 min 51 sec elapsed, 17 min 54 sec remaining)
    Iteration  165/1000 (3 min 52 sec elapsed, 17 min 52 sec remaining)
    Iteration  166/1000 (3 min 54 sec elapsed, 17 min 51 sec remaining)
    Iteration  167/1000 (3 min 55 sec elapsed, 17 min 49 sec remaining)
    Iteration  168/1000 (3 min 56 sec elapsed, 17 min 49 sec remaining)
    Iteration  169/1000 (3 min 57 sec elapsed, 17 min 48 sec remaining)
    Iteration  170/1000 (3 min 59 sec elapsed, 17 min 47 sec remaining)
    Iteration  171/1000 (4 min 0 sec elapsed, 17 min 45 sec remaining)
    Iteration  172/1000 (4 min 1 sec elapsed, 17 min 44 sec remaining)
    Iteration  173/1000 (4 min 3 sec elapsed, 17 min 43 sec remaining)
    Iteration  174/1000 (4 min 4 sec elapsed, 17 min 41 sec remaining)
    Iteration  175/1000 (4 min 5 sec elapsed, 17 min 40 sec remaining)
    Iteration  176/1000 (4 min 6 sec elapsed, 17 min 39 sec remaining)
    Iteration  177/1000 (4 min 8 sec elapsed, 17 min 38 sec remaining)
    Iteration  178/1000 (4 min 9 sec elapsed, 17 min 36 sec remaining)
    Iteration  179/1000 (4 min 10 sec elapsed, 17 min 35 sec remaining)
    Iteration  180/1000 (4 min 12 sec elapsed, 17 min 34 sec remaining)
    Iteration  181/1000 (4 min 13 sec elapsed, 17 min 33 sec remaining)
    Iteration  182/1000 (4 min 14 sec elapsed, 17 min 32 sec remaining)
    Iteration  183/1000 (4 min 15 sec elapsed, 17 min 30 sec remaining)
    Iteration  184/1000 (4 min 17 sec elapsed, 17 min 28 sec remaining)
    Iteration  185/1000 (4 min 18 sec elapsed, 17 min 27 sec remaining)
    Iteration  186/1000 (4 min 19 sec elapsed, 17 min 26 sec remaining)
    Iteration  187/1000 (4 min 21 sec elapsed, 17 min 25 sec remaining)
    Iteration  188/1000 (4 min 22 sec elapsed, 17 min 23 sec remaining)
    Iteration  189/1000 (4 min 23 sec elapsed, 17 min 22 sec remaining)
    Iteration  190/1000 (4 min 24 sec elapsed, 17 min 20 sec remaining)
    Iteration  191/1000 (4 min 26 sec elapsed, 17 min 19 sec remaining)
    Iteration  192/1000 (4 min 27 sec elapsed, 17 min 17 sec remaining)
    Iteration  193/1000 (4 min 28 sec elapsed, 17 min 16 sec remaining)
    Iteration  194/1000 (4 min 29 sec elapsed, 17 min 15 sec remaining)
    Iteration  195/1000 (4 min 31 sec elapsed, 17 min 14 sec remaining)
    Iteration  196/1000 (4 min 32 sec elapsed, 17 min 13 sec remaining)
    Iteration  197/1000 (4 min 33 sec elapsed, 17 min 12 sec remaining)
    Iteration  198/1000 (4 min 35 sec elapsed, 17 min 11 sec remaining)
    Iteration  199/1000 (4 min 36 sec elapsed, 17 min 9 sec remaining)
    Iteration  200/1000 (4 min 37 sec elapsed, 17 min 8 sec remaining)
    Iteration  201/1000 (4 min 38 sec elapsed, 17 min 6 sec remaining)
    Iteration  202/1000 (4 min 41 sec elapsed, 18 min 45 sec remaining)
    Iteration  203/1000 (4 min 42 sec elapsed, 18 min 44 sec remaining)
    Iteration  204/1000 (4 min 44 sec elapsed, 18 min 42 sec remaining)
    Iteration  205/1000 (4 min 45 sec elapsed, 18 min 40 sec remaining)
    Iteration  206/1000 (4 min 46 sec elapsed, 18 min 38 sec remaining)
    Iteration  207/1000 (4 min 47 sec elapsed, 18 min 37 sec remaining)
    Iteration  208/1000 (4 min 49 sec elapsed, 18 min 35 sec remaining)
    Iteration  209/1000 (4 min 50 sec elapsed, 18 min 34 sec remaining)
    Iteration  210/1000 (4 min 51 sec elapsed, 18 min 33 sec remaining)
    Iteration  211/1000 (4 min 53 sec elapsed, 18 min 32 sec remaining)
    Iteration  212/1000 (4 min 54 sec elapsed, 16 min 51 sec remaining)
    Iteration  213/1000 (4 min 55 sec elapsed, 16 min 50 sec remaining)
    Iteration  214/1000 (4 min 56 sec elapsed, 16 min 49 sec remaining)
    Iteration  215/1000 (4 min 58 sec elapsed, 16 min 47 sec remaining)
    Iteration  216/1000 (4 min 59 sec elapsed, 16 min 46 sec remaining)
    Iteration  217/1000 (5 min 0 sec elapsed, 16 min 45 sec remaining)
    Iteration  218/1000 (5 min 2 sec elapsed, 16 min 44 sec remaining)
    Iteration  219/1000 (5 min 3 sec elapsed, 16 min 43 sec remaining)
    Iteration  220/1000 (5 min 4 sec elapsed, 16 min 42 sec remaining)
    Iteration  221/1000 (5 min 5 sec elapsed, 16 min 40 sec remaining)
    Iteration  222/1000 (5 min 7 sec elapsed, 16 min 39 sec remaining)
    Iteration  223/1000 (5 min 8 sec elapsed, 16 min 37 sec remaining)
    Iteration  224/1000 (5 min 9 sec elapsed, 16 min 37 sec remaining)
    Iteration  225/1000 (5 min 11 sec elapsed, 16 min 36 sec remaining)
    Iteration  226/1000 (5 min 12 sec elapsed, 16 min 34 sec remaining)
    Iteration  227/1000 (5 min 13 sec elapsed, 16 min 33 sec remaining)
    Iteration  228/1000 (5 min 14 sec elapsed, 16 min 32 sec remaining)
    Iteration  229/1000 (5 min 16 sec elapsed, 16 min 31 sec remaining)
    Iteration  230/1000 (5 min 17 sec elapsed, 16 min 29 sec remaining)
    Iteration  231/1000 (5 min 18 sec elapsed, 16 min 27 sec remaining)
    Iteration  232/1000 (5 min 19 sec elapsed, 16 min 26 sec remaining)
    Iteration  233/1000 (5 min 21 sec elapsed, 16 min 25 sec remaining)
    Iteration  234/1000 (5 min 22 sec elapsed, 16 min 23 sec remaining)
    Iteration  235/1000 (5 min 23 sec elapsed, 16 min 22 sec remaining)
    Iteration  236/1000 (5 min 25 sec elapsed, 16 min 21 sec remaining)
    Iteration  237/1000 (5 min 26 sec elapsed, 16 min 19 sec remaining)
    Iteration  238/1000 (5 min 27 sec elapsed, 16 min 18 sec remaining)
    Iteration  239/1000 (5 min 28 sec elapsed, 16 min 16 sec remaining)
    Iteration  240/1000 (5 min 30 sec elapsed, 16 min 15 sec remaining)
    Iteration  241/1000 (5 min 31 sec elapsed, 16 min 14 sec remaining)
    Iteration  242/1000 (5 min 32 sec elapsed, 16 min 13 sec remaining)
    Iteration  243/1000 (5 min 34 sec elapsed, 16 min 12 sec remaining)
    Iteration  244/1000 (5 min 35 sec elapsed, 16 min 11 sec remaining)
    Iteration  245/1000 (5 min 36 sec elapsed, 16 min 9 sec remaining)
    Iteration  246/1000 (5 min 37 sec elapsed, 16 min 8 sec remaining)
    Iteration  247/1000 (5 min 39 sec elapsed, 16 min 7 sec remaining)
    Iteration  248/1000 (5 min 40 sec elapsed, 16 min 6 sec remaining)
    Iteration  249/1000 (5 min 41 sec elapsed, 16 min 5 sec remaining)
    Iteration  250/1000 (5 min 43 sec elapsed, 16 min 3 sec remaining)
    Iteration  251/1000 (5 min 44 sec elapsed, 16 min 3 sec remaining)
    Iteration  252/1000 (5 min 45 sec elapsed, 16 min 1 sec remaining)
    Iteration  253/1000 (5 min 46 sec elapsed, 16 min 0 sec remaining)
    Iteration  254/1000 (5 min 48 sec elapsed, 15 min 58 sec remaining)
    Iteration  255/1000 (5 min 49 sec elapsed, 15 min 57 sec remaining)
    Iteration  256/1000 (5 min 50 sec elapsed, 15 min 56 sec remaining)
    Iteration  257/1000 (5 min 52 sec elapsed, 15 min 54 sec remaining)
    Iteration  258/1000 (5 min 53 sec elapsed, 15 min 53 sec remaining)
    Iteration  259/1000 (5 min 54 sec elapsed, 15 min 52 sec remaining)
    Iteration  260/1000 (5 min 55 sec elapsed, 15 min 51 sec remaining)
    Iteration  261/1000 (5 min 57 sec elapsed, 15 min 50 sec remaining)
    Iteration  262/1000 (5 min 58 sec elapsed, 15 min 48 sec remaining)
    Iteration  263/1000 (5 min 59 sec elapsed, 15 min 47 sec remaining)
    Iteration  264/1000 (6 min 1 sec elapsed, 15 min 46 sec remaining)
    Iteration  265/1000 (6 min 2 sec elapsed, 15 min 45 sec remaining)
    Iteration  266/1000 (6 min 3 sec elapsed, 15 min 44 sec remaining)
    Iteration  267/1000 (6 min 4 sec elapsed, 15 min 43 sec remaining)
    Iteration  268/1000 (6 min 6 sec elapsed, 15 min 41 sec remaining)
    Iteration  269/1000 (6 min 7 sec elapsed, 15 min 40 sec remaining)
    Iteration  270/1000 (6 min 8 sec elapsed, 15 min 38 sec remaining)
    Iteration  271/1000 (6 min 10 sec elapsed, 15 min 36 sec remaining)
    Iteration  272/1000 (6 min 11 sec elapsed, 15 min 35 sec remaining)
    Iteration  273/1000 (6 min 12 sec elapsed, 15 min 34 sec remaining)
    Iteration  274/1000 (6 min 13 sec elapsed, 15 min 33 sec remaining)
    Iteration  275/1000 (6 min 15 sec elapsed, 15 min 32 sec remaining)
    Iteration  276/1000 (6 min 16 sec elapsed, 15 min 30 sec remaining)
    Iteration  277/1000 (6 min 17 sec elapsed, 15 min 29 sec remaining)
    Iteration  278/1000 (6 min 19 sec elapsed, 15 min 27 sec remaining)
    Iteration  279/1000 (6 min 20 sec elapsed, 15 min 26 sec remaining)
    Iteration  280/1000 (6 min 21 sec elapsed, 15 min 25 sec remaining)
    Iteration  281/1000 (6 min 22 sec elapsed, 15 min 24 sec remaining)
    Iteration  282/1000 (6 min 24 sec elapsed, 15 min 22 sec remaining)
    Iteration  283/1000 (6 min 25 sec elapsed, 15 min 21 sec remaining)
    Iteration  284/1000 (6 min 26 sec elapsed, 15 min 19 sec remaining)
    Iteration  285/1000 (6 min 28 sec elapsed, 15 min 17 sec remaining)
    Iteration  286/1000 (6 min 29 sec elapsed, 15 min 16 sec remaining)
    Iteration  287/1000 (6 min 30 sec elapsed, 15 min 15 sec remaining)
    Iteration  288/1000 (6 min 31 sec elapsed, 15 min 14 sec remaining)
    Iteration  289/1000 (6 min 33 sec elapsed, 15 min 13 sec remaining)
    Iteration  290/1000 (6 min 34 sec elapsed, 15 min 12 sec remaining)
    Iteration  291/1000 (6 min 35 sec elapsed, 15 min 11 sec remaining)
    Iteration  292/1000 (6 min 36 sec elapsed, 15 min 10 sec remaining)
    Iteration  293/1000 (6 min 38 sec elapsed, 15 min 8 sec remaining)
    Iteration  294/1000 (6 min 39 sec elapsed, 15 min 6 sec remaining)
    Iteration  295/1000 (6 min 40 sec elapsed, 15 min 5 sec remaining)
    Iteration  296/1000 (6 min 42 sec elapsed, 15 min 4 sec remaining)
    Iteration  297/1000 (6 min 43 sec elapsed, 15 min 3 sec remaining)
    Iteration  298/1000 (6 min 44 sec elapsed, 15 min 2 sec remaining)
    Iteration  299/1000 (6 min 45 sec elapsed, 15 min 0 sec remaining)
    Iteration  300/1000 (6 min 47 sec elapsed, 14 min 59 sec remaining)
    Iteration  301/1000 (6 min 48 sec elapsed, 14 min 58 sec remaining)
    Iteration  302/1000 (6 min 50 sec elapsed, 16 min 16 sec remaining)
    Iteration  303/1000 (6 min 52 sec elapsed, 16 min 15 sec remaining)
    Iteration  304/1000 (6 min 53 sec elapsed, 16 min 14 sec remaining)
    Iteration  305/1000 (6 min 54 sec elapsed, 16 min 12 sec remaining)
    Iteration  306/1000 (6 min 56 sec elapsed, 16 min 11 sec remaining)
    Iteration  307/1000 (6 min 57 sec elapsed, 16 min 10 sec remaining)
    Iteration  308/1000 (6 min 58 sec elapsed, 16 min 8 sec remaining)
    Iteration  309/1000 (6 min 59 sec elapsed, 16 min 7 sec remaining)
    Iteration  310/1000 (7 min 1 sec elapsed, 16 min 5 sec remaining)
    Iteration  311/1000 (7 min 2 sec elapsed, 16 min 4 sec remaining)
    Iteration  312/1000 (7 min 3 sec elapsed, 14 min 44 sec remaining)
    Iteration  313/1000 (7 min 5 sec elapsed, 14 min 42 sec remaining)
    Iteration  314/1000 (7 min 6 sec elapsed, 14 min 41 sec remaining)
    Iteration  315/1000 (7 min 7 sec elapsed, 14 min 40 sec remaining)
    Iteration  316/1000 (7 min 8 sec elapsed, 14 min 39 sec remaining)
    Iteration  317/1000 (7 min 10 sec elapsed, 14 min 38 sec remaining)
    Iteration  318/1000 (7 min 11 sec elapsed, 14 min 36 sec remaining)
    Iteration  319/1000 (7 min 12 sec elapsed, 14 min 35 sec remaining)
    Iteration  320/1000 (7 min 14 sec elapsed, 14 min 33 sec remaining)
    Iteration  321/1000 (7 min 15 sec elapsed, 14 min 32 sec remaining)
    Iteration  322/1000 (7 min 16 sec elapsed, 14 min 30 sec remaining)
    Iteration  323/1000 (7 min 17 sec elapsed, 14 min 29 sec remaining)
    Iteration  324/1000 (7 min 19 sec elapsed, 14 min 28 sec remaining)
    Iteration  325/1000 (7 min 20 sec elapsed, 14 min 27 sec remaining)
    Iteration  326/1000 (7 min 21 sec elapsed, 14 min 25 sec remaining)
    Iteration  327/1000 (7 min 23 sec elapsed, 14 min 23 sec remaining)
    Iteration  328/1000 (7 min 24 sec elapsed, 14 min 22 sec remaining)
    Iteration  329/1000 (7 min 25 sec elapsed, 14 min 21 sec remaining)
    Iteration  330/1000 (7 min 26 sec elapsed, 14 min 20 sec remaining)
    Iteration  331/1000 (7 min 28 sec elapsed, 14 min 19 sec remaining)
    Iteration  332/1000 (7 min 29 sec elapsed, 14 min 17 sec remaining)
    Iteration  333/1000 (7 min 30 sec elapsed, 14 min 16 sec remaining)
    Iteration  334/1000 (7 min 32 sec elapsed, 14 min 15 sec remaining)
    Iteration  335/1000 (7 min 33 sec elapsed, 14 min 13 sec remaining)
    Iteration  336/1000 (7 min 34 sec elapsed, 14 min 12 sec remaining)
    Iteration  337/1000 (7 min 35 sec elapsed, 14 min 11 sec remaining)
    Iteration  338/1000 (7 min 37 sec elapsed, 14 min 10 sec remaining)
    Iteration  339/1000 (7 min 38 sec elapsed, 14 min 8 sec remaining)
    Iteration  340/1000 (7 min 39 sec elapsed, 14 min 7 sec remaining)
    Iteration  341/1000 (7 min 40 sec elapsed, 14 min 6 sec remaining)
    Iteration  342/1000 (7 min 42 sec elapsed, 14 min 5 sec remaining)
    Iteration  343/1000 (7 min 43 sec elapsed, 14 min 3 sec remaining)
    Iteration  344/1000 (7 min 44 sec elapsed, 14 min 2 sec remaining)
    Iteration  345/1000 (7 min 46 sec elapsed, 14 min 0 sec remaining)
    Iteration  346/1000 (7 min 47 sec elapsed, 13 min 59 sec remaining)
    Iteration  347/1000 (7 min 48 sec elapsed, 13 min 58 sec remaining)
    Iteration  348/1000 (7 min 49 sec elapsed, 13 min 56 sec remaining)
    Iteration  349/1000 (7 min 51 sec elapsed, 13 min 55 sec remaining)
    Iteration  350/1000 (7 min 52 sec elapsed, 13 min 54 sec remaining)
    Iteration  351/1000 (7 min 53 sec elapsed, 13 min 53 sec remaining)
    Iteration  352/1000 (7 min 55 sec elapsed, 13 min 52 sec remaining)
    Iteration  353/1000 (7 min 56 sec elapsed, 13 min 51 sec remaining)
    Iteration  354/1000 (7 min 57 sec elapsed, 13 min 49 sec remaining)
    Iteration  355/1000 (7 min 58 sec elapsed, 13 min 49 sec remaining)
    Iteration  356/1000 (8 min 0 sec elapsed, 13 min 47 sec remaining)
    Iteration  357/1000 (8 min 1 sec elapsed, 13 min 46 sec remaining)
    Iteration  358/1000 (8 min 2 sec elapsed, 13 min 45 sec remaining)
    Iteration  359/1000 (8 min 4 sec elapsed, 13 min 44 sec remaining)
    Iteration  360/1000 (8 min 5 sec elapsed, 13 min 43 sec remaining)
    Iteration  361/1000 (8 min 6 sec elapsed, 13 min 41 sec remaining)
    Iteration  362/1000 (8 min 7 sec elapsed, 13 min 39 sec remaining)
    Iteration  363/1000 (8 min 9 sec elapsed, 13 min 38 sec remaining)
    Iteration  364/1000 (8 min 10 sec elapsed, 13 min 37 sec remaining)
    Iteration  365/1000 (8 min 11 sec elapsed, 13 min 35 sec remaining)
    Iteration  366/1000 (8 min 13 sec elapsed, 13 min 34 sec remaining)
    Iteration  367/1000 (8 min 14 sec elapsed, 13 min 32 sec remaining)
    Iteration  368/1000 (8 min 15 sec elapsed, 13 min 31 sec remaining)
    Iteration  369/1000 (8 min 16 sec elapsed, 13 min 30 sec remaining)
    Iteration  370/1000 (8 min 18 sec elapsed, 13 min 28 sec remaining)
    Iteration  371/1000 (8 min 19 sec elapsed, 13 min 27 sec remaining)
    Iteration  372/1000 (8 min 20 sec elapsed, 13 min 26 sec remaining)
    Iteration  373/1000 (8 min 22 sec elapsed, 13 min 25 sec remaining)
    Iteration  374/1000 (8 min 23 sec elapsed, 13 min 23 sec remaining)
    Iteration  375/1000 (8 min 24 sec elapsed, 13 min 22 sec remaining)
    Iteration  376/1000 (8 min 25 sec elapsed, 13 min 21 sec remaining)
    Iteration  377/1000 (8 min 27 sec elapsed, 13 min 20 sec remaining)
    Iteration  378/1000 (8 min 28 sec elapsed, 13 min 18 sec remaining)
    Iteration  379/1000 (8 min 29 sec elapsed, 13 min 17 sec remaining)
    Iteration  380/1000 (8 min 31 sec elapsed, 13 min 16 sec remaining)
    Iteration  381/1000 (8 min 32 sec elapsed, 13 min 15 sec remaining)
    Iteration  382/1000 (8 min 33 sec elapsed, 13 min 14 sec remaining)
    Iteration  383/1000 (8 min 34 sec elapsed, 13 min 13 sec remaining)
    Iteration  384/1000 (8 min 36 sec elapsed, 13 min 12 sec remaining)
    Iteration  385/1000 (8 min 37 sec elapsed, 13 min 10 sec remaining)
    Iteration  386/1000 (8 min 38 sec elapsed, 13 min 9 sec remaining)
    Iteration  387/1000 (8 min 39 sec elapsed, 13 min 7 sec remaining)
    Iteration  388/1000 (8 min 41 sec elapsed, 13 min 6 sec remaining)
    Iteration  389/1000 (8 min 42 sec elapsed, 13 min 5 sec remaining)
    Iteration  390/1000 (8 min 43 sec elapsed, 13 min 4 sec remaining)
    Iteration  391/1000 (8 min 45 sec elapsed, 13 min 2 sec remaining)
    Iteration  392/1000 (8 min 46 sec elapsed, 13 min 1 sec remaining)
    Iteration  393/1000 (8 min 47 sec elapsed, 12 min 59 sec remaining)
    Iteration  394/1000 (8 min 48 sec elapsed, 12 min 58 sec remaining)
    Iteration  395/1000 (8 min 50 sec elapsed, 12 min 57 sec remaining)
    Iteration  396/1000 (8 min 51 sec elapsed, 12 min 56 sec remaining)
    Iteration  397/1000 (8 min 52 sec elapsed, 12 min 55 sec remaining)
    Iteration  398/1000 (8 min 54 sec elapsed, 12 min 54 sec remaining)
    Iteration  399/1000 (8 min 55 sec elapsed, 12 min 53 sec remaining)
    Iteration  400/1000 (8 min 56 sec elapsed, 12 min 52 sec remaining)
    Iteration  401/1000 (8 min 57 sec elapsed, 12 min 50 sec remaining)
    Iteration  402/1000 (9 min 0 sec elapsed, 14 min 1 sec remaining)
    Iteration  403/1000 (9 min 1 sec elapsed, 14 min 0 sec remaining)
    Iteration  404/1000 (9 min 3 sec elapsed, 13 min 58 sec remaining)
    Iteration  405/1000 (9 min 4 sec elapsed, 13 min 57 sec remaining)
    Iteration  406/1000 (9 min 5 sec elapsed, 13 min 55 sec remaining)
    Iteration  407/1000 (9 min 6 sec elapsed, 13 min 54 sec remaining)
    Iteration  408/1000 (9 min 8 sec elapsed, 13 min 52 sec remaining)
    Iteration  409/1000 (9 min 9 sec elapsed, 13 min 51 sec remaining)
    Iteration  410/1000 (9 min 10 sec elapsed, 13 min 50 sec remaining)
    Iteration  411/1000 (9 min 12 sec elapsed, 13 min 49 sec remaining)
    Iteration  412/1000 (9 min 13 sec elapsed, 12 min 36 sec remaining)
    Iteration  413/1000 (9 min 14 sec elapsed, 12 min 35 sec remaining)
    Iteration  414/1000 (9 min 15 sec elapsed, 12 min 34 sec remaining)
    Iteration  415/1000 (9 min 17 sec elapsed, 12 min 32 sec remaining)
    Iteration  416/1000 (9 min 18 sec elapsed, 12 min 31 sec remaining)
    Iteration  417/1000 (9 min 19 sec elapsed, 12 min 30 sec remaining)
    Iteration  418/1000 (9 min 21 sec elapsed, 12 min 28 sec remaining)
    Iteration  419/1000 (9 min 22 sec elapsed, 12 min 27 sec remaining)
    Iteration  420/1000 (9 min 23 sec elapsed, 12 min 25 sec remaining)
    Iteration  421/1000 (9 min 24 sec elapsed, 12 min 23 sec remaining)
    Iteration  422/1000 (9 min 26 sec elapsed, 12 min 22 sec remaining)
    Iteration  423/1000 (9 min 27 sec elapsed, 12 min 21 sec remaining)
    Iteration  424/1000 (9 min 28 sec elapsed, 12 min 19 sec remaining)
    Iteration  425/1000 (9 min 29 sec elapsed, 12 min 17 sec remaining)
    Iteration  426/1000 (9 min 31 sec elapsed, 12 min 16 sec remaining)
    Iteration  427/1000 (9 min 32 sec elapsed, 12 min 14 sec remaining)
    Iteration  428/1000 (9 min 33 sec elapsed, 12 min 13 sec remaining)
    Iteration  429/1000 (9 min 35 sec elapsed, 12 min 12 sec remaining)
    Iteration  430/1000 (9 min 36 sec elapsed, 12 min 10 sec remaining)
    Iteration  431/1000 (9 min 37 sec elapsed, 12 min 9 sec remaining)
    Iteration  432/1000 (9 min 38 sec elapsed, 12 min 8 sec remaining)
    Iteration  433/1000 (9 min 40 sec elapsed, 12 min 6 sec remaining)
    Iteration  434/1000 (9 min 41 sec elapsed, 12 min 6 sec remaining)
    Iteration  435/1000 (9 min 42 sec elapsed, 12 min 5 sec remaining)
    Iteration  436/1000 (9 min 44 sec elapsed, 12 min 3 sec remaining)
    Iteration  437/1000 (9 min 45 sec elapsed, 12 min 2 sec remaining)
    Iteration  438/1000 (9 min 46 sec elapsed, 12 min 1 sec remaining)
    Iteration  439/1000 (9 min 47 sec elapsed, 11 min 59 sec remaining)
    Iteration  440/1000 (9 min 49 sec elapsed, 11 min 59 sec remaining)
    Iteration  441/1000 (9 min 50 sec elapsed, 11 min 57 sec remaining)
    Iteration  442/1000 (9 min 51 sec elapsed, 11 min 56 sec remaining)
    Iteration  443/1000 (9 min 53 sec elapsed, 11 min 55 sec remaining)
    Iteration  444/1000 (9 min 54 sec elapsed, 11 min 53 sec remaining)
    Iteration  445/1000 (9 min 55 sec elapsed, 11 min 52 sec remaining)
    Iteration  446/1000 (9 min 56 sec elapsed, 11 min 51 sec remaining)
    Iteration  447/1000 (9 min 58 sec elapsed, 11 min 50 sec remaining)
    Iteration  448/1000 (9 min 59 sec elapsed, 11 min 48 sec remaining)
    Iteration  449/1000 (10 min 0 sec elapsed, 11 min 47 sec remaining)
    Iteration  450/1000 (10 min 2 sec elapsed, 11 min 46 sec remaining)
    Iteration  451/1000 (10 min 3 sec elapsed, 11 min 45 sec remaining)
    Iteration  452/1000 (10 min 4 sec elapsed, 11 min 44 sec remaining)
    Iteration  453/1000 (10 min 5 sec elapsed, 11 min 42 sec remaining)
    Iteration  454/1000 (10 min 7 sec elapsed, 11 min 41 sec remaining)
    Iteration  455/1000 (10 min 8 sec elapsed, 11 min 40 sec remaining)
    Iteration  456/1000 (10 min 9 sec elapsed, 11 min 39 sec remaining)
    Iteration  457/1000 (10 min 10 sec elapsed, 11 min 37 sec remaining)
    Iteration  458/1000 (10 min 12 sec elapsed, 11 min 36 sec remaining)
    Iteration  459/1000 (10 min 13 sec elapsed, 11 min 35 sec remaining)
    Iteration  460/1000 (10 min 14 sec elapsed, 11 min 33 sec remaining)
    Iteration  461/1000 (10 min 16 sec elapsed, 11 min 32 sec remaining)
    Iteration  462/1000 (10 min 17 sec elapsed, 11 min 31 sec remaining)
    Iteration  463/1000 (10 min 18 sec elapsed, 11 min 30 sec remaining)
    Iteration  464/1000 (10 min 19 sec elapsed, 11 min 29 sec remaining)
    Iteration  465/1000 (10 min 21 sec elapsed, 11 min 28 sec remaining)
    Iteration  466/1000 (10 min 22 sec elapsed, 11 min 26 sec remaining)
    Iteration  467/1000 (10 min 23 sec elapsed, 11 min 25 sec remaining)
    Iteration  468/1000 (10 min 25 sec elapsed, 11 min 23 sec remaining)
    Iteration  469/1000 (10 min 26 sec elapsed, 11 min 22 sec remaining)
    Iteration  470/1000 (10 min 27 sec elapsed, 11 min 21 sec remaining)
    Iteration  471/1000 (10 min 28 sec elapsed, 11 min 19 sec remaining)
    Iteration  472/1000 (10 min 30 sec elapsed, 11 min 18 sec remaining)
    Iteration  473/1000 (10 min 31 sec elapsed, 11 min 17 sec remaining)
    Iteration  474/1000 (10 min 32 sec elapsed, 11 min 16 sec remaining)
    Iteration  475/1000 (10 min 34 sec elapsed, 11 min 14 sec remaining)
    Iteration  476/1000 (10 min 35 sec elapsed, 11 min 14 sec remaining)
    Iteration  477/1000 (10 min 36 sec elapsed, 11 min 12 sec remaining)
    Iteration  478/1000 (10 min 37 sec elapsed, 11 min 12 sec remaining)
    Iteration  479/1000 (10 min 39 sec elapsed, 11 min 10 sec remaining)
    Iteration  480/1000 (10 min 40 sec elapsed, 11 min 9 sec remaining)
    Iteration  481/1000 (10 min 41 sec elapsed, 11 min 7 sec remaining)
    Iteration  482/1000 (10 min 43 sec elapsed, 11 min 6 sec remaining)
    Iteration  483/1000 (10 min 44 sec elapsed, 11 min 5 sec remaining)
    Iteration  484/1000 (10 min 45 sec elapsed, 11 min 3 sec remaining)
    Iteration  485/1000 (10 min 46 sec elapsed, 11 min 2 sec remaining)
    Iteration  486/1000 (10 min 48 sec elapsed, 11 min 1 sec remaining)
    Iteration  487/1000 (10 min 49 sec elapsed, 10 min 59 sec remaining)
    Iteration  488/1000 (10 min 50 sec elapsed, 10 min 58 sec remaining)
    Iteration  489/1000 (10 min 52 sec elapsed, 10 min 57 sec remaining)
    Iteration  490/1000 (10 min 53 sec elapsed, 10 min 55 sec remaining)
    Iteration  491/1000 (10 min 54 sec elapsed, 10 min 54 sec remaining)
    Iteration  492/1000 (10 min 55 sec elapsed, 10 min 53 sec remaining)
    Iteration  493/1000 (10 min 57 sec elapsed, 10 min 52 sec remaining)
    Iteration  494/1000 (10 min 58 sec elapsed, 10 min 50 sec remaining)
    Iteration  495/1000 (10 min 59 sec elapsed, 10 min 49 sec remaining)
    Iteration  496/1000 (11 min 1 sec elapsed, 10 min 48 sec remaining)
    Iteration  497/1000 (11 min 2 sec elapsed, 10 min 47 sec remaining)
    Iteration  498/1000 (11 min 3 sec elapsed, 10 min 45 sec remaining)
    Iteration  499/1000 (11 min 4 sec elapsed, 10 min 44 sec remaining)
    Iteration  500/1000 (11 min 6 sec elapsed, 10 min 42 sec remaining)
    Iteration  501/1000 (11 min 7 sec elapsed, 10 min 42 sec remaining)
    Iteration  502/1000 (11 min 10 sec elapsed, 11 min 56 sec remaining)
    Iteration  503/1000 (11 min 11 sec elapsed, 11 min 55 sec remaining)
    Iteration  504/1000 (11 min 12 sec elapsed, 11 min 53 sec remaining)
    Iteration  505/1000 (11 min 14 sec elapsed, 11 min 52 sec remaining)
    Iteration  506/1000 (11 min 15 sec elapsed, 11 min 50 sec remaining)
    Iteration  507/1000 (11 min 16 sec elapsed, 11 min 48 sec remaining)
    Iteration  508/1000 (11 min 17 sec elapsed, 11 min 47 sec remaining)
    Iteration  509/1000 (11 min 19 sec elapsed, 11 min 45 sec remaining)
    Iteration  510/1000 (11 min 20 sec elapsed, 11 min 44 sec remaining)
    Iteration  511/1000 (11 min 21 sec elapsed, 11 min 42 sec remaining)
    Iteration  512/1000 (11 min 23 sec elapsed, 10 min 27 sec remaining)
    Iteration  513/1000 (11 min 24 sec elapsed, 10 min 25 sec remaining)
    Iteration  514/1000 (11 min 25 sec elapsed, 10 min 23 sec remaining)
    Iteration  515/1000 (11 min 26 sec elapsed, 10 min 22 sec remaining)
    Iteration  516/1000 (11 min 28 sec elapsed, 10 min 21 sec remaining)
    Iteration  517/1000 (11 min 29 sec elapsed, 10 min 19 sec remaining)
    Iteration  518/1000 (11 min 30 sec elapsed, 10 min 18 sec remaining)
    Iteration  519/1000 (11 min 32 sec elapsed, 10 min 17 sec remaining)
    Iteration  520/1000 (11 min 33 sec elapsed, 10 min 16 sec remaining)
    Iteration  521/1000 (11 min 34 sec elapsed, 10 min 15 sec remaining)
    Iteration  522/1000 (11 min 35 sec elapsed, 10 min 13 sec remaining)
    Iteration  523/1000 (11 min 37 sec elapsed, 10 min 12 sec remaining)
    Iteration  524/1000 (11 min 38 sec elapsed, 10 min 12 sec remaining)
    Iteration  525/1000 (11 min 39 sec elapsed, 10 min 10 sec remaining)
    Iteration  526/1000 (11 min 41 sec elapsed, 10 min 9 sec remaining)
    Iteration  527/1000 (11 min 42 sec elapsed, 10 min 8 sec remaining)
    Iteration  528/1000 (11 min 43 sec elapsed, 10 min 6 sec remaining)
    Iteration  529/1000 (11 min 44 sec elapsed, 10 min 5 sec remaining)
    Iteration  530/1000 (11 min 46 sec elapsed, 10 min 4 sec remaining)
    Iteration  531/1000 (11 min 47 sec elapsed, 10 min 2 sec remaining)
    Iteration  532/1000 (11 min 48 sec elapsed, 10 min 1 sec remaining)
    Iteration  533/1000 (11 min 50 sec elapsed, 9 min 59 sec remaining)
    Iteration  534/1000 (11 min 51 sec elapsed, 9 min 57 sec remaining)
    Iteration  535/1000 (11 min 52 sec elapsed, 9 min 56 sec remaining)
    Iteration  536/1000 (11 min 53 sec elapsed, 9 min 55 sec remaining)
    Iteration  537/1000 (11 min 55 sec elapsed, 9 min 54 sec remaining)
    Iteration  538/1000 (11 min 56 sec elapsed, 9 min 53 sec remaining)
    Iteration  539/1000 (11 min 57 sec elapsed, 9 min 52 sec remaining)
    Iteration  540/1000 (11 min 58 sec elapsed, 9 min 51 sec remaining)
    Iteration  541/1000 (12 min 0 sec elapsed, 9 min 50 sec remaining)
    Iteration  542/1000 (12 min 1 sec elapsed, 9 min 49 sec remaining)
    Iteration  543/1000 (12 min 2 sec elapsed, 9 min 47 sec remaining)
    Iteration  544/1000 (12 min 4 sec elapsed, 9 min 46 sec remaining)
    Iteration  545/1000 (12 min 5 sec elapsed, 9 min 45 sec remaining)
    Iteration  546/1000 (12 min 6 sec elapsed, 9 min 44 sec remaining)
    Iteration  547/1000 (12 min 7 sec elapsed, 9 min 43 sec remaining)
    Iteration  548/1000 (12 min 9 sec elapsed, 9 min 41 sec remaining)
    Iteration  549/1000 (12 min 10 sec elapsed, 9 min 40 sec remaining)
    Iteration  550/1000 (12 min 11 sec elapsed, 9 min 38 sec remaining)
    Iteration  551/1000 (12 min 13 sec elapsed, 9 min 37 sec remaining)
    Iteration  552/1000 (12 min 14 sec elapsed, 9 min 36 sec remaining)
    Iteration  553/1000 (12 min 15 sec elapsed, 9 min 34 sec remaining)
    Iteration  554/1000 (12 min 16 sec elapsed, 9 min 34 sec remaining)
    Iteration  555/1000 (12 min 18 sec elapsed, 9 min 32 sec remaining)
    Iteration  556/1000 (12 min 19 sec elapsed, 9 min 31 sec remaining)
    Iteration  557/1000 (12 min 20 sec elapsed, 9 min 29 sec remaining)
    Iteration  558/1000 (12 min 22 sec elapsed, 9 min 29 sec remaining)
    Iteration  559/1000 (12 min 23 sec elapsed, 9 min 27 sec remaining)
    Iteration  560/1000 (12 min 24 sec elapsed, 9 min 26 sec remaining)
    Iteration  561/1000 (12 min 25 sec elapsed, 9 min 25 sec remaining)
    Iteration  562/1000 (12 min 27 sec elapsed, 9 min 23 sec remaining)
    Iteration  563/1000 (12 min 28 sec elapsed, 9 min 22 sec remaining)
    Iteration  564/1000 (12 min 29 sec elapsed, 9 min 20 sec remaining)
    Iteration  565/1000 (12 min 31 sec elapsed, 9 min 19 sec remaining)
    Iteration  566/1000 (12 min 32 sec elapsed, 9 min 18 sec remaining)
    Iteration  567/1000 (12 min 33 sec elapsed, 9 min 16 sec remaining)
    Iteration  568/1000 (12 min 34 sec elapsed, 9 min 15 sec remaining)
    Iteration  569/1000 (12 min 36 sec elapsed, 9 min 14 sec remaining)
    Iteration  570/1000 (12 min 37 sec elapsed, 9 min 13 sec remaining)
    Iteration  571/1000 (12 min 38 sec elapsed, 9 min 11 sec remaining)
    Iteration  572/1000 (12 min 40 sec elapsed, 9 min 10 sec remaining)
    Iteration  573/1000 (12 min 41 sec elapsed, 9 min 9 sec remaining)
    Iteration  574/1000 (12 min 42 sec elapsed, 9 min 7 sec remaining)
    Iteration  575/1000 (12 min 43 sec elapsed, 9 min 6 sec remaining)
    Iteration  576/1000 (12 min 45 sec elapsed, 9 min 5 sec remaining)
    Iteration  577/1000 (12 min 46 sec elapsed, 9 min 4 sec remaining)
    Iteration  578/1000 (12 min 47 sec elapsed, 9 min 2 sec remaining)
    Iteration  579/1000 (12 min 49 sec elapsed, 9 min 1 sec remaining)
    Iteration  580/1000 (12 min 50 sec elapsed, 9 min 0 sec remaining)
    Iteration  581/1000 (12 min 51 sec elapsed, 8 min 59 sec remaining)
    Iteration  582/1000 (12 min 52 sec elapsed, 8 min 57 sec remaining)
    Iteration  583/1000 (12 min 54 sec elapsed, 8 min 56 sec remaining)
    Iteration  584/1000 (12 min 55 sec elapsed, 8 min 55 sec remaining)
    Iteration  585/1000 (12 min 56 sec elapsed, 8 min 54 sec remaining)
    Iteration  586/1000 (12 min 58 sec elapsed, 8 min 52 sec remaining)
    Iteration  587/1000 (12 min 59 sec elapsed, 8 min 51 sec remaining)
    Iteration  588/1000 (13 min 0 sec elapsed, 8 min 49 sec remaining)
    Iteration  589/1000 (13 min 1 sec elapsed, 8 min 48 sec remaining)
    Iteration  590/1000 (13 min 3 sec elapsed, 8 min 47 sec remaining)
    Iteration  591/1000 (13 min 4 sec elapsed, 8 min 46 sec remaining)
    Iteration  592/1000 (13 min 5 sec elapsed, 8 min 44 sec remaining)
    Iteration  593/1000 (13 min 7 sec elapsed, 8 min 43 sec remaining)
    Iteration  594/1000 (13 min 8 sec elapsed, 8 min 42 sec remaining)
    Iteration  595/1000 (13 min 9 sec elapsed, 8 min 41 sec remaining)
    Iteration  596/1000 (13 min 10 sec elapsed, 8 min 39 sec remaining)
    Iteration  597/1000 (13 min 12 sec elapsed, 8 min 38 sec remaining)
    Iteration  598/1000 (13 min 13 sec elapsed, 8 min 37 sec remaining)
    Iteration  599/1000 (13 min 14 sec elapsed, 8 min 35 sec remaining)
    Iteration  600/1000 (13 min 16 sec elapsed, 8 min 34 sec remaining)
    Iteration  601/1000 (13 min 17 sec elapsed, 8 min 33 sec remaining)
    Iteration  602/1000 (13 min 19 sec elapsed, 9 min 18 sec remaining)
    Iteration  603/1000 (13 min 21 sec elapsed, 9 min 17 sec remaining)
    Iteration  604/1000 (13 min 22 sec elapsed, 9 min 15 sec remaining)
    Iteration  605/1000 (13 min 23 sec elapsed, 9 min 14 sec remaining)
    Iteration  606/1000 (13 min 24 sec elapsed, 9 min 12 sec remaining)
    Iteration  607/1000 (13 min 26 sec elapsed, 9 min 11 sec remaining)
    Iteration  608/1000 (13 min 27 sec elapsed, 9 min 10 sec remaining)
    Iteration  609/1000 (13 min 28 sec elapsed, 9 min 8 sec remaining)
    Iteration  610/1000 (13 min 30 sec elapsed, 9 min 7 sec remaining)
    Iteration  611/1000 (13 min 31 sec elapsed, 9 min 6 sec remaining)
    Iteration  612/1000 (13 min 32 sec elapsed, 8 min 19 sec remaining)
    Iteration  613/1000 (13 min 33 sec elapsed, 8 min 18 sec remaining)
    Iteration  614/1000 (13 min 35 sec elapsed, 8 min 16 sec remaining)
    Iteration  615/1000 (13 min 36 sec elapsed, 8 min 15 sec remaining)
    Iteration  616/1000 (13 min 37 sec elapsed, 8 min 14 sec remaining)
    Iteration  617/1000 (13 min 38 sec elapsed, 8 min 12 sec remaining)
    Iteration  618/1000 (13 min 40 sec elapsed, 8 min 11 sec remaining)
    Iteration  619/1000 (13 min 41 sec elapsed, 8 min 10 sec remaining)
    Iteration  620/1000 (13 min 42 sec elapsed, 8 min 9 sec remaining)
    Iteration  621/1000 (13 min 44 sec elapsed, 8 min 7 sec remaining)
    Iteration  622/1000 (13 min 45 sec elapsed, 8 min 6 sec remaining)
    Iteration  623/1000 (13 min 46 sec elapsed, 8 min 5 sec remaining)
    Iteration  624/1000 (13 min 47 sec elapsed, 8 min 4 sec remaining)
    Iteration  625/1000 (13 min 49 sec elapsed, 8 min 2 sec remaining)
    Iteration  626/1000 (13 min 50 sec elapsed, 8 min 1 sec remaining)
    Iteration  627/1000 (13 min 51 sec elapsed, 7 min 59 sec remaining)
    Iteration  628/1000 (13 min 53 sec elapsed, 7 min 58 sec remaining)
    Iteration  629/1000 (13 min 54 sec elapsed, 7 min 57 sec remaining)
    Iteration  630/1000 (13 min 55 sec elapsed, 7 min 55 sec remaining)
    Iteration  631/1000 (13 min 56 sec elapsed, 7 min 54 sec remaining)
    Iteration  632/1000 (13 min 58 sec elapsed, 7 min 53 sec remaining)
    Iteration  633/1000 (13 min 59 sec elapsed, 7 min 52 sec remaining)
    Iteration  634/1000 (14 min 0 sec elapsed, 7 min 51 sec remaining)
    Iteration  635/1000 (14 min 2 sec elapsed, 7 min 49 sec remaining)
    Iteration  636/1000 (14 min 3 sec elapsed, 7 min 48 sec remaining)
    Iteration  637/1000 (14 min 4 sec elapsed, 7 min 47 sec remaining)
    Iteration  638/1000 (14 min 5 sec elapsed, 7 min 45 sec remaining)
    Iteration  639/1000 (14 min 7 sec elapsed, 7 min 44 sec remaining)
    Iteration  640/1000 (14 min 8 sec elapsed, 7 min 42 sec remaining)
    Iteration  641/1000 (14 min 9 sec elapsed, 7 min 41 sec remaining)
    Iteration  642/1000 (14 min 11 sec elapsed, 7 min 40 sec remaining)
    Iteration  643/1000 (14 min 12 sec elapsed, 7 min 39 sec remaining)
    Iteration  644/1000 (14 min 13 sec elapsed, 7 min 37 sec remaining)
    Iteration  645/1000 (14 min 14 sec elapsed, 7 min 36 sec remaining)
    Iteration  646/1000 (14 min 16 sec elapsed, 7 min 35 sec remaining)
    Iteration  647/1000 (14 min 17 sec elapsed, 7 min 34 sec remaining)
    Iteration  648/1000 (14 min 18 sec elapsed, 7 min 33 sec remaining)
    Iteration  649/1000 (14 min 20 sec elapsed, 7 min 32 sec remaining)
    Iteration  650/1000 (14 min 21 sec elapsed, 7 min 30 sec remaining)
    Iteration  651/1000 (14 min 22 sec elapsed, 7 min 29 sec remaining)
    Iteration  652/1000 (14 min 23 sec elapsed, 7 min 28 sec remaining)
    Iteration  653/1000 (14 min 25 sec elapsed, 7 min 26 sec remaining)
    Iteration  654/1000 (14 min 26 sec elapsed, 7 min 25 sec remaining)
    Iteration  655/1000 (14 min 27 sec elapsed, 7 min 24 sec remaining)
    Iteration  656/1000 (14 min 29 sec elapsed, 7 min 22 sec remaining)
    Iteration  657/1000 (14 min 30 sec elapsed, 7 min 21 sec remaining)
    Iteration  658/1000 (14 min 31 sec elapsed, 7 min 20 sec remaining)
    Iteration  659/1000 (14 min 32 sec elapsed, 7 min 18 sec remaining)
    Iteration  660/1000 (14 min 34 sec elapsed, 7 min 17 sec remaining)
    Iteration  661/1000 (14 min 35 sec elapsed, 7 min 16 sec remaining)
    Iteration  662/1000 (14 min 36 sec elapsed, 7 min 15 sec remaining)
    Iteration  663/1000 (14 min 38 sec elapsed, 7 min 13 sec remaining)
    Iteration  664/1000 (14 min 39 sec elapsed, 7 min 12 sec remaining)
    Iteration  665/1000 (14 min 40 sec elapsed, 7 min 11 sec remaining)
    Iteration  666/1000 (14 min 41 sec elapsed, 7 min 10 sec remaining)
    Iteration  667/1000 (14 min 43 sec elapsed, 7 min 9 sec remaining)
    Iteration  668/1000 (14 min 44 sec elapsed, 7 min 7 sec remaining)
    Iteration  669/1000 (14 min 45 sec elapsed, 7 min 6 sec remaining)
    Iteration  670/1000 (14 min 47 sec elapsed, 7 min 5 sec remaining)
    Iteration  671/1000 (14 min 48 sec elapsed, 7 min 3 sec remaining)
    Iteration  672/1000 (14 min 49 sec elapsed, 7 min 2 sec remaining)
    Iteration  673/1000 (14 min 50 sec elapsed, 7 min 1 sec remaining)
    Iteration  674/1000 (14 min 52 sec elapsed, 6 min 59 sec remaining)
    Iteration  675/1000 (14 min 53 sec elapsed, 6 min 58 sec remaining)
    Iteration  676/1000 (14 min 54 sec elapsed, 6 min 57 sec remaining)
    Iteration  677/1000 (14 min 56 sec elapsed, 6 min 55 sec remaining)
    Iteration  678/1000 (14 min 57 sec elapsed, 6 min 54 sec remaining)
    Iteration  679/1000 (14 min 58 sec elapsed, 6 min 53 sec remaining)
    Iteration  680/1000 (14 min 59 sec elapsed, 6 min 51 sec remaining)
    Iteration  681/1000 (15 min 1 sec elapsed, 6 min 50 sec remaining)
    Iteration  682/1000 (15 min 2 sec elapsed, 6 min 49 sec remaining)
    Iteration  683/1000 (15 min 3 sec elapsed, 6 min 47 sec remaining)
    Iteration  684/1000 (15 min 4 sec elapsed, 6 min 46 sec remaining)
    Iteration  685/1000 (15 min 6 sec elapsed, 6 min 45 sec remaining)
    Iteration  686/1000 (15 min 7 sec elapsed, 6 min 44 sec remaining)
    Iteration  687/1000 (15 min 8 sec elapsed, 6 min 42 sec remaining)
    Iteration  688/1000 (15 min 10 sec elapsed, 6 min 41 sec remaining)
    Iteration  689/1000 (15 min 11 sec elapsed, 6 min 40 sec remaining)
    Iteration  690/1000 (15 min 12 sec elapsed, 6 min 39 sec remaining)
    Iteration  691/1000 (15 min 13 sec elapsed, 6 min 37 sec remaining)
    Iteration  692/1000 (15 min 15 sec elapsed, 6 min 36 sec remaining)
    Iteration  693/1000 (15 min 16 sec elapsed, 6 min 35 sec remaining)
    Iteration  694/1000 (15 min 17 sec elapsed, 6 min 34 sec remaining)
    Iteration  695/1000 (15 min 19 sec elapsed, 6 min 32 sec remaining)
    Iteration  696/1000 (15 min 20 sec elapsed, 6 min 31 sec remaining)
    Iteration  697/1000 (15 min 21 sec elapsed, 6 min 30 sec remaining)
    Iteration  698/1000 (15 min 22 sec elapsed, 6 min 29 sec remaining)
    Iteration  699/1000 (15 min 24 sec elapsed, 6 min 27 sec remaining)
    Iteration  700/1000 (15 min 25 sec elapsed, 6 min 26 sec remaining)
    Iteration  701/1000 (15 min 26 sec elapsed, 6 min 25 sec remaining)
    Iteration  702/1000 (15 min 29 sec elapsed, 6 min 59 sec remaining)
    Iteration  703/1000 (15 min 30 sec elapsed, 6 min 58 sec remaining)
    Iteration  704/1000 (15 min 31 sec elapsed, 6 min 56 sec remaining)
    Iteration  705/1000 (15 min 33 sec elapsed, 6 min 55 sec remaining)
    Iteration  706/1000 (15 min 34 sec elapsed, 6 min 53 sec remaining)
    Iteration  707/1000 (15 min 35 sec elapsed, 6 min 52 sec remaining)
    Iteration  708/1000 (15 min 36 sec elapsed, 6 min 50 sec remaining)
    Iteration  709/1000 (15 min 38 sec elapsed, 6 min 49 sec remaining)
    Iteration  710/1000 (15 min 39 sec elapsed, 6 min 47 sec remaining)
    Iteration  711/1000 (15 min 40 sec elapsed, 6 min 46 sec remaining)
    Iteration  712/1000 (15 min 42 sec elapsed, 6 min 10 sec remaining)
    Iteration  713/1000 (15 min 43 sec elapsed, 6 min 9 sec remaining)
    Iteration  714/1000 (15 min 44 sec elapsed, 6 min 8 sec remaining)
    Iteration  715/1000 (15 min 45 sec elapsed, 6 min 7 sec remaining)
    Iteration  716/1000 (15 min 47 sec elapsed, 6 min 5 sec remaining)
    Iteration  717/1000 (15 min 48 sec elapsed, 6 min 4 sec remaining)
    Iteration  718/1000 (15 min 49 sec elapsed, 6 min 2 sec remaining)
    Iteration  719/1000 (15 min 51 sec elapsed, 6 min 1 sec remaining)
    Iteration  720/1000 (15 min 52 sec elapsed, 6 min 0 sec remaining)
    Iteration  721/1000 (15 min 53 sec elapsed, 5 min 58 sec remaining)
    Iteration  722/1000 (15 min 54 sec elapsed, 5 min 57 sec remaining)
    Iteration  723/1000 (15 min 56 sec elapsed, 5 min 56 sec remaining)
    Iteration  724/1000 (15 min 57 sec elapsed, 5 min 54 sec remaining)
    Iteration  725/1000 (15 min 58 sec elapsed, 5 min 53 sec remaining)
    Iteration  726/1000 (16 min 0 sec elapsed, 5 min 52 sec remaining)
    Iteration  727/1000 (16 min 1 sec elapsed, 5 min 51 sec remaining)
    Iteration  728/1000 (16 min 2 sec elapsed, 5 min 50 sec remaining)
    Iteration  729/1000 (16 min 3 sec elapsed, 5 min 48 sec remaining)
    Iteration  730/1000 (16 min 5 sec elapsed, 5 min 47 sec remaining)
    Iteration  731/1000 (16 min 6 sec elapsed, 5 min 46 sec remaining)
    Iteration  732/1000 (16 min 7 sec elapsed, 5 min 45 sec remaining)
    Iteration  733/1000 (16 min 9 sec elapsed, 5 min 43 sec remaining)
    Iteration  734/1000 (16 min 10 sec elapsed, 5 min 42 sec remaining)
    Iteration  735/1000 (16 min 11 sec elapsed, 5 min 41 sec remaining)
    Iteration  736/1000 (16 min 12 sec elapsed, 5 min 40 sec remaining)
    Iteration  737/1000 (16 min 14 sec elapsed, 5 min 38 sec remaining)
    Iteration  738/1000 (16 min 15 sec elapsed, 5 min 37 sec remaining)
    Iteration  739/1000 (16 min 16 sec elapsed, 5 min 36 sec remaining)
    Iteration  740/1000 (16 min 18 sec elapsed, 5 min 35 sec remaining)
    Iteration  741/1000 (16 min 19 sec elapsed, 5 min 34 sec remaining)
    Iteration  742/1000 (16 min 20 sec elapsed, 5 min 32 sec remaining)
    Iteration  743/1000 (16 min 21 sec elapsed, 5 min 31 sec remaining)
    Iteration  744/1000 (16 min 23 sec elapsed, 5 min 30 sec remaining)
    Iteration  745/1000 (16 min 24 sec elapsed, 5 min 28 sec remaining)
    Iteration  746/1000 (16 min 25 sec elapsed, 5 min 27 sec remaining)
    Iteration  747/1000 (16 min 27 sec elapsed, 5 min 26 sec remaining)
    Iteration  748/1000 (16 min 28 sec elapsed, 5 min 24 sec remaining)
    Iteration  749/1000 (16 min 29 sec elapsed, 5 min 23 sec remaining)
    Iteration  750/1000 (16 min 30 sec elapsed, 5 min 22 sec remaining)
    Iteration  751/1000 (16 min 32 sec elapsed, 5 min 20 sec remaining)
    Iteration  752/1000 (16 min 33 sec elapsed, 5 min 19 sec remaining)
    Iteration  753/1000 (16 min 34 sec elapsed, 5 min 17 sec remaining)
    Iteration  754/1000 (16 min 36 sec elapsed, 5 min 16 sec remaining)
    Iteration  755/1000 (16 min 37 sec elapsed, 5 min 15 sec remaining)
    Iteration  756/1000 (16 min 38 sec elapsed, 5 min 14 sec remaining)
    Iteration  757/1000 (16 min 39 sec elapsed, 5 min 12 sec remaining)
    Iteration  758/1000 (16 min 41 sec elapsed, 5 min 11 sec remaining)
    Iteration  759/1000 (16 min 42 sec elapsed, 5 min 10 sec remaining)
    Iteration  760/1000 (16 min 43 sec elapsed, 5 min 9 sec remaining)
    Iteration  761/1000 (16 min 44 sec elapsed, 5 min 7 sec remaining)
    Iteration  762/1000 (16 min 46 sec elapsed, 5 min 6 sec remaining)
    Iteration  763/1000 (16 min 47 sec elapsed, 5 min 5 sec remaining)
    Iteration  764/1000 (16 min 48 sec elapsed, 5 min 4 sec remaining)
    Iteration  765/1000 (16 min 50 sec elapsed, 5 min 2 sec remaining)
    Iteration  766/1000 (16 min 51 sec elapsed, 5 min 1 sec remaining)
    Iteration  767/1000 (16 min 52 sec elapsed, 5 min 0 sec remaining)
    Iteration  768/1000 (16 min 53 sec elapsed, 4 min 59 sec remaining)
    Iteration  769/1000 (16 min 55 sec elapsed, 4 min 57 sec remaining)
    Iteration  770/1000 (16 min 56 sec elapsed, 4 min 56 sec remaining)
    Iteration  771/1000 (16 min 57 sec elapsed, 4 min 55 sec remaining)
    Iteration  772/1000 (16 min 59 sec elapsed, 4 min 54 sec remaining)
    Iteration  773/1000 (17 min 0 sec elapsed, 4 min 52 sec remaining)
    Iteration  774/1000 (17 min 1 sec elapsed, 4 min 51 sec remaining)
    Iteration  775/1000 (17 min 2 sec elapsed, 4 min 50 sec remaining)
    Iteration  776/1000 (17 min 4 sec elapsed, 4 min 48 sec remaining)
    Iteration  777/1000 (17 min 5 sec elapsed, 4 min 47 sec remaining)
    Iteration  778/1000 (17 min 6 sec elapsed, 4 min 46 sec remaining)
    Iteration  779/1000 (17 min 8 sec elapsed, 4 min 44 sec remaining)
    Iteration  780/1000 (17 min 9 sec elapsed, 4 min 43 sec remaining)
    Iteration  781/1000 (17 min 10 sec elapsed, 4 min 42 sec remaining)
    Iteration  782/1000 (17 min 11 sec elapsed, 4 min 41 sec remaining)
    Iteration  783/1000 (17 min 13 sec elapsed, 4 min 39 sec remaining)
    Iteration  784/1000 (17 min 14 sec elapsed, 4 min 38 sec remaining)
    Iteration  785/1000 (17 min 15 sec elapsed, 4 min 37 sec remaining)
    Iteration  786/1000 (17 min 17 sec elapsed, 4 min 36 sec remaining)
    Iteration  787/1000 (17 min 18 sec elapsed, 4 min 34 sec remaining)
    Iteration  788/1000 (17 min 19 sec elapsed, 4 min 33 sec remaining)
    Iteration  789/1000 (17 min 20 sec elapsed, 4 min 32 sec remaining)
    Iteration  790/1000 (17 min 22 sec elapsed, 4 min 30 sec remaining)
    Iteration  791/1000 (17 min 23 sec elapsed, 4 min 29 sec remaining)
    Iteration  792/1000 (17 min 24 sec elapsed, 4 min 28 sec remaining)
    Iteration  793/1000 (17 min 26 sec elapsed, 4 min 26 sec remaining)
    Iteration  794/1000 (17 min 27 sec elapsed, 4 min 25 sec remaining)
    Iteration  795/1000 (17 min 28 sec elapsed, 4 min 24 sec remaining)
    Iteration  796/1000 (17 min 29 sec elapsed, 4 min 23 sec remaining)
    Iteration  797/1000 (17 min 31 sec elapsed, 4 min 21 sec remaining)
    Iteration  798/1000 (17 min 32 sec elapsed, 4 min 20 sec remaining)
    Iteration  799/1000 (17 min 33 sec elapsed, 4 min 19 sec remaining)
    Iteration  800/1000 (17 min 35 sec elapsed, 4 min 18 sec remaining)
    Iteration  801/1000 (17 min 36 sec elapsed, 4 min 16 sec remaining)
    Iteration  802/1000 (17 min 38 sec elapsed, 4 min 36 sec remaining)
    Iteration  803/1000 (17 min 39 sec elapsed, 4 min 35 sec remaining)
    Iteration  804/1000 (17 min 41 sec elapsed, 4 min 33 sec remaining)
    Iteration  805/1000 (17 min 42 sec elapsed, 4 min 32 sec remaining)
    Iteration  806/1000 (17 min 43 sec elapsed, 4 min 31 sec remaining)
    Iteration  807/1000 (17 min 45 sec elapsed, 4 min 29 sec remaining)
    Iteration  808/1000 (17 min 46 sec elapsed, 4 min 28 sec remaining)
    Iteration  809/1000 (17 min 47 sec elapsed, 4 min 27 sec remaining)
    Iteration  810/1000 (17 min 48 sec elapsed, 4 min 25 sec remaining)
    Iteration  811/1000 (17 min 50 sec elapsed, 4 min 24 sec remaining)
    Iteration  812/1000 (17 min 51 sec elapsed, 4 min 2 sec remaining)
    Iteration  813/1000 (17 min 52 sec elapsed, 4 min 1 sec remaining)
    Iteration  814/1000 (17 min 54 sec elapsed, 3 min 59 sec remaining)
    Iteration  815/1000 (17 min 55 sec elapsed, 3 min 58 sec remaining)
    Iteration  816/1000 (17 min 56 sec elapsed, 3 min 57 sec remaining)
    Iteration  817/1000 (17 min 57 sec elapsed, 3 min 56 sec remaining)
    Iteration  818/1000 (17 min 59 sec elapsed, 3 min 54 sec remaining)
    Iteration  819/1000 (18 min 0 sec elapsed, 3 min 53 sec remaining)
    Iteration  820/1000 (18 min 1 sec elapsed, 3 min 52 sec remaining)
    Iteration  821/1000 (18 min 3 sec elapsed, 3 min 51 sec remaining)
    Iteration  822/1000 (18 min 4 sec elapsed, 3 min 49 sec remaining)
    Iteration  823/1000 (18 min 5 sec elapsed, 3 min 48 sec remaining)
    Iteration  824/1000 (18 min 6 sec elapsed, 3 min 47 sec remaining)
    Iteration  825/1000 (18 min 8 sec elapsed, 3 min 46 sec remaining)
    Iteration  826/1000 (18 min 9 sec elapsed, 3 min 44 sec remaining)
    Iteration  827/1000 (18 min 10 sec elapsed, 3 min 43 sec remaining)
    Iteration  828/1000 (18 min 12 sec elapsed, 3 min 41 sec remaining)
    Iteration  829/1000 (18 min 13 sec elapsed, 3 min 40 sec remaining)
    Iteration  830/1000 (18 min 14 sec elapsed, 3 min 39 sec remaining)
    Iteration  831/1000 (18 min 15 sec elapsed, 3 min 38 sec remaining)
    Iteration  832/1000 (18 min 17 sec elapsed, 3 min 36 sec remaining)
    Iteration  833/1000 (18 min 18 sec elapsed, 3 min 35 sec remaining)
    Iteration  834/1000 (18 min 19 sec elapsed, 3 min 34 sec remaining)
    Iteration  835/1000 (18 min 21 sec elapsed, 3 min 33 sec remaining)
    Iteration  836/1000 (18 min 22 sec elapsed, 3 min 31 sec remaining)
    Iteration  837/1000 (18 min 23 sec elapsed, 3 min 30 sec remaining)
    Iteration  838/1000 (18 min 24 sec elapsed, 3 min 29 sec remaining)
    Iteration  839/1000 (18 min 26 sec elapsed, 3 min 27 sec remaining)
    Iteration  840/1000 (18 min 27 sec elapsed, 3 min 26 sec remaining)
    Iteration  841/1000 (18 min 28 sec elapsed, 3 min 25 sec remaining)
    Iteration  842/1000 (18 min 30 sec elapsed, 3 min 24 sec remaining)
    Iteration  843/1000 (18 min 31 sec elapsed, 3 min 22 sec remaining)
    Iteration  844/1000 (18 min 32 sec elapsed, 3 min 21 sec remaining)
    Iteration  845/1000 (18 min 33 sec elapsed, 3 min 20 sec remaining)
    Iteration  846/1000 (18 min 35 sec elapsed, 3 min 18 sec remaining)
    Iteration  847/1000 (18 min 36 sec elapsed, 3 min 17 sec remaining)
    Iteration  848/1000 (18 min 37 sec elapsed, 3 min 16 sec remaining)
    Iteration  849/1000 (18 min 39 sec elapsed, 3 min 15 sec remaining)
    Iteration  850/1000 (18 min 40 sec elapsed, 3 min 13 sec remaining)
    Iteration  851/1000 (18 min 41 sec elapsed, 3 min 12 sec remaining)
    Iteration  852/1000 (18 min 42 sec elapsed, 3 min 11 sec remaining)
    Iteration  853/1000 (18 min 44 sec elapsed, 3 min 10 sec remaining)
    Iteration  854/1000 (18 min 45 sec elapsed, 3 min 8 sec remaining)
    Iteration  855/1000 (18 min 46 sec elapsed, 3 min 7 sec remaining)
    Iteration  856/1000 (18 min 47 sec elapsed, 3 min 6 sec remaining)
    Iteration  857/1000 (18 min 49 sec elapsed, 3 min 4 sec remaining)
    Iteration  858/1000 (18 min 50 sec elapsed, 3 min 3 sec remaining)
    Iteration  859/1000 (18 min 51 sec elapsed, 3 min 2 sec remaining)
    Iteration  860/1000 (18 min 53 sec elapsed, 3 min 0 sec remaining)
    Iteration  861/1000 (18 min 54 sec elapsed, 2 min 59 sec remaining)
    Iteration  862/1000 (18 min 55 sec elapsed, 2 min 58 sec remaining)
    Iteration  863/1000 (18 min 56 sec elapsed, 2 min 57 sec remaining)
    Iteration  864/1000 (18 min 58 sec elapsed, 2 min 55 sec remaining)
    Iteration  865/1000 (18 min 59 sec elapsed, 2 min 54 sec remaining)
    Iteration  866/1000 (19 min 0 sec elapsed, 2 min 53 sec remaining)
    Iteration  867/1000 (19 min 2 sec elapsed, 2 min 51 sec remaining)
    Iteration  868/1000 (19 min 3 sec elapsed, 2 min 50 sec remaining)
    Iteration  869/1000 (19 min 4 sec elapsed, 2 min 49 sec remaining)
    Iteration  870/1000 (19 min 5 sec elapsed, 2 min 47 sec remaining)
    Iteration  871/1000 (19 min 7 sec elapsed, 2 min 46 sec remaining)
    Iteration  872/1000 (19 min 8 sec elapsed, 2 min 45 sec remaining)
    Iteration  873/1000 (19 min 9 sec elapsed, 2 min 44 sec remaining)
    Iteration  874/1000 (19 min 11 sec elapsed, 2 min 42 sec remaining)
    Iteration  875/1000 (19 min 12 sec elapsed, 2 min 41 sec remaining)
    Iteration  876/1000 (19 min 13 sec elapsed, 2 min 40 sec remaining)
    Iteration  877/1000 (19 min 14 sec elapsed, 2 min 39 sec remaining)
    Iteration  878/1000 (19 min 16 sec elapsed, 2 min 37 sec remaining)
    Iteration  879/1000 (19 min 17 sec elapsed, 2 min 36 sec remaining)
    Iteration  880/1000 (19 min 18 sec elapsed, 2 min 35 sec remaining)
    Iteration  881/1000 (19 min 20 sec elapsed, 2 min 33 sec remaining)
    Iteration  882/1000 (19 min 21 sec elapsed, 2 min 32 sec remaining)
    Iteration  883/1000 (19 min 22 sec elapsed, 2 min 31 sec remaining)
    Iteration  884/1000 (19 min 23 sec elapsed, 2 min 30 sec remaining)
    Iteration  885/1000 (19 min 25 sec elapsed, 2 min 28 sec remaining)
    Iteration  886/1000 (19 min 26 sec elapsed, 2 min 27 sec remaining)
    Iteration  887/1000 (19 min 27 sec elapsed, 2 min 26 sec remaining)
    Iteration  888/1000 (19 min 29 sec elapsed, 2 min 25 sec remaining)
    Iteration  889/1000 (19 min 30 sec elapsed, 2 min 23 sec remaining)
    Iteration  890/1000 (19 min 31 sec elapsed, 2 min 22 sec remaining)
    Iteration  891/1000 (19 min 32 sec elapsed, 2 min 21 sec remaining)
    Iteration  892/1000 (19 min 34 sec elapsed, 2 min 19 sec remaining)
    Iteration  893/1000 (19 min 35 sec elapsed, 2 min 18 sec remaining)
    Iteration  894/1000 (19 min 36 sec elapsed, 2 min 17 sec remaining)
    Iteration  895/1000 (19 min 38 sec elapsed, 2 min 16 sec remaining)
    Iteration  896/1000 (19 min 39 sec elapsed, 2 min 14 sec remaining)
    Iteration  897/1000 (19 min 40 sec elapsed, 2 min 13 sec remaining)
    Iteration  898/1000 (19 min 41 sec elapsed, 2 min 12 sec remaining)
    Iteration  899/1000 (19 min 43 sec elapsed, 2 min 10 sec remaining)
    Iteration  900/1000 (19 min 44 sec elapsed, 2 min 9 sec remaining)
    Iteration  901/1000 (19 min 45 sec elapsed, 2 min 8 sec remaining)
    Iteration  902/1000 (19 min 48 sec elapsed, 2 min 16 sec remaining)
    Iteration  903/1000 (19 min 49 sec elapsed, 2 min 15 sec remaining)
    Iteration  904/1000 (19 min 50 sec elapsed, 2 min 13 sec remaining)
    Iteration  905/1000 (19 min 51 sec elapsed, 2 min 12 sec remaining)
    Iteration  906/1000 (19 min 53 sec elapsed, 2 min 11 sec remaining)
    Iteration  907/1000 (19 min 54 sec elapsed, 2 min 9 sec remaining)
    Iteration  908/1000 (19 min 55 sec elapsed, 2 min 8 sec remaining)
    Iteration  909/1000 (19 min 56 sec elapsed, 2 min 7 sec remaining)
    Iteration  910/1000 (19 min 58 sec elapsed, 2 min 5 sec remaining)
    Iteration  911/1000 (19 min 59 sec elapsed, 2 min 4 sec remaining)
    Iteration  912/1000 (20 min 0 sec elapsed, 1 min 54 sec remaining)
    Iteration  913/1000 (20 min 2 sec elapsed, 1 min 52 sec remaining)
    Iteration  914/1000 (20 min 3 sec elapsed, 1 min 51 sec remaining)
    Iteration  915/1000 (20 min 4 sec elapsed, 1 min 50 sec remaining)
    Iteration  916/1000 (20 min 5 sec elapsed, 1 min 49 sec remaining)
    Iteration  917/1000 (20 min 7 sec elapsed, 1 min 47 sec remaining)
    Iteration  918/1000 (20 min 8 sec elapsed, 1 min 46 sec remaining)
    Iteration  919/1000 (20 min 9 sec elapsed, 1 min 45 sec remaining)
    Iteration  920/1000 (20 min 11 sec elapsed, 1 min 44 sec remaining)
    Iteration  921/1000 (20 min 12 sec elapsed, 1 min 42 sec remaining)
    Iteration  922/1000 (20 min 13 sec elapsed, 1 min 41 sec remaining)
    Iteration  923/1000 (20 min 14 sec elapsed, 1 min 40 sec remaining)
    Iteration  924/1000 (20 min 16 sec elapsed, 1 min 38 sec remaining)
    Iteration  925/1000 (20 min 17 sec elapsed, 1 min 37 sec remaining)
    Iteration  926/1000 (20 min 18 sec elapsed, 1 min 36 sec remaining)
    Iteration  927/1000 (20 min 20 sec elapsed, 1 min 34 sec remaining)
    Iteration  928/1000 (20 min 21 sec elapsed, 1 min 33 sec remaining)
    Iteration  929/1000 (20 min 22 sec elapsed, 1 min 32 sec remaining)
    Iteration  930/1000 (20 min 23 sec elapsed, 1 min 31 sec remaining)
    Iteration  931/1000 (20 min 25 sec elapsed, 1 min 29 sec remaining)
    Iteration  932/1000 (20 min 26 sec elapsed, 1 min 28 sec remaining)
    Iteration  933/1000 (20 min 27 sec elapsed, 1 min 27 sec remaining)
    Iteration  934/1000 (20 min 29 sec elapsed, 1 min 25 sec remaining)
    Iteration  935/1000 (20 min 30 sec elapsed, 1 min 24 sec remaining)
    Iteration  936/1000 (20 min 31 sec elapsed, 1 min 23 sec remaining)
    Iteration  937/1000 (20 min 32 sec elapsed, 1 min 22 sec remaining)
    Iteration  938/1000 (20 min 34 sec elapsed, 1 min 20 sec remaining)
    Iteration  939/1000 (20 min 35 sec elapsed, 1 min 19 sec remaining)
    Iteration  940/1000 (20 min 36 sec elapsed, 1 min 18 sec remaining)
    Iteration  941/1000 (20 min 38 sec elapsed, 1 min 16 sec remaining)
    Iteration  942/1000 (20 min 39 sec elapsed, 1 min 15 sec remaining)
    Iteration  943/1000 (20 min 40 sec elapsed, 1 min 14 sec remaining)
    Iteration  944/1000 (20 min 41 sec elapsed, 1 min 13 sec remaining)
    Iteration  945/1000 (20 min 43 sec elapsed, 1 min 11 sec remaining)
    Iteration  946/1000 (20 min 44 sec elapsed, 1 min 10 sec remaining)
    Iteration  947/1000 (20 min 45 sec elapsed, 1 min 9 sec remaining)
    Iteration  948/1000 (20 min 47 sec elapsed, 1 min 7 sec remaining)
    Iteration  949/1000 (20 min 48 sec elapsed, 1 min 6 sec remaining)
    Iteration  950/1000 (20 min 49 sec elapsed, 1 min 5 sec remaining)
    Iteration  951/1000 (20 min 50 sec elapsed, 1 min 4 sec remaining)
    Iteration  952/1000 (20 min 52 sec elapsed, 1 min 2 sec remaining)
    Iteration  953/1000 (20 min 53 sec elapsed, 1 min 1 sec remaining)
    Iteration  954/1000 (20 min 54 sec elapsed, 1 min 0 sec remaining)
    Iteration  955/1000 (20 min 56 sec elapsed, 59 sec remaining)
    Iteration  956/1000 (20 min 57 sec elapsed, 57 sec remaining)
    Iteration  957/1000 (20 min 58 sec elapsed, 56 sec remaining)
    Iteration  958/1000 (20 min 59 sec elapsed, 55 sec remaining)
    Iteration  959/1000 (21 min 1 sec elapsed, 53 sec remaining)
    Iteration  960/1000 (21 min 2 sec elapsed, 52 sec remaining)
    Iteration  961/1000 (21 min 3 sec elapsed, 51 sec remaining)
    Iteration  962/1000 (21 min 4 sec elapsed, 50 sec remaining)
    Iteration  963/1000 (21 min 6 sec elapsed, 48 sec remaining)
    Iteration  964/1000 (21 min 7 sec elapsed, 47 sec remaining)
    Iteration  965/1000 (21 min 8 sec elapsed, 46 sec remaining)
    Iteration  966/1000 (21 min 10 sec elapsed, 44 sec remaining)
    Iteration  967/1000 (21 min 11 sec elapsed, 43 sec remaining)
    Iteration  968/1000 (21 min 12 sec elapsed, 42 sec remaining)
    Iteration  969/1000 (21 min 13 sec elapsed, 41 sec remaining)
    Iteration  970/1000 (21 min 15 sec elapsed, 39 sec remaining)
    Iteration  971/1000 (21 min 16 sec elapsed, 38 sec remaining)
    Iteration  972/1000 (21 min 17 sec elapsed, 37 sec remaining)
    Iteration  973/1000 (21 min 19 sec elapsed, 35 sec remaining)
    Iteration  974/1000 (21 min 20 sec elapsed, 34 sec remaining)
    Iteration  975/1000 (21 min 21 sec elapsed, 33 sec remaining)
    Iteration  976/1000 (21 min 22 sec elapsed, 32 sec remaining)
    Iteration  977/1000 (21 min 24 sec elapsed, 30 sec remaining)
    Iteration  978/1000 (21 min 25 sec elapsed, 29 sec remaining)
    Iteration  979/1000 (21 min 26 sec elapsed, 28 sec remaining)
    Iteration  980/1000 (21 min 28 sec elapsed, 26 sec remaining)
    Iteration  981/1000 (21 min 29 sec elapsed, 25 sec remaining)
    Iteration  982/1000 (21 min 30 sec elapsed, 24 sec remaining)
    Iteration  983/1000 (21 min 31 sec elapsed, 23 sec remaining)
    Iteration  984/1000 (21 min 33 sec elapsed, 21 sec remaining)
    Iteration  985/1000 (21 min 34 sec elapsed, 20 sec remaining)
    Iteration  986/1000 (21 min 35 sec elapsed, 19 sec remaining)
    Iteration  987/1000 (21 min 37 sec elapsed, 17 sec remaining)
    Iteration  988/1000 (21 min 38 sec elapsed, 16 sec remaining)
    Iteration  989/1000 (21 min 39 sec elapsed, 15 sec remaining)
    Iteration  990/1000 (21 min 40 sec elapsed, 14 sec remaining)
    Iteration  991/1000 (21 min 42 sec elapsed, 12 sec remaining)
    Iteration  992/1000 (21 min 43 sec elapsed, 11 sec remaining)
    Iteration  993/1000 (21 min 44 sec elapsed, 10 sec remaining)
    Iteration  994/1000 (21 min 46 sec elapsed, 8 sec remaining)
    Iteration  995/1000 (21 min 47 sec elapsed, 7 sec remaining)
    Iteration  996/1000 (21 min 48 sec elapsed, 6 sec remaining)
    Iteration  997/1000 (21 min 49 sec elapsed, 5 sec remaining)
    Iteration  998/1000 (21 min 51 sec elapsed, 3 sec remaining)
    Iteration  999/1000 (21 min 52 sec elapsed, 2 sec remaining)
    Iteration 1000/1000 (21 min 53 sec elapsed, 1 sec remaining)
    content loss: 5.22943e+06
      style loss: 1.14158e+06
         tv loss: 62979.8
      total loss: 6.43399e+06
    


```
from IPython.display import Image
display(Image('./Nueral_Style_Images/new.jpg'))
display(Image('./Nueral_Style_Images/mid.jpg'))

```


    
![jpeg](/images/Nueral_Style_13_0.jpg)
    



    
![jpeg](/images/Nueral_Style_13_1.jpg)
    

