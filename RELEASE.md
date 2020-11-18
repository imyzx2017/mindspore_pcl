# Release 1.0.0

## Major Features and Improvements
### MindSpore Training and Inference Framework
#### Ascend 910
* New models
    * DenseNet121: a dense convolutional neural network, which connects each layer to every other layer in a feed-forward fashion for object recognition on ImageNet dataset.
    * UNet2D-Medical: Unet Medical model for 2D image segmentation, Convolutional Networks for Biomedical Image Segmentation on ISBI Challenge database.
* Frontend and user interface
    * Second-Order Optimization
      * Enable second-order optimization for Bert on Ascend 910, which can achieve a masked lm accuracy of 71.3% in 800 seconds using 8 Ascend 910 (Bert-Large @MLPerf v0.7 dataset).
    * New GNN model BGCF
      * Bayesian Graph Convolutional Filtering network which naturally incorporate the uncertainty in the user-item interaction graph shows excellent recommendation performance on Amazon-Beauty dataset.
    * Add append interface for SequentialCell.
    * Add a level `auto` for AMP.
* Executor and performance optimization
    * Support quantitative network (Resnet50 & YoloV3 & MobileNetV2).
    * Project ease of use optimization: project compilation time optimization, CMakelist regularization, cudnn, cuda independent compilation and installation independent.
* Data processing, augmentation, and save format
    * Support GeneratorDataset return string type

#### Other Hardware Support
* GPU platform
    * Enable second-order optimization for resnet50 on GPU, which achieve 30% improvement on training time compared to SGD with Momentum (Resnet50 @ImageNet).
#### User interfaces change log
* Remove global object GradOperation in Autodiff([!5011](https://gitee.com/mindspore/mindspore/pulls/5011))
* Remove useless attribute 'name' in Autodiff([!5172](https://gitee.com/mindspore/mindspore/pulls/5172))
* Rectification distributed init([!5350](https://gitee.com/mindspore/mindspore/pulls/5350))
* Move the setting of ParalleMode from train.parallel_utils to context([!5351](https://gitee.com/mindspore/mindspore/pulls/5351))
* Modification of save_checkpoint([!5482](https://gitee.com/mindspore/mindspore/pulls/5482))
* Wrap numpy random seed into an api([!5634](https://gitee.com/mindspore/mindspore/pulls/5634))
* Delete enable_fused_layernorm in some modelzoo scripts([!5665](https://gitee.com/mindspore/mindspore/pulls/5665))
* Move 'multi-subgraphs' interface to internal([!5696](https://gitee.com/mindspore/mindspore/pulls/5696))
* Rename mirror_mean to gradient_mean([!5700](https://gitee.com/mindspore/mindspore/pulls/5700))
* Remove default value of 'group' of DepthWiseConv2d([!5865](https://gitee.com/mindspore/mindspore/pulls/5865))
* Modify interface for function and remove duplicated def([!5958](https://gitee.com/mindspore/mindspore/pulls/5958))
* Unify Conv2d and DepthwiseConv2d([!5916](https://gitee.com/mindspore/mindspore/pulls/5916))
* Modification of SoftmaxCrossEntropyWithLogits([!5502](https://gitee.com/mindspore/mindspore/pulls/5502))
* Change API set_strategy() to shard()([!5991](https://gitee.com/mindspore/mindspore/pulls/5991))
* Move batch_size from bert_cfg_cfg to cfg([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
* Remove unused parameters from SummaryRecord __init__([!5548](https://gitee.com/mindspore/mindspore/pulls/5548))
* remove sens parameter of TrainOneStepWithLossScaleCell([!5753](https://gitee.com/mindspore/mindspore/pulls/5753))
* optimize the TrainOneStepCell for user's define([!6159](https://gitee.com/mindspore/mindspore/pulls/6159))
* delete seed0 and seed1 of nn.Dropout([!5735](https://gitee.com/mindspore/mindspore/pulls/5735))
* delete DataWrapper([!6101](https://gitee.com/mindspore/mindspore/pulls/6101))
* LSTM API optimization([!6374](https://gitee.com/mindspore/mindspore/pulls/6374))
* Merge P\C\F of ops([!5645](https://gitee.com/mindspore/mindspore/pulls/5645))
* delete SoftmaxCrossEntropyExpand interface([!6607](https://gitee.com/mindspore/mindspore/pulls/6607))
* Adjust GroupNorm interface([!6329](https://gitee.com/mindspore/mindspore/pulls/6329))
* Modify init interface to internal interface([!6651](https://gitee.com/mindspore/mindspore/pulls/6651))
* Log optimization([!5842](https://gitee.com/mindspore/mindspore/pulls/5842))
* Remove useless API dataset.set_dataset_size（[!5806](https://gitee.com/mindspore/mindspore/pulls/5806))
* Some of Dataset API add usage parameter（[!5605](https://gitee.com/mindspore/mindspore/pulls/5605))
* Change the import path, such as from mindspore.dataset.transforms.vision to mindspore.dataset.vision.transforms（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
* Rename ImageFolderDatasetV2 to ImageFolderDataset（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
* Dataset.map parameter optimization（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
* Add new api dataset.get_col_names（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
* Add new api dataset.get_col_names（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
* Remove useless API MindRecord finish（[!5580](https://gitee.com/mindspore/mindspore/pulls/5580))

### MindSpore Lite
* Converter
    * Add 6 TFLite op, 7 Caffe op, 1 ONNX op.
    * Add support for Windows.
    * Support parallel inference of multiple sessions to adapt to more scenarios
    * Support 8bits only weight-quantization, most main-stream models has small accuracy loss (less than 0.5%) when compared to non-qunantized fp32 model.

* CPU & GPU
    * Add 20 CPU ops，include FP32, int8/uint8, FP16 and int32 ops.
    * Add supporting FP16 for GPU, add 14 GPU ops include FP32/FP16.
    * Add Buffer/Image2D transform op for GPU
    * Performance optimization for CPU ops focus on ARM32.
    * Performance optimization for GPU Convolution using winograd.

* Tool & example
    * Add object detection Android Demo.

## Bugfixes
* Models
    * fix the constant folding problem in multiply.([!6092](https://gitee.com/mindspore/mindspore/pulls/6092))
    * move batch_size from bert_net_cfg to cfg in bert scripts.([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
    * modify the checkpoint file path.([!6137](https://gitee.com/mindspore/mindspore/pulls/6137))
* Python API
    * fix semi auto parallel parameter of reshape has another user([!5722](https://gitee.com/mindspore/mindspore/pulls/5722))
    * raise ValueError when call hook function in graph mode([!5831](https://gitee.com/mindspore/mindspore/pulls/5831))
* Executor
    * fix pynative mode to build temporary nn objects.（[!6189](https://gitee.com/mindspore/mindspore/pulls/6189))
    * fix the accuracy problem of multiple inputs of multi-card communication operator broadcast.([!6522](https://gitee.com/mindspore/mindspore/pulls/5622))
    * fix the problem that the sample distribution interface categorical does not support graph mode.([!5772](https://gitee.com/mindspore/mindspore/pulls/5772))
    * fix the random seed failure problem of the polynomial downsampling distribution operator.([!5948](https://gitee.com/mindspore/mindspore/pulls/5948))
    * fix unnecessary address binding issues in GPU heterogeneous scenarios.([!6232](https://gitee.com/mindspore/mindspore/pulls/6232))
* GPU platform
    * fix for kernel resource leak([!5315](https://gitee.com/mindspore/mindspore/pulls/5315))
    * fix for insufficient memory for continuous unit test running([!5617](https://gitee.com/mindspore/mindspore/pulls/5617))
    * fix for the memory leak in the sparse slicer([!5578](https://gitee.com/mindspore/mindspore/pulls/5578))
* Data processing
    * fix hang when use pyfunc([!6346](https://gitee.com/mindspore/mindspore/pulls/6346))
    * fix GPU device queue does not release GIL during resource clean up([!5964](https://gitee.com/mindspore/mindspore/pulls/5964))
    * fix hang if scripte exit unnormally([!6441](https://gitee.com/mindspore/mindspore/pulls/6441))
* Third party
    * Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    * Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors
Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, danish, Danish, dayschan, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, gongdaguo, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, root, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, Zichun, Zirui, Ziyan, zjun, ZPaC

Contributions of any kind are welcome!

# Release 0.7.0-beta

## Major Features and Improvements
### MindSpore Training and Inference Framework
#### Ascend 910
* New models
    * TinyBert: a smaller and faster version of BERT using transformer distillation for natural language understanding on GLUE benchmark.
    * SE-ResNet50: add Squeeze-and-Excitation blocks(SE-Blocks) to the resnet50 network to improve channel interdependencies for image classification on ImageNet 2012 dataset.
    * Inception V3: the third version of Inception convolutional architectures for image classification on ImageNet 2012 dataset.
* Frontend and user interface
    * Embedding operator high-level packaging to support segmented by field for Wide&Deep.
    * Load multi-node checkpoint into single-process to support host-device hybrid inference.
    * Support Concat/Tile/Strideslice distributed operators.
    * Support cumulative gradient and batch training split.
    * Support variable parameter input for Cell object.
    * Parameter mixed calculation optimization for pynative mode.
    * Deep Probabilistic Programming
      * Support statistical distributions classes used to generate stochastic tensors.
      * Support probabilistic inference algorithms.
      * Support BNN layers used to construct BNN in Graph mode.
      * Support interfaces for the transformation between BNN and DNN in Graph mode.
      * Support uncertainty estimation to estimate epistemic uncertainty and aleatoric uncertainty.
    * User interfaces change log
      * change base class of parameter([!3473](https://gitee.com/mindspore/mindspore/pulls/3473))
      * change binary to mindir([!4258](https://gitee.com/mindspore/mindspore/pulls/4258))
      * change export from geir to air([!4269](https://gitee.com/mindspore/mindspore/pulls/4269))
      * Init parameter data by default([!3967](https://gitee.com/mindspore/mindspore/pulls/3967))
      * change IndexedSlices to RowTensor([!4031](https://gitee.com/mindspore/mindspore/pulls/4031))
      * Must set or change parallel mode before any Initializer created([!4801](https://gitee.com/mindspore/mindspore/pulls/4801))
* Executor and performance optimization
    * MindSpore graph compilation process performance improved by 20%.
    * Decoupling C++ and Python modules to achieve separate compilation of core modules.
* Data processing, augmentation, and save format
    * Support automatic data augmentation
    * Support GNN distributed cache in single node
    * Support ConcatDataset using distributed sampler

#### Other Hardware Support
* GPU platform
    * New model supported: VGG16, ResNet101, DeepFM.
    * Support some distributed operators in ResNet50 and Wide&Deep.
    * Support automatic parallel for Wide&Deep.
    * Support function funcs[i](*inputs) (such as switch-case).
    * Support distributed training with parameter server.
    * Support GPU operator profiling.
    * Performance optimization of the distributed training with allreduce.
    * Performance optimization of the mixed precision training.
    * Performance optimization of the pynative mode.
    * Performance optimization of the convolution operator, batch normalization operator.
* CPU platform
    * Support MobileNetV2 Re-Training: Re-train the network with different class number.

### MindSpore Lite
* Converter
    * Support third-party models, including TFLite/Caffe/ONNX.
    * Add 93 TFLite op.
    * Add 24 Caffe op.
    * Add 62 ONNX op.
    * Add 11 optimized passes, include fusion/const fold.
    * Support aware-training and Post-training quantization.
* CPU
    * Add 100+ops，support fp32, int8/uint8, FP16 ops
    * Support fast convolution algorithms: Sliding Window, Img2col + Gemm, Strassen, Winograd
    * Support assembly/neon instruction.
    * Support CPU fp16 and sdot on ARM v8.2+.
* GPU
    * Add 20+ ops for OpenCL.
    * Support image2D/buffer format.
    * Optimize online initialization time.
    * add optimized convolution1X1/3X3/depthwise/convolution_transposed for OpenCL.
* Tool & example
    * Add benchmark and TimeProfile tools.
    * Add image classification Android Demo.

## Bugfixes
* Models
  * normalize the readme file([!5410](https://gitee.com/mindspore/mindspore/pulls/5410))
  * fix a sink_size bug for transformer([!5393](https://gitee.com/mindspore/mindspore/pulls/5393))
  * fix bool type optional for resnet50([!5363](https://gitee.com/mindspore/mindspore/pulls/5363))
* Python API
  * improve interface '__bool__' for tensor([!4000](https://gitee.com/mindspore/mindspore/pulls/4000))
  * fix GPU-ResizeNearestNeighbor([!3760](https://gitee.com/mindspore/mindspore/pulls/3760))
  * fix topK multi dimention grad func([!3711](https://gitee.com/mindspore/mindspore/pulls/3711))
  * fix scatterop error msg([!3699](https://gitee.com/mindspore/mindspore/pulls/3699))
  * fix bug of cast dtype when using mix_presion in pynative mode([!3730](https://gitee.com/mindspore/mindspore/pulls/3730))
* Executor
  * fix etsnet train error when UnsegmentSum's first input shape is (1,) ([!4573](https://gitee.com/mindspore/mindspore/pulls/4573))
  * fix bug of result error in while control flow because of unsupporting for value reference ([!4103](https://gitee.com/mindspore/mindspore/pulls/4103))
  * fix bug of the output tensor does not carry device data type ([!3774](https://gitee.com/mindspore/mindspore/pulls/3774))
  * fix bug of avoiding multi attr value are eliminated in pynative mode ([!4225](https://gitee.com/mindspore/mindspore/pulls/4225))
  * fix bug of AssignAdd unable to work normally in multi-cases ([!5171](https://gitee.com/mindspore/mindspore/pulls/5171))
* GPU platform
  * improve the environment variable checking for nvcc compiler path ([!5140](https://gitee.com/mindspore/mindspore/pulls/5140))
  * fix bug of error in cast operator conversion from fp16 to fp32 ([!4147](https://gitee.com/mindspore/mindspore/pulls/4147))
  * fix bug of the array out of bound in case of make_tuple operator ([!5219](https://gitee.com/mindspore/mindspore/pulls/5219))
* Data processing and Pro
  * fix GeneratorDataset time out([!3624](https://gitee.com/mindspore/mindspore/pulls/3624))
  * fix concat operator get_dataset_size error([!4701](https://gitee.com/mindspore/mindspore/pulls/4701))
  * fixing python validator for Repeat Op([!4366](https://gitee.com/mindspore/mindspore/pulls/4366))
* Third party
    * Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    * Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors
Thanks goes to these wonderful people:

Adel, Alexey, andy, andy_wangrui, anthonyaje, anzhengqi, askmiao, avakh, baihuawei, bingyaweng, BowenK, buxue, caifubi, CaoJian, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chenzupeng, chujinjin, cjh9368, Corleone, cristoval, danish, dengyutao, eric, Eric, ervinzhang, etone-chan, fangzehua, fary86, fuzhiye, gengdongjie, genglishuai, Giancarlo, gongdaguo, gukecai, guohongzilong, GuoMengHao, hangq, hanhaocheng, hanhuifeng2020, hanjun996, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, hongxing, huangdongrun, huanghui, huangxinjing, islam_amin, Jesse, jianghui58, jiangzhiwen, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, kai00, kingfo, kpy, kswang, laiyongqiang, leilei_snow, leopz, Li, liangzelang, lianliguang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, lingyunli63, linqingke, lirongzhen1, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuzhongkai, Lixia, lixian, liyong, lizhenyu, looop5, luoyang, lvchangquan, lvliang, lvwenyuan, lyvette, mahdi, Mahdi, mamba_ni, maning202007, Margaret_wangrui, mayang, meixiaowei, meng_chunyang, ms_yan, nhussain, panbingao, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, pengyongrong, Pengyongrong, qianlong, qujianwei, root, shenwei41, shibeiji, simson, songhonglei413, Su, sunsuodong, suteng, tao_yunhao, TFbunny, tinazhang, tom__chen, tony_liu2, tronzhang, VectorSL, wandongdong, wangdongxu, wanghua, wangmin, wangshaocong, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, wuyongkang, xiefangqi, xuanyue, Xun, xutianchun, xuyongfei, yanghaitao, yangjie159, YangLuo, yangruoqi713, yangyongjie, yangzhenzhang, yankai, yao_yf, yelihua, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zhangxuetong, zhaizhiqiang, Zhang, zhangxinfeng3, zhangxuetong, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaoting, zhaozhenlong, zhengjun10, zhongligeng, zhoufeng, zhousiyi, zhouyaqiang, zhouyuanshen, Zichun, Zirui, zjun, zongha, ZPaC, lijiaqi, liangchenghui, wangminggui

Contributions of any kind are welcome!


# Release 0.6.0-beta

## Major Features and Improvements
### Ascend 910 Training and Inference Framework
* New models
    * There are official, research and community under modelzoo.
        * Official is maintained  with the newest APIs by MindSpore team,  MaskRCNN are added.
        * Research is uploaded by researchers for official review, and APIs may not  be updated in time.
        * Community reprints the relevant links of partner research results.
    * Hub added on the same level as modelzoo, synchronous storage of materials needed for official hub web pages which will be launched soon.
    * Support pre-trained models, few lines of code can be used to download and load pre-trained models, supporting inference or transfer learning.
* Frontend and user interface
    * Supports user side operator compilation and graph execution error rendering.
    * Uniform definition dynamic learning rate behavior in optimizers.
    * Support IndexSlice in sparse expression.
    * Support use parent construct method during construct.
    * Support asynchronous execution save checkpoint file.
    * Support implicit type conversion in pynative mode.
    * User interfaces change log
      * unform learning rate behavior in optimizers([!2755](https://gitee.com/mindspore/mindspore/pulls/2755))
      * rename operator of sparse optimizer([!3217](https://gitee.com/mindspore/mindspore/pulls/3217))
      * move profiler module from mindinsight to mindspore([!3075](https://gitee.com/mindspore/mindspore/pulls/3075))
      * VOCDataset output change to multi-columns([!3093](https://gitee.com/mindspore/mindspore/pulls/3093))
      * GetDatasize feature([!3212](https://gitee.com/mindspore/mindspore/pulls/3212))
      * dataset: modify config api([!2936](https://gitee.com/mindspore/mindspore/pulls/2936))
* Executor and performance optimization
    * Decouple C++ and python, so make the architecture more extensible.
    * Parameter Server for distributed deep learning supported.
    * Serving：a flexible service deployment framework for deep learning models.
    * Memory reuse is enhanced, and the batch size of Bert large model is increased from 96 to 160 on a single server.
* Data processing, augmentation, and save format
    * Support MindRecord save operator after  date processing
    * Support automatic fusion operator, such as decode/resize/crop
    * Support CSV dataset loading
### Other Hardware Support
* GPU platform
    * New model supported: ResNext50, WarpCTC and GoogLeNet.
    * Support hyperparametric search and data enhanced automl on GPU.
    * Support Resnet50 automatic parallel in GPU backend.

## Bugfixes
* Models
  * Improved the performance and accuracy on ResNet50([!3456](https://gitee.com/mindspore/mindspore/pulls/3456))
  * Fixed the performance test case of bert([!3486](https://gitee.com/mindspore/mindspore/pulls/3486))
* Python API
  * Fix assign used in while loop([!2720](https://gitee.com/mindspore/mindspore/pulls/2720))
  * Revert optimize the graph output of all nop node.([!2857](https://gitee.com/mindspore/mindspore/pulls/2857))
  * Print tensor as numpy.([!2859](https://gitee.com/mindspore/mindspore/pulls/2859))
  * Support weight decay for sparse optimizer([!2668](https://gitee.com/mindspore/mindspore/pulls/2668))
  * Fix BatchToSpaceND([!2741](https://gitee.com/mindspore/mindspore/pulls/2741))
  * Fixing type check mistakes of InplaceAdd and Inplace Sub ops([!2744](https://gitee.com/mindspore/mindspore/pulls/2744]))
  * Change order param only equal to group param([!2748](https://gitee.com/mindspore/mindspore/pulls/2748))
* Executor
  * The performance of graph whith control flow is optimized([!2931](https://gitee.com/mindspore/mindspore/pulls/2931))
  * Fix bug of wrong number of tuple layers([!3390](https://gitee.com/mindspore/mindspore/pulls/3390))
  * Fix cpu multi graph memory exception([!3631](https://gitee.com/mindspore/mindspore/pulls/3631))
  * Enable data sync when calling operator without defining a cell([!3081](https://gitee.com/mindspore/mindspore/pulls/3081))
  * Fix argmaxwith value error in pynative mode on GPU([!3082](https://gitee.com/mindspore/mindspore/pulls/3082))
  * Fix precision error with fp16 input on pynative mode([!3196](https://gitee.com/mindspore/mindspore/pulls/3196))
* Data processing
    * Fix bug of RandomColor and RandomSharpness default parameter checking  ([!2833](https://gitee.com/mindspore/mindspore/pulls/2833))
    * Fix process hung when training and eval  ([!3469](https://gitee.com/mindspore/mindspore/pulls/3469))
* Third party
    * Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    * Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors
Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# Release 0.5.2-beta

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* New models
    * DenseNet121: a convolution based neural network for the task of image classification on ImageNet 2012 dataset.

## Bugfixes
* Models
    * VGG16,Alexnet,GoogleNet,optimize network for better performance. ([!5539](https://gitee.com/mindspore/mindspore/pulls/5539))
    * YOLOV3, fix yolov3_darknet53 dataset bug. ([!5658](https://gitee.com/mindspore/mindspore/pulls/5658))

## Contributors
Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!


# Release 0.5.0-beta

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* New models
    * ResNext50: a simple, highly modularized network architecture using aggregated resdiual transformations for image classification on ImageNet 2012 dataset.
    * MASS: a pre-training method for sequence to sequence based language generation tasks on Text Summarization and Conversational Response Generation using News Crawls 2007-2017 dataset, Gigaword corpus and Cornell movie dialog corpus.
    * Transformer: a neural network architecture for language understanding on WMT 2014 English-German dataset.
    * GCN：Graph Convolutional Networks for the task of classification of nodes in a graph on Cora and Citeseer datasets.
    * GAT：an attention-based graph neural network for node classification on Cora and CiteSeer dataset.
* Frontend and user interface
    * Support tensor value and assignment of mixed tensor index in graph mode.
    * Support tensor comparison, len operator, constexpr syntax, value and assignment of tensor index in pynative mode.
    * Support converting MindSpore IR to pb format for infer model.
    * Support print operator to write data directly on the hard disk.
    * Add the double recursive programming solution for very high speed parallel strategy search in automatic parallel.
    * User interfaces change log
      * Allow the learning rate of AdamWeightDecayDynamicLR and Lamb to be 0([!1826](https://gitee.com/mindspore/mindspore/pulls/1826))
      * Restricting the entire network input parameter is Tensor([!1967](https://gitee.com/mindspore/mindspore/pulls/1967))
      * Turn shape and dtype into attributes instead of interfaces([!1919](https://gitee.com/mindspore/mindspore/pulls/1919))
      * Delete multitypefungraph([!2116](https://gitee.com/mindspore/mindspore/pulls/2116))
      * Refactor the callback module in an encapsulated way, use _CallbackManager instead of _build_callbacks([!2236](https://gitee.com/mindspore/mindspore/pulls/2236))
      * Delete EmbeddingLookup([!2163](https://gitee.com/mindspore/mindspore/pulls/2163))
      * Checkpoint add model_type([!2517](https://gitee.com/mindspore/mindspore/pulls/2517))
* Executor and performance optimization
    * Heterogeneous execution on CPU and Ascend devices supported, and is verified in Wide&Deep model.
    * Quantitative training of MobileNetV2, Lenet and Resnet50 on Ascend-910 are supported.
    * Support new fusion architecture, which can do fusion optimization across graphs and kernels to improve execution speed.
* Data processing, augmentation, and save format
    * Support data processing pipeline performance profiling.
    * Support public dataset loading, such as CLUE and Coco.
    * Support more text processing, such as more tokenizers and vocab data.
    * Support MindRecord padded data.
### Other Hardware Support
* GPU platform
    * New model supported: Bert / Wide&Deep.
    * Support setting max device memory.
* CPU platform
    * New model supported: LSTM.

## Bugfixes
* Models
    * Bert, Move Bert from `example` to `model_zoo`, optimize network for better performance. ([!1902](https://gitee.com/mindspore/mindspore/pulls/1902))
    * VGG16, Move VGG16 from `example` to `model_zoo`, optimize network for better accuracy. ([!2645](https://gitee.com/mindspore/mindspore/pulls/2645))
    * Alexnet, modify parameter setting to improve accuracy ([!1364](https://gitee.com/mindspore/mindspore/pulls/2370))
    * Wide&Deep, Move Wide&Deep from `example` to `model_zoo`, optimize network for better performance. ([!2221](https://gitee.com/mindspore/mindspore/pulls/2221))
* Python API
    * Fix bug in auto cast([!1766](https://gitee.com/mindspore/mindspore/pulls/1766))
    * Fix bug of register_backward_hook([!2148](https://gitee.com/mindspore/mindspore/pulls/2148))
    * Fix bug of tuple args in pynative mode([!1878](https://gitee.com/mindspore/mindspore/pulls/1878))
    * Fix bug of checking numbers of arguments and graph parameters([!1701](https://gitee.com/mindspore/mindspore/pulls/1701))
* Executor
    * Fix bug of loading input data repeatedly in pynative mode([!1966](https://gitee.com/mindspore/mindspore/pulls/1966))
    * Fix bug of list cannot be used as input in pynative mode([!1765](https://gitee.com/mindspore/mindspore/pulls/1765))
    * Fix bug of kernel select ([!2103](https://gitee.com/mindspore/mindspore/pulls/2103))
    * Fix bug of pattern matching for batchnorm fusion in the case of auto mix precision.([!1851](https://gitee.com/mindspore/mindspore/pulls/1851))
    * Fix bug of generate hccl's kernel info.([!2393](https://gitee.com/mindspore/mindspore/mindspore/pulls/2393))
* GPU platform
    * Fix bug of summary feature invalid([!2173](https://gitee.com/mindspore/mindspore/pulls/2173))
* Data processing
    * Fix bug of Cifar dataset reading([!2096](https://gitee.com/mindspore/mindspore/pulls/2096))
    * Fix bug of C++ behavior in RandomCropAndResize([!2026](https://gitee.com/mindspore/mindspore/pulls/2026))
    * Fix the bug of mindrecord shuffle([!2420](https://gitee.com/mindspore/mindspore/pulls/2420))
* Third party
    * Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).

## Contributors
Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# Release 0.3.1-alpha

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* Frontend and User Interface
    * Independent model init interface.
* Data processing, augmentation, and save format
    * Support sample padding for minddataset.

## Bugfixes
* Python API
    * Fix bugs in the lars optimizer([!1894](https://gitee.com/mindspore/mindspore/pulls/1894))
* Data processing
    * Fix accuracy problem of RandomCropDecodeResize ([!2340](https://gitee.com/mindspore/mindspore/pulls/2340))

# Release 0.3.0-alpha

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* New models
    * DeepFM: a factorization-machine based neural network for CTR prediction on Criteo dataset.
    * DeepLabV3: significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2007 semantic image segmentation benchmark.
    * Faster-RCNN: towards real-time object detection with region proposal networks on COCO 2017 dataset.
    * SSD: a single stage object detection methods on COCO 2017 dataset.
    * GoogLeNet: a deep convolutional neural network architecture codenamed Inception V1 for classification and detection on CIFAR-10 dataset.
    * Wide&Deep: jointly trained wide linear models and deep neural networks for recommender systems on Criteo dataset.
* Frontend and User Interface
    * Complete numpy advanced indexing method. Supports value and assignment through tensor index.
    * Some optimizers support separating parameter groups. Different parameter groups can set different `learning_rate` and `weight_decay`.
    * Support setting submodule's logging level independently, e.g. you can set logging level of module `A` to warning and set logging level of module `B` to info.
    * Support weights to be compiled according to shape to solve the problem of large memory overhead.
    * Add some operators implement and grammar support in pynative mode. To be consistent with graph mode.
    * User interfaces change log
      * Learning rate and weight decay making group params([!637](https://gitee.com/mindspore/mindspore/pulls/637))
      * Support weights to be compiled according to shape([!1015](https://gitee.com/mindspore/mindspore/pulls/1015))
      * delete some context param([!1100](https://gitee.com/mindspore/mindspore/pulls/1100))
      * ImageSummary/ScalarSummary/TensorSummary/HistogramSummary([!1329](https://gitee.com/mindspore/mindspore/pulls/1329))([!1425](https://gitee.com/mindspore/mindspore/pulls/1425))
* Executor and Performance Optimization
    * Support doing evaluation while in training process, so that the accuracy of training can be easily obtained.
    * Enable second-order optimization for resnet50, which can achieve 75.9% accuracy in 45 epochs (Resnet50 @ImageNet).
    * Optimize pynative implementation and improve it's execution performance.
    * Optimize summary record implementation and improve its performance.
* Data processing, augmentation, and save format
    * Support simple text processing, such as tokenizer/buildvocab/lookup.
    * Support padding batch.
    * Support split or concat dataset.
    * Support MindDataset reading from file list.

### Other Hardware Support
* GPU platform
    * New models supported: MobileNetV2, MobileNetV3.
    * Support mixed precision training.
    * Support device memory swapping.

## Bugfixes
* Python API
    * An exception to the broadcast input data type check([!712](https://gitee.com/mindspore/mindspore/pulls/712))
    * Fix issues assignsub return value 0([!1036](https://gitee.com/mindspore/mindspore/pulls/1036))
    * Fix issue Conv2dBackpropInput bprop should return 3 instead of 2 items([!1001](https://gitee.com/mindspore/mindspore/pulls/1001))
    * Fix sens shape error of TrainOneStepWithLossScaleCell([!1050](https://gitee.com/mindspore/mindspore/pulls/1050))
    * Fix BatchNormGrad operator([!1344](https://gitee.com/mindspore/mindspore/pulls/1344))
* Executor
    * Fix dropout，topK and addn errors in PyNative mode ([!1285](https://gitee.com/mindspore/mindspore/pulls/1285), [!1138](https://gitee.com/mindspore/mindspore/pulls/1138), [!1033](https://gitee.com/mindspore/mindspore/pulls/1033)).
    * Fix memory leaks after execution in PyNatvie mode ([!1201](https://gitee.com/mindspore/mindspore/pulls/1201)).
    * Fix HCCL failure in some special scenes ([!1204](https://gitee.com/mindspore/mindspore/pulls/1204), [!1252](https://gitee.com/mindspore/mindspore/pulls/1252)).
    * Fix SSD network when Select failed, cann't find kernel info([!1449](https://gitee.com/mindspore/mindspore/pulls/1449)).
    * Fix Topk operator selection strategy bug between aicore and aicpu([!1367](https://gitee.com/mindspore/mindspore/pulls/1367)).
    * Fix input memory size of 'assign' op unequal in control sink mode when assigning a data from one child graph to another child graph([!802](https://gitee.com/mindspore/mindspore/pulls/802)).
    * Fix allreduce ir inconsistency([!989](https://gitee.com/mindspore/mindspore/pulls/989)).
* GPU platform
    * Fix summary for gradient collection ([!1364](https://gitee.com/mindspore/mindspore/pulls/1364))
    * Fix the slice operator ([!1489](https://gitee.com/mindspore/mindspore/pulls/1489))
* Data processing
    * Fix memory problems of GeneratorDataset of sub-process ([!907](https://gitee.com/mindspore/mindspore/pulls/907))
    * Fix getting data timeout when training the cifar10 dataset under the lenet([!1391](https://gitee.com/mindspore/mindspore/pulls/1391))

## Contributors
Thanks goes to these wonderful people:

Alexey Shevlyakov, Amir Lashkari, anthony, baihuawei, biffex, buxue, caifubi, candanzg, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenzomi, chujinjin, cristoval, dengwentao, eric, etone-chan, fary86, gaojing, gengdongjie, gongchen, guohongzilong, guozhijian, heleiwang, hesham, He Wei, Hoai Linh Tran, hongxing, huangdongrun, huanghui, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jonwe, jonyguo, Junhan Hu, Kang, kingfo, kswang, laiyongqiang, leopz, lichenever, lihongkang, limingqi107, liubuyu, liuliyan2, liuwenhao4, liuxiao, liuxiao, liyong, lizhenyu, lvliang, Margaret_wangrui, meixiaowei, ms_yan, Nat Sutyanyong, ougongchang, panfengfeng, panyifeng, Peilin Wang, peixu_ren, qianlong, rick_sanchez, seatea, sheng, shijianning, simson, sunsuodong, Tinazhang, VectorSL, wandongdong, wangcong, wanghua, wangnan39, Wei Luning, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuxuejian, Xiaoda Zhang, xiefangqi, xulei2020, Yang, yangjie159, yangruoqi713, yangyongjie, yangzhenzhang, Yanjun Peng, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yujianfeng, YuJianfeng, yvetteliu, zhangdengcheng, Zhang Qinghua, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, zhouyuanshen, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang

Contributions of any kind are welcome!

# Release 0.2.0-alpha

## Major Features and Improvements

### Ascend 910 Training and Inference Framework
* New models
    * MobileNetV2: Inverted Residuals and Linear Bottlenecks.
    * ResNet101: Deep Residual Learning for Image Recognition.

* Frontend and User Interface
   * Support for all python comparison operators.
   * Support for math operators **,//,%. Support for other python operators like and/or/not/is/is not/ in/ not in.
   * Support for the gradients of function with variable arguments.
   * Support for tensor indexing assignment for certain indexing type.
   * Support for dynamic learning rate.
   * User interfaces change log
     * DepthwiseConv2dNative, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropInput([!424](https://gitee.com/mindspore/mindspore/pulls/424))
     * ReLU6, ReLU6Grad([!224](https://gitee.com/mindspore/mindspore/pulls/224))
     * GeneratorDataset([!183](https://gitee.com/mindspore/mindspore/pulls/183))
     * VOCDataset([!477](https://gitee.com/mindspore/mindspore/pulls/477))
     * MindDataset, PKSampler([!514](https://gitee.com/mindspore/mindspore/pulls/514))
     * map([!506](https://gitee.com/mindspore/mindspore/pulls/506))
     * Conv([!226](https://gitee.com/mindspore/mindspore/pulls/226))
     * Adam([!253](https://gitee.com/mindspore/mindspore/pulls/253))
     * _set_fusion_strategy_by_idx, _set_fusion_strategy_by_size([!189](https://gitee.com/mindspore/mindspore/pulls/189))
     * CheckpointConfig([!122](https://gitee.com/mindspore/mindspore/pulls/122))
     * Constant([!54](https://gitee.com/mindspore/mindspore/pulls/54))
* Executor and Performance Optimization
    * Support parallel execution of data prefetching and forward/backward computing.
    * Support parallel execution of gradient aggregation and forward/backward computing in distributed training scenarios.
    * Support operator fusion optimization.
    * Optimize compilation process and improve the performance.
* Data processing, augmentation, and save format
    * Support multi-process of GeneratorDataset/PyFunc for high performance
    * Support variable batchsize
    * Support new Dataset operators, such as filter,skip,take,TextLineDataset

### Other Hardware Support
* GPU platform
    * Use dynamic memory pool by default on GPU.
    * Support parallel execution of computation and communication.
    * Support continuous address allocation by memory pool.
* CPU platform
    * Support for windows 10 OS.

## Bugfixes
* Models
    * Fix mixed precision bug for VGG16 model ([!629](https://gitee.com/mindspore/mindspore/pulls/629)).
* Python API
    * Fix ControlDepend operator bugs on CPU and GPU ([!396](https://gitee.com/mindspore/mindspore/pulls/396)).
    * Fix ArgMinWithValue operator bugs ([!338](https://gitee.com/mindspore/mindspore/pulls/338)).
    * Fix Dense operator bugs on PyNative mode ([!276](https://gitee.com/mindspore/mindspore/pulls/276)).
    * Fix MatMul operator bugs on PyNative mode ([!288](https://gitee.com/mindspore/mindspore/pulls/288)).
* Executor
    * Fix operator selection bugs and make it general ([!300](https://gitee.com/mindspore/mindspore/pulls/300)).
    * Fix memory reuse bug for GetNext op ([!291](https://gitee.com/mindspore/mindspore/pulls/291)).
* GPU platform
    * Fix memory allocation in multi-graph scenarios ([!444](https://gitee.com/mindspore/mindspore/pulls/444)).
    * Fix bias_add_grad under fp16 precision ([!598](https://gitee.com/mindspore/mindspore/pulls/598)).
    * Fix support for fp16 kernels on nvidia 1080Ti([!571](https://gitee.com/mindspore/mindspore/pulls/571)).
    * Fix parsing of tuple type parameters ([!316](https://gitee.com/mindspore/mindspore/pulls/316)).
* Data processing
    * Fix TypeErrors about can't pickle mindspore._c_dataengine.DEPipeline objects([!434](https://gitee.com/mindspore/mindspore/pulls/434)).
    * Add TFRecord file verification([!406](https://gitee.com/mindspore/mindspore/pulls/406)).

## Contributors
Thanks goes to these wonderful people:

Alexey_Shevlyakov, Cathy, Chong, Hoai, Jonathan, Junhan, JunhanHu, Peilin, SanjayChan, StrawNoBerry, VectorSL, Wei, WeibiaoYu, Xiaoda, Yanjun, YuJianfeng, ZPaC, Zhang, ZhangQinghua, ZiruiWu, amongo, anthonyaje, anzhengqi, biffex, caifubi, candanzg, caojian05, casgj, cathwong, ch-l, chang, changzherui, chenfei, chengang, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, dengwentao, dinghao, fanglei, fary86, flywind, gaojing, geekun, gengdongjie, ghzl, gong, gongchen, gukecai, guohongzilong, guozhijian, gziyan, h.farahat, hesham, huangdongrun, huanghui, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, jonathan_yan, jonyguo, jzw, kingfo, kisnwang, laiyongqiang, leonwanghui, lianliguang, lichen, lichenever, limingqi107, liubuyu, liuxiao, liyong, liyong126, lizhenyu, lupengcheng, lvliang, maoweiyong, ms_yan, mxm, ougongchang, panfengfeng, panyifeng, pengyanjun, penn, qianlong, seatea, simson, suteng, thlinh, vlne-v1, wangchengke, wanghua, wangnan39, wangqiuliang, wenchunjiang, wenkai, wukesong, xiefangqi, xulei, yanghaitao, yanghaoran, yangjie159, yangzhenzhang, yankai10, yanzhenxiang2020, yao_yf, yoonlee666, zhangbuxue, zhangz0911gm, zhangzheng, zhaojichen, zhaoting, zhaozhenlong, zhongligeng, zhoufeng, zhousiyi, zjun, zyli2020, yuhuijun, limingqi107, lizhenyu, chenweifeng.

Contributions of any kind are welcome!

# Release 0.1.0-alpha

## Main Features

### Ascend 910 Training and Inference Framework
* Recommended OS: Ubuntu 16.04 (or later) or EulerOS 2.5 or EulerOS 2.8
* Python version: 3.7.5
* Preset models
    * ResNet-50: residual structure-based convolutional neural network (CNN) for image classification, which is widely used.
    * AlexNet: classic CNN for image classification, achieving historical results in ImageNet LSVRC-2012.
    * LeNet: classic CNN for image classification, which was proposed by Yann LeCun.
    * VGG16: classic CNN for image classification, which was proposed by Oxford Visual Geometry Group.
    * YoloV3: real-time object detection network.
    * NEZHA: BERT-based Chinese pre-training network produced by Huawei Noah's Ark Laboratory.
* Execution modes
    * Graph mode: provides graph optimization methods such as memory overcommitment, IR fusion, and buffer fusion to achieve optimal execution performance.
    * PyNative mode: single-step execution mode, facilitating process debugging.
* Debugging capability and methods
    * Save CheckPoints and Summary data during training.
    * Support asynchronous printing.
    * Dump the computing data.
    * Support profiling analysis of the execution process performance.
* Distributed execution
    * Support AllReduce, AllGather, and BroadCast collective communication.
    * AllReduce data parallel: Each device obtains different training data, which accelerates the overall training process.
    * Collective communication-based layerwise parallel: Models are divided and allocated to different devices to solve the problem of insufficient memory for large model processing and improve the training speed.
    * Automatic parallel mode: The better data and model parallel mode can be predicted based on the cost model. It is recommended that this mode be used on ResNet series networks.
* Automatic differentiation
    * Implement automatic differentiation based on Source to Source.
    * Support distributed scenarios and automatic insertion of reverse communication operators.
* Data processing, augmentation, and save format
    * Load common datasets such as ImageNet, MNIST, CIFAR-10, and CIFAR-100.
    * Support common data loading pipeline operations, such as shuffle, repeat, batch, map, and sampler.
    * Provide basic operator libraries to cover common CV scenarios.
    * Support users to customize Python data augmentation operators through the Pyfunc mechanism.
    * Support the access of user-defined datasets through the GeneratorDataset mechanism.
    * Provide the MindSpore data format, data aggregation and storage, random access example, data partition, efficient parallel read, user-defined index, and dataset search.
    * Convert user datasets to the MindSpore data format.
    * After data processing and augmentation, provide training applications in feed and graph modes.
* FP32/16 mixed precision computation, supporting automatic and manual configuration
* Provide common operators such as nn, math, and array, which can be customized.

### Inference Deployment
* Deploy models in MindSpore format on the Ascend 310 platform for inference.
* Save models in ONNX format.
* Support saving models in LITE format and running models based on the lightweight inference framework.
    * Recommended OS: Android 4.3 or later
    * Supported network type: LeNet
    * Provide the generalization operators generated by TVM and operators generated after specific networks are tuned.

### Other Hardware Support
* GPU platform training
    * Recommended OS: Ubuntu 16.04
    * CUDA version: 9.2 or 10.1
    * CuDNN version: 7.6 or later
    * Python version: 3.7.5
    * NCCL version: 2.4.8-1
    * OpenMPI version: 3.1.5
    * Supported models: AlexNet, LeNet, and LSTM
    * Supported datasets: MNIST and CIFAR-10
    * Support data parallel.
* CPU platform training
    * Recommended OS: Ubuntu 16.04
    * Python version: 3.7.5
    * Supported model: LeNet
    * Supported dataset: MNIST
    * Provide only the stand-alone operation version.

## Peripherals and Tools
* [MindSpore Official Website] (https://www.mindspore.cn/)
* [MindInsight Visualization Debugging and Optimization] (https://gitee.com/mindspore/mindinsight)
* [MindArmour Model Security Hardening Package] (https://gitee.com/mindspore/mindarmour)
* [GraphEngine Computational Graph Engine] (https://gitee.com/mindspore/graphengine)
