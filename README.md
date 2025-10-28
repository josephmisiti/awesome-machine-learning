# Awesome Machine Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Track Awesome List](https://www.trackawesomelist.com/badge.svg)](https://www.trackawesomelist.com/josephmisiti/awesome-machine-learning/)

A curated list of awesome machine learning frameworks, libraries and software (by language). Inspired by `awesome-php`.

_If you want to contribute to this list (please do), send me a pull request or contact me [@josephmisiti](https://twitter.com/josephmisiti)._
Also, a listed repository should be deprecated if:

* Repository's owner explicitly says that "this library is not maintained".
* Not committed for a long time (2~3 years).

Further resources:

* For a list of free machine learning books available for download, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md).

* For a list of professional machine learning events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/events.md).

* For a list of (mostly) free machine learning courses available online, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md).

* For a list of blogs and newsletters on data science and machine learning, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/blogs.md).

* For a list of free-to-attend meetups and local events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/meetups.md).

## Table of Contents

### Frameworks and Libraries
<!-- MarkdownTOC depth=4 -->
<!-- Contents-->
- [Awesome Machine Learning ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](#awesome-machine-learning-)
  - [Table of Contents](#table-of-contents)
    - [Frameworks and Libraries](#frameworks-and-libraries)
    - [Tools](#tools)
  - [APL](#apl)
      - [General-Purpose Machine Learning](#apl-general-purpose-machine-learning)
  - [C](#c)
      - [General-Purpose Machine Learning](#c-general-purpose-machine-learning)
      - [Computer Vision](#c-computer-vision)
  - [C++](#cpp)
      - [Computer Vision](#cpp-computer-vision)
      - [General-Purpose Machine Learning](#cpp-general-purpose-machine-learning)
      - [Natural Language Processing](#cpp-natural-language-processing)
      - [Speech Recognition](#cpp-speech-recognition)
      - [Sequence Analysis](#cpp-sequence-analysis)
      - [Gesture Detection](#cpp-gesture-detection)
      - [Reinforcement Learning](#cpp-reinforcement-learning)
  - [Common Lisp](#common-lisp)
      - [General-Purpose Machine Learning](#common-lisp-general-purpose-machine-learning)
  - [Clojure](#clojure)
      - [Natural Language Processing](#clojure-natural-language-processing)
      - [General-Purpose Machine Learning](#clojure-general-purpose-machine-learning)
      - [Deep Learning](#clojure-deep-learning)
      - [Data Analysis](#clojure-data-analysis--data-visualization)
      - [Data Visualization](#clojure-data-visualization)
      - [Interop](#clojure-interop)
      - [Misc](#clojure-misc)
      - [Extra](#clojure-extra)
  - [Crystal](#crystal)
      - [General-Purpose Machine Learning](#crystal-general-purpose-machine-learning)
  - [Elixir](#elixir)
      - [General-Purpose Machine Learning](#elixir-general-purpose-machine-learning)
      - [Natural Language Processing](#elixir-natural-language-processing)
  - [Erlang](#erlang)
      - [General-Purpose Machine Learning](#erlang-general-purpose-machine-learning)
  - [Fortran](#fortran)
      - [General-Purpose Machine Learning](#fortran-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#fortran-data-analysis--data-visualization)
  - [Go](#go)
      - [Natural Language Processing](#go-natural-language-processing)
      - [General-Purpose Machine Learning](#go-general-purpose-machine-learning)
      - [Spatial analysis and geometry](#go-spatial-analysis-and-geometry)
      - [Data Analysis / Data Visualization](#go-data-analysis--data-visualization)
      - [Computer vision](#go-computer-vision)
      - [Reinforcement learning](#go-reinforcement-learning)
  - [Haskell](#haskell)
      - [General-Purpose Machine Learning](#haskell-general-purpose-machine-learning)
  - [Java](#java)
      - [Natural Language Processing](#java-natural-language-processing)
      - [General-Purpose Machine Learning](#java-general-purpose-machine-learning)
      - [Speech Recognition](#java-speech-recognition)
      - [Data Analysis / Data Visualization](#java-data-analysis--data-visualization)
      - [Deep Learning](#java-deep-learning)
  - [Javascript](#javascript)
      - [Natural Language Processing](#javascript-natural-language-processing)
      - [Data Analysis / Data Visualization](#javascript-data-analysis--data-visualization)
      - [General-Purpose Machine Learning](#javascript-general-purpose-machine-learning)
      - [Misc](#javascript-misc)
      - [Demos and Scripts](#javascript-demos-and-scripts)
  - [Julia](#julia)
      - [General-Purpose Machine Learning](#julia-general-purpose-machine-learning)
      - [Natural Language Processing](#julia-natural-language-processing)
      - [Data Analysis / Data Visualization](#julia-data-analysis--data-visualization)
      - [Misc Stuff / Presentations](#julia-misc-stuff--presentations)
  - [Kotlin](#kotlin)
      - [Deep Learning](#kotlin-deep-learning)
  - [Lua](#lua)
      - [General-Purpose Machine Learning](#lua-general-purpose-machine-learning)
      - [Demos and Scripts](#lua-demos-and-scripts)
  - [Matlab](#matlab)
      - [Computer Vision](#matlab-computer-vision)
      - [Natural Language Processing](#matlab-natural-language-processing)
      - [General-Purpose Machine Learning](#matlab-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#matlab-data-analysis--data-visualization)
  - [.NET](#net)
      - [Computer Vision](#net-computer-vision)
      - [Natural Language Processing](#net-natural-language-processing)
      - [General-Purpose Machine Learning](#net-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#net-data-analysis--data-visualization)
  - [Objective C](#objective-c)
    - [General-Purpose Machine Learning](#objective-c-general-purpose-machine-learning)
  - [OCaml](#ocaml)
    - [General-Purpose Machine Learning](#ocaml-general-purpose-machine-learning)
  - [OpenCV](#opencv)
    - [Computer Vision](#opencv-Computer-Vision)
    - [Text-Detection](#Text-Character-Number-Detection)
  - [Perl](#perl)
    - [Data Analysis / Data Visualization](#perl-data-analysis--data-visualization)
    - [General-Purpose Machine Learning](#perl-general-purpose-machine-learning)
  - [Perl 6](#perl-6)
    - [Data Analysis / Data Visualization](#perl-6-data-analysis--data-visualization)
    - [General-Purpose Machine Learning](#perl-6-general-purpose-machine-learning)
  - [PHP](#php)
    - [Natural Language Processing](#php-natural-language-processing)
    - [General-Purpose Machine Learning](#php-general-purpose-machine-learning)
  - [Python](#python)
      - [Computer Vision](#python-computer-vision)
      - [Natural Language Processing](#python-natural-language-processing)
      - [General-Purpose Machine Learning](#python-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#python-data-analysis--data-visualization)
      - [Misc Scripts / iPython Notebooks / Codebases](#python-misc-scripts--ipython-notebooks--codebases)
      - [Neural Networks](#python-neural-networks)
      - [Survival Analysis](#python-survival-analysis)
      - [Federated Learning](#python-federated-learning)
      - [Kaggle Competition Source Code](#python-kaggle-competition-source-code)
      - [Reinforcement Learning](#python-reinforcement-learning)
      - [Speech Recognition](#python-speech-recognition)
  - [Ruby](#ruby)
      - [Natural Language Processing](#ruby-natural-language-processing)
      - [General-Purpose Machine Learning](#ruby-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#ruby-data-analysis--data-visualization)
      - [Misc](#ruby-misc)
  - [Rust](#rust)
      - [General-Purpose Machine Learning](#rust-general-purpose-machine-learning)
      - [Deep Learning](#rust-deep-learning)
      - [Natural Language Processing](#rust-natural-language-processing)
  - [R](#r)
      - [General-Purpose Machine Learning](#r-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#r-data-analysis--data-visualization)
  - [SAS](#sas)
      - [General-Purpose Machine Learning](#sas-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#sas-data-analysis--data-visualization)
      - [Natural Language Processing](#sas-natural-language-processing)
      - [Demos and Scripts](#sas-demos-and-scripts)
  - [Scala](#scala)
      - [Natural Language Processing](#scala-natural-language-processing)
      - [Data Analysis / Data Visualization](#scala-data-analysis--data-visualization)
      - [General-Purpose Machine Learning](#scala-general-purpose-machine-learning)
  - [Scheme](#scheme)
      - [Neural Networks](#scheme-neural-networks)
  - [Swift](#swift)
      - [General-Purpose Machine Learning](#swift-general-purpose-machine-learning)
  - [TensorFlow](#tensorflow)
      - [General-Purpose Machine Learning](#tensorflow-general-purpose-machine-learning)

### [Tools](#tools-1)

- [Neural Networks](#tools-neural-networks)
- [Misc](#tools-misc)


[Credits](#credits)

<!-- /MarkdownTOC -->

<a name="apl"></a>
## APL

<a name="apl-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [naive-apl](https://github.com/mattcunningham/naive-apl) - Naive Bayesian Classifier implementation in APL. **[Deprecated]**

<a name="c"></a>
## C

<a name="c-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [Darknet](https://github.com/pjreddie/darknet) - Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).
* [Hybrid Recommender System](https://github.com/SeniorSA/hybrid-rs-trainner) - A hybrid recommender system based upon scikit-learn algorithms. **[Deprecated]**
* [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [cONNXr](https://github.com/alrevuelta/cONNXr) - An `ONNX` runtime written in pure C (99) with zero dependencies focused on small embedded devices. Run inference on your machine learning models no matter which framework you train it with. Easy to install and compiles everywhere, even in very old devices.
* [libonnx](https://github.com/xboot/libonnx) - A lightweight, portable pure C99 onnx inference engine for embedded devices with hardware acceleration support.
* [onnx-c](https://github.com/onnx/onnx-c) - A lightweight C library for ONNX model inference, optimized for performance and portability across platforms.

<a name="c-computer-vision"></a>
#### Computer Vision

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library.
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has a Matlab toolbox.
* [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics' YOLOv8 implementation with C++ support for real-time object detection and tracking, optimized for edge devices.

<a name="cpp"></a>
## C++

<a name="cpp-computer-vision"></a>
#### Computer Vision

* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models **[Deprecated]**
* [OpenCV](https://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
* [VIGRA](https://github.com/ukoethe/vigra) - VIGRA is a genertic cross-platform C++ computer vision and machine learning library for volumes of arbitrary dimensionality with Python bindings.
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation

<a name="cpp-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Speedster](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster) -Automatically apply SOTA optimization techniques to achieve the maximum inference speed-up on your hardware. [DEEP LEARNING]
* [BanditLib](https://github.com/jkomiyama/banditlib) - A simple Multi-armed Bandit library. **[Deprecated]**
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind. [DEEP LEARNING]
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, contains fast inference implementation and supports CPU and GPU (even multi-GPU) computation.
* [CNTK](https://github.com/Microsoft/CNTK) - The Computational Network Toolkit (CNTK) by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph.
* [CUDA](https://code.google.com/p/cuda-convnet/) - This is a fast C++/CUDA implementation of convolutional [DEEP LEARNING]
* [DeepDetect](https://github.com/jolibrain/deepdetect) - A machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.
* [Distributed Machine learning Tool Kit (DMTK)](http://www.dmtk.io/) - A distributed machine learning (parameter server) framework by Microsoft. Enables training models on large data sets across multiple machines. Current tools bundled with it include: LightLDA and Distributed (Multisense) Word Embedding.
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications.
* [DSSTNE](https://github.com/amznlabs/amazon-dsstne) - A software library created by Amazon for training and deploying deep neural networks using GPUs which emphasizes speed and scale over experimental flexibility.
* [DyNet](https://github.com/clab/dynet) - A dynamic neural network library working well with networks that have dynamic structures that change for every training instance. Written in C++ with bindings in Python.
* [Fido](https://github.com/FidoProject/Fido) - A highly-modular C++ machine learning library for embedded electronics and robotics.
* [FlexML](https://github.com/ozguraslank/flexml) - Easy-to-use and flexible AutoML library for Python.
* [igraph](http://igraph.org/) - General purpose graph library.
* [Intel® oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL) - A high performance software library developed by Intel and optimized for Intel's architectures. Library provides algorithmic building blocks for all stages of data analytics and allows to process data in batch, online and distributed modes.
* [LightGBM](https://github.com/Microsoft/LightGBM) - Microsoft's fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
* [libfm](https://github.com/srendle/libfm) - A generic approach that allows to mimic most factorization models by feature engineering.
* [MLDB](https://mldb.ai) - The Machine Learning Database is a database designed for machine learning. Send it commands over a RESTful API to store data, explore it using SQL, then train machine learning models and expose them as APIs.
* [mlpack](https://www.mlpack.org/) - A scalable C++ machine learning library.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [N2D2](https://github.com/CEA-LIST/N2D2) - CEA-List's CAD framework for designing and simulating Deep Neural Network, and building full DNN-based applications on embedded platforms
* [oneDNN](https://github.com/oneapi-src/oneDNN) - An open-source cross-platform performance library for deep learning applications.
* [Opik](https://www.comet.com/site/products/opik/) - Open source engineering platform to debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards. ([Source Code](https://github.com/comet-ml/opik/))
* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose library with C/C++ interface for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [proNet-core](https://github.com/cnclabs/proNet-core) - A general-purpose network embedding framework: pair-wise representations optimization Network Edit.
* [PyCaret](https://github.com/pycaret/pycaret) - An open-source, low-code machine learning library in Python that automates machine learning workflows.
* [PyCUDA](https://mathema.tician.de/software/pycuda/) - Python interface to CUDA
* [ROOT](https://root.cern.ch) - A modular scientific software framework. It provides all the functionalities needed to deal with big data processing, statistical analysis, visualization and storage.
* [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html) - A fast, modular, feature-rich open-source C++ machine learning library.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [sofia-ml](https://code.google.com/archive/p/sofia-ml) - Suite of fast incremental algorithms.
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling.
* [Timbl](https://languagemachines.github.io/timbl/) - A software package/C++ library implementing several memory-based learning algorithms, among which IB1-IG, an implementation of k-nearest neighbor classification, and IGTree, a decision-tree approximation of IB1-IG. Commonly used for NLP.
* [Vowpal Wabbit (VW)](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast out-of-core learning system.
* [Warp-CTC](https://github.com/baidu-research/warp-ctc) - A fast parallel implementation of Connectionist Temporal Classification (CTC), on both CPU and GPU.
* [XGBoost](https://github.com/dmlc/xgboost) - A parallelized optimized general purpose gradient boosting library.
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) - A fast library for GBDTs and Random Forests on GPUs.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - A fast SVM library on GPUs and CPUs.
* [LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) - A header-only C++11 Neural Network library. Low dependency, native traditional chinese document.
* [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertising and recommender systems.
* [Featuretools](https://github.com/featuretools/featuretools) - A library for automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning using reusable feature engineering "primitives".
* [skynet](https://github.com/Tyill/skynet) - A library for learning neural networks, has C-interface, net set in JSON. Written in C++ with bindings in Python, C++ and C#.
* [Feast](https://github.com/gojek/feast) - A feature store for the management, discovery, and access of machine learning features. Feast provides a consistent view of feature data for both model training and model serving.
* [Hopsworks](https://github.com/logicalclocks/hopsworks) - A data-intensive platform for AI with the industry's first open-source feature store. The Hopsworks Feature Store provides both a feature warehouse for training and batch based on Apache Hive and a feature serving database, based on MySQL Cluster, for online applications.
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform for reproducible and scalable machine learning and deep learning.
* [QuestDB](https://questdb.io/) - A relational column-oriented database designed for real-time analytics on time series and event data.
* [Phoenix](https://phoenix.arize.com) - Uncover insights, surface problems, monitor and fine tune your generative LLM, CV and tabular models.
* [XAD](https://github.com/auto-differentiation/XAD) - Comprehensive backpropagation tool for C++.
* [Truss](https://truss.baseten.co) - An open source framework for packaging and serving ML models.

<a name="cpp-natural-language-processing"></a>
#### Natural Language Processing

* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser).
* [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks. **[Deprecated]**
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data. **[Deprecated]**
* [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
* [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia/)
* [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
* [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
* [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
* [SentencePiece](https://github.com/google/sentencepiece) - A C++ library for unsupervised text tokenization and detokenization, widely used in modern NLP models.

<a name="cpp-speech-recognition"></a>
#### Speech Recognition
* [Kaldi](https://github.com/kaldi-asr/kaldi) - Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.
* [Vosk](https://github.com/alphacep/vosk-api) - An offline speech recognition toolkit with C++ support, designed for low-resource devices and multiple languages.

<a name="cpp-sequence-analysis"></a>
#### Sequence Analysis
* [ToPS](https://github.com/ayoshiaki/tops) - This is an object-oriented framework that facilitates the integration of probabilistic models for sequences over a user defined alphabet. **[Deprecated]**

<a name="cpp-gesture-detection"></a>
#### Gesture Detection
* [grt](https://github.com/nickgillian/grt) - The Gesture Recognition Toolkit (GRT) is a cross-platform, open-source, C++ machine learning library designed for real-time gesture recognition.

<a name="cpp-reinforcement-learning"></a>
#### Reinforcement Learning
* [RLtools](https://github.com/rl-tools/rl-tools) - The fastest deep reinforcement learning library for continuous control, implemented header-only in pure, dependency-free C++ (Python bindings available as well).

<a name="common-lisp"></a>
## Common Lisp

<a name="common-lisp-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [mgl](https://github.com/melisgl/mgl/) - Neural networks (boltzmann machines, feed-forward and recurrent nets), Gaussian Processes.
* [mgl-gpr](https://github.com/melisgl/mgl-gpr/) - Evolutionary algorithms. **[Deprecated]**
* [cl-libsvm](https://github.com/melisgl/cl-libsvm/) - Wrapper for the libsvm support vector machine library. **[Deprecated]**
* [cl-online-learning](https://github.com/masatoi/cl-online-learning) - Online learning algorithms (Perceptron, AROW, SCW, Logistic Regression).
* [cl-random-forest](https://github.com/masatoi/cl-random-forest) - Implementation of Random Forest in Common Lisp.

<a name="clojure"></a>
## Clojure

<a name="clojure-natural-language-processing"></a>
#### Natural Language Processing

* [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp).
* [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript.

<a name="clojure-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [scicloj.ml](https://github.com/scicloj/scicloj.ml) -  A idiomatic Clojure machine learning library based on tech.ml.dataset with a unique approach for immutable data processing pipelines.
* [clj-ml](https://github.com/joshuaeckroth/clj-ml/) - A machine learning library for Clojure built on top of Weka and friends.
* [clj-boost](https://gitlab.com/alanmarazzi/clj-boost) - Wrapper for XGBoost
* [Touchstone](https://github.com/ptaoussanis/touchstone) - Clojure A/B testing library.
* [Clojush](https://github.com/lspector/Clojush) - The Push programming language and the PushGP genetic programming system implemented in Clojure.
* [lambda-ml](https://github.com/cloudkj/lambda-ml) - Simple, concise implementations of machine learning techniques and utilities in Clojure.
* [Infer](https://github.com/aria42/infer) - Inference and machine learning in Clojure. **[Deprecated]**
* [Encog](https://github.com/jimpil/enclog) - Clojure wrapper for Encog (v3) (Machine-Learning framework that specializes in neural-nets). **[Deprecated]**
* [Fungp](https://github.com/vollmerm/fungp) - A genetic programming library for Clojure. **[Deprecated]**
* [Statistiker](https://github.com/clojurewerkz/statistiker) - Basic Machine Learning algorithms in Clojure. **[Deprecated]**
* [clortex](https://github.com/htm-community/clortex) - General Machine Learning library using Numenta’s Cortical Learning Algorithm. **[Deprecated]**
* [comportex](https://github.com/htm-community/comportex) - Functionally composable Machine Learning library using Numenta’s Cortical Learning Algorithm. **[Deprecated]**

<a name="clojure-deep-learning"></a>
#### Deep Learning
* [MXNet](https://mxnet.apache.org/versions/1.7.0/api/clojure) - Bindings to Apache MXNet - part of the MXNet project
* [Deep Diamond](https://github.com/uncomplicate/deep-diamond) - A fast Clojure Tensor & Deep Learning library
* [jutsu.ai](https://github.com/hswick/jutsu.ai) - Clojure wrapper for deeplearning4j with some added syntactic sugar.
* [cortex](https://github.com/originrose/cortex) - Neural networks, regression and feature learning in Clojure.
* [Flare](https://github.com/aria42/flare) - Dynamic Tensor Graph library in Clojure (think PyTorch, DynNet, etc.)
* [dl4clj](https://github.com/yetanalytics/dl4clj) - Clojure wrapper for Deeplearning4j.

<a name="clojure-data-analysis--data-visualization"></a>
#### Data Analysis
* [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) - Clojure dataframe library and pipeline for data processing and machine learning
* [Tablecloth](https://github.com/scicloj/tablecloth) - A dataframe grammar wrapping tech.ml.dataset, inspired by several R libraries
* [Panthera](https://github.com/alanmarazzi/panthera) - Clojure API wrapping Python's Pandas library
* [Incanter](http://incanter.org/) - Incanter is a Clojure-based, R-like platform for statistical computing and graphics.
* [PigPen](https://github.com/Netflix/PigPen) - Map-Reduce for Clojure.
* [Geni](https://github.com/zero-one-group/geni) - a Clojure dataframe library that runs on Apache Spark

<a name="clojure-data-visualization"></a>
#### Data Visualization
* [Hanami](https://github.com/jsa-aerial/hanami) : Clojure(Script) library and framework for creating interactive visualization applications based in Vega-Lite (VGL) and/or Vega (VG) specifications. Automatic framing and layouts along with a powerful templating system for abstracting visualization specs
* [Saite](https://github.com/jsa-aerial/saite) -  Clojure(Script) client/server application for dynamic interactive explorations and the creation of live shareable documents capturing them using Vega/Vega-Lite, CodeMirror, markdown, and LaTeX
* [Oz](https://github.com/metasoarous/oz) - Data visualisation using Vega/Vega-Lite and Hiccup, and a live-reload platform for literate-programming
* [Envision](https://github.com/clojurewerkz/envision) - Clojure Data Visualisation library, based on Statistiker and D3.
* [Pink Gorilla Notebook](https://github.com/pink-gorilla/gorilla-notebook) - A Clojure/Clojurescript notebook application/-library based on Gorilla-REPL
* [clojupyter](https://github.com/clojupyter/clojupyter) -  A Jupyter kernel for Clojure - run Clojure code in Jupyter Lab, Notebook and Console.
* [notespace](https://github.com/scicloj/notespace) - Notebook experience in your Clojure namespace
* [Delight](https://github.com/datamechanics/delight) - A listener that streams your spark events logs to delight, a free and improved spark UI

<a name="clojure-interop"></a>
#### Interop

* [Java Interop](https://clojure.org/reference/java_interop) - Clojure has Native Java Interop from which Java's ML ecosystem can be accessed
* [JavaScript Interop](https://clojurescript.org/reference/javascript-api) - ClojureScript has Native JavaScript Interop from which JavaScript's ML ecosystem can be accessed
* [Libpython-clj](https://github.com/clj-python/libpython-clj) - Interop with Python
* [ClojisR](https://github.com/scicloj/clojisr) - Interop with R and Renjin (R on the JVM)

<a name="clojure-misc"></a>
#### Misc
* [Neanderthal](https://neanderthal.uncomplicate.org/) - Fast Clojure Matrix Library (native CPU, GPU, OpenCL, CUDA)
* [kixistats](https://github.com/MastodonC/kixi.stats) - A library of statistical distribution sampling and transducing functions
* [fastmath](https://github.com/generateme/fastmath) - A collection of functions for mathematical and statistical computing, macine learning, etc., wrapping several JVM libraries
* [matlib](https://github.com/atisharma/matlib) - A Clojure library of optimisation and control theory tools and convenience functions based on Neanderthal.

<a name="clojure-extra"></a>
#### Extra
* [Scicloj](https://scicloj.github.io/pages/libraries/) - Curated list of ML related resources for Clojure.

<a name="crystal"></a>
## Crystal

<a name="crystal-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [machine](https://github.com/mathieulaporte/machine) - Simple machine learning algorithm.
* [crystal-fann](https://github.com/NeuraLegion/crystal-fann) - FANN (Fast Artificial Neural Network) binding.

<a name="elixir"></a>
## Elixir

<a name="elixir-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Simple Bayes](https://github.com/fredwu/simple_bayes) - A Simple Bayes / Naive Bayes implementation in Elixir.
* [emel](https://github.com/mrdimosthenis/emel) - A simple and functional machine learning library written in Elixir.
* [Tensorflex](https://github.com/anshuman23/tensorflex) - Tensorflow bindings for the Elixir programming language.

<a name="elixir-natural-language-processing"></a>
#### Natural Language Processing

* [Stemmer](https://github.com/fredwu/stemmer) - An English (Porter2) stemming implementation in Elixir.

<a name="erlang"></a>
## Erlang

<a name="erlang-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Disco](https://github.com/discoproject/disco/) - Map Reduce in Erlang. **[Deprecated]**

<a name="fortran"></a>
## Fortran

<a name="fortran-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [neural-fortran](https://github.com/modern-fortran/neural-fortran) - A parallel neural net microframework.
Read the paper [here](https://arxiv.org/abs/1902.06714).

<a name="fortran-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose Fortran library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).

<a name="go"></a>
## Go

<a name="go-natural-language-processing"></a>
#### Natural Language Processing

* [Cybertron](https://github.com/nlpodyssey/cybertron) - Cybertron: the home planet of the Transformers in Go.
* [snowball](https://github.com/tebeka/snowball) - Snowball Stemmer for Go.
* [word-embedding](https://github.com/ynqa/word-embedding) - Word Embeddings: the full implementation of word2vec, GloVe in Go.
* [sentences](https://github.com/neurosnap/sentences) - Golang implementation of Punkt sentence tokenizer.
* [go-ngram](https://github.com/Lazin/go-ngram) - In-memory n-gram index with compression. *[Deprecated]*
* [paicehusk](https://github.com/Rookii/paicehusk) - Golang implementation of the Paice/Husk Stemming Algorithm. *[Deprecated]*
* [go-porterstemmer](https://github.com/reiver/go-porterstemmer) - A native Go clean room implementation of the Porter Stemming algorithm. **[Deprecated]**

<a name="go-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Spago](https://github.com/nlpodyssey/spago) - Self-contained Machine Learning and Natural Language Processing library in Go.
* [birdland](https://github.com/rlouf/birdland) - A recommendation library in Go.
* [eaopt](https://github.com/MaxHalford/eaopt) - An evolutionary optimization library.
* [leaves](https://github.com/dmitryikh/leaves) - A pure Go implementation of the prediction part of GBRTs, including XGBoost and LightGBM.
* [gobrain](https://github.com/goml/gobrain) - Neural Networks written in Go.
* [go-featureprocessing](https://github.com/nikolaydubina/go-featureprocessing) - Fast and convenient feature processing for low latency machine learning in Go.
* [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor) - Go binding for MXNet c_predict_api to do inference with a pre-trained model.
* [go-ml-benchmarks](https://github.com/nikolaydubina/go-ml-benchmarks) — benchmarks of machine learning inference for Go.
* [go-ml-transpiler](https://github.com/znly/go-ml-transpiler) - An open source Go transpiler for machine learning models.
* [golearn](https://github.com/sjwhitworth/golearn) - Machine learning for Go.
* [goml](https://github.com/cdipaolo/goml) - Machine learning library written in pure Go.
* [gorgonia](https://github.com/gorgonia/gorgonia) - Deep learning in Go.
* [goro](https://github.com/aunum/goro) - A high-level machine learning library in the vein of Keras.
* [gorse](https://github.com/zhenghaoz/gorse) - An offline recommender system backend based on collaborative filtering written in Go.
* [therfoo](https://github.com/therfoo/therfoo) - An embedded deep learning library for Go.
* [neat](https://github.com/jinyeom/neat) - Plug-and-play, parallel Go framework for NeuroEvolution of Augmenting Topologies (NEAT). **[Deprecated]**
* [go-pr](https://github.com/daviddengcn/go-pr) - Pattern recognition package in Go lang. **[Deprecated]**
* [go-ml](https://github.com/alonsovidales/go_ml) - Linear / Logistic regression, Neural Networks, Collaborative Filtering and Gaussian Multivariate Distribution. **[Deprecated]**
* [GoNN](https://github.com/fxsjy/gonn) - GoNN is an implementation of Neural Network in Go Language, which includes BPNN, RBF, PCN. **[Deprecated]**
* [bayesian](https://github.com/jbrukh/bayesian) - Naive Bayesian Classification for Golang. **[Deprecated]**
* [go-galib](https://github.com/thoj/go-galib) - Genetic Algorithms library written in Go / Golang. **[Deprecated]**
* [Cloudforest](https://github.com/ryanbressler/CloudForest) - Ensembles of decision trees in Go/Golang. **[Deprecated]**
* [go-dnn](https://github.com/sudachen/go-dnn) - Deep Neural Networks for Golang (powered by MXNet)

<a name="go-spatial-analysis-and-geometry"></a>
#### Spatial analysis and geometry

* [go-geom](https://github.com/twpayne/go-geom) - Go library to handle geometries.
* [gogeo](https://github.com/golang/geo) - Spherical geometry in Go.

<a name="go-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [dataframe-go](https://github.com/rocketlaunchr/dataframe-go) - Dataframes for machine-learning and statistics (similar to pandas).
* [gota](https://github.com/go-gota/gota) - Dataframes.
* [gonum/mat](https://godoc.org/gonum.org/v1/gonum/mat) - A linear algebra package for Go.
* [gonum/optimize](https://godoc.org/gonum.org/v1/gonum/optimize) - Implementations of optimization algorithms.
* [gonum/plot](https://godoc.org/gonum.org/v1/plot) - A plotting library.
* [gonum/stat](https://godoc.org/gonum.org/v1/gonum/stat) - A statistics library.
* [SVGo](https://github.com/ajstarks/svgo) - The Go Language library for SVG generation.
* [glot](https://github.com/arafatk/glot) - Glot is a plotting library for Golang built on top of gnuplot.
* [globe](https://github.com/mmcloughlin/globe) - Globe wireframe visualization.
* [gonum/graph](https://godoc.org/gonum.org/v1/gonum/graph) - General-purpose graph library.
* [go-graph](https://github.com/StepLg/go-graph) - Graph library for Go/Golang language. **[Deprecated]**
* [RF](https://github.com/fxsjy/RF.go) - Random forests implementation in Go. **[Deprecated]**

<a name="go-computer-vision"></a>
#### Computer vision

* [GoCV](https://github.com/hybridgroup/gocv) - Package for computer vision using OpenCV 4 and beyond.

<a name="go-reinforcement-learning"></a>
#### Reinforcement learning

* [gold](https://github.com/aunum/gold) - A reinforcement learning library.
* [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) - PyTorch implementations of Stable Baselines (deep) reinforcement learning algorithms.

<a name="haskell"></a>
## Haskell

<a name="haskell-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [haskell-ml](https://github.com/ajtulloch/haskell-ml) - Haskell implementations of various ML algorithms. **[Deprecated]**
* [HLearn](https://github.com/mikeizbicki/HLearn) - a suite of libraries for interpreting machine learning models according to their algebraic structure. **[Deprecated]**
* [hnn](https://github.com/alpmestan/HNN) - Haskell Neural Network library.
* [hopfield-networks](https://github.com/ajtulloch/hopfield-networks) - Hopfield Networks for unsupervised learning in Haskell. **[Deprecated]**
* [DNNGraph](https://github.com/ajtulloch/dnngraph) - A DSL for deep neural networks. **[Deprecated]**
* [LambdaNet](https://github.com/jbarrow/LambdaNet) - Configurable Neural Networks in Haskell. **[Deprecated]**

<a name="java"></a>
## Java

<a name="java-natural-language-processing"></a>
#### Natural Language Processing
* [Cortical.io](https://www.cortical.io/) - Retina: an API performing complex NLP operations (disambiguation, classification, streaming text filtering, etc...) as quickly and intuitively as the brain.
* [IRIS](https://github.com/cortical-io/Iris) - [Cortical.io's](https://cortical.io) FREE NLP, Retina API Analysis Tool (written in JavaFX!) - [See the Tutorial Video](https://www.youtube.com/watch?v=CsF4pd7fGF0).
* [CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) - Stanford CoreNLP provides a set of natural language analysis tools which can take raw English language text input and give the base forms of words.
* [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml) - A natural language parser is a program that works out the grammatical structure of sentences.
* [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml) - A Part-Of-Speech Tagger (POS Tagger).
* [Stanford Name Entity Recognizer](https://nlp.stanford.edu/software/CRF-NER.shtml) - Stanford NER is a Java implementation of a Named Entity Recognizer.
* [Stanford Word Segmenter](https://nlp.stanford.edu/software/segmenter.shtml) - Tokenization of raw text is a standard pre-processing step for many NLP tasks.
* [Tregex, Tsurgeon and Semgrex](https://nlp.stanford.edu/software/tregex.shtml) - Tregex is a utility for matching patterns in trees, based on tree relationships and regular expression matches on nodes (the name is short for "tree regular expressions").
* [Stanford Phrasal: A Phrase-Based Translation System](https://nlp.stanford.edu/phrasal/)
* [Stanford English Tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml) - Stanford Phrasal is a state-of-the-art statistical phrase-based machine translation system, written in Java.
* [Stanford Tokens Regex](https://nlp.stanford.edu/software/tokensregex.shtml) - A tokenizer divides text into a sequence of tokens, which roughly correspond to "words".
* [Stanford Temporal Tagger](https://nlp.stanford.edu/software/sutime.shtml) - SUTime is a library for recognizing and normalizing time expressions.
* [Stanford SPIED](https://nlp.stanford.edu/software/patternslearning.shtml) - Learning entities from unlabeled text starting with seed sets using patterns in an iterative fashion.
* [Twitter Text Java](https://github.com/twitter/twitter-text/tree/master/java) - A Java implementation of Twitter's text processing library.
* [MALLET](http://mallet.cs.umass.edu/) - A Java-based package for statistical natural language processing, document classification, clustering, topic modelling, information extraction, and other machine learning applications to text.
* [OpenNLP](https://opennlp.apache.org/) - A machine learning based toolkit for the processing of natural language text.
* [LingPipe](http://alias-i.com/lingpipe/index.html) - A tool kit for processing text using computational linguistics.
* [ClearTK](https://github.com/ClearTK/cleartk) - ClearTK provides a framework for developing statistical natural language processing (NLP) components in Java and is built on top of Apache UIMA. **[Deprecated]**
* [Apache cTAKES](https://ctakes.apache.org/) - Apache Clinical Text Analysis and Knowledge Extraction System (cTAKES) is an open-source natural language processing system for information extraction from electronic medical record clinical free-text.
* [NLP4J](https://github.com/emorynlp/nlp4j) - The NLP4J project provides software and resources for natural language processing. The project started at the Center for Computational Language and EducAtion Research, and is currently developed by the Center for Language and Information Research at Emory University. **[Deprecated]**
* [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - This project collects a number of core libraries for Natural Language Processing (NLP) developed in the University of Illinois' Cognitive Computation Group, for example `illinois-core-utilities` which provides a set of NLP-friendly data structures and a number of NLP-related utilities that support writing NLP applications, running experiments, etc, `illinois-edison` a library for feature extraction from illinois-core-utilities data structures and many other packages.

<a name="java-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [aerosolve](https://github.com/airbnb/aerosolve) - A machine learning library by Airbnb designed from the ground up to be human friendly.
* [AMIDST Toolbox](http://www.amidsttoolbox.com/) - A Java Toolbox for Scalable Probabilistic Machine Learning.
* [Chips-n-Salsa](https://github.com/cicirello/Chips-n-Salsa) - A Java library for genetic algorithms, evolutionary computation, and stochastic local search, with a focus on self-adaptation / self-tuning, as well as parallel execution.
* [Datumbox](https://github.com/datumbox/datumbox-framework) - Machine Learning framework for rapid development of Machine Learning and Statistical applications.
* [ELKI](https://elki-project.github.io/) - Java toolkit for data mining. (unsupervised: clustering, outlier detection etc.)
* [Encog](https://github.com/encog/encog-java-core) - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trainings using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [FlinkML in Apache Flink](https://ci.apache.org/projects/flink/flink-docs-master/dev/libs/ml/index.html) - Distributed machine learning library in Flink.
* [H2O](https://github.com/h2oai/h2o-3) - ML engine that supports distributed learning on Hadoop, Spark or your laptop via APIs in R, Python, Scala, REST/JSON.
* [htm.java](https://github.com/numenta/htm.java) - General Machine Learning library using Numenta’s Cortical Learning Algorithm.
* [liblinear-java](https://github.com/bwaldvogel/liblinear-java) - Java version of liblinear.
* [Mahout](https://github.com/apache/mahout) - Distributed machine learning.
* [Meka](http://meka.sourceforge.net/) - An open source implementation of methods for multi-label classification and evaluation (extension to Weka).
* [MLlib in Apache Spark](https://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark.
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [Neuroph](http://neuroph.sourceforge.net/) - Neuroph is lightweight Java neural network framework.
* [ORYX](https://github.com/oryxproject/oryx) - Lambda Architecture Framework using Apache Spark and Apache Kafka with a specialization for real-time large-scale machine learning.
* [Samoa](https://samoa.incubator.apache.org/) SAMOA is a framework that includes distributed machine learning for data streams with an interface to plug-in different stream processing platforms.
* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) - RankLib is a library of learning to rank algorithms. **[Deprecated]**
* [rapaio](https://github.com/padreati/rapaio) - statistics, data mining and machine learning toolbox in Java.
* [RapidMiner](https://rapidminer.com) - RapidMiner integration into Java code.
* [Stanford Classifier](https://nlp.stanford.edu/software/classifier.shtml) - A classifier is a machine learning tool that will take data items and place them into one of k classes.
* [Smile](https://haifengl.github.io/) - Statistical Machine Intelligence & Learning Engine.
* [SystemML](https://github.com/apache/systemml) - flexible, scalable machine learning (ML) language.
* [Tribou](https://tribuo.org) - A machine learning library written in Java by Oracle.
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/) - Weka is a collection of machine learning algorithms for data mining tasks.
* [LBJava](https://github.com/CogComp/lbjava) - Learning Based Java is a modelling language for the rapid development of software systems, offers a convenient, declarative syntax for classifier and constraint definition directly in terms of the objects in the programmer's application.
* [knn-java-library](https://github.com/felipexw/knn-java-library) - Just a simple implementation of K-Nearest Neighbors algorithm using with a bunch of similarity measures.

<a name="java-speech-recognition"></a>
#### Speech Recognition
* [CMU Sphinx](https://cmusphinx.github.io) - Open Source Toolkit For Speech Recognition purely based on Java speech recognition library.

<a name="java-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [Flink](https://flink.apache.org/) - Open source platform for distributed stream and batch data processing.
* [Hadoop](https://github.com/apache/hadoop) - Hadoop/HDFS.
* [Onyx](https://github.com/onyx-platform/onyx) - Distributed, masterless, high performance, fault tolerant data processing. Written entirely in Clojure.
* [Spark](https://github.com/apache/spark) - Spark is a fast and general engine for large-scale data processing.
* [Storm](https://storm.apache.org/) - Storm is a distributed realtime computation system.
* [Impala](https://github.com/cloudera/impala) - Real-time Query for Hadoop.
* [DataMelt](https://jwork.org/dmelt/) - Mathematics software for numeric computation, statistics, symbolic calculations, data analysis and data visualization.
* [Dr. Michael Thomas Flanagan's Java Scientific Library.](https://www.ee.ucl.ac.uk/~mflanaga/java/) **[Deprecated]**

<a name="java-deep-learning"></a>
#### Deep Learning

* [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) - Scalable deep learning for industry with parallel GPUs.
* [Keras Beginner Tutorial](https://victorzhou.com/blog/keras-neural-network-tutorial/) - Friendly guide on using Keras to implement a simple Neural Network in Python.
* [deepjavalibrary/djl](https://github.com/deepjavalibrary/djl) - Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning, designed to be easy to get started with and simple to use for Java developers.

<a name="javascript"></a>
## JavaScript

<a name="javascript-natural-language-processing"></a>
#### Natural Language Processing

* [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library.
* [natural](https://github.com/NaturalNode/natural) - General natural language facilities for node.
* [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS.
* [Retext](https://github.com/retextjs/retext) - Extensible system for analyzing and manipulating natural language.
* [NLP Compromise](https://github.com/spencermountain/compromise) - Natural Language processing in the browser.
* [nlp.js](https://github.com/axa-group/nlp.js) - An NLP library built in node over Natural, with entity extraction, sentiment analysis, automatic language identify, and so more.



<a name="javascript-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [D3.js](https://d3js.org/)
* [High Charts](https://www.highcharts.com/)
* [NVD3.js](http://nvd3.org/)
* [dc.js](https://dc-js.github.io/dc.js/)
* [chartjs](https://www.chartjs.org/)
* [dimple](http://dimplejs.org/)
* [amCharts](https://www.amcharts.com/)
* [D3xter](https://github.com/NathanEpstein/D3xter) - Straight forward plotting built on D3. **[Deprecated]**
* [statkit](https://github.com/rigtorp/statkit) - Statistics kit for JavaScript. **[Deprecated]**
* [datakit](https://github.com/nathanepstein/datakit) - A lightweight framework for data analysis in JavaScript
* [science.js](https://github.com/jasondavies/science.js/) - Scientific and statistical computing in JavaScript. **[Deprecated]**
* [Z3d](https://github.com/NathanEpstein/Z3d) - Easily make interactive 3d plots built on Three.js **[Deprecated]**
* [Sigma.js](http://sigmajs.org/) - JavaScript library dedicated to graph drawing.
* [C3.js](https://c3js.org/) - customizable library based on D3.js for easy chart drawing.
* [Datamaps](https://datamaps.github.io/) - Customizable SVG map/geo visualizations using D3.js. **[Deprecated]**
* [ZingChart](https://www.zingchart.com/) - library written on Vanilla JS for big data visualization.
* [cheminfo](https://www.cheminfo.org/) - Platform for data visualization and analysis, using the [visualizer](https://github.com/npellet/visualizer) project.
* [Learn JS Data](http://learnjsdata.com/)
* [AnyChart](https://www.anychart.com/)
* [FusionCharts](https://www.fusioncharts.com/)
* [Nivo](https://nivo.rocks) - built on top of the awesome d3 and Reactjs libraries


<a name="javascript-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Auto ML](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning, data formatting, ensembling, and hyperparameter optimization for competitions and exploration- just give it a .csv file! **[Deprecated]**
* [Convnet.js](https://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a JavaScript library for training Deep Learning models[DEEP LEARNING] **[Deprecated]**
* [Creatify MCP](https://github.com/TSavo/creatify-mcp) - Model Context Protocol server that exposes Creatify AI's video generation capabilities to AI assistants, enabling natural language video creation workflows.
* [Clusterfck](https://harthur.github.io/clusterfck/) - Agglomerative hierarchical clustering implemented in JavaScript for Node.js and the browser. **[Deprecated]**
* [Clustering.js](https://github.com/emilbayes/clustering.js) - Clustering algorithms implemented in JavaScript for Node.js and the browser. **[Deprecated]**
* [Decision Trees](https://github.com/serendipious/nodejs-decision-tree-id3) - NodeJS Implementation of Decision Tree using ID3 Algorithm. **[Deprecated]**
* [DN2A](https://github.com/antoniodeluca/dn2a.js) - Digital Neural Networks Architecture. **[Deprecated]**
* [figue](https://code.google.com/archive/p/figue) - K-means, fuzzy c-means and agglomerative clustering.
* [Gaussian Mixture Model](https://github.com/lukapopijac/gaussian-mixture-model) - Unsupervised machine learning with multivariate Gaussian mixture model.
* [Node-fann](https://github.com/rlidwka/node-fann) - FANN (Fast Artificial Neural Network Library) bindings for Node.js **[Deprecated]**
* [Keras.js](https://github.com/transcranial/keras-js) - Run Keras models in the browser, with GPU support provided by WebGL 2.
* [Kmeans.js](https://github.com/emilbayes/kMeans.js) - Simple JavaScript implementation of the k-means algorithm, for node.js and the browser. **[Deprecated]**
* [LDA.js](https://github.com/primaryobjects/lda) - LDA topic modelling for Node.js
* [Learning.js](https://github.com/yandongliu/learningjs) - JavaScript implementation of logistic regression/c4.5 decision tree **[Deprecated]**
* [machinelearn.js](https://github.com/machinelearnjs/machinelearnjs) - Machine Learning library for the web, Node.js and developers
* [mil-tokyo](https://github.com/mil-tokyo) - List of several machine learning libraries.
* [Node-SVM](https://github.com/nicolaspanel/node-svm) - Support Vector Machine for Node.js
* [Brain](https://github.com/harthur/brain) - Neural networks in JavaScript **[Deprecated]**
* [Brain.js](https://github.com/BrainJS/brain.js) - Neural networks in JavaScript - continued community fork of [Brain](https://github.com/harthur/brain).
* [Bayesian-Bandit](https://github.com/omphalos/bayesian-bandit.js) - Bayesian bandit implementation for Node and the browser. **[Deprecated]**
* [Synaptic](https://github.com/cazala/synaptic) - Architecture-free neural network library for Node.js and the browser.
* [kNear](https://github.com/NathanEpstein/kNear) - JavaScript implementation of the k nearest neighbors algorithm for supervised learning.
* [NeuralN](https://github.com/totemstech/neuraln) - C++ Neural Network library for Node.js. It has advantage on large dataset and multi-threaded training. **[Deprecated]**
* [kalman](https://github.com/itamarwe/kalman) - Kalman filter for JavaScript. **[Deprecated]**
* [shaman](https://github.com/luccastera/shaman) - Node.js library with support for both simple and multiple linear regression. **[Deprecated]**
* [ml.js](https://github.com/mljs/ml) - Machine learning and numerical analysis tools for Node.js and the Browser!
* [ml5](https://github.com/ml5js/ml5-library) - Friendly machine learning for the web!
* [Pavlov.js](https://github.com/NathanEpstein/Pavlov.js) - Reinforcement learning using Markov Decision Processes.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [TensorFlow.js](https://js.tensorflow.org/) - A WebGL accelerated, browser based JavaScript library for training and deploying ML models.
* [JSMLT](https://github.com/jsmlt/jsmlt) - Machine learning toolkit with classification and clustering for Node.js; supports visualization (see [visualml.io](https://visualml.io)).
* [xgboost-node](https://github.com/nuanio/xgboost-node) - Run XGBoost model and make predictions in Node.js.
* [Netron](https://github.com/lutzroeder/netron) - Visualizer for machine learning models.
* [tensor-js](https://github.com/Hoff97/tensorjs) - A deep learning library for the browser, accelerated by WebGL and WebAssembly.
* [WebDNN](https://github.com/mil-tokyo/webdnn) - Fast Deep Neural Network JavaScript Framework. WebDNN uses next generation JavaScript API, WebGPU for GPU execution, and WebAssembly for CPU execution.
* [WebNN](https://webnn.dev) - A new web standard that allows web apps and frameworks to accelerate deep neural networks with on-device hardware such as GPUs, CPUs, or purpose-built AI accelerators.

<a name="javascript-misc"></a>
#### Misc

* [stdlib](https://github.com/stdlib-js/stdlib) - A standard library for JavaScript and Node.js, with an emphasis on numeric computing. The library provides a collection of robust, high performance libraries for mathematics, statistics, streams, utilities, and more.
* [sylvester](https://github.com/jcoglan/sylvester) - Vector and Matrix math for JavaScript. **[Deprecated]**
* [simple-statistics](https://github.com/simple-statistics/simple-statistics) - A JavaScript implementation of descriptive, regression, and inference statistics. Implemented in literate JavaScript with no dependencies, designed to work in all modern browsers (including IE) as well as in Node.js.
* [regression-js](https://github.com/Tom-Alexander/regression-js) - A javascript library containing a collection of least squares fitting methods for finding a trend in a set of data.
* [Lyric](https://github.com/flurry/Lyric) - Linear Regression library. **[Deprecated]**
* [GreatCircle](https://github.com/mwgg/GreatCircle) - Library for calculating great circle distance.
* [MLPleaseHelp](https://github.com/jgreenemi/MLPleaseHelp) - MLPleaseHelp is a simple ML resource search engine. You can use this search engine right now at [https://jgreenemi.github.io/MLPleaseHelp/](https://jgreenemi.github.io/MLPleaseHelp/), provided via GitHub Pages.
* [Pipcook](https://github.com/alibaba/pipcook) - A JavaScript application framework for machine learning and its engineering.

<a name="javascript-demos-and-scripts"></a>
#### Demos and Scripts
* [The Bot](https://github.com/sta-ger/TheBot) - Example of how the neural network learns to predict the angle between two points created with [Synaptic](https://github.com/cazala/synaptic).
* [Half Beer](https://github.com/sta-ger/HalfBeer) - Beer glass classifier created with [Synaptic](https://github.com/cazala/synaptic).
* [NSFWJS](http://nsfwjs.com) - Indecent content checker with TensorFlow.js
* [Rock Paper Scissors](https://rps-tfjs.netlify.com/) - Rock Paper Scissors trained in the browser with TensorFlow.js
* [Heroes Wear Masks](https://heroeswearmasks.fun/) - A fun TensorFlow.js-based oracle that tells, whether one wears a face mask or not. It can even tell when one wears the mask incorrectly.

<a name="julia"></a>
## Julia

<a name="julia-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [MachineLearning](https://github.com/benhamner/MachineLearning.jl) - Julia Machine Learning library. **[Deprecated]**
* [MLBase](https://github.com/JuliaStats/MLBase.jl) - A set of functions to support the development of machine learning algorithms.
* [PGM](https://github.com/JuliaStats/PGM.jl) - A Julia framework for probabilistic graphical models.
* [DA](https://github.com/trthatcher/DiscriminantAnalysis.jl) - Julia package for Regularized Discriminant Analysis.
* [Regression](https://github.com/lindahua/Regression.jl) - Algorithms for regression analysis (e.g. linear regression and logistic regression). **[Deprecated]**
* [Local Regression](https://github.com/JuliaStats/Loess.jl) - Local regression, so smooooth!
* [Naive Bayes](https://github.com/nutsiepully/NaiveBayes.jl) - Simple Naive Bayes implementation in Julia. **[Deprecated]**
* [Mixed Models](https://github.com/dmbates/MixedModels.jl) - A Julia package for fitting (statistical) mixed-effects models.
* [Simple MCMC](https://github.com/fredo-dedup/SimpleMCMC.jl) - basic MCMC sampler implemented in Julia. **[Deprecated]**
* [Distances](https://github.com/JuliaStats/Distances.jl) - Julia module for Distance evaluation.
* [Decision Tree](https://github.com/bensadeghi/DecisionTree.jl) - Decision Tree Classifier and Regressor.
* [Neural](https://github.com/compressed/BackpropNeuralNet.jl) - A neural network in Julia.
* [MCMC](https://github.com/doobwa/MCMC.jl) - MCMC tools for Julia. **[Deprecated]**
* [Mamba](https://github.com/brian-j-smith/Mamba.jl) - Markov chain Monte Carlo (MCMC) for Bayesian analysis in Julia.
* [GLM](https://github.com/JuliaStats/GLM.jl) - Generalized linear models in Julia.
* [Gaussian Processes](https://github.com/STOR-i/GaussianProcesses.jl) - Julia package for Gaussian processes.
* [Online Learning](https://github.com/lendle/OnlineLearning.jl) **[Deprecated]**
* [GLMNet](https://github.com/simonster/GLMNet.jl) - Julia wrapper for fitting Lasso/ElasticNet GLM models using glmnet.
* [Clustering](https://github.com/JuliaStats/Clustering.jl) - Basic functions for clustering data: k-means, dp-means, etc.
* [SVM](https://github.com/JuliaStats/SVM.jl) - SVM for Julia. **[Deprecated]**
* [Kernel Density](https://github.com/JuliaStats/KernelDensity.jl) - Kernel density estimators for Julia.
* [MultivariateStats](https://github.com/JuliaStats/MultivariateStats.jl) - Methods for dimensionality reduction.
* [NMF](https://github.com/JuliaStats/NMF.jl) - A Julia package for non-negative matrix factorization.
* [ANN](https://github.com/EricChiang/ANN.jl) - Julia artificial neural networks. **[Deprecated]**
* [Mocha](https://github.com/pluskid/Mocha.jl) - Deep Learning framework for Julia inspired by Caffe. **[Deprecated]**
* [XGBoost](https://github.com/dmlc/XGBoost.jl) - eXtreme Gradient Boosting Package in Julia.
* [ManifoldLearning](https://github.com/wildart/ManifoldLearning.jl) - A Julia package for manifold learning and nonlinear dimensionality reduction.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Merlin](https://github.com/hshindo/Merlin.jl) - Flexible Deep Learning Framework in Julia.
* [ROCAnalysis](https://github.com/davidavdav/ROCAnalysis.jl) - Receiver Operating Characteristics and functions for evaluation probabilistic binary classifiers.
* [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl) - Large scale Gaussian Mixture Models.
* [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl) - Julia implementation of the scikit-learn API.
* [Knet](https://github.com/denizyuret/Knet.jl) - Koç University Deep Learning Framework.
* [Flux](https://fluxml.ai/) - Relax! Flux is the ML library that doesn't make you tensor
* [MLJ](https://github.com/alan-turing-institute/MLJ.jl) - A Julia machine learning framework.
* [CluGen](https://github.com/clugen/CluGen.jl/) - Multidimensional cluster generation in Julia.

<a name="julia-natural-language-processing"></a>
#### Natural Language Processing

* [Topic Models](https://github.com/slycoder/TopicModels.jl) - TopicModels for Julia. **[Deprecated]**
* [Text Analysis](https://github.com/JuliaText/TextAnalysis.jl) - Julia package for text analysis.
* [Word Tokenizers](https://github.com/JuliaText/WordTokenizers.jl) - Tokenizers for Natural Language Processing in Julia
* [Corpus Loaders](https://github.com/JuliaText/CorpusLoaders.jl) - A Julia package providing a variety of loaders for various NLP corpora.
* [Embeddings](https://github.com/JuliaText/Embeddings.jl) - Functions and data dependencies for loading various word embeddings
* [Languages](https://github.com/JuliaText/Languages.jl) - Julia package for working with various human languages
* [WordNet](https://github.com/JuliaText/WordNet.jl) - A Julia package for Princeton's WordNet

<a name="julia-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [Graph Layout](https://github.com/IainNZ/GraphLayout.jl) - Graph layout algorithms in pure Julia.
* [LightGraphs](https://github.com/JuliaGraphs/LightGraphs.jl) - Graph modelling and analysis.
* [Data Frames Meta](https://github.com/JuliaData/DataFramesMeta.jl) - Metaprogramming tools for DataFrames.
* [Julia Data](https://github.com/nfoti/JuliaData) - library for working with tabular data in Julia. **[Deprecated]**
* [Data Read](https://github.com/queryverse/ReadStat.jl) - Read files from Stata, SAS, and SPSS.
* [Hypothesis Tests](https://github.com/JuliaStats/HypothesisTests.jl) - Hypothesis tests for Julia.
* [Gadfly](https://github.com/GiovineItalia/Gadfly.jl) - Crafty statistical graphics for Julia.
* [Stats](https://github.com/JuliaStats/StatsKit.jl) - Statistical tests for Julia.
* [RDataSets](https://github.com/johnmyleswhite/RDatasets.jl) - Julia package for loading many of the data sets available in R.
* [DataFrames](https://github.com/JuliaData/DataFrames.jl) - library for working with tabular data in Julia.
* [Distributions](https://github.com/JuliaStats/Distributions.jl) - A Julia package for probability distributions and associated functions.
* [Data Arrays](https://github.com/JuliaStats/DataArrays.jl) - Data structures that allow missing values. **[Deprecated]**
* [Time Series](https://github.com/JuliaStats/TimeSeries.jl) - Time series toolkit for Julia.
* [Sampling](https://github.com/lindahua/Sampling.jl) - Basic sampling algorithms for Julia.

<a name="julia-misc-stuff--presentations"></a>
#### Misc Stuff / Presentations

* [DSP](https://github.com/JuliaDSP/DSP.jl) - Digital Signal Processing (filtering, periodograms, spectrograms, window functions).
* [JuliaCon Presentations](https://github.com/JuliaCon/presentations) - Presentations for JuliaCon.
* [SignalProcessing](https://github.com/JuliaDSP/DSP.jl) - Signal Processing tools for Julia.
* [Images](https://github.com/JuliaImages/Images.jl) - An image library for Julia.
* [DataDeps](https://github.com/oxinabox/DataDeps.jl) - Reproducible data setup for reproducible science.

<a name="kotlin"></a>
## Kotlin

<a name="kotlin-deep-learning"></a>
#### Deep Learning
* [KotlinDL](https://github.com/JetBrains/KotlinDL) - Deep learning framework written in Kotlin.

<a name="lua"></a>
## Lua

<a name="lua-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Torch7](http://torch.ch/)
  * [cephes](https://github.com/deepmind/torch-cephes) - Cephes mathematical functions library, wrapped for Torch. Provides and wraps the 180+ special mathematical functions from the Cephes mathematical library, developed by Stephen L. Moshier. It is used, among many other places, at the heart of SciPy. **[Deprecated]**
  * [autograd](https://github.com/twitter/torch-autograd) - Autograd automatically differentiates native Torch code. Inspired by the original Python version.
  * [graph](https://github.com/torch/graph) - Graph package for Torch. **[Deprecated]**
  * [randomkit](https://github.com/deepmind/torch-randomkit) - Numpy's randomkit, wrapped for Torch. **[Deprecated]**
  * [signal](https://github.com/soumith/torch-signal) - A signal processing toolbox for Torch-7. FFT, DCT, Hilbert, cepstrums, stft.
  * [nn](https://github.com/torch/nn) - Neural Network package for Torch.
  * [torchnet](https://github.com/torchnet/torchnet) - framework for torch which provides a set of abstractions aiming at encouraging code re-use as well as encouraging modular programming.
  * [nngraph](https://github.com/torch/nngraph) - This package provides graphical computation for nn library in Torch7.
  * [nnx](https://github.com/clementfarabet/lua---nnx) - A completely unstable and experimental package that extends Torch's builtin nn library.
  * [rnn](https://github.com/Element-Research/rnn) - A Recurrent Neural Network library that extends Torch's nn. RNNs, LSTMs, GRUs, BRNNs, BLSTMs, etc.
  * [dpnn](https://github.com/Element-Research/dpnn) - Many useful features that aren't part of the main nn package.
  * [dp](https://github.com/nicholas-leonard/dp) - A deep learning library designed for streamlining research and development using the Torch7 distribution. It emphasizes flexibility through the elegant use of object-oriented design patterns. **[Deprecated]**
  * [optim](https://github.com/torch/optim) - An optimization library for Torch. SGD, Adagrad, Conjugate-Gradient, LBFGS, RProp and more.
  * [unsup](https://github.com/koraykv/unsup) - A package for unsupervised learning in Torch. Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA). **[Deprecated]**
  * [manifold](https://github.com/clementfarabet/manifold) - A package to manipulate manifolds.
  * [svm](https://github.com/koraykv/torch-svm) - Torch-SVM library. **[Deprecated]**
  * [lbfgs](https://github.com/clementfarabet/lbfgs) - FFI Wrapper for liblbfgs. **[Deprecated]**
  * [vowpalwabbit](https://github.com/clementfarabet/vowpal_wabbit) - An old vowpalwabbit interface to torch. **[Deprecated]**
  * [OpenGM](https://github.com/clementfarabet/lua---opengm) - OpenGM is a C++ library for graphical modelling, and inference. The Lua bindings provide a simple way of describing graphs, from Lua, and then optimizing them with OpenGM. **[Deprecated]**
  * [spaghetti](https://github.com/MichaelMathieu/lua---spaghetti) - Spaghetti (sparse linear) module for torch7 by @MichaelMathieu **[Deprecated]**
  * [LuaSHKit](https://github.com/ocallaco/LuaSHkit) - A Lua wrapper around the Locality sensitive hashing library SHKit **[Deprecated]**
  * [kernel smoothing](https://github.com/rlowrance/kernel-smoothers) - KNN, kernel-weighted average, local linear regression smoothers. **[Deprecated]**
  * [cutorch](https://github.com/torch/cutorch) - Torch CUDA Implementation.
  * [cunn](https://github.com/torch/cunn) - Torch CUDA Neural Network Implementation.
  * [imgraph](https://github.com/clementfarabet/lua---imgraph) - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images. **[Deprecated]**
  * [videograph](https://github.com/clementfarabet/videograph) - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos. **[Deprecated]**
  * [saliency](https://github.com/marcoscoffier/torch-saliency) - code and tools around integral images. A library for finding interest points based on fast integral histograms. **[Deprecated]**
  * [stitch](https://github.com/marcoscoffier/lua---stitch) - allows us to use hugin to stitch images and apply same stitching to a video sequence. **[Deprecated]**
  * [sfm](https://github.com/marcoscoffier/lua---sfm) - A bundle adjustment/structure from motion package. **[Deprecated]**
  * [fex](https://github.com/koraykv/fex) - A package for feature extraction in Torch. Provides SIFT and dSIFT modules. **[Deprecated]**
  * [OverFeat](https://github.com/sermanet/OverFeat) - A state-of-the-art generic dense feature extractor. **[Deprecated]**
  * [wav2letter](https://github.com/facebookresearch/wav2letter) - a simple and efficient end-to-end Automatic Speech Recognition (ASR) system from Facebook AI Research.
* [Numeric Lua](http://numlua.luaforge.net/)
* [Lunatic Python](https://labix.org/lunatic-python)
* [SciLua](http://scilua.org/)
* [Lua - Numerical Algorithms](https://bitbucket.org/lucashnegri/lna) **[Deprecated]**
* [Lunum](https://github.com/jzrake/lunum) **[Deprecated]**
* [Keras GPT Copilot](https://github.com/fabprezja/keras-gpt-copilot) - A python package that integrates an LLM copilot inside the keras model development workflow.

<a name="lua-demos-and-scripts"></a>
#### Demos and Scripts
* [Core torch7 demos repository](https://github.com/e-lab/torch7-demos).
  * linear-regression, logistic-regression
  * face detector (training and detection as separate demos)
  * mst-based-segmenter
  * train-a-digit-classifier
  * train-autoencoder
  * optical flow demo
  * train-on-housenumbers
  * train-on-cifar
  * tracking with deep nets
  * kinect demo
  * filter-bank visualization
  * saliency-networks
* [Training a Convnet for the Galaxy-Zoo Kaggle challenge(CUDA demo)](https://github.com/soumith/galaxyzoo)
* [torch-datasets](https://github.com/rosejn/torch-datasets) - Scripts to load several popular datasets including:
  * BSR 500
  * CIFAR-10
  * COIL
  * Street View House Numbers
  * MNIST
  * NORB
* [Atari2600](https://github.com/fidlej/aledataset) - Scripts to generate a dataset with static frames from the Arcade Learning Environment.



<a name="matlab"></a>
## Matlab

<a name="matlab-computer-vision"></a>
#### Computer Vision

* [Contourlets](http://www.ifp.illinois.edu/~minhdo/software/contourlet_toolbox.tar) - MATLAB source code that implements the contourlet transform and its utility functions.
* [Shearlets](https://www3.math.tu-berlin.de/numerik/www.shearlab.org/software) - MATLAB code for shearlet transform.
* [Curvelets](http://www.curvelet.org/software.html) - The Curvelet transform is a higher dimensional generalization of the Wavelet transform designed to represent images at different scales and different angles.
* [Bandlets](http://www.cmap.polytechnique.fr/~peyre/download/) - MATLAB code for bandlet transform.
* [mexopencv](https://kyamagu.github.io/mexopencv/) - Collection and a development kit of MATLAB mex functions for OpenCV library.

<a name="matlab-natural-language-processing"></a>
#### Natural Language Processing

* [NLP](https://amplab.cs.berkeley.edu/an-nlp-library-for-matlab/) - A NLP library for Matlab.

<a name="matlab-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Training a deep autoencoder or a classifier
on MNIST digits](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) - Training a deep autoencoder or a classifier
on MNIST digits[DEEP LEARNING].
* [Convolutional-Recursive Deep Learning for 3D Object Classification](https://www.socher.org/index.php/Main/Convolutional-RecursiveDeepLearningFor3DObjectClassification) - Convolutional-Recursive Deep Learning for 3D Object Classification[DEEP LEARNING].
* [Spider](https://people.kyb.tuebingen.mpg.de/spider/) - The spider is intended to be a complete object orientated environment for machine learning in Matlab.
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/#matlab) - A Library for Support Vector Machines.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - An Open-Source SVM Library on GPUs and CPUs
* [LibLinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/#download) - A Library for Large Linear Classification.
* [Machine Learning Module](https://github.com/josephmisiti/machine-learning-module) - Class on machine w/ PDF, lectures, code
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [Pattern Recognition Toolbox](https://github.com/covartech/PRT) - A complete object-oriented environment for machine learning in Matlab.
* [Pattern Recognition and Machine Learning](https://github.com/PRML/PRMLT) - This package contains the matlab implementation of the algorithms described in the book Pattern Recognition and Machine Learning by C. Bishop.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search. Optunity is written in Python but interfaces seamlessly with MATLAB.
* [MXNet](https://github.com/apache/incubator-mxnet/) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Machine Learning in MatLab/Octave](https://github.com/trekhleb/machine-learning-octave) - Examples of popular machine learning algorithms (neural networks, linear/logistic regressions, K-Means, etc.) with code examples and mathematics behind them being explained.
* [MOCluGen](https://github.com/clugen/MOCluGen/) - Multidimensional cluster generation in MATLAB/Octave.

<a name="matlab-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose MATLAB library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [matlab_bgl](https://www.cs.purdue.edu/homes/dgleich/packages/matlab_bgl/) - MatlabBGL is a Matlab package for working with graphs.
* [gaimc](https://www.mathworks.com/matlabcentral/fileexchange/24134-gaimc---graph-algorithms-in-matlab-code) - Efficient pure-Matlab implementations of graph algorithms to complement MatlabBGL's mex functions.

<a name="net"></a>
## .NET

<a name="net-computer-vision"></a>
#### Computer Vision

* [OpenCVDotNet](https://code.google.com/archive/p/opencvdotnet) - A wrapper for the OpenCV project to be used with .NET applications.
* [Emgu CV](http://www.emgu.com/wiki/index.php/Main_Page) - Cross platform wrapper of OpenCV which can be compiled in Mono to be run on Windows, Linus, Mac OS X, iOS, and Android.
* [AForge.NET](http://www.aforgenet.com/framework/) - Open source C# framework for developers and researchers in the fields of Computer Vision and Artificial Intelligence. Development has now shifted to GitHub.
* [Accord.NET](http://accord-framework.net) - Together with AForge.NET, this library can provide image processing and computer vision algorithms to Windows, Windows RT and Windows Phone. Some components are also available for Java and Android.

<a name="net-natural-language-processing"></a>
#### Natural Language Processing

* [Stanford.NLP for .NET](https://github.com/sergey-tihon/Stanford.NLP.NET/) - A full port of Stanford NLP packages to .NET and also available precompiled as a NuGet package.

<a name="net-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Accord-Framework](http://accord-framework.net/) -The Accord.NET Framework is a complete framework for building machine learning, computer vision, computer audition, signal processing and statistical applications.
* [Accord.MachineLearning](https://www.nuget.org/packages/Accord.MachineLearning/) - Support Vector Machines, Decision Trees, Naive Bayesian models, K-means, Gaussian Mixture models and general algorithms such as Ransac, Cross-validation and Grid-Search for machine-learning applications. This package is part of the Accord.NET Framework.
* [DiffSharp](https://diffsharp.github.io/DiffSharp/) - An automatic differentiation (AD) library providing exact and efficient derivatives (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products) for machine learning and optimization applications. Operations can be nested to any level, meaning that you can compute exact higher-order derivatives and differentiate functions that are internally making use of differentiation, for applications such as hyperparameter optimization.
* [Encog](https://www.nuget.org/packages/encog-dotnet-core/) - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trains using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [GeneticSharp](https://github.com/giacomelli/GeneticSharp) - Multi-platform genetic algorithm library for .NET Core and .NET Framework. The library has several implementations of GA operators, like: selection, crossover, mutation, reinsertion and termination.
* [Infer.NET](https://dotnet.github.io/infer/) - Infer.NET is a framework for running Bayesian inference in graphical models. One can use Infer.NET to solve many different kinds of machine learning problems, from standard problems like classification, recommendation or clustering through customized solutions to domain-specific problems. Infer.NET has been used in a wide variety of domains including information retrieval, bioinformatics, epidemiology, vision, and many others.
* [ML.NET](https://github.com/dotnet/machinelearning) - ML.NET is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers. ML.NET was originally developed in Microsoft Research and evolved into a significant framework over the last decade and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and more.
* [Neural Network Designer](https://sourceforge.net/projects/nnd/) - DBMS management system and designer for neural networks. The designer application is developed using WPF, and is a user interface which allows you to design your neural network, query the network, create and configure chat bots that are capable of asking questions and learning from your feedback. The chat bots can even scrape the internet for information to return in their output as well as to use for learning.
* [Synapses](https://github.com/mrdimosthenis/Synapses) - Neural network library in F#.
* [Vulpes](https://github.com/fsprojects/Vulpes) - Deep belief and deep learning implementation written in F# and leverages CUDA GPU execution with Alea.cuBase.
* [MxNet.Sharp](https://github.com/tech-quantum/MxNet.Sharp) - .NET Standard bindings for Apache MxNet with Imperative, Symbolic and Gluon Interface for developing, training and deploying Machine Learning models in C#. https://mxnet.tech-quantum.com/

<a name="net-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [numl](https://www.nuget.org/packages/numl/) - numl is a machine learning library intended to ease the use of using standard modelling techniques for both prediction and clustering.
* [Math.NET Numerics](https://www.nuget.org/packages/MathNet.Numerics/) - Numerical foundation of the Math.NET project, aiming to provide methods and algorithms for numerical computations in science, engineering and everyday use. Supports .Net 4.0, .Net 3.5 and Mono on Windows, Linux and Mac; Silverlight 5, WindowsPhone/SL 8, WindowsPhone 8.1 and Windows 8 with PCL Portable Profiles 47 and 344; Android/iOS with Xamarin.
* [Sho](https://www.microsoft.com/en-us/research/project/sho-the-net-playground-for-data/) - Sho is an interactive environment for data analysis and scientific computing that lets you seamlessly connect scripts (in IronPython) with compiled code (in .NET) to enable fast and flexible prototyping. The environment includes powerful and efficient libraries for linear algebra as well as data visualization that can be used from any .NET language, as well as a feature-rich interactive shell for rapid development.

<a name="objective-c"></a>
## Objective C

<a name="objective-c-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [YCML](https://github.com/yconst/YCML) - A Machine Learning framework for Objective-C and Swift (OS X / iOS).
* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural networks. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available. **[Deprecated]**
* [MAChineLearning](https://github.com/gianlucabertani/MAChineLearning) - An Objective-C multilayer perceptron library, with full support for training through backpropagation. Implemented using vDSP and vecLib, it's 20 times faster than its Java equivalent. Includes sample code for use from Swift.
* [BPN-NeuralNetwork](https://github.com/Kalvar/ios-BPN-NeuralNetwork) - It implemented 3 layers of neural networks ( Input Layer, Hidden Layer and Output Layer ) and it was named Back Propagation Neural Networks (BPN). This network can be used in products recommendation, user behavior analysis, data mining and data analysis. **[Deprecated]**
* [Multi-Perceptron-NeuralNetwork](https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork) - It implemented multi-perceptrons neural network (ニューラルネットワーク) based on Back Propagation Neural Networks (BPN) and designed unlimited-hidden-layers.
* [KRHebbian-Algorithm](https://github.com/Kalvar/ios-KRHebbian-Algorithm) - It is a non-supervisory and self-learning algorithm (adjust the weights) in the neural network of Machine Learning. **[Deprecated]**
* [KRKmeans-Algorithm](https://github.com/Kalvar/ios-KRKmeans-Algorithm) - It implemented K-Means  clustering and classification algorithm. It could be used in data mining and image compression. **[Deprecated]**
* [KRFuzzyCMeans-Algorithm](https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm) - It implemented Fuzzy C-Means (FCM) the fuzzy clustering / classification algorithm on Machine Learning. It could be used in data mining and image compression. **[Deprecated]**

<a name="ocaml"></a>
## OCaml

<a name="ocaml-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [Oml](https://github.com/rleonid/oml) - A general statistics and machine learning library.
* [GPR](https://mmottl.github.io/gpr/) - Efficient Gaussian Process Regression in OCaml.
* [Libra-Tk](https://libra.cs.uoregon.edu) - Algorithms for learning and inference with discrete probabilistic models.
* [TensorFlow](https://github.com/LaurentMazare/tensorflow-ocaml) - OCaml bindings for TensorFlow.

<a name="opencv"></a>
## OpenCV

<a name="opencv-ComputerVision and Text Detection"></a>
### OpenSource-Computer-Vision

* [OpenCV](https://github.com/opencv/opencv) - A OpenSource Computer Vision Library

<a name="perl"></a>
## Perl

<a name="perl-data-analysis--data-visualization"></a>
### Data Analysis / Data Visualization

* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning), a pluggable architecture for data and image processing, which can
be [used for machine learning](https://github.com/zenogantner/PDL-ML).

<a name="perl-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [MXnet for Deep Learning, in Perl](https://github.com/apache/incubator-mxnet/tree/master/perl-package),
also [released in CPAN](https://metacpan.org/pod/AI::MXNet).
* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning),
using AWS machine learning platform from Perl.
* [Algorithm::SVMLight](https://metacpan.org/pod/Algorithm::SVMLight),
  implementation of Support Vector Machines with SVMLight under it. **[Deprecated]**
* Several machine learning and artificial intelligence models are
  included in the [`AI`](https://metacpan.org/search?size=20&q=AI)
  namespace. For instance, you can
  find [Naïve Bayes](https://metacpan.org/pod/AI::NaiveBayes).

<a name="perl6"></a>
## Perl 6

* [Support Vector Machines](https://github.com/titsuki/p6-Algorithm-LibSVM)
* [Naïve Bayes](https://github.com/titsuki/p6-Algorithm-NaiveBayes)

<a name="perl-6-data-analysis--data-visualization"></a>
### Data Analysis / Data Visualization

* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning),
a pluggable architecture for data and image processing, which can
be
[used for machine learning](https://github.com/zenogantner/PDL-ML).

<a name="perl-6-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

<a name="php"></a>
## PHP

<a name="php-natural-language-processing"></a>
### Natural Language Processing

* [jieba-php](https://github.com/fukuball/jieba-php) - Chinese Words Segmentation Utilities.

<a name="php-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [PHP-ML](https://gitlab.com/php-ai/php-ml) - Machine Learning library for PHP. Algorithms, Cross Validation, Neural Network, Preprocessing, Feature Extraction and much more in one library.
* [PredictionBuilder](https://github.com/denissimon/prediction-builder) - A library for machine learning that builds predictions using a linear regression.
* [Rubix ML](https://github.com/RubixML) - A high-level machine learning (ML) library that lets you build programs that learn from data using the PHP language.
* [19 Questions](https://github.com/fulldecent/19-questions) - A machine learning / bayesian inference assigning attributes to objects.

<a name="python"></a>
## Python

<a name="python-computer-vision"></a>
#### Computer Vision

* [LightlyTrain](https://github.com/lightly-ai/lightly-train) - Pretrain computer vision models on unlabeled data for industrial applications
* [Scikit-Image](https://github.com/scikit-image/scikit-image) - A collection of algorithms for image processing in Python.
* [Scikit-Opt](https://github.com/guofei9987/scikit-opt) - Swarm Intelligence in Python (Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm, Immune Algorithm, Artificial Fish Swarm Algorithm in Python)
* [SimpleCV](http://simplecv.org/) - An open source computer vision framework that gives access to several high-powered computer vision libraries, such as OpenCV. Written on Python and runs on Mac, Windows, and Ubuntu Linux.
* [Vigranumpy](https://github.com/ukoethe/vigra) - Python bindings for the VIGRA C++ computer vision library.
* [OpenFace](https://cmusatyalab.github.io/openface/) - Free and open source face recognition with deep neural networks.
* [PCV](https://github.com/jesolem/PCV) - Open source Python module for computer vision. **[Deprecated]**
* [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library that recognizes and manipulates faces from Python or from the command line.
* [deepface](https://github.com/serengil/deepface) - A lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for Python covering cutting-edge models such as VGG-Face, FaceNet, OpenFace, DeepFace, DeepID, Dlib and ArcFace.
* [retinaface](https://github.com/serengil/retinaface) - deep learning based cutting-edge facial detector for Python coming with facial landmarks
* [dockerface](https://github.com/natanielruiz/dockerface) - Easy to install and use deep learning Faster R-CNN face detection for images and video in a docker container. **[Deprecated]**
* [Detectron](https://github.com/facebookresearch/Detectron) - FAIR's software system that implements state-of-the-art object detection algorithms, including Mask R-CNN. It is written in Python and powered by the Caffe2 deep learning framework. **[Deprecated]**
* [detectron2](https://github.com/facebookresearch/detectron2) - FAIR's next-generation research platform for object detection and segmentation. It is a ground-up rewrite of the previous version, Detectron, and is powered by the PyTorch deep learning framework.
* [albumentations](https://github.com/albu/albumentations) - А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops.
* [pytessarct](https://github.com/madmaze/pytesseract) - Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and "read" the text embedded in images. Python-tesseract is a wrapper for [Google's Tesseract-OCR Engine](https://github.com/tesseract-ocr/tesseract).
* [imutils](https://github.com/jrosebr1/imutils) - A library containing Convenience functions to make basic image processing operations such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and Python.
* [PyTorchCV](https://github.com/donnyyou/PyTorchCV) - A PyTorch-Based Framework for Deep Learning in Computer Vision.
* [joliGEN](https://github.com/jolibrain/joliGEN) - Generative AI Image Toolset with GANs and Diffusion for Real-World Applications.
* [Self-supervised learning](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html)
* [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt) - A PyTorch implementation of Justin Johnson's neural-style (neural style transfer).
* [Detecto](https://github.com/alankbi/detecto) - Train and run a computer vision model with 5-10 lines of code.
* [neural-dream](https://github.com/ProGamerGov/neural-dream) - A PyTorch implementation of DeepDream.
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation
* [Deep High-Resolution-Net](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) - A PyTorch implementation of CVPR2019 paper "Deep High-Resolution Representation Learning for Human Pose Estimation"
* [TF-GAN](https://github.com/tensorflow/gan) - TF-GAN is a lightweight library for training and evaluating Generative Adversarial Networks (GANs).
* [dream-creator](https://github.com/ProGamerGov/dream-creator) - A PyTorch implementation of DeepDream. Allows individuals to quickly and easily train their own custom GoogleNet models with custom datasets for DeepDream.
* [Lucent](https://github.com/greentfrapp/lucent) - Tensorflow and OpenAI Clarity's Lucid adapted for PyTorch.
* [lightly](https://github.com/lightly-ai/lightly) - Lightly is a computer vision framework for self-supervised learning.
* [Learnergy](https://github.com/gugarosa/learnergy) - Energy-based machine learning models built upon PyTorch.
* [OpenVisionAPI](https://github.com/openvisionapi) - Open source computer vision API based on open source models.
* [IoT Owl](https://github.com/Ret2Me/IoT-Owl) - Light face detection and recognition system with huge possibilities, based on Microsoft Face API and TensorFlow made for small IoT devices like raspberry pi.
* [Exadel CompreFace](https://github.com/exadel-inc/CompreFace) - face recognition system that can be easily integrated into any system without prior machine learning skills. CompreFace provides REST API for face recognition, face verification, face detection, face mask detection, landmark detection, age, and gender recognition and is easily deployed with docker.
* [computer-vision-in-action](https://github.com/Charmve/computer-vision-in-action) - as known as ``L0CV``, is a new generation of computer vision open source online learning media, a cross-platform interactive learning framework integrating graphics, source code and HTML. the L0CV ecosystem — Notebook, Datasets, Source Code, and from Diving-in to Advanced — as well as the L0CV Hub.
* [timm](https://github.com/rwightman/pytorch-image-models) - PyTorch image models, scripts, pretrained weights -- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more.
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - A PyTorch-based toolkit that offers pre-trained segmentation models for computer vision tasks. It simplifies the development of image segmentation applications by providing a collection of popular architecture implementations, such as UNet and PSPNet, along with pre-trained weights, making it easier for researchers and developers to achieve high-quality pixel-level object segmentation in images.
* [segmentation_models](https://github.com/qubvel/segmentation_models) - A TensorFlow Keras-based toolkit that offers pre-trained segmentation models for computer vision tasks. It simplifies the development of image segmentation applications by providing a collection of popular architecture implementations, such as UNet and PSPNet, along with pre-trained weights, making it easier for researchers and developers to achieve high-quality pixel-level object segmentation in images.
* [MLX](https://github.com/ml-explore/mlx)- MLX is an array framework for machine learning on Apple silicon, developed by Apple machine learning research.

<a name="python-natural-language-processing"></a>
#### Natural Language Processing

* [pkuseg-python](https://github.com/lancopku/pkuseg-python) - A better version of Jieba, developed by Peking University.
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [Pattern](https://github.com/clips/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [Quepy](https://github.com/machinalis/quepy) - A python framework to transform natural language questions to queries in a database query language.
* [TextBlob](http://textblob.readthedocs.io/en/dev/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
* [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora. **[Deprecated]**
* [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
* [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
* [spammy](https://github.com/tasdikrahman/spammy) - A library for email Spam filtering built on top of NLTK
* [loso](https://github.com/fangpenlin/loso) - Another Chinese segmentation library. **[Deprecated]**
* [genius](https://github.com/duanhongyi/genius) - A Chinese segment based on Conditional Random Field.
* [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
* [nut](https://github.com/pprett/nut) - Natural language Understanding Toolkit. **[Deprecated]**
* [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
* [BLLIP Parser](https://pypi.org/project/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser). **[Deprecated]**
* [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
* [PySS3](https://github.com/sergioburdisso/pyss3) - Python package that implements a novel white-box machine learning model for text classification, called SS3. Since SS3 has the ability to visually explain its rationale, this package also comes with easy-to-use interactive visualizations tools ([online demos](http://tworld.io/ss3/)).
* [python-ucto](https://github.com/proycon/python-ucto) - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages).
* [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
* [python-zpar](https://github.com/EducationalTestingService/python-zpar) - Python bindings for [ZPar](https://github.com/frcchang/zpar), a statistical part-of-speech-tagger, constituency parser, and dependency parser for English.
* [colibri-core](https://github.com/proycon/colibri-core) - Python binding to C++ library for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython.
* [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.
* [Distance](https://github.com/doukremt/distance) - Levenshtein and Hamming distance computation. **[Deprecated]**
* [Fuzzy Wuzzy](https://github.com/seatgeek/fuzzywuzzy) - Fuzzy String Matching in Python.
* [Neofuzz](https://github.com/x-tabdeveloping/neofuzz) - Blazing fast, lightweight and customizable fuzzy and semantic text search in Python with fuzzywuzzy/thefuzz compatible API.
* [jellyfish](https://github.com/jamesturk/jellyfish) - a python library for doing approximate and phonetic matching of strings.
* [editdistance](https://pypi.org/project/editdistance/) - fast implementation of edit distance.
* [textacy](https://github.com/chartbeat-labs/textacy) - higher-level NLP built on Spacy.
* [stanford-corenlp-python](https://github.com/dasmith/stanford-corenlp-python) - Python wrapper for [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) **[Deprecated]**
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkit.
* [Rasa](https://github.com/RasaHQ/rasa) - A "machine learning framework to automate text-and voice-based conversations."
* [yase](https://github.com/PPACI/yase) - Transcode sentence (or other sequence) to list of word vector.
* [Polyglot](https://github.com/aboSamoor/polyglot) - Multilingual text (NLP) processing toolkit.
* [DrQA](https://github.com/facebookresearch/DrQA) - Reading Wikipedia to answer open-domain questions.
* [Dedupe](https://github.com/dedupeio/dedupe) - A python library for accurate and scalable fuzzy matching, record deduplication and entity-resolution.
* [Snips NLU](https://github.com/snipsco/snips-nlu) - Natural Language Understanding library for intent classification and entity extraction
* [NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) - Named-entity recognition using neural networks providing state-of-the-art-results
* [DeepPavlov](https://github.com/deepmipt/DeepPavlov/) - conversational AI library with many pre-trained Russian NLP models.
* [BigARTM](https://github.com/bigartm/bigartm) - topic modelling platform.
* [NALP](https://github.com/gugarosa/nalp) - A Natural Adversarial Language Processing framework built over Tensorflow.
* [DL Translate](https://github.com/xhlulu/dl-translate) - A deep learning-based translation library between 50 languages, built with `transformers`.
* [Haystack](https://github.com/deepset-ai/haystack) - A framework for building industrial-strength applications with Transformer models and LLMs.
* [CometLLM](https://github.com/comet-ml/comet-llm) - Track, log, visualize and evaluate your LLM prompts and prompt chains.
* [Transformers](https://github.com/huggingface/transformers) - A deep learning library containing thousands of pre-trained models on different tasks. The goto place for anything related to Large Language Models.
* [TextCL](https://github.com/alinapetukhova/textcl) - Text preprocessing package for use in NLP tasks.

<a name="python-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [ray3.run](https://ray3.run) - AI-powered tools and applications for developers and businesses to enhance productivity and workflow automation. * [XAD](https://pypi.org/project/xad/) -> Fast and easy-to-use backpropagation tool.
 * [Aim](https://github.com/aimhubio/aim) -> An easy-to-use & supercharged open-source AI metadata tracker.
 * [RexMex](https://github.com/AstraZeneca/rexmex) -> A general purpose recommender metrics library for fair evaluation.
 * * [TopFreePrompts by LucyBrain](https://topfreeprompts.com) -> 10,000+ professional AI prompts across 23 categories with systematic training for automating ML workflows and analysis.
 * [ChemicalX](https://github.com/AstraZeneca/chemicalx) -> A PyTorch based deep learning library for drug pair scoring
 * [Microsoft ML for Apache Spark](https://github.com/Azure/mmlspark) -> A distributed machine learning framework Apache Spark
 * [Shapley](https://github.com/benedekrozemberczki/shapley) -> A data-driven framework to quantify the value of classifiers in a machine learning ensemble.
 * [igel](https://github.com/nidhaloff/igel) -> A delightful machine learning tool that allows you to train/fit, test and use models **without writing code**
 * [ML Model building](https://github.com/Shanky-21/Machine_learning) -> A Repository Containing Classification, Clustering, Regression, Recommender Notebooks with illustration to make them.
 * [ML/DL project template](https://github.com/PyTorchLightning/deep-learning-project-template)
 * [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) -> A Modular Framework for Multi-Modal Tabular Learning.
 * [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) -> Graph Neural Network Library for PyTorch.
 * [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) -> A temporal extension of PyTorch Geometric for dynamic graph representation learning.
 * [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) -> A graph sampling extension library for NetworkX with a Scikit-Learn like API.
 * [Karate Club](https://github.com/benedekrozemberczki/karateclub) -> An unsupervised machine learning extension library for NetworkX with a Scikit-Learn like API.
* [Auto_ViML](https://github.com/AutoViML/Auto_ViML) -> Automatically Build Variant Interpretable ML models fast! Auto_ViML is pronounced "auto vimal", is a comprehensive and scalable Python AutoML toolkit with imbalanced handling, ensembling, stacking and built-in feature selection. Featured in <a href="https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46?source=friends_link&sk=d03a0cc55c23deb497d546d6b9be0653">Medium article</a>.
* [PyOD](https://github.com/yzhao062/pyod) -> Python Outlier Detection, comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. Featured for Advanced models, including Neural Networks/Deep Learning and Outlier Ensembles.
* [steppy](https://github.com/neptune-ml/steppy) -> Lightweight, Python library for fast and reproducible machine learning experimentation. Introduces a very simple interface that enables clean machine learning pipeline design.
* [steppy-toolkit](https://github.com/neptune-ml/steppy-toolkit) -> Curated collection of the neural networks, transformers and models that make your machine learning work faster and more effective.
* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit. Documentation can be found [here](https://docs.microsoft.com/cognitive-toolkit/).
* [Couler](https://github.com/couler-proj/couler) - Unified interface for constructing and managing machine learning workflows on different workflow engines, such as Argo Workflows, Tekton Pipelines, and Apache Airflow.
* [auto_ml](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning for production and analytics. Lets you focus on the fun parts of ML, while outputting production-ready code, and detailed analytics of your dataset and results. Includes support for NLP, XGBoost, CatBoost, LightGBM, and soon, deep learning.
* [dtaidistance](https://github.com/wannesm/dtaidistance) - High performance library for time series distances (DTW) and time series clustering.
* [einops](https://github.com/arogozhnikov/einops) - Deep learning operations reinvented (for pytorch, tensorflow, jax and others).
* [machine learning](https://github.com/jeff1evesque/machine-learning) - automated build consisting of a [web-interface](https://github.com/jeff1evesque/machine-learning#web-interface), and set of [programmatic-interface](https://github.com/jeff1evesque/machine-learning#programmatic-interface) API, for support vector machines. Corresponding dataset(s) are stored into a SQL database, then generated model(s) used for prediction(s), are stored into a NoSQL datastore.
* [XGBoost](https://github.com/dmlc/xgboost) - Python bindings for eXtreme Gradient Boosting (Tree) Library.
* [ChefBoost](https://github.com/serengil/chefboost) - a lightweight decision tree framework for Python with categorical feature support covering regular decision tree algorithms such as ID3, C4.5, CART, CHAID and regression tree; also some advanced bagging and boosting techniques such as gradient boosting, random forest and adaboost.
* [Apache SINGA](https://singa.apache.org) - An Apache Incubating project for developing an open source machine learning library.
* [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Book/iPython notebooks on Probabilistic Programming in Python.
* [Featureforge](https://github.com/machinalis/featureforge) A set of tools for creating and testing machine learning features, with a scikit-learn compatible API.
* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - A service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [Towhee](https://towhee.io) - A Python module that encode unstructured data into embeddings.
* [scikit-learn](https://scikit-learn.org/) - A Python module for machine learning built on top of SciPy.
* [metric-learn](https://github.com/metric-learn/metric-learn) - A Python module for metric learning.
* [MCP Memory Service](https://github.com/doobidoo/mcp-memory-service) - Universal memory service with semantic search, autonomous consolidation, and multi-client support for AI applications.
* [OpenMetricLearning](https://github.com/OML-Team/open-metric-learning) - A PyTorch-based framework to train and validate the models producing high-quality embeddings.
* [Intel(R) Extension for Scikit-learn](https://github.com/intel/scikit-learn-intelex) - A seamless way to speed up your Scikit-learn applications with no accuracy loss and code changes.
* [SimpleAI](https://github.com/simpleai-team/simpleai) Python implementation of many of the artificial intelligence algorithms described in the book "Artificial Intelligence, a Modern Approach". It focuses on providing an easy to use, well documented and tested library.
* [astroML](https://www.astroml.org/) - Machine Learning and Data Mining for Astronomy.
* [graphlab-create](https://turi.com/products/create/docs/) - A library with various machine learning models (regression, clustering, recommender systems, graph analytics, etc.) implemented on top of a disk-backed DataFrame.
* [BigML](https://bigml.com) - A library that contacts external servers.
* [pattern](https://github.com/clips/pattern) - Web mining module for Python.
* [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
* [Pylearn2](https://github.com/lisa-lab/pylearn2) - A Machine Learning library based on [Theano](https://github.com/Theano/Theano). **[Deprecated]**
* [keras](https://github.com/keras-team/keras) - High-level neural networks frontend for [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/CNTK) and [Theano](https://github.com/Theano/Theano).
* [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano.
* [hebel](https://github.com/hannes-brt/hebel) - GPU-Accelerated Deep Learning Library in Python. **[Deprecated]**
* [Chainer](https://github.com/chainer/chainer) - Flexible neural network framework.
* [prophet](https://facebook.github.io/prophet/) - Fast and automated time series forecasting framework by Facebook.
* [skforecast](https://github.com/skforecast/skforecast) - Python library for time series forecasting using machine learning models. It works with any regressor compatible with the scikit-learn API, including popular options like LightGBM, XGBoost, CatBoost, Keras, and many others.
* [Feature-engine](https://github.com/feature-engine/feature_engine) - Open source library with an exhaustive battery of feature engineering and selection methods based on pandas and scikit-learn.
* [gensim](https://github.com/RaRe-Technologies/gensim) - Topic Modelling for Humans.
* [Gower Express](https://github.com/momonga-ml/gower-express.git) - The Fastest Gower Distance Implementation for Python. GPU-accelerated similarity matching for mixed data types, 15-25% faster than alternatives with production-ready reliability.
* [tweetopic](https://centre-for-humanities-computing.github.io/tweetopic/) - Blazing fast short-text-topic-modelling for Python.
* [topicwizard](https://github.com/x-tabdeveloping/topic-wizard) - Interactive topic model visualization/interpretation framework.
* [topik](https://github.com/ContinuumIO/topik) - Topic modelling toolkit. **[Deprecated]**
* [PyBrain](https://github.com/pybrain/pybrain) - Another Python Machine Learning Library.
* [Brainstorm](https://github.com/IDSIA/brainstorm) - Fast, flexible and fun neural networks. This is the successor of PyBrain.
* [Surprise](https://surpriselib.com) - A scikit for building and analyzing recommender systems.
* [implicit](https://implicit.readthedocs.io/en/latest/quickstart.html) - Fast Python Collaborative Filtering for Implicit Datasets.
* [LightFM](https://making.lyst.com/lightfm/docs/home.html) -  A Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.
* [Crab](https://github.com/muricoca/crab) - A flexible, fast recommender engine. **[Deprecated]**
* [python-recsys](https://github.com/ocelma/python-recsys) - A Python library for implementing a Recommender System.
* [thinking bayes](https://github.com/AllenDowney/ThinkBayes) - Book on Bayesian Analysis.
* [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/williamFalcon/pix2pix-keras) - Implementation of image to image (pix2pix) translation from the paper by [isola et al](https://arxiv.org/pdf/1611.07004.pdf).[DEEP LEARNING]
* [Restricted Boltzmann Machines](https://github.com/echen/restricted-boltzmann-machines) -Restricted Boltzmann Machines in Python. [DEEP LEARNING]
* [Bolt](https://github.com/pprett/bolt) - Bolt Online Learning Toolbox. **[Deprecated]**
* [CoverTree](https://github.com/patvarilly/CoverTree) - Python implementation of cover trees, near-drop-in replacement for scipy.spatial.kdtree **[Deprecated]**
* [nilearn](https://github.com/nilearn/nilearn) - Machine learning for NeuroImaging in Python.
* [neuropredict](https://github.com/raamana/neuropredict) - Aimed at novice machine learners and non-expert programmers, this package offers easy (no coding needed) and comprehensive machine learning (evaluation and full report of predictive performance WITHOUT requiring you to code) in Python for NeuroImaging and any other type of features. This is aimed at absorbing much of the ML workflow, unlike other packages like nilearn and pymvpa, which require you to learn their API and code to produce anything useful.
* [imbalanced-learn](https://imbalanced-learn.org/stable/) - Python module to perform under sampling and oversampling with various techniques.
* [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble) - Python toolbox for quick implementation, modification, evaluation, and visualization of ensemble learning algorithms for class-imbalanced data. Supports out-of-the-box multi-class imbalanced (long-tailed) classification.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [Pyevolve](https://github.com/perone/Pyevolve) - Genetic algorithm framework. **[Deprecated]**
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [breze](https://github.com/breze-no-salt/breze) - Theano based library for deep and recurrent neural networks.
* [Cortex](https://github.com/cortexlabs/cortex) - Open source platform for deploying machine learning models in production.
* [pyhsmm](https://github.com/mattjj/pyhsmm) - library for approximate unsupervised inference in Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM and HDP-HSMM, mostly with weak-limit approximations.
* [SKLL](https://github.com/EducationalTestingService/skll) - A wrapper around scikit-learn that makes it simpler to conduct experiments.
* [neurolab](https://github.com/zueve/neurolab)
* [Spearmint](https://github.com/HIPS/Spearmint) - Spearmint is a package to perform Bayesian optimization according to the algorithms outlined in the paper: Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan P. Adams. Advances in Neural Information Processing Systems, 2012. **[Deprecated]**
* [Pebl](https://github.com/abhik/pebl/) - Python Environment for Bayesian Learning. **[Deprecated]**
* [Theano](https://github.com/Theano/Theano/) - Optimizing GPU-meta-programming code generating array oriented optimizing math compiler in Python.
* [TensorFlow](https://github.com/tensorflow/tensorflow/) - Open source software library for numerical computation using data flow graphs.
* [pomegranate](https://github.com/jmschrei/pomegranate) - Hidden Markov Models for Python, implemented in Cython for speed and efficiency.
* [python-timbl](https://github.com/proycon/python-timbl) - A Python extension module wrapping the full TiMBL C++ programming interface. Timbl is an elaborate k-Nearest Neighbours machine learning toolkit.
* [deap](https://github.com/deap/deap) - Evolutionary algorithm framework.
* [pydeep](https://github.com/andersbll/deeppy) - Deep Learning In Python. **[Deprecated]**
* [mlxtend](https://github.com/rasbt/mlxtend) - A library consisting of useful tools for data science and machine learning tasks.
* [neon](https://github.com/NervanaSystems/neon) - Nervana's [high-performance](https://github.com/soumith/convnet-benchmarks) Python-based Deep Learning framework [DEEP LEARNING]. **[Deprecated]**
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search.
* [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) - Code samples for my book "Neural Networks and Deep Learning" [DEEP LEARNING].
* [Annoy](https://github.com/spotify/annoy) - Approximate nearest neighbours implementation.
* [TPOT](https://github.com/EpistasisLab/tpot) - Tool that automatically creates and optimizes machine learning pipelines using genetic programming. Consider it your personal data science assistant, automating a tedious part of machine learning.
* [pgmpy](https://github.com/pgmpy/pgmpy) A python library for working with Probabilistic Graphical Models.
* [DIGITS](https://github.com/NVIDIA/DIGITS) - The Deep Learning GPU Training System (DIGITS) is a web application for training deep learning models.
* [Orange](https://orange.biolab.si/) - Open source data visualization and data analysis for novices and experts.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [milk](https://github.com/luispedro/milk) - Machine learning toolkit focused on supervised classification. **[Deprecated]**
* [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow.
* [REP](https://github.com/yandex/rep) - an IPython-based environment for conducting data-driven research in a consistent and reproducible way. REP is not trying to substitute scikit-learn, but extends it and provides better user experience. **[Deprecated]**
* [rgf_python](https://github.com/RGF-team/rgf) - Python bindings for Regularized Greedy Forest (Tree) Library.
* [skbayes](https://github.com/AmazaspShumik/sklearn-bayes) - Python package for Bayesian Machine Learning with scikit-learn API.
* [fuku-ml](https://github.com/fukuball/fuku-ml) - Simple machine learning library, including Perceptron, Regression, Support Vector Machine, Decision Tree and more, it's easy to use and easy to learn for beginners.
* [Xcessiv](https://github.com/reiinakano/xcessiv) - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - The lightweight PyTorch wrapper for high-performance AI research.
* [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts) - Toolbox of models, callbacks, and datasets for AI/ML researchers.
* [skorch](https://github.com/skorch-dev/skorch) - A scikit-learn compatible neural network library that wraps PyTorch.
* [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) - Implementations of Machine Learning models from scratch in Python with a focus on transparency. Aims to showcase the nuts and bolts of ML in an accessible way.
* [Edward](http://edwardlib.org/) - A library for probabilistic modelling, inference, and criticism. Built on top of TensorFlow.
* [xRBM](https://github.com/omimo/xRBM) - A library for Restricted Boltzmann Machine (RBM) and its conditional variants in Tensorflow.
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, well documented and supports CPU and GPU (even multi-GPU) computation.
* [stacked_generalization](https://github.com/fukatani/stacked_generalization) - Implementation of machine learning stacking technique as a handy library in Python.
* [modAL](https://github.com/modAL-python/modAL) - A modular active learning framework for Python, built on top of scikit-learn.
* [Cogitare](https://github.com/cogitare-ai/cogitare): A Modern, Fast, and Modular Deep Learning and Machine Learning framework for Python.
* [Parris](https://github.com/jgreenemi/Parris) - Parris, the automated infrastructure setup tool for machine learning algorithms.
* [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [Turi Create](https://github.com/apple/turicreate) - Machine learning from Apple. Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
* [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement and recommender systems.
* [mlens](https://github.com/flennerhag/mlens) - A high performance, memory efficient, maximally parallelized ensemble learning, integrated with scikit-learn.
* [Thampi](https://github.com/scoremedia/thampi) - Machine Learning Prediction System on AWS Lambda
* [MindsDB](https://github.com/mindsdb/mindsdb) - Open Source framework to streamline use of neural networks.
* [Microsoft Recommenders](https://github.com/Microsoft/Recommenders): Examples and best practices for building recommendation systems, provided as Jupyter notebooks. The repo contains some of the latest state of the art algorithms from Microsoft Research as well as from other companies and institutions.
* [StellarGraph](https://github.com/stellargraph/stellargraph): Machine Learning on Graphs, a Python library for machine learning on graph-structured (network-structured) data.
* [BentoML](https://github.com/bentoml/bentoml): Toolkit for package and deploy machine learning models for serving in production
* [MiraiML](https://github.com/arthurpaulino/miraiml): An asynchronous engine for continuous & autonomous machine learning, built for real-time usage.
* [numpy-ML](https://github.com/ddbourgin/numpy-ml): Reference implementations of ML models written in numpy
* [Neuraxle](https://github.com/Neuraxio/Neuraxle): A framework providing the right abstractions to ease research, development, and deployment of your ML pipelines.
* [Cornac](https://github.com/PreferredAI/cornac) - A comparative framework for multimodal recommender systems with a focus on models leveraging auxiliary data.
* [JAX](https://github.com/google/jax) - JAX is Autograd and XLA, brought together for high-performance machine learning research.
* [Catalyst](https://github.com/catalyst-team/catalyst) - High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
* [Fastai](https://github.com/fastai/fastai) - High-level wrapper built on the top of Pytorch which supports vision, text, tabular data and collaborative filtering.
* [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) - A machine learning framework for multi-output/multi-label and stream data.
* [Lightwood](https://github.com/mindsdb/lightwood) - A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with objective to build predictive models with one line of code.
* [bayeso](https://github.com/jungtaekkim/bayeso) - A simple, but essential Bayesian optimization package, written in Python.
* [mljar-supervised](https://github.com/mljar/mljar-supervised) - An Automated Machine Learning (AutoML) python package for tabular data. It can handle: Binary Classification, MultiClass Classification and Regression. It provides explanations and markdown reports.
* [evostra](https://github.com/alirezamika/evostra) - A fast Evolution Strategy implementation in Python.
* [Determined](https://github.com/determined-ai/determined) - Scalable deep learning training platform, including integrated support for distributed training, hyperparameter tuning, experiment tracking, and model management.
* [PySyft](https://github.com/OpenMined/PySyft) - A Python library for secure and private Deep Learning built on PyTorch and TensorFlow.
* [PyGrid](https://github.com/OpenMined/PyGrid/) - Peer-to-peer network of data owners and data scientists who can collectively train AI models using PySyft
* [sktime](https://github.com/alan-turing-institute/sktime) - A unified framework for machine learning with time series
* [OPFython](https://github.com/gugarosa/opfython) - A Python-inspired implementation of the Optimum-Path Forest classifier.
* [Opytimizer](https://github.com/gugarosa/opytimizer) - Python-based meta-heuristic optimization techniques.
* [Gradio](https://github.com/gradio-app/gradio) - A Python library for quickly creating and sharing demos of models. Debug models interactively in your browser, get feedback from collaborators, and generate public links without deploying anything.
* [Hub](https://github.com/activeloopai/Hub) - Fastest unstructured dataset management for TensorFlow/PyTorch. Stream & version-control data. Store even petabyte-scale data in a single numpy-like array on the cloud accessible on any machine. Visit [activeloop.ai](https://activeloop.ai) for more info.
* [Synthia](https://github.com/dmey/synthia) - Multidimensional synthetic data generation in Python.
* [ByteHub](https://github.com/bytehub-ai/bytehub) - An easy-to-use, Python-based feature store. Optimized for time-series data.
* [Backprop](https://github.com/backprop-ai/backprop) - Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.
* [River](https://github.com/online-ml/river): A framework for general purpose online machine learning.
* [FEDOT](https://github.com/nccr-itmo/FEDOT): An AutoML framework for the automated design of composite modelling pipelines. It can handle classification, regression, and time series forecasting tasks on different types of data (including multi-modal datasets).
* [Sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt): An AutoML package for hyperparameters tuning using evolutionary algorithms, with built-in callbacks, plotting, remote logging and more.
* [Evidently](https://github.com/evidentlyai/evidently): Interactive reports to analyze machine learning models during validation or production monitoring.
* [Streamlit](https://github.com/streamlit/streamlit): Streamlit is an framework to create beautiful data apps in hours, not weeks.
* [Optuna](https://github.com/optuna/optuna): Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.
* [Deepchecks](https://github.com/deepchecks/deepchecks): Validation & testing of machine learning models and data during model development, deployment, and production. This includes checks and suites related to various types of issues, such as model performance, data integrity, distribution mismatches, and more.
* [Shapash](https://github.com/MAIF/shapash) : Shapash is a Python library that provides several types of visualization that display explicit labels that everyone can understand.
* [Eurybia](https://github.com/MAIF/eurybia): Eurybia monitors data and model drift over time and securizes model deployment with data validation.
* [Colossal-AI](https://github.com/hpcaitech/ColossalAI): An open-source deep learning system for large-scale model training and inference with high efficiency and low cost.
* [skrub](https://github.com/skrub-data/skrub) - Skrub is a Python library that eases preprocessing and feature engineering for machine learning on dataframes.
* [Upgini](https://github.com/upgini/upgini): Free automated data & feature enrichment library for machine learning - automatically searches through thousands of ready-to-use features from public and community shared data sources and enriches your training dataset with only the accuracy improving features.
* [AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics](https://github.com/Western-OC2-Lab/AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics): A tutorial to help machine learning researchers to automatically obtain optimized machine learning models with the optimal learning performance on any specific task.
* [SKBEL](https://github.com/robinthibaut/skbel): A Python library for Bayesian Evidential Learning (BEL) in order to estimate the uncertainty of a prediction.
* [NannyML](https://bit.ly/nannyml-github-machinelearning): Python library capable of fully capturing the impact of data drift on performance. Allows estimation of post-deployment model performance without access to targets.
* [cleanlab](https://github.com/cleanlab/cleanlab): The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels.
* [AutoGluon](https://github.com/awslabs/autogluon): AutoML for Image, Text, Tabular, Time-Series, and MultiModal Data.
* [PyBroker](https://github.com/edtechre/pybroker) - Algorithmic Trading with Machine Learning.
* [Frouros](https://github.com/IFCA/frouros): Frouros is an open source Python library for drift detection in machine learning systems.
* [CometML](https://github.com/comet-ml/comet-examples): The best-in-class MLOps platform with experiment tracking, model production monitoring, a model registry, and data lineage from training straight through to production.
* [Okrolearn](https://github.com/Okerew/okrolearn): A python machine learning library created to combine powefull data analasys features with tensors and machine learning components, while maintaining support for other libraries.
* [Opik](https://github.com/comet-ml/opik): Evaluate, trace, test, and ship LLM applications across your dev and production lifecycles.
* [pyclugen](https://github.com/clugen/pyclugen) - Multidimensional cluster generation in Python.
* [mlforgex](https://github.com/dhgefergfefruiwefhjhcduc/ML_Forgex) - Lightweight ML utility for automated training, evaluation, and prediction with CLI and Python API support.

<a name="python-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization
* [DataComPy](https://github.com/capitalone/datacompy) - A library to compare Pandas, Polars, and Spark data frames. It provides stats and lets users adjust for match accuracy.
* [DataVisualization](https://github.com/Shanky-21/Data_visualization) - A GitHub Repository Where you can Learn Datavisualizatoin Basics to Intermediate level.
* [Cartopy](https://scitools.org.uk/cartopy/docs/latest/) - Cartopy is a Python package designed for geospatial data processing in order to produce maps and other geospatial data analyses.
* [SciPy](https://www.scipy.org/) - A Python-based ecosystem of open-source software for mathematics, science, and engineering.
* [NumPy](https://www.numpy.org/) - A fundamental package for scientific computing with Python.
* [AutoViz](https://github.com/AutoViML/AutoViz) AutoViz performs automatic visualization of any dataset with a single line of Python code. Give it any input file (CSV, txt or JSON) of any size and AutoViz will visualize it. See <a href="https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad?source=friends_link&sk=c9e9503ec424b191c6096d7e3f515d10">Medium article</a>.
* [Numba](https://numba.pydata.org/) - Python JIT (just in time) compiler to LLVM aimed at scientific Python by the developers of Cython and NumPy.
* [Mars](https://github.com/mars-project/mars) - A tensor-based framework for large-scale data computation which is often regarded as a parallel and distributed version of NumPy.
* [NetworkX](https://networkx.github.io/) - A high-productivity software for complex networks.
* [igraph](https://igraph.org/python/) - binding to igraph library - General purpose graph library.
* [Pandas](https://pandas.pydata.org/) - A library providing high-performance, easy-to-use data structures and data analysis tools.
* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose Python library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [Vaex](https://github.com/vaexio/vaex) - A high performance Python library for lazy Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets. Documentation can be found [here](https://vaex.io/docs/index.html).
* [Open Mining](https://github.com/mining/mining) - Business Intelligence (BI) in Python (Pandas web interface) **[Deprecated]**
* [PyMC](https://github.com/pymc-devs/pymc) - Markov Chain Monte Carlo sampling toolkit.
* [zipline](https://github.com/quantopian/zipline) - A Pythonic algorithmic trading library.
* [PyDy](https://www.pydy.org/) - Short for Python Dynamics, used to assist with workflow in the modelling of dynamic motion based around NumPy, SciPy, IPython, and matplotlib.
* [SymPy](https://github.com/sympy/sympy) - A Python library for symbolic mathematics.
* [statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modelling and econometrics in Python.
* [astropy](https://www.astropy.org/) - A community Python library for Astronomy.
* [matplotlib](https://matplotlib.org/) - A Python 2D plotting library.
* [bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python.
* [plotly](https://plot.ly/python/) - Collaborative web plotting for Python and matplotlib.
* [altair](https://github.com/altair-viz/altair) - A Python to Vega translator.
* [d3py](https://github.com/mikedewar/d3py) - A plotting library for Python, based on [D3.js](https://d3js.org/).
* [PyDexter](https://github.com/D3xterjs/pydexter) - Simple plotting for Python. Wrapper for D3xterjs; easily render charts in-browser.
* [ggplot](https://github.com/yhat/ggpy) - Same API as ggplot2 for R. **[Deprecated]**
* [ggfortify](https://github.com/sinhrks/ggfortify) - Unified interface to ggplot2 popular R packages.
* [Kartograph.py](https://github.com/kartograph/kartograph.py) - Rendering beautiful SVG maps in Python.
* [pygal](http://pygal.org/en/stable/) - A Python SVG Charts Creator.
* [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) - A pure-python graphics and GUI library built on PyQt4 / PySide and NumPy.
* [pycascading](https://github.com/twitter/pycascading) **[Deprecated]**
* [Petrel](https://github.com/AirSage/Petrel) - Tools for writing, submitting, debugging, and monitoring Storm topologies in pure Python.
* [Blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data.
* [emcee](https://github.com/dfm/emcee) - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [windML](https://github.com/cigroup-ol/windml) - A Python Framework for Wind Energy Analysis and Prediction.
* [vispy](https://github.com/vispy/vispy) - GPU-based high-performance interactive OpenGL 2D/3D data visualization library.
* [cerebro2](https://github.com/numenta/nupic.cerebro2) A web-based visualization and debugging platform for NuPIC. **[Deprecated]**
* [NuPIC Studio](https://github.com/htm-community/nupic.studio) An all-in-one NuPIC Hierarchical Temporal Memory visualization and debugging super-tool! **[Deprecated]**
* [SparklingPandas](https://github.com/sparklingpandas/sparklingpandas) Pandas on PySpark (POPS).
* [Seaborn](https://seaborn.pydata.org/) - A python visualization library based on matplotlib.
* [ipychart](https://github.com/nicohlr/ipychart) - The power of Chart.js in Jupyter Notebook.
* [bqplot](https://github.com/bloomberg/bqplot) - An API for plotting in Jupyter (IPython).
* [pastalog](https://github.com/rewonc/pastalog) - Simple, realtime visualization of neural network training performance.
* [Superset](https://github.com/apache/incubator-superset) - A data exploration platform designed to be visual, intuitive, and interactive.
* [Dora](https://github.com/nathanepstein/dora) - Tools for exploratory data analysis in Python.
* [Ruffus](http://www.ruffus.org.uk) - Computation Pipeline library for python.
* [SOMPY](https://github.com/sevamoo/SOMPY) - Self Organizing Map written in Python (Uses neural networks for data analysis).
* [somoclu](https://github.com/peterwittek/somoclu) Massively parallel self-organizing maps: accelerate training on multicore CPUs, GPUs, and clusters, has python API.
* [HDBScan](https://github.com/lmcinnes/hdbscan) - implementation of the hdbscan algorithm in Python - used for clustering
* [visualize_ML](https://github.com/ayush1997/visualize_ML) - A python package for data exploration and data analysis. **[Deprecated]**
* [scikit-plot](https://github.com/reiinakano/scikit-plot) - A visualization library for quick and easy generation of common plots in data analysis and machine learning.
* [Bowtie](https://github.com/jwkvam/bowtie) - A dashboard library for interactive visualizations using flask socketio and react.
* [lime](https://github.com/marcotcr/lime) - Lime is about explaining what machine learning classifiers (or models) are doing. It is able to explain any black box classifier, with two or more classes.
* [PyCM](https://github.com/sepandhaghighi/pycm) - PyCM is a multi-class confusion matrix library written in Python that supports both input data vectors and direct matrix, and a proper tool for post-classification model evaluation that supports most classes and overall statistics parameters
* [Dash](https://github.com/plotly/dash) - A framework for creating analytical web applications built on top of Plotly.js, React, and Flask
* [Lambdo](https://github.com/asavinov/lambdo) - A workflow engine for solving machine learning problems by combining in one analysis pipeline (i) feature engineering and machine learning (ii) model training and prediction (iii) table population and column evaluation via user-defined (Python) functions.
* [TensorWatch](https://github.com/microsoft/tensorwatch) - Debugging and visualization tool for machine learning and data science. It extensively leverages Jupyter Notebook to show real-time visualizations of data in running processes such as machine learning training.
* [dowel](https://github.com/rlworkgroup/dowel) - A little logger for machine learning research. Output any object to the terminal, CSV, TensorBoard, text logs on disk, and more with just one call to `logger.log()`.
* [Flama](https://github.com/vortico/flama) - Ignite your models into blazing-fast machine learning APIs with a modern framework.

<a name="python-misc-scripts--ipython-notebooks--codebases"></a>
#### Misc Scripts / iPython Notebooks / Codebases
* [minidiff](https://github.com/ahoynodnarb/minidiff) - A slightly larger, somewhat feature-complete, PyTorch-inspired, NumPy implementation of a tensor reverse-mode automatic differentiation engine.
* [MiniGrad](https://github.com/kennysong/minigrad) – A minimal, educational, Pythonic implementation of autograd (~100 loc).
* [Map/Reduce implementations of common ML algorithms](https://github.com/Yannael/BigDataAnalytics_INFOH515): Jupyter notebooks that cover how to implement from scratch different ML algorithms (ordinary least squares, gradient descent, k-means, alternating least squares), using Python NumPy, and how to then make these implementations scalable using Map/Reduce and Spark.
* [BioPy](https://github.com/jaredthecoder/BioPy) - Biologically-Inspired and Machine Learning Algorithms in Python. **[Deprecated]**
* [CAEs for Data Assimilation](https://github.com/julianmack/Data_Assimilation) - Convolutional autoencoders for 3D image/field compression applied to reduced order [Data Assimilation](https://en.wikipedia.org/wiki/Data_assimilation).
* [handsonml](https://github.com/ageron/handson-ml) - Fundamentals of machine learning in python.
* [SVM Explorer](https://github.com/plotly/dash-svm) - Interactive SVM Explorer, using Dash and scikit-learn
* [pattern_classification](https://github.com/rasbt/pattern_classification)
* [thinking stats 2](https://github.com/Wavelets/ThinkStats2)
* [hyperopt](https://github.com/hyperopt/hyperopt-sklearn)
* [numpic](https://github.com/numenta/nupic)
* [2012-paper-diginorm](https://github.com/dib-lab/2012-paper-diginorm)
* [A gallery of interesting IPython notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)
* [ipython-notebooks](https://github.com/ogrisel/notebooks)
* [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) - Continually updated Data Science Python Notebooks: Spark, Hadoop MapReduce, HDFS, AWS, Kaggle, scikit-learn, matplotlib, pandas, NumPy, SciPy, and various command lines.
* [decision-weights](https://github.com/CamDavidsonPilon/decision-weights)
* [Sarah Palin LDA](https://github.com/Wavelets/sarah-palin-lda) - Topic Modelling the Sarah Palin emails.
* [Diffusion Segmentation](https://github.com/Wavelets/diffusion-segmentation) - A collection of image segmentation algorithms based on diffusion methods.
* [Scipy Tutorials](https://github.com/Wavelets/scipy-tutorials) - SciPy tutorials. This is outdated, check out scipy-lecture-notes.
* [Crab](https://github.com/marcelcaraciolo/crab) - A recommendation engine library for Python.
* [BayesPy](https://github.com/maxsklar/BayesPy) - Bayesian Inference Tools in Python.
* [scikit-learn tutorials](https://github.com/GaelVaroquaux/scikit-learn-tutorial) - Series of notebooks for learning scikit-learn.
* [sentiment-analyzer](https://github.com/madhusudancs/sentiment-analyzer) - Tweets Sentiment Analyzer
* [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - Sentiment classifier using word sense disambiguation.
* [group-lasso](https://github.com/fabianp/group_lasso) - Some experiments with the coordinate descent algorithm used in the (Sparse) Group Lasso model.
* [jProcessing](https://github.com/kevincobain2000/jProcessing) - Kanji / Hiragana / Katakana to Romaji Converter. Edict Dictionary & parallel sentences Search. Sentence Similarity between two JP Sentences. Sentiment Analysis of Japanese Text. Run Cabocha(ISO--8859-1 configured) in Python.
* [mne-python-notebooks](https://github.com/mne-tools/mne-python-notebooks) - IPython notebooks for EEG/MEG data processing using mne-python.
* [Neon Course](https://github.com/NervanaSystems/neon_course) - IPython notebooks for a complete course around understanding Nervana's Neon.
* [pandas cookbook](https://github.com/jvns/pandas-cookbook) - Recipes for using Python's pandas library.
* [climin](https://github.com/BRML/climin) - Optimization library focused on machine learning, pythonic implementations of gradient descent, LBFGS, rmsprop, adadelta and others.
* [Allen Downey’s Data Science Course](https://github.com/AllenDowney/DataScience) - Code for Data Science at Olin College, Spring 2014.
* [Allen Downey’s Think Bayes Code](https://github.com/AllenDowney/ThinkBayes) - Code repository for Think Bayes.
* [Allen Downey’s Think Complexity Code](https://github.com/AllenDowney/ThinkComplexity) - Code for Allen Downey's book Think Complexity.
* [Allen Downey’s Think OS Code](https://github.com/AllenDowney/ThinkOS) - Text and supporting code for Think OS: A Brief Introduction to Operating Systems.
* [Python Programming for the Humanities](https://www.karsdorp.io/python-course/) - Course for Python programming for the Humanities, assuming no prior knowledge. Heavy focus on text processing / NLP.
* [GreatCircle](https://github.com/mwgg/GreatCircle) - Library for calculating great circle distance.
* [Optunity examples](http://optunity.readthedocs.io/en/latest/notebooks/index.html) - Examples demonstrating how to use Optunity in synergy with machine learning libraries.
* [Dive into Machine Learning  with Python Jupyter notebook and scikit-learn](https://github.com/hangtwenty/dive-into-machine-learning) - "I learned Python by hacking first, and getting serious *later.* I wanted to do this with Machine Learning. If this is your style, join me in getting a bit ahead of yourself."
* [TDB](https://github.com/ericjang/tdb) - TensorDebugger (TDB) is a visual debugger for deep learning. It features interactive, node-by-node debugging and visualization for TensorFlow.
* [Suiron](https://github.com/kendricktan/suiron/) - Machine Learning for RC Cars.
* [Introduction to machine learning with scikit-learn](https://github.com/justmarkham/scikit-learn-videos) - IPython notebooks from Data School's video tutorials on scikit-learn.
* [Practical XGBoost in Python](https://parrotprediction.teachable.com/p/practical-xgboost-in-python) - comprehensive online course about using XGBoost in Python.
* [Introduction to Machine Learning with Python](https://github.com/amueller/introduction_to_ml_with_python) - Notebooks and code for the book "Introduction to Machine Learning with Python"
* [Pydata book](https://github.com/wesm/pydata-book) - Materials and IPython notebooks for "Python for Data Analysis" by Wes McKinney, published by O'Reilly Media
* [Homemade Machine Learning](https://github.com/trekhleb/homemade-machine-learning) - Python examples of popular machine learning algorithms with interactive Jupyter demos and math being explained
* [Prodmodel](https://github.com/prodmodel/prodmodel) - Build tool for data science pipelines.
* [the-elements-of-statistical-learning](https://github.com/maitbayev/the-elements-of-statistical-learning) - This repository contains Jupyter notebooks implementing the algorithms found in the book and summary of the textbook.
* [Hyperparameter-Optimization-of-Machine-Learning-Algorithms](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms) - Code for hyperparameter tuning/optimization of machine learning and deep learning algorithms.
* [Heart_Disease-Prediction](https://github.com/ShivamChoudhary17/Heart_Disease) - Given clinical parameters about a patient, can we predict whether or not they have heart disease?
* [Flight Fare Prediction](https://github.com/ShivamChoudhary17/Flight_Fare_Prediction) - This basically to gauge the understanding of Machine Learning Workflow and Regression technique in specific.
* [Keras Tuner](https://github.com/keras-team/keras-tuner) - An easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search.



<a name="python-neural-networks"></a>
#### Neural Networks

* [Kinho](https://github.com/kinhosz/Neural) - Simple API for Neural Network. Better for image processing with CPU/GPU + Transfer Learning.
* [nn_builder](https://github.com/p-christ/nn_builder) - nn_builder is a python package that lets you build neural networks in 1 line
* [NeuralTalk](https://github.com/karpathy/neuraltalk) - NeuralTalk is a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences.
* [NeuralTalk](https://github.com/karpathy/neuraltalk2) - NeuralTalk is a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences. **[Deprecated]**
* [Neuron](https://github.com/molcik/python-neuron) - Neuron is simple class for time series predictions. It's utilize LNU (Linear Neural Unit), QNU (Quadratic Neural Unit), RBF (Radial Basis Function), MLP (Multi Layer Perceptron), MLP-ELM (Multi Layer Perceptron - Extreme Learning Machine) neural networks learned with Gradient descent or LeLevenberg–Marquardt algorithm. **[Deprecated]**
* [Data Driven Code](https://github.com/atmb4u/data-driven-code) - Very simple implementation of neural networks for dummies in python without using any libraries, with detailed comments.
* [Machine Learning, Data Science and Deep Learning with Python](https://www.manning.com/livevideo/machine-learning-data-science-and-deep-learning-with-python) - LiveVideo course that covers machine learning, Tensorflow, artificial intelligence, and neural networks.
* [TResNet: High Performance GPU-Dedicated Architecture](https://github.com/mrT23/TResNet) - TResNet models were designed and optimized to give the best speed-accuracy tradeoff out there on GPUs.
* [TResNet: Simple and powerful neural network library for python](https://github.com/zueve/neurolab) - Variety of supported types of Artificial Neural Network and learning algorithms.
* [Jina AI](https://jina.ai/) An easier way to build neural search in the cloud. Compatible with Jupyter Notebooks.
* [sequitur](https://github.com/shobrook/sequitur) PyTorch library for creating and training sequence autoencoders in just two lines of code


<a name="python-spiking-neural-networks"></a>
#### Spiking Neural Networks

* [Rockpool](https://github.com/synsense/rockpool) - A machine learning library for spiking neural networks. Supports training with both torch and jax pipelines, and deployment to neuromorphic hardware.
* [Sinabs](https://github.com/synsense/sinabs) - A deep learning library for spiking neural networks which is based on PyTorch, focuses on fast training and supports inference on neuromorphic hardware.
* [Tonic](https://github.com/neuromorphs/tonic) - A library that makes downloading publicly available neuromorphic datasets a breeze and provides event-based data transformation/augmentation pipelines.

<a name="python-survival-analysis"></a>
#### Python Survival Analysis
* [lifelines](https://github.com/CamDavidsonPilon/lifelines) - lifelines is a complete survival analysis library, written in pure Python
* [Scikit-Survival](https://github.com/sebp/scikit-survival) - scikit-survival is a Python module for survival analysis built on top of scikit-learn. It allows doing survival analysis while utilizing the power of scikit-learn, e.g., for pre-processing or doing cross-validation.

<a name="python-federated-learning"></a>
#### Federated Learning
* [Flower](https://flower.dev/) - A unified approach to federated learning, analytics, and evaluation. Federate any workload, any ML framework, and any programming language.
* [PySyft](https://github.com/OpenMined/PySyft) - A Python library for secure and private Deep Learning.
* [Tensorflow-Federated](https://www.tensorflow.org/federated) A federated learning framework for machine learning and other computations on decentralized data.

<a name="python-kaggle-competition-source-code"></a>
#### Kaggle Competition Source Code
* [open-solution-home-credit](https://github.com/neptune-ml/open-solution-home-credit) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk) for [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
* [open-solution-googleai-object-detection](https://github.com/neptune-ml/open-solution-googleai-object-detection) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Google-AI-Object-Detection-Challenge) for [Google AI Open Images - Object Detection Track](https://www.kaggle.com/c/google-ai-open-images-object-detection-track).
* [open-solution-salt-identification](https://github.com/neptune-ml/open-solution-salt-identification) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Salt-Detection) for [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).
* [open-solution-ship-detection](https://github.com/neptune-ml/open-solution-ship-detection) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Ships) for [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).
* [open-solution-data-science-bowl-2018](https://github.com/neptune-ml/open-solution-data-science-bowl-2018) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Data-Science-Bowl-2018) for [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).
* [open-solution-value-prediction](https://github.com/neptune-ml/open-solution-value-prediction) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Santander-Value-Prediction-Challenge) for [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge).
* [open-solution-toxic-comments](https://github.com/neptune-ml/open-solution-toxic-comments) -> source code for [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
* [wiki challenge](https://github.com/hammer/wikichallenge) - An implementation of Dell Zhang's solution to Wikipedia's Participation Challenge on Kaggle.
* [kaggle insults](https://github.com/amueller/kaggle_insults) - Kaggle Submission for "Detecting Insults in Social Commentary".
* [kaggle_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) - Code for the Kaggle acquire valued shoppers challenge.
* [kaggle-cifar](https://github.com/zygmuntz/kaggle-cifar) - Code for the CIFAR-10 competition at Kaggle, uses cuda-convnet.
* [kaggle-blackbox](https://github.com/zygmuntz/kaggle-blackbox) - Deep learning made easy.
* [kaggle-accelerometer](https://github.com/zygmuntz/kaggle-accelerometer) - Code for Accelerometer Biometric Competition at Kaggle.
* [kaggle-advertised-salaries](https://github.com/zygmuntz/kaggle-advertised-salaries) - Predicting job salaries from ads - a Kaggle competition.
* [kaggle amazon](https://github.com/zygmuntz/kaggle-amazon) - Amazon access control challenge.
* [kaggle-bestbuy_big](https://github.com/zygmuntz/kaggle-bestbuy_big) - Code for the Best Buy competition at Kaggle.
* [kaggle-bestbuy_small](https://github.com/zygmuntz/kaggle-bestbuy_small)
* [Kaggle Dogs vs. Cats](https://github.com/kastnerkyle/kaggle-dogs-vs-cats) - Code for Kaggle Dogs vs. Cats competition.
* [Kaggle Galaxy Challenge](https://github.com/benanne/kaggle-galaxies) - Winning solution for the Galaxy Challenge on Kaggle.
* [Kaggle Gender](https://github.com/zygmuntz/kaggle-gender) - A Kaggle competition: discriminate gender based on handwriting.
* [Kaggle Merck](https://github.com/zygmuntz/kaggle-merck) - Merck challenge at Kaggle.
* [Kaggle Stackoverflow](https://github.com/zygmuntz/kaggle-stackoverflow) - Predicting closed questions on Stack Overflow.
* [kaggle_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) - Code for the Kaggle acquire valued shoppers challenge.
* [wine-quality](https://github.com/zygmuntz/wine-quality) - Predicting wine quality.

<a name="python-reinforcement-learning"></a>
#### Reinforcement Learning
* [DeepMind Lab](https://github.com/deepmind/lab) - DeepMind Lab is a 3D learning environment based on id Software's Quake III Arena via ioquake3 and other open source software. Its primary purpose is to act as a testbed for research in artificial intelligence, especially deep reinforcement learning.
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - A library for developing and comparing reinforcement learning algorithms (successor of [gym])(https://github.com/openai/gym).
* [Serpent.AI](https://github.com/SerpentAI/SerpentAI) - Serpent.AI is a game agent framework that allows you to turn any video game you own into a sandbox to develop AI and machine learning experiments. For both researchers and hobbyists.
* [ViZDoom](https://github.com/mwydmuch/ViZDoom) - ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.
* [Roboschool](https://github.com/openai/roboschool) - Open-source software for robot simulation, integrated with OpenAI Gym.
* [Retro](https://github.com/openai/retro) - Retro Games in Gym
* [SLM Lab](https://github.com/kengz/SLM-Lab) - Modular Deep Reinforcement Learning framework in PyTorch.
* [Coach](https://github.com/NervanaSystems/coach) - Reinforcement Learning Coach by Intel® AI Lab enables easy experimentation with state of the art Reinforcement Learning algorithms
* [garage](https://github.com/rlworkgroup/garage) - A toolkit for reproducible reinforcement learning research
* [metaworld](https://github.com/rlworkgroup/metaworld) - An open source robotics benchmark for meta- and multi-task reinforcement learning
* [acme](https://deepmind.com/research/publications/Acme) - An Open Source Distributed Framework for Reinforcement Learning that makes build and train your agents easily.
* [Spinning Up](https://spinningup.openai.com) - An educational resource designed to let anyone learn to become a skilled practitioner in deep reinforcement learning
* [Maze](https://github.com/enlite-ai/maze) - Application-oriented deep reinforcement learning framework addressing real-world decision problems.
* [RLlib](https://github.com/ray-project/ray) - RLlib is an industry level, highly scalable RL library for tf and torch, based on Ray. It's used by companies like Amazon and Microsoft to solve real-world decision making problems at scale.
* [DI-engine](https://github.com/opendilab/DI-engine) - DI-engine is a generalized Decision Intelligence engine. It supports most basic deep reinforcement learning (DRL) algorithms, such as DQN, PPO, SAC, and domain-specific algorithms like QMIX in multi-agent RL, GAIL in inverse RL, and RND in exploration problems.
* [Gym4ReaL](https://github.com/Daveonwave/gym4ReaL) - Gym4ReaL is a comprehensive suite of realistic environments designed to support the development and evaluation of RL algorithms that can operate in real-world scenarios. The suite includes a diverse set of tasks exposing RL algorithms to a variety of practical challenges.

<a name="python-speech-recognition"></a>
#### Speech Recognition
* [EspNet](https://github.com/espnet/espnet) - ESPnet is an end-to-end speech processing toolkit for tasks like speech recognition, translation, and enhancement, using PyTorch and Kaldi-style data processing.

<a name="ruby"></a>
## Ruby

<a name="ruby-natural-language-processing"></a>
#### Natural Language Processing

* [Awesome NLP with Ruby](https://github.com/arbox/nlp-with-ruby) - Curated link list for practical natural language processing in Ruby.
* [Treat](https://github.com/louismullie/treat) - Text Retrieval and Annotation Toolkit, definitely the most comprehensive toolkit I’ve encountered so far for Ruby.
* [Stemmer](https://github.com/aurelian/ruby-stemmer) - Expose libstemmer_c to Ruby. **[Deprecated]**
* [Raspell](https://sourceforge.net/projects/raspell/) - raspell is an interface binding for ruby. **[Deprecated]**
* [UEA Stemmer](https://github.com/ealdent/uea-stemmer) - Ruby port of UEALite Stemmer - a conservative stemmer for search and indexing.
* [Twitter-text-rb](https://github.com/twitter/twitter-text/tree/master/rb) - A library that does auto linking and extraction of usernames, lists and hashtags in tweets.

<a name="ruby-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Awesome Machine Learning with Ruby](https://github.com/arbox/machine-learning-with-ruby) - Curated list of ML related resources for Ruby.
* [Ruby Machine Learning](https://github.com/tsycho/ruby-machine-learning) - Some Machine Learning algorithms, implemented in Ruby. **[Deprecated]**
* [Machine Learning Ruby](https://github.com/mizoR/machine-learning-ruby) **[Deprecated]**
* [jRuby Mahout](https://github.com/vasinov/jruby_mahout) - JRuby Mahout is a gem that unleashes the power of Apache Mahout in the world of JRuby. **[Deprecated]**
* [CardMagic-Classifier](https://github.com/cardmagic/classifier) - A general classifier module to allow Bayesian and other types of classifications.
* [rb-libsvm](https://github.com/febeling/rb-libsvm) - Ruby language bindings for LIBSVM which is a Library for Support Vector Machines.
* [Scoruby](https://github.com/asafschers/scoruby) - Creates Random Forest classifiers from PMML files.
* [rumale](https://github.com/yoshoku/rumale) - Rumale is a machine learning library in Ruby

<a name="ruby-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [rsruby](https://github.com/alexgutteridge/rsruby) - Ruby - R bridge.
* [data-visualization-ruby](https://github.com/chrislo/data_visualisation_ruby) - Source code and supporting content for my Ruby Manor presentation on Data Visualisation with Ruby. **[Deprecated]**
* [ruby-plot](https://www.ruby-toolbox.com/projects/ruby-plot) - gnuplot wrapper for Ruby, especially for plotting ROC curves into SVG files. **[Deprecated]**
* [plot-rb](https://github.com/zuhao/plotrb) - A plotting library in Ruby built on top of Vega and D3. **[Deprecated]**
* [scruffy](https://github.com/delano/scruffy) - A beautiful graphing toolkit for Ruby.
* [SciRuby](http://sciruby.com/)
* [Glean](https://github.com/glean/glean) - A data management tool for humans. **[Deprecated]**
* [Bioruby](https://github.com/bioruby/bioruby)
* [Arel](https://github.com/nkallen/arel) **[Deprecated]**

<a name="ruby-misc"></a>
#### Misc

* [Big Data For Chimps](https://github.com/infochimps-labs/big_data_for_chimps)
* [Listof](https://github.com/kevincobain2000/listof) - Community based data collection, packed in gem. Get list of pretty much anything (stop words, countries, non words) in txt, JSON or hash. [Demo/Search for a list](http://kevincobain2000.github.io/listof/)


<a name="rust"></a>
## Rust

<a name="rust-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [smartcore](https://github.com/smartcorelib/smartcore) - "The Most Advanced Machine Learning Library In Rust."
* [linfa](https://github.com/rust-ml/linfa) - a comprehensive toolkit to build Machine Learning applications with Rust
* [deeplearn-rs](https://github.com/tedsta/deeplearn-rs) - deeplearn-rs provides simple networks that use matrix multiplication, addition, and ReLU under the MIT license.
* [rustlearn](https://github.com/maciejkula/rustlearn) - a machine learning framework featuring logistic regression, support vector machines, decision trees and random forests.
* [rusty-machine](https://github.com/AtheMathmo/rusty-machine) - a pure-rust machine learning library.
* [leaf](https://github.com/autumnai/leaf) - open source framework for machine intelligence, sharing concepts from TensorFlow and Caffe. Available under the MIT license. [**[Deprecated]**](https://medium.com/@mjhirn/tensorflow-wins-89b78b29aafb#.s0a3uy4cc)
* [RustNN](https://github.com/jackm321/RustNN) - RustNN is a feedforward neural network library. **[Deprecated]**
* [RusticSOM](https://github.com/avinashshenoy97/RusticSOM) - A Rust library for Self Organising Maps (SOM).
* [candle](https://github.com/huggingface/candle) - Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) and ease of use.
* [linfa](https://github.com/rust-ml/linfa) - `linfa` aims to provide a comprehensive toolkit to build Machine Learning applications with Rust
* [delta](https://github.com/delta-rs/delta) - An open source machine learning framework in Rust Δ

#### Deep Learning

* [tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for the C++ API of PyTorch
* [dfdx](https://github.com/coreylowman/dfdx) - Deep learning in Rust, with shape checked tensors and neural networks
* [burn](https://github.com/tracel-ai/burn) - Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals

#### Natural Language Processing

* [huggingface/tokenizers](https://github.com/huggingface/tokenizers) - Fast State-of-the-Art Tokenizers optimized for Research and Production
* [rust-bert](https://github.com/guillaume-be/rust-bert) - Rust native ready-to-use NLP pipelines and transformer-based models (BERT, DistilBERT, GPT2,...)

<a name="r"></a>
## R

<a name="r-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [ahaz](https://cran.r-project.org/web/packages/ahaz/index.html) - ahaz: Regularization for semiparametric additive hazards regression. **[Deprecated]**
* [arules](https://cran.r-project.org/web/packages/arules/index.html) - arules: Mining Association Rules and Frequent Itemsets
* [biglasso](https://cran.r-project.org/web/packages/biglasso/index.html) - biglasso: Extending Lasso Model Fitting to Big Data in R.
* [bmrm](https://cran.r-project.org/web/packages/bmrm/index.html) - bmrm: Bundle Methods for Regularized Risk Minimization Package.
* [Boruta](https://cran.r-project.org/web/packages/Boruta/index.html) - Boruta: A wrapper algorithm for all-relevant feature selection.
* [bst](https://cran.r-project.org/web/packages/bst/index.html) - bst: Gradient Boosting.
* [C50](https://cran.r-project.org/web/packages/C50/index.html) - C50: C5.0 Decision Trees and Rule-Based Models.
* [caret](https://topepo.github.io/caret/index.html) - Classification and Regression Training: Unified interface to ~150 ML algorithms in R.
* [caretEnsemble](https://cran.r-project.org/web/packages/caretEnsemble/index.html) - caretEnsemble: Framework for fitting multiple caret models as well as creating ensembles of such models. **[Deprecated]**
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box for R.
* [Clever Algorithms For Machine Learning](https://machinelearningmastery.com/)
* [CORElearn](https://cran.r-project.org/web/packages/CORElearn/index.html) - CORElearn: Classification, regression, feature evaluation and ordinal evaluation.
-* [CoxBoost](https://cran.r-project.org/web/packages/CoxBoost/index.html) - CoxBoost: Cox models by likelihood based boosting for a single survival endpoint or competing risks **[Deprecated]**
* [Cubist](https://cran.r-project.org/web/packages/Cubist/index.html) - Cubist: Rule- and Instance-Based Regression Modelling.
* [e1071](https://cran.r-project.org/web/packages/e1071/index.html) - e1071: Misc Functions of the Department of Statistics (e1071), TU Wien
* [earth](https://cran.r-project.org/web/packages/earth/index.html) - earth: Multivariate Adaptive Regression Spline Models
* [elasticnet](https://cran.r-project.org/web/packages/elasticnet/index.html) - elasticnet: Elastic-Net for Sparse Estimation and Sparse PCA.
* [ElemStatLearn](https://cran.r-project.org/web/packages/ElemStatLearn/index.html) - ElemStatLearn: Data sets, functions and examples from the book: "The Elements of Statistical Learning, Data Mining, Inference, and Prediction" by Trevor Hastie, Robert Tibshirani and Jerome Friedman Prediction" by Trevor Hastie, Robert Tibshirani and Jerome Friedman.
* [evtree](https://cran.r-project.org/web/packages/evtree/index.html) - evtree: Evolutionary Learning of Globally Optimal Trees.
* [forecast](https://cran.r-project.org/web/packages/forecast/index.html) - forecast: Timeseries forecasting using ARIMA, ETS, STLM, TBATS, and neural network models.
* [forecastHybrid](https://cran.r-project.org/web/packages/forecastHybrid/index.html) - forecastHybrid: Automatic ensemble and cross validation of ARIMA, ETS, STLM, TBATS, and neural network models from the "forecast" package.
* [fpc](https://cran.r-project.org/web/packages/fpc/index.html) - fpc: Flexible procedures for clustering.
* [frbs](https://cran.r-project.org/web/packages/frbs/index.html) - frbs: Fuzzy Rule-based Systems for Classification and Regression Tasks. **[Deprecated]**
* [GAMBoost](https://cran.r-project.org/web/packages/GAMBoost/index.html) - GAMBoost: Generalized linear and additive models by likelihood based boosting. **[Deprecated]**
* [gamboostLSS](https://cran.r-project.org/web/packages/gamboostLSS/index.html) - gamboostLSS: Boosting Methods for GAMLSS.
* [gbm](https://cran.r-project.org/web/packages/gbm/index.html) - gbm: Generalized Boosted Regression Models.
* [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html) - glmnet: Lasso and elastic-net regularized generalized linear models.
* [glmpath](https://cran.r-project.org/web/packages/glmpath/index.html) - glmpath: L1 Regularization Path for Generalized Linear Models and Cox Proportional Hazards Model.
* [GMMBoost](https://cran.r-project.org/web/packages/GMMBoost/index.html) - GMMBoost: Likelihood-based Boosting for Generalized mixed models. **[Deprecated]**
* [grplasso](https://cran.r-project.org/web/packages/grplasso/index.html) - grplasso: Fitting user specified models with Group Lasso penalty.
* [grpreg](https://cran.r-project.org/web/packages/grpreg/index.html) - grpreg: Regularization paths for regression models with grouped covariates.
* [h2o](https://cran.r-project.org/web/packages/h2o/index.html) - A framework for fast, parallel, and distributed machine learning algorithms at scale -- Deeplearning, Random forests, GBM, KMeans, PCA, GLM.
* [hda](https://cran.r-project.org/web/packages/hda/index.html) - hda: Heteroscedastic Discriminant Analysis. **[Deprecated]**
* [Introduction to Statistical Learning](https://www-bcf.usc.edu/~gareth/ISL/)
* [ipred](https://cran.r-project.org/web/packages/ipred/index.html) - ipred: Improved Predictors.
* [kernlab](https://cran.r-project.org/web/packages/kernlab/index.html) - kernlab: Kernel-based Machine Learning Lab.
* [klaR](https://cran.r-project.org/web/packages/klaR/index.html) - klaR: Classification and visualization.
* [L0Learn](https://cran.r-project.org/web/packages/L0Learn/index.html) - L0Learn: Fast algorithms for best subset selection.
* [lars](https://cran.r-project.org/web/packages/lars/index.html) - lars: Least Angle Regression, Lasso and Forward Stagewise. **[Deprecated]**
* [lasso2](https://cran.r-project.org/web/packages/lasso2/index.html) - lasso2: L1 constrained estimation aka ‘lasso’.
* [LiblineaR](https://cran.r-project.org/web/packages/LiblineaR/index.html) - LiblineaR: Linear Predictive Models Based On The Liblinear C/C++ Library.
* [LogicReg](https://cran.r-project.org/web/packages/LogicReg/index.html) - LogicReg: Logic Regression.
* [Machine Learning For Hackers](https://github.com/johnmyleswhite/ML_for_Hackers)
* [maptree](https://cran.r-project.org/web/packages/maptree/index.html) - maptree: Mapping, pruning, and graphing tree models. **[Deprecated]**
* [mboost](https://cran.r-project.org/web/packages/mboost/index.html) - mboost: Model-Based Boosting.
* [medley](https://www.kaggle.com/general/3661) - medley: Blending regression models, using a greedy stepwise approach.
* [mlr](https://cran.r-project.org/web/packages/mlr/index.html) - mlr: Machine Learning in R.
* [ncvreg](https://cran.r-project.org/web/packages/ncvreg/index.html) - ncvreg: Regularization paths for SCAD- and MCP-penalized regression models.
* [nnet](https://cran.r-project.org/web/packages/nnet/index.html) - nnet: Feed-forward Neural Networks and Multinomial Log-Linear Models. **[Deprecated]**
* [pamr](https://cran.r-project.org/web/packages/pamr/index.html) - pamr: Pam: prediction analysis for microarrays. **[Deprecated]**
* [party](https://cran.r-project.org/web/packages/party/index.html) - party: A Laboratory for Recursive Partitioning
* [partykit](https://cran.r-project.org/web/packages/partykit/index.html) - partykit: A Toolkit for Recursive Partitioning.
* [penalized](https://cran.r-project.org/web/packages/penalized/index.html) - penalized: L1 (lasso and fused lasso) and L2 (ridge) penalized estimation in GLMs and in the Cox model.
* [penalizedLDA](https://cran.r-project.org/web/packages/penalizedLDA/index.html) - penalizedLDA: Penalized classification using Fisher's linear discriminant. **[Deprecated]**
* [penalizedSVM](https://cran.r-project.org/web/packages/penalizedSVM/index.html) - penalizedSVM: Feature Selection SVM using penalty functions.
* [quantregForest](https://cran.r-project.org/web/packages/quantregForest/index.html) - quantregForest: Quantile Regression Forests.
* [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html) - randomForest: Breiman and Cutler's random forests for classification and regression.
* [randomForestSRC](https://cran.r-project.org/web/packages/randomForestSRC/index.html) - randomForestSRC: Random Forests for Survival, Regression and Classification (RF-SRC).
* [rattle](https://cran.r-project.org/web/packages/rattle/index.html) - rattle: Graphical user interface for data mining in R.
* [rda](https://cran.r-project.org/web/packages/rda/index.html) - rda: Shrunken Centroids Regularized Discriminant Analysis.
* [rdetools](https://cran.r-project.org/web/packages/rdetools/index.html) - rdetools: Relevant Dimension Estimation (RDE) in Feature Spaces. **[Deprecated]**
* [REEMtree](https://cran.r-project.org/web/packages/REEMtree/index.html) - REEMtree: Regression Trees with Random Effects for Longitudinal (Panel) Data. **[Deprecated]**
* [relaxo](https://cran.r-project.org/web/packages/relaxo/index.html) - relaxo: Relaxed Lasso. **[Deprecated]**
* [rgenoud](https://cran.r-project.org/web/packages/rgenoud/index.html) - rgenoud: R version of GENetic Optimization Using Derivatives
* [Rmalschains](https://cran.r-project.org/web/packages/Rmalschains/index.html) - Rmalschains: Continuous Optimization using Memetic Algorithms with Local Search Chains (MA-LS-Chains) in R.
* [rminer](https://cran.r-project.org/web/packages/rminer/index.html) - rminer: Simpler use of data mining methods (e.g. NN and SVM) in classification and regression. **[Deprecated]**
* [ROCR](https://cran.r-project.org/web/packages/ROCR/index.html) - ROCR: Visualizing the performance of scoring classifiers. **[Deprecated]**
* [RoughSets](https://cran.r-project.org/web/packages/RoughSets/index.html) - RoughSets: Data Analysis Using Rough Set and Fuzzy Rough Set Theories. **[Deprecated]**
* [rpart](https://cran.r-project.org/web/packages/rpart/index.html) - rpart: Recursive Partitioning and Regression Trees.
* [RPMM](https://cran.r-project.org/web/packages/RPMM/index.html) - RPMM: Recursively Partitioned Mixture Model.
* [RSNNS](https://cran.r-project.org/web/packages/RSNNS/index.html) - RSNNS: Neural Networks in R using the Stuttgart Neural Network Simulator (SNNS).
* [RWeka](https://cran.r-project.org/web/packages/RWeka/index.html) - RWeka: R/Weka interface.
* [RXshrink](https://cran.r-project.org/web/packages/RXshrink/index.html) - RXshrink: Maximum Likelihood Shrinkage via Generalized Ridge or Least Angle Regression.
* [sda](https://cran.r-project.org/web/packages/sda/index.html) - sda: Shrinkage Discriminant Analysis and CAT Score Variable Selection. **[Deprecated]**
* [spectralGraphTopology](https://cran.r-project.org/web/packages/spectralGraphTopology/index.html) - spectralGraphTopology: Learning Graphs from Data via Spectral Constraints.
* [SuperLearner](https://github.com/ecpolley/SuperLearner) - Multi-algorithm ensemble learning packages.
* [svmpath](https://cran.r-project.org/web/packages/svmpath/index.html) - svmpath: svmpath: the SVM Path algorithm. **[Deprecated]**
* [tgp](https://cran.r-project.org/web/packages/tgp/index.html) - tgp: Bayesian treed Gaussian process models. **[Deprecated]**
* [tree](https://cran.r-project.org/web/packages/tree/index.html) - tree: Classification and regression trees.
* [varSelRF](https://cran.r-project.org/web/packages/varSelRF/index.html) - varSelRF: Variable selection using random forests.
* [XGBoost.R](https://github.com/tqchen/xgboost/tree/master/R-package) - R binding for eXtreme Gradient Boosting (Tree) Library.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search. Optunity is written in Python but interfaces seamlessly to R.
* [igraph](https://igraph.org/r/) - binding to igraph library - General purpose graph library.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [TDSP-Utilities](https://github.com/Azure/Azure-TDSP-Utilities) - Two data science utilities in R from Microsoft: 1) Interactive Data Exploration, Analysis, and Reporting (IDEAR) ; 2) Automated Modelling and Reporting (AMR).
* [clugenr](https://github.com/clugen/clugenr/) - Multidimensional cluster generation in R.

<a name="r-data-analysis--data-visualization"></a>
#### Data Manipulation | Data Analysis | Data Visualization

* [data.table](https://rdatatable.gitlab.io/data.table/) - `data.table` provides a high-performance version of base R’s `data.frame` with syntax and feature enhancements for ease of use, convenience and programming speed.
* [dplyr](https://www.rdocumentation.org/packages/dplyr/versions/0.7.8) - A data manipulation package that helps to solve the most common data manipulation problems.
* [ggplot2](https://ggplot2.tidyverse.org/) - A data visualization package based on the grammar of graphics.
* [tmap](https://cran.r-project.org/web/packages/tmap/vignettes/tmap-getstarted.html) for visualizing geospatial data with static maps and [leaflet](https://rstudio.github.io/leaflet/) for interactive maps
* [tm](https://www.rdocumentation.org/packages/tm/) and [quanteda](https://quanteda.io/) are the main packages for managing,  analyzing, and visualizing textual data.
* [shiny](https://shiny.rstudio.com/) is the basis for truly interactive displays and dashboards in R. However, some measure of interactivity can be achieved with [htmlwidgets](https://www.htmlwidgets.org/) bringing javascript libraries to R. These include, [plotly](https://plot.ly/r/), [dygraphs](http://rstudio.github.io/dygraphs), [highcharter](http://jkunst.com/highcharter/), and several others.

<a name="sas"></a>
## SAS

<a name="sas-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Visual Data Mining and Machine Learning](https://www.sas.com/en_us/software/visual-data-mining-machine-learning.html) - Interactive, automated, and programmatic modelling with the latest machine learning algorithms in and end-to-end analytics environment, from data prep to deployment. Free trial available.
* [Enterprise Miner](https://www.sas.com/en_us/software/enterprise-miner.html) - Data mining and machine learning that creates deployable models using a GUI or code.
* [Factory Miner](https://www.sas.com/en_us/software/factory-miner.html) - Automatically creates deployable machine learning models across numerous market or customer segments using a GUI.

<a name="sas-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [SAS/STAT](https://www.sas.com/en_us/software/stat.html) - For conducting advanced statistical analysis.
* [University Edition](https://www.sas.com/en_us/software/university-edition.html) - FREE! Includes all SAS packages necessary for data analysis and visualization, and includes online SAS courses.

<a name="sas-natural-language-processing"></a>
#### Natural Language Processing

* [Contextual Analysis](https://www.sas.com/en_us/software/contextual-analysis.html) - Add structure to unstructured text using a GUI.
* [Sentiment Analysis](https://www.sas.com/en_us/software/sentiment-analysis.html) - Extract sentiment from text using a GUI.
* [Text Miner](https://www.sas.com/en_us/software/text-miner.html) - Text mining using a GUI or code.

<a name="sas-demos-and-scripts"></a>
#### Demos and Scripts

* [ML_Tables](https://github.com/sassoftware/enlighten-apply/tree/master/ML_tables) - Concise cheat sheets containing machine learning best practices.
* [enlighten-apply](https://github.com/sassoftware/enlighten-apply) - Example code and materials that illustrate applications of SAS machine learning techniques.
* [enlighten-integration](https://github.com/sassoftware/enlighten-integration) - Example code and materials that illustrate techniques for integrating SAS with other analytics technologies in Java, PMML, Python and R.
* [enlighten-deep](https://github.com/sassoftware/enlighten-deep) - Example code and materials that illustrate using neural networks with several hidden layers in SAS.
* [dm-flow](https://github.com/sassoftware/dm-flow) - Library of SAS Enterprise Miner process flow diagrams to help you learn by example about specific data mining topics.


<a name="scala"></a>
## Scala

<a name="scala-natural-language-processing"></a>
#### Natural Language Processing

* [ScalaNLP](http://www.scalanlp.org/) - ScalaNLP is a suite of machine learning and numerical computing libraries.
* [Breeze](https://github.com/scalanlp/breeze) - Breeze is a numerical processing library for Scala.
* [Chalk](https://github.com/scalanlp/chalk) - Chalk is a natural language processing library. **[Deprecated]**
* [FACTORIE](https://github.com/factorie/factorie) - FACTORIE is a toolkit for deployable probabilistic modelling, implemented as a software library in Scala. It provides its users with a succinct language for creating relational factor graphs, estimating parameters and performing inference.
* [Montague](https://github.com/Workday/upshot-montague) - Montague is a semantic parsing library for Scala with an easy-to-use DSL.
* [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) - Natural language processing library built on top of Apache Spark ML to provide simple, performant, and accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.

<a name="scala-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [NDScala](https://github.com/SciScala/NDScala) - N-dimensional arrays in Scala 3. Think NumPy ndarray, but with compile-time type-checking/inference over shapes, tensor/axis labels & numeric data types
* [MLlib in Apache Spark](https://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [Scalding](https://github.com/twitter/scalding) - A Scala API for Cascading.
* [Summing Bird](https://github.com/twitter/summingbird) - Streaming MapReduce with Scalding and Storm.
* [Algebird](https://github.com/twitter/algebird) - Abstract Algebra for Scala.
* [xerial](https://github.com/xerial/xerial) - Data management utilities for Scala. **[Deprecated]**
* [PredictionIO](https://github.com/apache/predictionio) - PredictionIO, a machine learning server for software developers and data engineers.
* [BIDMat](https://github.com/BIDData/BIDMat) - CPU and GPU-accelerated matrix library intended to support large-scale exploratory data analysis.
* [Flink](https://flink.apache.org/) - Open source platform for distributed stream and batch data processing.
* [Spark Notebook](http://spark-notebook.io) - Interactive and Reactive Data Science using Scala and Spark.

<a name="scala-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Microsoft ML for Apache Spark](https://github.com/Azure/mmlspark) -> A distributed machine learning framework Apache Spark
* [ONNX-Scala](https://github.com/EmergentOrder/onnx-scala) - An ONNX (Open Neural Network eXchange) API and backend for typeful, functional deep learning in Scala (3).
* [DeepLearning.scala](https://deeplearning.thoughtworks.school/) - Creating statically typed dynamic neural networks from object-oriented & functional programming constructs.
* [Conjecture](https://github.com/etsy/Conjecture) - Scalable Machine Learning in Scalding.
* [brushfire](https://github.com/stripe/brushfire) - Distributed decision tree ensemble learning in Scala.
* [ganitha](https://github.com/tresata/ganitha) - Scalding powered machine learning. **[Deprecated]**
* [adam](https://github.com/bigdatagenomics/adam) - A genomics processing engine and specialized file format built using Apache Avro, Apache Spark and Parquet. Apache 2 licensed.
* [bioscala](https://github.com/bioscala/bioscala) - Bioinformatics for the Scala programming language
* [BIDMach](https://github.com/BIDData/BIDMach) - CPU and GPU-accelerated Machine Learning Library.
* [Figaro](https://github.com/p2t2/figaro) - a Scala library for constructing probabilistic models.
* [H2O Sparkling Water](https://github.com/h2oai/sparkling-water) - H2O and Spark interoperability.
* [FlinkML in Apache Flink](https://ci.apache.org/projects/flink/flink-docs-master/dev/libs/ml/index.html) - Distributed machine learning library in Flink.
* [DynaML](https://github.com/transcendent-ai-labs/DynaML) - Scala Library/REPL for Machine Learning Research.
* [Saul](https://github.com/CogComp/saul) - Flexible Declarative Learning-Based Programming.
* [SwiftLearner](https://github.com/valdanylchuk/swiftlearner/) - Simply written algorithms to help study ML or write your own implementations.
* [Smile](https://haifengl.github.io/) - Statistical Machine Intelligence and Learning Engine.
* [doddle-model](https://github.com/picnicml/doddle-model) - An in-memory machine learning library built on top of Breeze. It provides immutable objects and exposes its functionality through a scikit-learn-like API.
* [TensorFlow Scala](https://github.com/eaplatanios/tensorflow_scala) - Strongly-typed Scala API for TensorFlow.
* [isolation-forest](https://github.com/linkedin/isolation-forest) - A distributed Spark/Scala implementation of the isolation forest algorithm for unsupervised outlier detection, featuring support for scalable training and ONNX export for easy cross-platform inference.

<a name="scheme"></a>
## Scheme

<a name="scheme-neural-networks"></a>
#### Neural Networks

* [layer](https://github.com/cloudkj/layer) - Neural network inference from the command line, implemented in [CHICKEN Scheme](https://www.call-cc.org/).

<a name="swift"></a>
## Swift

<a name="swift-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Bender](https://github.com/xmartlabs/Bender) - Fast Neural Networks framework built on top of Metal. Supports TensorFlow models.
* [Swift AI](https://github.com/Swift-AI/Swift-AI) - Highly optimized artificial intelligence and machine learning library written in Swift.
* [Swift for Tensorflow](https://github.com/tensorflow/swift) - a next-generation platform for machine learning, incorporating the latest research across machine learning, compilers, differentiable programming, systems design, and beyond.
* [BrainCore](https://github.com/alejandro-isaza/BrainCore) - The iOS and OS X neural network framework.
* [swix](https://github.com/stsievert/swix) - A bare bones library that includes a general matrix language and wraps some OpenCV for iOS development. **[Deprecated]**
* [AIToolbox](https://github.com/KevinCoble/AIToolbox) - A toolbox framework of AI modules written in Swift: Graphs/Trees, Linear Regression, Support Vector Machines, Neural Networks, PCA, KMeans, Genetic Algorithms, MDP, Mixture of Gaussians.
* [MLKit](https://github.com/Somnibyte/MLKit) - A simple Machine Learning Framework written in Swift. Currently features Simple Linear Regression, Polynomial Regression, and Ridge Regression.
* [Swift Brain](https://github.com/vlall/Swift-Brain) - The first neural network / machine learning library written in Swift. This is a project for AI algorithms in Swift for iOS and OS X development. This project includes algorithms focused on Bayes theorem, neural networks, SVMs, Matrices, etc...
* [Perfect TensorFlow](https://github.com/PerfectlySoft/Perfect-TensorFlow) - Swift Language Bindings of TensorFlow. Using native TensorFlow models on both macOS / Linux.
* [PredictionBuilder](https://github.com/denissimon/prediction-builder-swift) - A library for machine learning that builds predictions using a linear regression.
* [Awesome CoreML](https://github.com/SwiftBrain/awesome-CoreML-models) - A curated list of pretrained CoreML models.
* [Awesome Core ML Models](https://github.com/likedan/Awesome-CoreML-Models) - A curated list of machine learning models in CoreML format.

<a name="tensorflow"></a>
## TensorFlow

<a name="tensorflow-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [Awesome Keras](https://github.com/markusschanta/awesome-keras) - A curated list of awesome Keras projects, libraries and resources.
* [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow) - A list of all things related to TensorFlow.
* [Golden TensorFlow](https://golden.com/wiki/TensorFlow) - A page of content on TensorFlow, including academic papers and links to related topics.

<a name="tools"></a>
## Tools

<a name="tools-neural-networks"></a>
#### Neural Networks
* [layer](https://github.com/cloudkj/layer) - Neural network inference from the command line

<a name="tools-misc"></a>
#### Misc

* [Wallaroo.AI](https://wallaroo.ai/) - Production AI plaftorm for deploying, managing, and observing any model at scale across any environment from cloud to edge. Let's go from python notebook to inferencing in minutes. 
* [Infinity](https://github.com/infiniflow/infinity) - The AI-native database built for LLM applications, providing incredibly fast vector and full-text search. Developed using C++20
* [Synthical](https://synthical.com) - AI-powered collaborative research environment. You can use it to get recommendations of articles based on reading history, simplify papers, find out what articles are trending, search articles by meaning (not just keywords), create and share folders of articles, see lists of articles from specific companies and universities, and add highlights.
* [Humanloop](https://humanloop.com) – Humanloop is a platform for prompt experimentation, finetuning models for better performance, cost optimization, and collecting model generated data and user feedback.
* [Qdrant](https://qdrant.tech) – Qdrant is [open source](https://github.com/qdrant/qdrant) vector similarity search engine with extended filtering support, written in Rust.
* [Localforge](https://localforge.dev/) – Is an [open source](https://github.com/rockbite/localforge) on-prem AI coding autonomous assistant that lives inside your repo, edits and tests files at SSD speed. Think Claude Code but with UI. plug in any LLM (OpenAI, Gemini, Ollama, etc.) and let it work for you.
* [milvus](https://milvus.io) – Milvus is [open source](https://github.com/milvus-io/milvus) vector database for production AI, written in Go and C++, scalable and blazing fast for billions of embedding vectors.
* [Weaviate](https://www.semi.technology/developers/weaviate/current/) – Weaviate is an [open source](https://github.com/semi-technologies/weaviate) vector search engine and vector database. Weaviate uses machine learning to vectorize and store data, and to find answers to natural language queries. With Weaviate you can also bring your custom ML models to production scale.
* [txtai](https://github.com/neuml/txtai) - Build semantic search applications and workflows.
* [MLReef](https://about.mlreef.com/) - MLReef is an end-to-end development platform using the power of git to give structure and deep collaboration possibilities to the ML development process.
* [Chroma](https://www.trychroma.com/) - Open-source search and retrieval database for AI applications. Vector, full-text, regex, and metadata search. [Self-host](https://docs.trychroma.com) or [Cloud](https://trychroma.com/signup) available.
* [Pinecone](https://www.pinecone.io/) - Vector database for applications that require real-time, scalable vector embedding and similarity search.
* [CatalyzeX](https://chrome.google.com/webstore/detail/code-finder-for-research/aikkeehnlfpamidigaffhfmgbkdeheil) - Browser extension ([Chrome](https://chrome.google.com/webstore/detail/code-finder-for-research/aikkeehnlfpamidigaffhfmgbkdeheil) and [Firefox](https://addons.mozilla.org/en-US/firefox/addon/code-finder-catalyzex/)) that automatically finds and shows code implementations for machine learning papers anywhere: Google, Twitter, Arxiv, Scholar, etc.
* [ML Workspace](https://github.com/ml-tooling/ml-workspace) - All-in-one web-based IDE for machine learning and data science. The workspace is deployed as a docker container and is preloaded with a variety of popular data science libraries (e.g., Tensorflow, PyTorch) and dev tools (e.g., Jupyter, VS Code).
* [Notebooks](https://github.com/rlan/notebooks) - A starter kit for Jupyter notebooks and machine learning. Companion docker images consist of all combinations of python versions, machine learning frameworks (Keras, PyTorch and Tensorflow) and CPU/CUDA versions.
* [DVC](https://github.com/iterative/dvc) - Data Science Version Control is an open-source version control system for machine learning projects with pipelines support. It makes ML projects reproducible and shareable.
* [DVClive](https://github.com/iterative/dvclive) - Python library for experiment metrics logging into simply formatted local files.
* [VDP](https://github.com/instill-ai/vdp) - open source visual data ETL to streamline the end-to-end visual data processing pipeline: extract unstructured visual data from pre-built data sources, transform it into analysable structured insights by Vision AI models imported from various ML platforms, and load the insights into warehouses or applications.
* [Kedro](https://github.com/quantumblacklabs/kedro/) - Kedro is a data and development workflow framework that implements best practices for data pipelines with an eye towards productionizing machine learning models.
* [Hamilton](https://github.com/dagworks-inc/hamilton) - a lightweight library to define data transformations as a directed-acyclic graph (DAG). It helps author reliable feature engineering and machine learning pipelines, and more.
* [guild.ai](https://guild.ai/) - Tool to log, analyze, compare and "optimize" experiments. It's cross-platform and framework independent, and provided integrated visualizers such as tensorboard.
* [Sacred](https://github.com/IDSIA/sacred) - Python tool to help  you configure, organize, log and reproduce experiments. Like a notebook lab in the context of Chemistry/Biology. The community has built multiple add-ons leveraging the proposed standard.
* [Comet](https://www.comet.com/) -  ML platform for tracking experiments, hyper-parameters, artifacts and more. It's deeply integrated with over 15+ deep learning frameworks and orchestration tools. Users can also use the platform to monitor their models in production.
* [MLFlow](https://mlflow.org/) - platform to manage the ML lifecycle, including experimentation, reproducibility and deployment. Framework and language agnostic, take a look at all the built-in integrations.
* [Weights & Biases](https://www.wandb.com/) - Machine learning experiment tracking, dataset versioning, hyperparameter search, visualization, and collaboration
* More tools to improve the ML lifecycle: [Catalyst](https://github.com/catalyst-team/catalyst), [PachydermIO](https://www.pachyderm.io/). The following are GitHub-alike and targeting teams [Weights & Biases](https://www.wandb.com/), [Neptune.ai](https://neptune.ai/), [Comet.ml](https://www.comet.ml/), [Valohai.ai](https://valohai.com/), [DAGsHub](https://DAGsHub.com/).
* [Arize AI](https://www.arize.com) - Model validation and performance monitoring, drift detection, explainability, visualization across structured and unstructured data
* [MachineLearningWithTensorFlow2ed](https://www.manning.com/books/machine-learning-with-tensorflow-second-edition) - a book on general purpose machine learning techniques regression, classification, unsupervised clustering, reinforcement learning, auto encoders, convolutional neural networks, RNNs, LSTMs, using TensorFlow 1.14.1.
* [m2cgen](https://github.com/BayesWitnesses/m2cgen) - A tool that allows the conversion of ML models into native code (Java, C, Python, Go, JavaScript, Visual Basic, C#, R, PowerShell, PHP, Dart) with zero dependencies.
* [CML](https://github.com/iterative/cml) - A library for doing continuous integration with ML projects. Use GitHub Actions & GitLab CI to train and evaluate models in production like environments and automatically generate visual reports with metrics and graphs in pull/merge requests. Framework & language agnostic.
* [Pythonizr](https://pythonizr.com) - An online tool to generate boilerplate machine learning code that uses scikit-learn.
* [Flyte](https://flyte.org/) - Flyte makes it easy to create concurrent, scalable, and maintainable workflows for machine learning and data processing.
* [Chaos Genius](https://github.com/chaos-genius/chaos_genius/) - ML powered analytics engine for outlier/anomaly detection and root cause analysis.
* [MLEM](https://github.com/iterative/mlem) - Version and deploy your ML models following GitOps principles
* [DockerDL](https://github.com/matifali/dockerdl) - Ready to use deeplearning docker images.
* [Aqueduct](https://github.com/aqueducthq/aqueduct) - Aqueduct enables you to easily define, run, and manage AI & ML tasks on any cloud infrastructure.
* [Ambrosia](https://github.com/reactorsh/ambrosia) - Ambrosia helps you clean up your LLM datasets using _other_ LLMs.
* [Fiddler AI](https://www.fiddler.ai) - The all-in-one AI Observability and Security platform for responsible AI. It provides monitoring, analytics, and centralized controls to operationalize ML, GenAI, and LLM applications with trust. Fiddler helps enterprises scale LLM and ML deployments to deliver high performance AI, reduce costs, and be responsible in governance.
* [Maxim AI](https://getmaxim.ai) - The agent simulation, evaluation, and observability platform helping product teams ship their AI applications with the quality and speed needed for real-world use.
* [Agentic Radar](https://github.com/splx-ai/agentic-radar) -  Open-source CLI security scanner for agentic workflows. Scans your workflow’s source code, detects vulnerabilities, and generates an interactive visualization along with a detailed security report. Supports LangGraph, CrewAI, n8n, OpenAI Agents, and more.
* [Agentic Signal](https://github.com/code-forge-temple/agentic-signal) - Visual AI agent workflow automation platform with local LLM integration. Build intelligent workflows using drag-and-drop, no cloud required.
* [ScribePal](https://github.com/code-forge-temple/scribe-pal) - Chrome extension that uses local LLMs to assist with writing and drafting responses based on the context of your open tabs.
* [Local LLM NPC](https://github.com/code-forge-temple/local-llm-npc) - Godot 4.x asset that enables NPCs to interact with players using local LLMs for structured, offline-first learning conversations in games.
* [Awesome Hugging Face Models](https://github.com/JehoshuaM/awesome-huggingface-models) - Curated list of top Hugging Face models for NLP, vision, and audio tasks with demos and benchmarks.

<a name="books"></a>
## Books

* [Distributed Machine Learning Patterns](https://github.com/terrytangyuan/distributed-ml-patterns)  - This book teaches you how to take machine learning models from your personal laptop to large distributed clusters. You’ll explore key concepts and patterns behind successful distributed machine learning systems, and learn technologies like TensorFlow, Kubernetes, Kubeflow, and Argo Workflows directly from a key maintainer and contributor, with real-world scenarios and hands-on projects.
* [Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning) - Grokking Machine Learning teaches you how to apply ML to your projects using only standard Python code and high school-level math.
* [Machine Learning Bookcamp](https://www.manning.com/books/machine-learning-bookcamp) - Learn the essentials of machine learning by completing a carefully designed set of real-world projects.
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975) - Through a recent series of breakthroughs, deep learning has boosted the entire field of machine learning. Now, even programmers who know close to nothing about this technology can use simple, efficient tools to implement programs capable of learning from data. This bestselling book uses concrete examples, minimal theory, and production-ready Python frameworks (Scikit-Learn, Keras, and TensorFlow) to help you gain an intuitive understanding of the concepts and tools for building intelligent systems.
* [Machine Learning Books for Beginners](https://www.appliedaicourse.com/blog/machine-learning-books/) - This blog provides a curated list of introductory books to help aspiring ML professionals to grasp foundational machine learning concepts and techniques.


<a name="credits"></a>
* [Netron](https://netron.app/) - An opensource viewer for neural network, deep learning and machine learning models
* [Teachable Machine](https://teachablemachine.withgoogle.com/) - Train Machine Learning models on the fly to recognize your own images, sounds, & poses.
* [Pollinations.AI](https://pollinations.ai) - Free, no-signup APIs for text, image, and audio generation with no API keys required. Offers OpenAI-compatible interfaces and React hooks for easy integration.
* [Model Zoo](https://modelzoo.co/) - Discover open source deep learning code and pretrained models.

## Credits

* Some of the python libraries were cut-and-pasted from [vinta](https://github.com/vinta/awesome-python)
* References for Go were mostly cut-and-pasted from [gopherdata](https://github.com/gopherdata/resources/tree/master/tooling)
