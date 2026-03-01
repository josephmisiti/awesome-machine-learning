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
  - [CUDA PTX](#cuda-ptx)
      - [Neurosymbolic AI](#cuda-ptx-neurosymbolic-ai)
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

* * [Agentic Context Engine](https://github.com/kayba-ai/agentic-context-engine) -In-context learning framework that allows agents to learn from execution feedback.
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
* [nndeploy](https://github.com/nndeploy/nndeploy) - An Easy-to-Use and High-Performance AI deployment framework.

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