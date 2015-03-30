A curated list of awesome machine learning frameworks, libraries and software (by language). Inspired by awesome-php.
Other awesome lists can be found in the [awesome-awesomeness](https://github.com/bayandin/awesome-awesomeness) list.

If you want to contribute to this list (please do), send me a pull request or contact me [@josephmisiti](https://www.twitter.com/josephmisiti)

For a list of free machine learning books available for download, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md).

## Table of Contents

<!-- MarkdownTOC depth=4 -->

- [C](#c)
    - [General-Purpose Machine Learning](#c-general-purpose)
    - [Computer Vision](#c-cv)
- [C++](#cpp)
    - [Computer Vision](#cpp-cv)
    - [General-Purpose Machine Learning](#cpp-general-purpose)
    - [Natural Language Processing](#cpp-nlp)
    - [Sequence Analysis](#cpp-sequence)
- [Common Lisp](#common-lisp)
    - [General-Purpose Machine Learning](#common-lisp-general-purpose)
- [Clojure](#clojure)
    - [Natural Language Processing](#clojure-nlp)
    - [General-Purpose Machine Learning](#clojure-general-purpose)
    - [Data Analysis / Data Visualization](#clojure-data-analysis)
- [Erlang](#erlang)
    - [General-Purpose Machine Learning](#erlang-general-purpose)
- [Go](#go)
    - [Natural Language Processing](#go-nlp)
    - [General-Purpose Machine Learning](#go-general-purpose)
    - [Data Analysis / Data Visualization](#go-data-analysis)
- [Haskell](#haskell)
    - [General-Purpose Machine Learning](#haskell-general-purpose)
- [Java](#java)
    - [Natural Language Processing](#java-nlp)
    - [General-Purpose Machine Learning](#java-general-purpose)
    - [Data Analysis / Data Visualization](#java-data-analysis)
    - [Deep Learning](#java-deep-learning)
- [Javascript](#javascript)
    - [Natural Language Processing](#javascript-nlp)
    - [Data Analysis / Data Visualization](#javascript-data-analysis)
    - [General-Purpose Machine Learning](#javascript-general-purpose)
- [Julia](#julia)
    - [General-Purpose Machine Learning](#julia-general-purpose)
    - [Natural Language Processing](#julia-nlp)
    - [Data Analysis / Data Visualization](#julia-data-analysis)
    - [Misc Stuff / Presentations](#julia-misc)
- [Lua](#lua)
    - [General-Purpose Machine Learning](#lua-general-purpose)
    - [Demos and Scripts](#lua-demos)
- [Matlab](#matlab)
    - [Computer Vision](#matlab-cv)
    - [Natural Language Processing](#matlab-nlp)
    - [General-Purpose Machine Learning](#matlab-general-purpose)
    - [Data Analysis / Data Visualization](#matlab-data-analysis)
- [.NET](#net)
    - [Computer Vision](#net-cv)
    - [Natural Language Processing](#net-nlp)
    - [General-Purpose Machine Learning](#net-general-purpose)
    - [Data Analysis / Data Visualization](#net-data-analysis)
- [Objective C](#objectivec)
    - [General-Purpose Machine Learning](#objectivec-general-purpose)
- [Python](#python)
    - [Computer Vision](#python-P)
    - [Natural Language Processing](#python-nlp)
    - [General-Purpose Machine Learning](#python-general-purpose)
    - [Data Analysis / Data Visualization](#python-data-analysis)
    - [Misc Scripts / iPython Notebooks / Codebases](#python-misc)
    - [Kaggle Competition Source Code](#python-kaggle)
- [Ruby](#ruby)
    - [Natural Language Processing](#ruby-nlp)
    - [General-Purpose Machine Learning](#ruby-general-purpose)
    - [Data Analysis / Data Visualization](#ruby-data-analysis)
    - [Misc](#ruby-misc)
- [R](#r)
    - [General-Purpose Machine Learning](#r-general-purpose)
    - [Data Analysis / Data Visualization](#r-data-analysis)
- [Scala](#scala)
    - [Natural Language Processing](#scala-nlp)
    - [Data Analysis / Data Visualization](#scala-data-analysis)
    - [General-Purpose Machine Learning](#scala-general-purpose)
- [Swift](#swift)
    - [General-Purpose Machine Learning](#swift-general-purpose)
- [Credits](#credits)

<!-- /MarkdownTOC -->

<a name="c" />
## C

<a name="c-general-purpose" />
#### General-Purpose Machine Learning
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).


<a name="c-cv" />
#### Computer Vision

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has Matlab toolbox

<a name="cpp" />
## C++

<a name="cpp-cv" />
#### Computer Vision

* [OpenCV](http://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models

<a name="cpp-general-purpose" />
#### General-Purpose Machine Learning

* [MLPack](http://www.mlpack.org/)
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications
* [encog-cpp](https://code.google.com/p/encog-cpp/)
* [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html)
* [Vowpal Wabbit (VW)](https://github.com/JohnLangford/vowpal_wabbit/wiki) - A fast out-of-core learning system.
* [sofia-ml](https://code.google.com/p/sofia-ml/) - Suite of fast incremental algorithms.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox
* [Caffe](http://caffe.berkeleyvision.org)  - A deep learning framework developed with cleanliness, readability, and speed in mind. [DEEP LEARNING]
* [CXXNET](https://github.com/antinucleon/cxxnet) - Yet another deep learning framework with less than 1000 lines core code [DEEP LEARNING]
* [XGBoost](https://github.com/tqchen/xgboost) - A parallelized optimized general purpose gradient boosting library.
* [CUDA](https://code.google.com/p/cuda-convnet/) - This is a fast C++/CUDA implementation of convolutional [DEEP LEARNING]
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling
* [BanditLib](https://github.com/jkomiyama/banditlib) - A simple Multi-armed Bandit library.
* [Timbl](http://ilk.uvt.nl/timbl) - A software package/C++ library implementing several memory-based learning algorithms, among which IB1-IG, an implementation of k-nearest neighbor classification, and IGTree, a decision-tree approximation of IB1-IG. Commonly used for NLP.

<a name="cpp-nlp" />
#### Natural Language Processing
* [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
* [CRF++](http://crfpp.googlecode.com/svn/trunk/doc/index.html) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
* [BLLIP Parser](http://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
* [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [ucto](https://github.com/proycon/ucto) - Unicode-aware regular-expression based tokeniser for various languages. Tool and C++ library. Supports FoLiA format.
* [libfolia](https://github.com/proycon/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia)
* [frog](https://github.com/proycon/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyser.

#### Speech Recognition
* [Kaldi](http://kaldi.sourceforge.net/) - Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.

<a name="cpp-sequence" />
#### Sequence Analysis
* [ToPS](https://github.com/ayoshiaki/tops) - This is an objected-oriented framework that facilitates the integration of probabilistic models for sequences over a user defined alphabet.

<a name="common-lisp" />
## Common Lisp

<a name="common-lisp-general-purpose" />
#### General-Purpose Machine Learning

* [mgl](https://github.com/melisgl/mgl/) - Neural networks  (boltzmann machines, feed-forward and recurrent nets), Gaussian Processes
* [mgl-gpr](https://github.com/melisgl/mgl-gpr/) - Evolutionary algorithms
* [cl-libsvm](https://github.com/melisgl/cl-libsvm/) - Wrapper for the libsvm support vector machine library

<a name="clojure" />
## Clojure

<a name="clojure-nlp" />
#### Natural Language Processing

* [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp)
* [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript

<a name="clojure-general-purpose" />
#### General-Purpose Machine Learning

* [Touchstone](https://github.com/ptaoussanis/touchstone) - Clojure A/B testing library
* [Clojush](https://github.com/lspector/Clojush) -  he Push programming language and the PushGP genetic programming system implemented in Clojure
* [Infer](https://github.com/aria42/infer) - Inference and machine learning in clojure
* [Clj-ML](https://github.com/antoniogarrote/clj-ml) - A machine learning library for Clojure built on top of Weka and friends
* [Encog](https://github.com/jimpil/enclog) - Clojure wrapper for Encog (v3) (Machine-Learning framework that specialises in neural-nets)
* [Fungp](https://github.com/vollmerm/fungp) - A genetic programming library for Clojure
* [Statistiker](https://github.com/clojurewerkz/statistiker) - Basic Machine Learning algorithms in Clojure.
* [clortex](https://github.com/nupic-community/clortex) - General Machine Learning library using Numenta’s Cortical Learning Algorithm
* [comportex](https://github.com/nupic-community/comportex) - Functionally composable Machine Learning library using Numenta’s Cortical Learning Algorithm

<a name="clojure-data-analysis" />
#### Data Analysis / Data Visualization

* [Incanter](http://incanter.org/) - Incanter is a Clojure-based, R-like platform for statistical computing and graphics.
* [PigPen](https://github.com/Netflix/PigPen) - Map-Reduce for Clojure.
* [Envision] (https://github.com/clojurewerkz/envision) - Clojure Data Visualisation library, based on Statistiker and D3

<a name="erlang" />
## Erlang

<a name="erlang-general-purpose" />
#### General-Purpose Machine Learning
* [Disco](https://github.com/discoproject/disco/) - Map Reduce in Erlang

<a name="go" />
## Go

<a name="go-nlp" />
#### Natural Language Processing

* [go-porterstemmer](https://github.com/reiver/go-porterstemmer) - A native Go clean room implementation of the Porter Stemming algorithm.
* [paicehusk](https://github.com/Rookii/paicehusk) - Golang implementation of the Paice/Husk Stemming Algorithm.
* [snowball](https://bitbucket.org/tebeka/snowball) - Snowball Stemmer for Go.
* [go-ngram](https://github.com/Lazin/go-ngram) - In-memory n-gram index with compression.

<a name="go-general-purpose" />
#### General-Purpose Machine Learning

* [Go Learn](https://github.com/sjwhitworth/golearn) - Machine Learning for Go
* [go-pr](https://github.com/daviddengcn/go-pr) - Pattern recognition package in Go lang.
* [go-ml](https://github.com/alonsovidales/go_ml) - Linear / Logistic regression, Neural Networks, Collaborative Filtering and Gaussian Multivariate Distribution
* [bayesian](https://github.com/jbrukh/bayesian) - Naive Bayesian Classification for Golang.
* [go-galib](https://github.com/thoj/go-galib) - Genetic Algorithms library written in Go / golang
* [Cloudforest](https://github.com/ryanbressler/CloudForest) - Ensembles of decision trees in go/golang.
* [gobrain](https://github.com/jonhkr/gobrain) - Neural Networks written in go

<a name="go-data-analysis" />
#### Data Analysis / Data Visualization

* [go-graph](https://github.com/StepLg/go-graph) - Graph library for Go/golang language.
* [SVGo](http://www.svgopen.org/2011/papers/34-SVGo_a_Go_Library_for_SVG_generation/) - The Go Language library for SVG generation


<a name="haskell" />
## Haskell

<a name="haskell-general-purpose" />
#### General-Purpose Machine Learning
* [haskell-ml](https://github.com/ajtulloch/haskell-ml) - Haskell implementations of various ML algorithms.
* [HLearn](https://github.com/mikeizbicki/HLearn) - a suite of libraries for interpreting machine learning models according to their algebraic structure.
* [hnn](http://www.haskell.org/haskellwiki/HNN) - Haskell Neural Network library.
* [hopfield-networks](https://github.com/ajtulloch/hopfield-networks) - Hopfield Networks for unsupervised learning in Haskell.
* [caffegraph](https://github.com/ajtulloch/caffegraph) - A DSL for deep neural networks


<a name="java" />
## Java

<a name="java-nlp" />
#### Natural Language Processing

* [CoreNLP] (http://nlp.stanford.edu/software/corenlp.shtml) - Stanford CoreNLP provides a set of natural language analysis tools which can take raw English language text input and give the base forms of words
* [Stanford Parser] (http://nlp.stanford.edu/software/lex-parser.shtml) - A natural language parser is a program that works out the grammatical structure of sentences
* [Stanford POS Tagger] (http://nlp.stanford.edu/software/tagger.shtml) - A Part-Of-Speech Tagger (POS Tagger
* [Stanford Name Entity Recognizer] (http://nlp.stanford.edu/software/CRF-NER.shtml) - Stanford NER is a Java implementation of a Named Entity Recognizer.
* [Stanford Word Segmenter] (http://nlp.stanford.edu/software/segmenter.shtml) - Tokenization of raw text is a standard pre-processing step for many NLP tasks.
* [Tregex, Tsurgeon and Semgrex](http://nlp.stanford.edu/software/tregex.shtml) - Tregex is a utility for matching patterns in trees, based on tree relationships and regular expression matches on nodes (the name is short for "tree regular expressions").
* [Stanford Phrasal: A Phrase-Based Translation System](http://nlp.stanford.edu/software/phrasal/)
* [Stanford English Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) - Stanford Phrasal is a state-of-the-art statistical phrase-based machine translation system, written in Java.
* [Stanford Tokens Regex](http://nlp.stanford.edu/software/tokensregex.shtml) - A tokenizer divides text into a sequence of tokens, which roughly correspond to "words"
* [Stanford Temporal Tagger](http://nlp.stanford.edu/software/sutime.shtml) - SUTime is a library for recognizing and normalizing time expressions.
* [Stanford SPIED](http://nlp.stanford.edu/software/patternslearning.shtml) - Learning entities from unlabeled text starting with seed sets using patterns in an iterative fashion
* [Stanford Topic Modeling Toolbox](http://nlp.stanford.edu/software/tmt/tmt-0.4/) - Topic modeling tools to social scientists and others who wish to perform analysis on datasets
* [Twitter Text Java](https://github.com/twitter/twitter-text-java) - A Java implementation of Twitter's text processing library
* [MALLET](http://mallet.cs.umass.edu/) - A Java-based package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text.
* [OpenNLP](https://opennlp.apache.org/) - a machine learning based toolkit for the processing of natural language text.
* [LingPipe](http://alias-i.com/lingpipe/index.html) - A tool kit for processing text using computational linguistics.
* [ClearTK](https://code.google.com/p/cleartk/) - ClearTK provides a framework for developing statistical natural language processing (NLP) components in Java and is built on top of Apache UIMA.
* [Apache cTAKES](http://ctakes.apache.org/) - Apache clinical Text Analysis and Knowledge Extraction System (cTAKES) is an open-source natural language processing system for information extraction from electronic medical record clinical free-text.

<a name="java-general-purpose" />
#### General-Purpose Machine Learning

* [Datumbox](https://github.com/datumbox/datumbox-framework) - Machine Learning framework for rapid development of Machine Learning and Statistical applications
* [ELKI](http://elki.dbs.ifi.lmu.de/) - Java toolkit for data mining. (unsupervised: clustering, outlier detection etc.)
* [Encog](https://github.com/encog/encog-java-core) - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trains using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [H2O](https://github.com/0xdata/h2o) - ML engine that supports distributed learning on data stored in HDFS.
* [htm.java](https://github.com/numenta/htm.java) - General Machine Learning library using Numenta’s Cortical Learning Algorithm
* [java-deeplearning](https://github.com/agibsonccc/java-deeplearning) - Distributed Deep Learning Platform for Java, Clojure,Scala
* [JAVA-ML](http://java-ml.sourceforge.net/) - A general ML library with a common interface for all algorithms in Java
* [JSAT](https://code.google.com/p/java-statistical-analysis-tool/) - Numerous Machine Learning algoirhtms for classification, regresion, and  clustering.
* [Mahout](https://github.com/apache/mahout) - Distributed machine learning
* [Meka](http://meka.sourceforge.net/) - An open source implementation of methods for multi-label classification and evaluation (extension to Weka).
* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Neuroph](http://neuroph.sourceforge.net/) - Neuroph is lightweight Java neural network framework
* [ORYX](https://github.com/oryxproject/oryx) - Lambda Architecture Framework using Apache Spark and Apache Kafka with a specialization for real-time large-scale machine learning.
* [RankLib](http://sourceforge.net/p/lemur/wiki/RankLib/) - RankLib is a library of learning to rank algorithms
* [RapidMiner](http://rapid-i.com/wiki/index.php?title=Integrating_RapidMiner_into_your_application) - RapidMiner integration into Java code
* [Stanford Classifier](http://nlp.stanford.edu/software/classifier.shtml) - A classifier is a machine learning tool that will take data items and place them into one of k classes.
* [WalnutiQ](https://github.com/WalnutiQ/WalnutiQ) - object oriented model of the human brain
* [Weka](http://www.cs.waikato.ac.nz/ml/weka/) - Weka is a collection of machine learning algorithms for data mining tasks

#### Speech Recognition
* [CMU Sphinx](http://cmusphinx.sourceforge.net/) - Open Source Toolkit For Speech Recognition purely based on Java speech recognition library.

<a name="java-data-analysis" />
#### Data Analysis / Data Visualization

* [Hadoop](https://github.com/apache/hadoop-mapreduce) - Hadoop/HDFS
* [Spark](https://github.com/apache/spark) - Spark is a fast and general engine for large-scale data processing.
* [Impala](https://github.com/cloudera/impala) - Real-time Query for Hadoop

<a name="java-deep-learning" />
#### Deep Learning

* [Deeplearning4j](https://github.com/SkymindIO/deeplearning4j/) - Scalable deep learning for industry with parallel GPUs

<a name="javascript" />
## Javascript

<a name="javascript-nlp" />
#### Natural Language Processing

* [Twitter-text-js](https://github.com/twitter/twitter-text-js) - A JavaScript implementation of Twitter's text processing library
* [NLP.js](https://github.com/nicktesla/nlpjs) - NLP utilities in javascript and coffeescript
* [natural](https://github.com/NaturalNode/natural) - General natural language facilities for node
* [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS
* [Retext](http://github.com/wooorm/retext) - Extensible system for analysing and manipulating natural language
* [TextProcessing](https://www.mashape.com/japerk/text-processing/support) - Sentiment analysis, stemming and lemmatization, part-of-speech tagging and chunking, phrase extraction and named entity recognition.


<a name="javascript-data-analysis" />
#### Data Analysis / Data Visualization

* [D3.js](http://d3js.org/)
* [High Charts](http://www.highcharts.com/)
* [NVD3.js](http://nvd3.org/)
* [dc.js](http://dc-js.github.io/dc.js/)
* [chartjs](http://www.chartjs.org/)
* [dimple](http://dimplejs.org/)
* [amCharts](http://www.amcharts.com/)
* [D3xter](https://github.com/NathanEpstein/D3xter) - Straight forward plotting built on D3
* [statkit](https://github.com/rigtorp/statkit) - Statistics kit for JavaScript
* [science.js](https://github.com/jasondavies/science.js/) - Scientific and statistical computing in JavaScript.
* [Z3d](https://github.com/NathanEpstein/Z3d) - Easily make interactive 3d plots built on Three.js

<a name="javascript-general-purpose" />
#### General-Purpose Machine Learning

* [Convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a Javascript library for training Deep Learning models[DEEP LEARNING]
* [Clustering.js](https://github.com/tixz/clustering.js) - Clustering algorithms implemented in Javascript for Node.js and the browser
* [Decision Trees](https://github.com/serendipious/nodejs-decision-tree-id3) - NodeJS Implementation of Decision Tree using ID3 Algorithm
* [Node-fann](https://github.com/rlidwka/node-fann) - FANN (Fast Artificial Neural Network Library) bindings for Node.js
* [Kmeans.js](https://github.com/tixz/kmeans.js) - Simple Javascript implementation of the k-means algorithm, for node.js and the browser
* [LDA.js](https://github.com/primaryobjects/lda) - LDA topic modeling for node.js
* [Learning.js](https://github.com/yandongliu/learningjs) - Javascript implementation of logistic regression/c4.5 decision tree
* [Machine Learning](http://joonku.com/project/machine_learning) - Machine learning library for Node.js
* [Node-SVM](https://github.com/nicolaspanel/node-svm) - Support Vector Machine for nodejs
* [Brain](https://github.com/harthur/brain) - Neural networks in JavaScript
* [Bayesian-Bandit](https://github.com/omphalos/bayesian-bandit.js) - Bayesian bandit implementation for Node and the browser.
* [Synaptic](https://github.com/cazala/synaptic) - Architecture-free neural network library for node.js and the browser
* [kNear](https://github.com/NathanEpstein/kNear) - JavaScript implementation of the k nearest neighbors algorithm for supervised learning
* [NeuralN](https://github.com/totemstech/neuraln) - C++ Neural Network library for Node.js. It has advantage on large dataset and multi-threaded training.

<a name="julia" />
## Julia

<a name="julia-general-purpose" />
#### General-Purpose Machine Learning

* [PGM](https://github.com/JuliaStats/PGM.jl) - A Julia framework for probabilistic graphical models.
* [DA](https://github.com/trthatcher/DA.jl) - Julia package for Regularized Discriminant Analysis
* [Regression](https://github.com/lindahua/Regression.jl) - Algorithms for regression analysis (e.g. linear regression and logistic regression)
* [Local Regression](https://github.com/dcjones/Loess.jl) - Local regression, so smooooth!
* [Naive Bayes](https://github.com/nutsiepully/NaiveBayes.jl) - Simple Naive Bayes implementation in Julia
* [Mixed Models](https://github.com/dmbates/MixedModels.jl) - A Julia package for fitting (statistical) mixed-effects models
* [Simple MCMC](https://github.com/fredo-dedup/SimpleMCMC.jl) - basic mcmc sampler implemented in Julia
* [Distance](https://github.com/JuliaStats/Distance.jl) - Julia module for Distance evaluation
* [Decision Tree](https://github.com/bensadeghi/DecisionTree.jl) - Decision Tree Classifier and Regressor
* [Neural](https://github.com/compressed/neural.jl) - A neural network in Julia
* [MCMC](https://github.com/doobwa/MCMC.jl) - MCMC tools for Julia
* [GLM](https://github.com/JuliaStats/GLM.jl) - Generalized linear models in Julia
* [Online Learning](https://github.com/lendle/OnlineLearning.jl)
* [GLMNet](https://github.com/simonster/GLMNet.jl) - Julia wrapper for fitting Lasso/ElasticNet GLM models using glmnet
* [Clustering](https://github.com/JuliaStats/Clustering.jl) - Basic functions for clustering data: k-means, dp-means, etc.
* [SVM](https://github.com/JuliaStats/SVM.jl) - SVM's for Julia
* [Kernal Density](https://github.com/JuliaStats/KernelDensity.jl) - Kernel density estimators for julia
* [Dimensionality Reduction](https://github.com/JuliaStats/DimensionalityReduction.jl) - Methods for dimensionality reduction
* [NMF](https://github.com/JuliaStats/NMF.jl) - A Julia package for non-negative matrix factorization
* [ANN](https://github.com/EricChiang/ANN.jl) - Julia artificial neural networks
* [Mocha.jl](https://github.com/pluskid/Mocha.jl) - Deep Learning framework for Julia inspired by Caffe
* [XGBoost.jl](https://github.com/antinucleon/XGBoost.jl) - eXtreme Gradient Boosting Package in Julia

<a name="julia-nlp" />
#### Natural Language Processing

* [Topic Models](https://github.com/slycoder/TopicModels.jl) - TopicModels for Julia
* [Text Analysis](https://github.com/johnmyleswhite/TextAnalysis.jl) - Julia package for text analysis


<a name="julia-data-analysis" />
#### Data Analysis / Data Visualization

* [Graph Layout](https://github.com/IainNZ/GraphLayout.jl) - Graph layout algorithms in pure Julia
* [Data Frames Meta](https://github.com/JuliaStats/DataFramesMeta.jl) - Metaprogramming tools for DataFrames
* [Julia Data](https://github.com/nfoti/JuliaData) - library for working with tabular data in Julia
* [Data Read](https://github.com/WizardMac/DataRead.jl) - Read files from Stata, SAS, and SPSS
* [Hypothesis Tests](https://github.com/JuliaStats/HypothesisTests.jl) - Hypothesis tests for Julia
* [Gadfly](https://github.com/dcjones/Gadfly.jl) - Crafty statistical graphics for Julia.
* [Stats](https://github.com/johnmyleswhite/stats.jl) - Statistical tests for Julia

* [RDataSets](https://github.com/johnmyleswhite/RDatasets.jl) - Julia package for loading many of the data sets available in R
* [DataFrames](https://github.com/JuliaStats/DataFrames.jl) - library for working with tabular data in Julia
* [Distributions](https://github.com/JuliaStats/Distributions.jl) - A Julia package for probability distributions and associated functions.
* [Data Arrays](https://github.com/JuliaStats/DataArrays.jl) - Data structures that allow missing values
* [Time Series](https://github.com/JuliaStats/TimeSeries.jl) - Time series toolkit for Julia
* [Sampling](https://github.com/JuliaStats/Sampling.jl) - Basic sampling algorithms for Julia

<a name="julia-misc" />
#### Misc Stuff / Presentations

* [DSP](https://github.com/JuliaDSP/DSP.jl) - Digital Signal Processing (filtering, periodograms, spectrograms, window functions).
* [JuliaCon Presentations](https://github.com/JuliaCon/presentations) - Presentations for JuliaCon
* [SignalProcessing](https://github.com/davidavdav/SignalProcessing) - Signal Processing tools for Julia
* [Images](https://github.com/timholy/Images.jl) - An image library for Julia

<a name="lua" />
## Lua

<a name="lua-general-purpose" />
#### General-Purpose Machine Learning

* [Torch7](http://torch.ch/)
  * [cephes](http://jucor.github.io/torch-cephes) - Cephes mathematical functions library, wrapped for Torch. Provides and wraps the 180+ special mathematical functions from the Cephes mathematical library, developed by Stephen L. Moshier. It is used, among many other places, at the heart of SciPy.
  * [graph](https://github.com/torch/graph) - Graph package for Torch
  * [randomkit](http://jucor.github.io/torch-randomkit/) - Numpy's randomkit, wrapped for Torch
  * [signal](http://soumith.ch/torch-signal/signal/) - A signal processing toolbox for Torch-7. FFT, DCT, Hilbert, cepstrums, stft

  * [nn](https://github.com/torch/nn) - Neural Network package for Torch
  * [nngraph](https://github.com/torch/nngraph) - This package provides graphical computation for nn library in Torch7.
  * [nnx](https://github.com/clementfarabet/lua---nnx) - A completely unstable and experimental package that extends Torch's builtin nn library
  * [optim](https://github.com/torch/optim) - An optimization library for Torch. SGD, Adagrad, Conjugate-Gradient, LBFGS, RProp and more.
  * [unsup](https://github.com/koraykv/unsup) - A package for unsupervised learning in Torch. Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA).
  * [manifold](https://github.com/clementfarabet/manifold) - A package to manipulate manifolds
  * [svm](https://github.com/koraykv/torch-svm) - Torch-SVM library
  * [lbfgs](https://github.com/clementfarabet/lbfgs) - FFI Wrapper for liblbfgs
  * [vowpalwabbit](https://github.com/clementfarabet/vowpal_wabbit) - An old vowpalwabbit interface to torch.
  * [OpenGM](https://github.com/clementfarabet/lua---opengm) - OpenGM is a C++ library for graphical modeling, and inference. The Lua bindings provide a simple way of describing graphs, from Lua, and then optimizing them with OpenGM.
  * [sphagetti](https://github.com/MichaelMathieu/lua---spaghetti) - Spaghetti (sparse linear) module for torch7 by @MichaelMathieu
  * [LuaSHKit](https://github.com/ocallaco/LuaSHkit) - A lua wrapper around the Locality sensitive hashing library SHKit
  * [kernel smoothing](https://github.com/rlowrance/kernel-smoothers) - KNN, kernel-weighted average, local linear regression smoothers
  * [cutorch](https://github.com/torch/cutorch) - Torch CUDA Implementation
  * [cunn](https://github.com/torch/cunn) - Torch CUDA Neural Network Implementation
  * [imgraph](https://github.com/clementfarabet/lua---imgraph) - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images.
  * [videograph](https://github.com/clementfarabet/videograph) - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos.
  * [saliency](https://github.com/marcoscoffier/torch-saliency) - code and tools around integral images. A library for finding interest points based on fast integral histograms.
  * [stitch](https://github.com/marcoscoffier/lua---stitch) - allows us to use hugin to stitch images and apply same stitching to a video sequence
  * [sfm](https://github.com/marcoscoffier/lua---sfm) - A bundle adjustment/structure from motion package
  * [fex](https://github.com/koraykv/fex) - A package for feature extraction in Torch. Provides SIFT and dSIFT modules.
  * [OverFeat](https://github.com/sermanet/OverFeat) - A state-of-the-art generic dense feature extractor
* [Numeric Lua](http://numlua.luaforge.net/)
* [Lunatic Python](http://labix.org/lunatic-python)
* [SciLua](http://www.scilua.org/)
* [Lua - Numerical Algorithms](https://bitbucket.org/lucashnegri/lna)
* [Lunum](http://zrake.webfactional.com/projects/lunum)

<a name="lua-demos" />
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
* [Music Tagging](https://github.com/mbhenaff/MusicTagging) - Music Tagging scripts for torch7
* [torch-datasets](https://github.com/rosejn/torch-datasets) - Scripts to load several popular datasets including:
  * BSR 500
  * CIFAR-10
  * COIL
  * Street View House Numbers
  * MNIST
  * NORB
* [Atari2600](https://github.com/fidlej/aledataset) - Scripts to generate a dataset with static frames from the Arcade Learning Environment



<a name="matlab" />
## Matlab

<a name="matlab-cv" />
#### Computer Vision

* [Contourlets](http://www.ifp.illinois.edu/~minhdo/software/contourlet_toolbox.tar) - MATLAB source code that implements the contourlet transform and its utility functions.
* [Shearlets](http://www.shearlab.org/index_software.html) - MATLAB code for shearlet transform
* [Curvelets](http://www.curvelet.org/software.html) - The Curvelet transform is a higher dimensional generalization of the Wavelet transform designed to represent images at different scales and different angles.
* [Bandlets](http://www.cmap.polytechnique.fr/~peyre/download/) - MATLAB code for bandlet transform

<a name="matlab-nlp" />
#### Natural Language Processing

* [NLP](https://amplab.cs.berkeley.edu/2012/05/05/an-nlp-library-for-matlab/) - An NLP library for Matlab

<a name="matlab-general-purpose" />
#### General-Purpose Machine Learning

* [Training a deep autoencoder or a classifier
on MNIST digits](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) - Training a deep autoencoder or a classifier
on MNIST digits[DEEP LEARNING]
* [t-Distributed Stochastic Neighbor Embedding](http://homepage.tudelft.nl/19j49/t-SNE.html) - t-Distributed Stochastic Neighbor Embedding (t-SNE) is a (prize-winning) technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.
* [Spider](http://people.kyb.tuebingen.mpg.de/spider/) - The spider is intended to be a complete object orientated environment for machine learning in Matlab.
* [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/#matlab) - A Library for Support Vector Machines
* [LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/#download) - A Library for Large Linear Classification
* [Machine Learning Module](https://github.com/josephmisiti/machine-learning-module) - Class on machine w/ PDF,lectures,code
* [Caffe](http://caffe.berkeleyvision.org)  - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [Pattern Recognition Toolbox](https://github.com/newfolder/PRT)  - A complete object-oriented environment for machine learning in Matlab.
* [Pattern Recognition and Machine Learning](https://github.com/PRML/PRML) - This package contains the matlab implementation of the algorithms described in the book Pattern Recognition and Machine Learning by C. Bishop.

<a name="matlab-data-analysis" />
#### Data Analysis / Data Visualization

* [matlab_gbl](https://www.cs.purdue.edu/homes/dgleich/packages/matlab_bgl/) - MatlabBGL is a Matlab package for working with graphs.
* [gamic](http://www.mathworks.com/matlabcentral/fileexchange/24134-gaimc---graph-algorithms-in-matlab-code) - Efficient pure-Matlab implementations of graph algorithms to complement MatlabBGL's mex functions.

<a name="net" />
## .NET

<a name="net-cv" />
#### Computer Vision

* [OpenCVDotNet](https://code.google.com/p/opencvdotnet/) - A wrapper for the OpenCV project to be used with .NET applications.
* [Emgu CV](http://www.emgu.com/wiki/index.php/Main_Page) - Cross platform wrapper of OpenCV which can be compiled in Mono to e run on Windows, Linus, Mac OS X, iOS, and Android.
* [AForge.NET](http://www.aforgenet.com/framework/) - Open source C# framework for developers and researchers in the fields of Computer Vision and Artificial Intelligence. Development has now shifted to GitHub.
* [Accord.NET](http://accord-framework.net) - Together with AForge.NET, this library can provide image processing and computer vision algorithms to Windows, Windows RT and Windows Phone. Some components are also available for Java and Android.

<a name="net-nlp" />
#### Natural Language Processing

* [Stanford.NLP for .NET](https://github.com/sergey-tihon/Stanford.NLP.NET/) - A full port of Stanford NLP packages to .NET and also available precompiled as a NuGet package.

<a name="net-general-purpose" />
#### General-Purpose Machine Learning

* [Accord-Framework](http://accord-framework.net/) -The Accord.NET Framework is a complete framework for building machine learning, computer vision, computer audition, signal processing and statistical applications.
* [Accord.MachineLearning](http://www.nuget.org/packages/Accord.MachineLearning/) - Support Vector Machines, Decision Trees, Naive Bayesian models, K-means, Gaussian Mixture models and general algorithms such as Ransac, Cross-validation and Grid-Search for machine-learning applications. This package is part of the Accord.NET Framework.
* [Vulpes](https://github.com/fsprojects/Vulpes) - Deep belief and deep learning implementation written in F# and leverages CUDA GPU execution with Alea.cuBase.
* [Encog](http://www.nuget.org/packages/encog-dotnet-core/) -  An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trains using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [Neural Network Designer](http://bragisoft.com/) - DBMS management system and designer for neural networks. The designer application is developed using WPF, and is a user interface which allows you to design your neural network, query the network, create and configure chat bots that are capable of asking questions and learning from your feed back.  The chat bots can even scrape the internet for information to return in their output as well as to use for learning.

<a name="net-data-analysis" />
#### Data Analysis / Data Visualization

* [numl](http://www.nuget.org/packages/numl/) - numl is a machine learning library intended to ease the use of using standard modeling techniques for both prediction and clustering.
* [Math.NET Numerics](http://www.nuget.org/packages/MathNet.Numerics/) - Numerical foundation of the Math.NET project, aiming to provide methods and algorithms for numerical computations in science, engineering and every day use. Supports .Net 4.0, .Net 3.5 and Mono on Windows, Linux and Mac; Silverlight 5, WindowsPhone/SL 8, WindowsPhone 8.1 and Windows 8 with PCL Portable Profiles 47 and 344; Android/iOS with Xamarin.
* [Sho](http://research.microsoft.com/en-us/projects/sho/) - Sho is an interactive environment for data analysis and scientific computing that lets you seamlessly connect scripts (in IronPython) with compiled code (in .NET) to enable fast and flexible prototyping. The environment includes powerful and efficient libraries for linear algebra as well as data visualization that can be used from any .NET language, as well as a feature-rich interactive shell for rapid development.

<a name="objectivec">
## Objective C

<a name="objectivec-general-purpose">
### General-Purpose Machine Learning

* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural network. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available.

<a name="python" />
## Python

<a name="python-cv" />
#### Computer Vision

* [SimpleCV](http://simplecv.org/) - An open source computer vision framework that gives access to several high-powered computer vision libraries, such as OpenCV. Written on Python and runs on Mac, Windows, and Ubuntu Linux.

<a name="python-nlp" />
#### Natural Language Processing

* [NLTK](http://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [Quepy](https://github.com/machinalis/quepy) - A python framework to transform natural language questions to queries in a database query language
* [TextBlob](http://textblob.readthedocs.org/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
* [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora.
* [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
* [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
* [loso](https://github.com/victorlin/loso) - Another Chinese segmentation library.
* [genius](https://github.com/duanhongyi/genius) - A Chinese segment base on Conditional Random Field.
* [nut](https://github.com/pprett/nut) - Natural language Understanding Toolkit
* [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
* [BLLIP Parser](https://pypi.python.org/pypi/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
* [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
* [python-ucto](https://github.com/proycon/python-ucto) - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages)
* [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
* [colibri-core](https://github.com/proycon/colibri-core) - Python binding to C++ library for extracting and working with with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [spaCy](https://github.com/honnibal/spaCy/) - Industrial strength NLP with Python and Cython.
* [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.

<a name="python-general-purpose" />
#### General-Purpose Machine Learning
* [XGBoost](https://github.com/tqchen/xgboost) - Python bindings for eXtreme Gradient Boosting (Tree) Library

* [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Book/iPython notebooks on Probabilistic Programming in Python
* [Featureforge](https://github.com/machinalis/featureforge) A set of tools for creating and testing machine learning features, with a scikit-learn compatible API
* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [scikit-learn](http://scikit-learn.org/) - A Python module for machine learning built on top of SciPy.
* [SimpleAI](http://github.com/simpleai-team/simpleai) Python implementation of many of the artificial intelligence algorithms described on the book "Artificial Intelligence, a Modern Approach". It focuses on providing an easy to use, well documented and tested library.
* [astroML](http://www.astroml.org/) - Machine Learning and Data Mining for Astronomy.
* [graphlab-create](http://graphlab.com/products/create/docs/) - A library with various machine learning models (regression, clustering, recommender systems, graph analytics, etc.) implemented on top of a disk-backed DataFrame.
* [BigML](https://bigml.com) - A library that contacts external servers.
* [pattern](https://github.com/clips/pattern) - Web mining module for Python.
* [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
* [Pylearn2](https://github.com/lisa-lab/pylearn2) - A Machine Learning library based on [Theano](https://github.com/Theano/Theano).
* [hebel](https://github.com/hannes-brt/hebel) - GPU-Accelerated Deep Learning Library in Python.
* [gensim](https://github.com/piskvorky/gensim) - Topic Modelling for Humans.
* [PyBrain](https://github.com/pybrain/pybrain) - Another Python Machine Learning Library.
* [Crab](https://github.com/muricoca/crab) - A ﬂexible, fast recommender engine.
* [python-recsys](https://github.com/ocelma/python-recsys) - A Python library for implementing a Recommender System.
* [thinking bayes](https://github.com/AllenDowney/ThinkBayes) - Book on Bayesian Analysis
* [Restricted Boltzmann Machines](https://github.com/echen/restricted-boltzmann-machines) -Restricted Boltzmann Machines in Python. [DEEP LEARNING]
* [Bolt](https://github.com/pprett/bolt) - Bolt Online Learning Toolbox
* [CoverTree](https://github.com/patvarilly/CoverTree) - Python implementation of cover trees, near-drop-in replacement for scipy.spatial.kdtree
* [nilearn](https://github.com/nilearn/nilearn) - Machine learning for NeuroImaging in Python
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox
* [Pyevolve](https://github.com/perone/Pyevolve) - Genetic algorithm framework.
* [Caffe](http://caffe.berkeleyvision.org)  - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [breze](https://github.com/breze-no-salt/breze) - Theano based library for deep and recurrent neural networks
* [pyhsmm](https://github.com/mattjj/pyhsmm) - library for approximate unsupervised inference in Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM and HDP-HSMM, mostly with weak-limit approximations.
* [mrjob](https://pythonhosted.org/mrjob/) - A library to let Python program run on Hadoop.
* [SKLL](https://github.com/EducationalTestingService/skll) - A wrapper around scikit-learn that makes it simpler to conduct experiments.
* [neurolab](https://code.google.com/p/neurolab/) - https://code.google.com/p/neurolab/
* [Spearmint](https://github.com/JasperSnoek/spearmint) - Spearmint is a package to perform Bayesian optimization according to the algorithms outlined in the paper: Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan P. Adams. Advances in Neural Information Processing Systems, 2012.
* [Pebl](https://github.com/abhik/pebl/) - Python Environment for Bayesian Learning
* [Theano](https://github.com/Theano/Theano/) - Optimizing GPU-meta-programming code generating array oriented optimizing math compiler in Python
* [yahmm](https://github.com/jmschrei/yahmm/) - Hidden Markov Models for Python, implemented in Cython for speed and efficiency.
* [python-timbl](https://github.com/proycon/python-timbl) - A Python extension module wrapping the full TiMBL C++ programming interface. Timbl is an elaborate k-Nearest Neighbours machine learning toolkit.
* [deap](https://github.com/deap/deap) - Evolutionary algorithm framework.
* [pydeep](https://github.com/andersbll/deeppy) - Deep Learning In Python

<a name="python-data-analysis" />
#### Data Analysis / Data Visualization

* [SciPy](http://www.scipy.org/) - A Python-based ecosystem of open-source software for mathematics, science, and engineering.
* [NumPy](http://www.numpy.org/) - A fundamental package for scientific computing with Python.
* [Numba](http://numba.pydata.org/) - Python JIT (just in time) complier to LLVM aimed at scientific Python by the developers of Cython and NumPy.
* [NetworkX](https://networkx.github.io/) - A high-productivity software for complex networks.
* [Pandas](http://pandas.pydata.org/) - A library providing high-performance, easy-to-use data structures and data analysis tools.
* [Open Mining](https://github.com/avelino/mining) - Business Intelligence (BI) in Python (Pandas web interface)
* [PyMC](https://github.com/pymc-devs/pymc) - Markov Chain Monte Carlo sampling toolkit.
* [zipline](https://github.com/quantopian/zipline) - A Pythonic algorithmic trading library.
* [PyDy](https://pydy.org/) - Short for Python Dynamics, used to assist with workflow in the modeling of dynamic motion based around NumPy, SciPy, IPython, and matplotlib.
* [SymPy](https://github.com/sympy/sympy) - A Python library for symbolic mathematics.
* [statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modeling and econometrics in Python.
* [astropy](http://www.astropy.org/) - A community Python library for Astronomy.
* [matplotlib](http://matplotlib.org/) - A Python 2D plotting library.
* [bokeh](https://github.com/ContinuumIO/bokeh) - Interactive Web Plotting for Python.
* [plotly](https://plot.ly/python) - Collaborative web plotting for Python and matplotlib.
* [vincent](https://github.com/wrobstory/vincent) - A Python to Vega translator.
* [d3py](https://github.com/mikedewar/d3py) - A plottling library for Python, based on [D3.js](http://d3js.org/).
* [ggplot](https://github.com/yhat/ggplot) - Same API as ggplot2 for R.
* [Kartograph.py](https://github.com/kartograph/kartograph.py) - Rendering beautiful SVG maps in Python.
* [pygal](http://pygal.org/) - A Python SVG Charts Creator.
* [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) - A pure-python graphics and GUI library built on PyQt4 / PySide and NumPy.
* [pycascading](https://github.com/twitter/pycascading)
* [Petrel](https://github.com/AirSage/Petrel) - Tools for writing, submitting, debugging, and monitoring Storm topologies in pure Python.
* [Blaze](https://github.com/ContinuumIO/blaze) - NumPy and Pandas interface to Big Data.
* [emcee](https://github.com/dfm/emcee) - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [windML](http://www.windml.org) - A Python Framework for Wind Energy Analysis and Prediction
* [vispy](https://github.com/vispy/vispy) - GPU-based high-performance interactive OpenGL 2D/3D data visualization library
* [cerebro2](https://github.com/numenta/nupic.cerebro2) A web-based visualization and debugging platform for NuPIC.
* [NuPIC Studio](https://github.com/nupic-community/nupic.studio) An all-in-one NuPIC Hierarchical Temporal Memory visualization and debugging super-tool!

<a name="python-misc" />
#### Misc Scripts / iPython Notebooks / Codebases

* [pattern_classification](https://github.com/rasbt/pattern_classification)
* [thinking stats 2](https://github.com/Wavelets/ThinkStats2)
* [hyperopt](https://github.com/hyperopt/hyperopt-sklearn)
* [numpic](https://github.com/numenta/nupic)
* [2012-paper-diginorm](https://github.com/ged-lab/2012-paper-diginorm)
* [ipython-notebooks](https://github.com/ogrisel/notebooks)
* [decision-weights](https://github.com/CamDavidsonPilon/decision-weights)
* [Sarah Palin LDA](https://github.com/Wavelets/sarah-palin-lda) - Topic Modeling the Sarah Palin emails.
* [Diffusion Segmentation](https://github.com/Wavelets/diffusion-segmentation) - A collection of image segmentation algorithms based on diffusion methods
* [Scipy Tutorials](https://github.com/Wavelets/scipy-tutorials) - SciPy tutorials. This is outdated, check out scipy-lecture-notes
* [Crab](https://github.com/marcelcaraciolo/crab) - A recommendation engine library for Python
* [BayesPy](https://github.com/maxsklar/BayesPy) - Bayesian Inference Tools in Python
* [scikit-learn tutorials](https://github.com/GaelVaroquaux/scikit-learn-tutorial) - Series of notebooks for learning scikit-learn
* [sentiment-analyzer](https://github.com/madhusudancs/sentiment-analyzer) - Tweets Sentiment Analyzer
* [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - Sentiment classifier using word sense disambiguation.
* [group-lasso](https://github.com/fabianp/group_lasso) - Some experiments with the coordinate descent algorithm used in the (Sparse) Group Lasso model
* [jProcessing](https://github.com/kevincobain2000/jProcessing) - Kanji / Hiragana / Katakana to Romaji Converter. Edict Dictionary & parallel sentences Search. Sentence Similarity between two JP Sentences. Sentiment Analysis of Japanese Text. Run Cabocha(ISO--8859-1 configured) in Python.
* [mne-python-notebooks](https://github.com/mne-tools/mne-python-notebooks) - IPython notebooks for EEG/MEG data processing using mne-python
* [pandas cookbook](https://github.com/jvns/pandas-cookbook) - Recipes for using Python's pandas library
* [climin](https://github.com/BRML/climin) - Optimization library focused on machine learning, pythonic implementations of gradient descent, LBFGS, rmsprop, adadelta and others
* [Allen Downey’s Data Science Course](https://github.com/AllenDowney/DataScience) - Code for Data Science at Olin College, Spring 2014.
* [Allen Downey’s Think Bayes Code](https://github.com/AllenDowney/ThinkBayes) - Code repository for Think Bayes.
* [Allen Downey’s Think Complexity Code](https://github.com/AllenDowney/ThinkComplexity) - Code for Allen Downey's book Think Complexity.
* [Allen Downey’s Think OS Code](https://github.com/AllenDowney/ThinkOS) - Text and supporting code for Think OS: A Brief Introduction to Operating Systems.
* [Python Programming for the Humanities](http://fbkarsdorp.github.io/python-course/) - Course for Python programming for the Humanities, assuming no prior knowledge. Heavy focus on text processing / NLP.

<a name="python-kaggle" />
#### Kaggle Competition Source Code

* [wiki challenge](https://github.com/hammer/wikichallenge) - An implementation of Dell Zhang's solution to Wikipedia's Participation Challenge on Kaggle
* [kaggle insults](https://github.com/amueller/kaggle_insults) - Kaggle Submission for "Detecting Insults in Social Commentary"
* [kaggle_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) - Code for the Kaggle acquire valued shoppers challenge
* [kaggle-cifar](https://github.com/zygmuntz/kaggle-cifar) - Code for the CIFAR-10 competition at Kaggle, uses cuda-convnet
* [kaggle-blackbox](https://github.com/zygmuntz/kaggle-blackbox) - Deep learning made easy
* [kaggle-accelerometer](https://github.com/zygmuntz/kaggle-accelerometer) - Code for Accelerometer Biometric Competition at Kaggle
* [kaggle-advertised-salaries](https://github.com/zygmuntz/kaggle-advertised-salaries) - Predicting job salaries from ads - a Kaggle competition
* [kaggle amazon](https://github.com/zygmuntz/kaggle-amazon) - Amazon access control challenge
* [kaggle-bestbuy_big](https://github.com/zygmuntz/kaggle-bestbuy_big) - Code for the Best Buy competition at Kaggle
* [kaggle-bestbuy_small](https://github.com/zygmuntz/kaggle-bestbuy_small)
* [Kaggle Dogs vs. Cats](https://github.com/kastnerkyle/kaggle-dogs-vs-cats) - Code for Kaggle Dovs vs. Cats competition
* [Kaggle Galaxy Challenge](https://github.com/benanne/kaggle-galaxies) - Winning solution for the Galaxy Challenge on Kaggle
* [Kaggle Gender](https://github.com/zygmuntz/kaggle-gender) - A Kaggle competition: discriminate gender based on handwriting
* [Kaggle Merck](https://github.com/zygmuntz/kaggle-merck) - Merck challenge at Kaggle
* [Kaggle Stackoverflow](https://github.com/zygmuntz/kaggle-stackoverflow) - Predicting closed questions on Stack Overflow
* [kaggle_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) - Code for the Kaggle acquire valued shoppers challenge
* [wine-quality](https://github.com/zygmuntz/wine-quality) - Predicting wine quality

<a name="ruby" />
## Ruby

<a name="ruby-nlp" />
#### Natural Language Processing

* [Treat](https://github.com/louismullie/treat) -  Text REtrieval and Annotation Toolkit, definitely the most comprehensive toolkit I’ve encountered so far for Ruby
* [Ruby Linguistics](http://www.deveiate.org/projects/Linguistics/) -  Linguistics is a framework for building linguistic utilities for Ruby objects in any language. It includes a generic language-independent front end, a module for mapping language codes into language names, and a module which contains various English-language utilities.
* [Stemmer](https://github.com/aurelian/ruby-stemmer) - Expose libstemmer_c to Ruby
* [Ruby Wordnet](http://www.deveiate.org/projects/Ruby-WordNet/) - This library is a Ruby interface to WordNet
* [Raspel](http://sourceforge.net/projects/raspell/) - raspell is an interface binding for ruby
* [UEA Stemmer](https://github.com/ealdent/uea-stemmer) - Ruby port of UEALite Stemmer - a conservative stemmer for search and indexing
* [Twitter-text-rb](https://github.com/twitter/twitter-text-rb) - A library that does auto linking and extraction of usernames, lists and hashtags in tweets

<a name="ruby-general-purpose" />
#### General-Purpose Machine Learning

* [Ruby Machine Learning](https://github.com/tsycho/ruby-machine-learning) - Some Machine Learning algorithms, implemented in Ruby
* [Machine Learning Ruby](https://github.com/mizoR/machine-learning-ruby)
* [jRuby Mahout](https://github.com/vasinov/jruby_mahout) - JRuby Mahout is a gem that unleashes the power of Apache Mahout in the world of JRuby.
* [CardMagic-Classifier](https://github.com/cardmagic/classifier) - A general classifier module to allow Bayesian and other types of classifications.
* [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) - Code samples for my book "Neural Networks and Deep Learning" [DEEP LEARNING]

<a name="ruby-data-analysis" />
#### Data Analysis / Data Visualization

* [rsruby](https://github.com/alexgutteridge/rsruby) - Ruby - R bridge
* [data-visualization-ruby](https://github.com/chrislo/data_visualisation_ruby) - Source code and supporting content for my Ruby Manor presentation on Data Visualisation with Ruby
* [ruby-plot](https://www.ruby-toolbox.com/projects/ruby-plot) - gnuplot wrapper for ruby, especially for plotting roc curves into svg files
* [plot-rb](https://github.com/zuhao/plotrb) - A plotting library in Ruby built on top of Vega and D3.
* [scruffy](http://www.rubyinside.com/scruffy-a-beautiful-graphing-toolkit-for-ruby-194.html) - A beautiful graphing toolkit for Ruby
* [SciRuby](http://sciruby.com/)
* [Glean](https://github.com/glean/glean) - A data management tool for humans
* [Bioruby](https://github.com/bioruby/bioruby)
* [Arel](https://github.com/nkallen/arel)

<a name="ruby-misc" />
#### Misc

* [Big Data For Chimps](https://github.com/infochimps-labs/big_data_for_chimps)
* [Listof](https://github.com/kevincobain2000/listof) - Community based data collection, packed in gem. Get list of pretty much anything (stop words, countries, non words) in txt, json or hash. [Demo/Search for a list](http://listof.herokuapp.com/)

<a name="r" />
## R

<a name="r-general-purpose" />
#### General-Purpose Machine Learning

* [ahaz](http://cran.r-project.org/web/packages/ahaz/index.html) - ahaz: Regularization for semiparametric additive hazards regression
* [arules](http://cran.r-project.org/web/packages/arules/index.html) - arules: Mining Association Rules and Frequent Itemsets
* [bigrf](http://cran.r-project.org/web/packages/bigrf/index.html) - bigrf: Big Random Forests: Classification and Regression Forests for Large Data Sets
* [bigRR](http://cran.r-project.org/web/packages/bigRR/index.html) - bigRR: Generalized Ridge Regression (with special advantage for p >> n cases)
* [bmrm](http://cran.r-project.org/web/packages/bmrm/index.html) - bmrm: Bundle Methods for Regularized Risk Minimization Package
* [Boruta](http://cran.r-project.org/web/packages/Boruta/index.html) - Boruta: A wrapper algorithm for all-relevant feature selection
* [bst](http://cran.r-project.org/web/packages/bst/index.html) - bst: Gradient Boosting
* [C50](http://cran.r-project.org/web/packages/C50/index.html) - C50: C5.0 Decision Trees and Rule-Based Models
* [caret](http://caret.r-forge.r-project.org/) - Classification and Regression Training: Unified interface to ~150 ML algorithms in R.
* [Clever Algorithms For Machine Learning](https://github.com/jbrownlee/CleverAlgorithmsMachineLearning)
* [CORElearn](http://cran.r-project.org/web/packages/CORElearn/index.html) - CORElearn: Classification, regression, feature evaluation and ordinal evaluation
* [CoxBoost](http://cran.r-project.org/web/packages/CoxBoost/index.html) - CoxBoost: Cox models by likelihood based boosting for a single survival endpoint or competing risks
* [Cubist](http://cran.r-project.org/web/packages/Cubist/index.html) - Cubist: Rule- and Instance-Based Regression Modeling
* [e1071](http://cran.r-project.org/web/packages/e1071/index.html) - e1071: Misc Functions of the Department of Statistics (e1071), TU Wien
* [earth](http://cran.r-project.org/web/packages/earth/index.html) - earth: Multivariate Adaptive Regression Spline Models
* [elasticnet](http://cran.r-project.org/web/packages/elasticnet/index.html) - elasticnet: Elastic-Net for Sparse Estimation and Sparse PCA
* [ElemStatLearn](http://cran.r-project.org/web/packages/ElemStatLearn/index.html) - ElemStatLearn: Data sets, functions and examples from the book: "The Elements of Statistical Learning, Data Mining, Inference, and Prediction" by Trevor Hastie, Robert Tibshirani and Jerome Friedman Prediction" by Trevor Hastie, Robert Tibshirani and Jerome Friedman
* [evtree](http://cran.r-project.org/web/packages/evtree/index.html) - evtree: Evolutionary Learning of Globally Optimal Trees
* [fpc](http://cran.r-project.org/web/packages/fpc/index.html) - fpc: Flexible procedures for clustering
* [frbs](http://cran.r-project.org/web/packages/frbs/index.html) - frbs: Fuzzy Rule-based Systems for Classification and Regression Tasks
* [GAMBoost](http://cran.r-project.org/web/packages/GAMBoost/index.html) - GAMBoost: Generalized linear and additive models by likelihood based boosting
* [gamboostLSS](http://cran.r-project.org/web/packages/gamboostLSS/index.html) - gamboostLSS: Boosting Methods for GAMLSS
* [gbm](http://cran.r-project.org/web/packages/gbm/index.html) - gbm: Generalized Boosted Regression Models
* [glmnet](http://cran.r-project.org/web/packages/glmnet/index.html) - glmnet: Lasso and elastic-net regularized generalized linear models
* [glmpath](http://cran.r-project.org/web/packages/glmpath/index.html) - glmpath: L1 Regularization Path for Generalized Linear Models and Cox Proportional Hazards Model
* [GMMBoost](http://cran.r-project.org/web/packages/GMMBoost/index.html) - GMMBoost: Likelihood-based Boosting for Generalized mixed models
* [grplasso](http://cran.r-project.org/web/packages/grplasso/index.html) - grplasso: Fitting user specified models with Group Lasso penalty
* [grpreg](http://cran.r-project.org/web/packages/grpreg/index.html) - grpreg: Regularization paths for regression models with grouped covariates
* [h2o](http://cran.r-project.org/web/packages/h2o/index.html) - A framework for fast, parallel, and distributed machine learning algorithms at scale -- Deeplearning, Random forests, GBM, KMeans, PCA, GLM
* [hda](http://cran.r-project.org/web/packages/hda/index.html) - hda: Heteroscedastic Discriminant Analysis
* [Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
* [ipred](http://cran.r-project.org/web/packages/ipred/index.html) - ipred: Improved Predictors
* [kernlab](http://cran.r-project.org/web/packages/kernlab/index.html) - kernlab: Kernel-based Machine Learning Lab
* [klaR](http://cran.r-project.org/web/packages/klaR/index.html) - klaR: Classification and visualization
* [lars](http://cran.r-project.org/web/packages/lars/index.html) - lars: Least Angle Regression, Lasso and Forward Stagewise
* [lasso2](http://cran.r-project.org/web/packages/lasso2/index.html) - lasso2: L1 constrained estimation aka ‘lasso’
* [LiblineaR](http://cran.r-project.org/web/packages/LiblineaR/index.html) - LiblineaR: Linear Predictive Models Based On The Liblinear C/C++ Library
* [LogicReg](http://cran.r-project.org/web/packages/LogicReg/index.html) - LogicReg: Logic Regression
* [Machine Learning For Hackers](https://github.com/johnmyleswhite/ML_for_Hackers)
* [maptree](http://cran.r-project.org/web/packages/maptree/index.html) - maptree: Mapping, pruning, and graphing tree models
* [mboost](http://cran.r-project.org/web/packages/mboost/index.html) - mboost: Model-Based Boosting
* [mlr](http://cran.r-project.org/web/packages/mlr/index.html) - mlr: Machine Learning in R
* [mvpart](http://cran.r-project.org/web/packages/mvpart/index.html) - mvpart: Multivariate partitioning
* [ncvreg](http://cran.r-project.org/web/packages/ncvreg/index.html) - ncvreg: Regularization paths for SCAD- and MCP-penalized regression models
* [nnet](http://cran.r-project.org/web/packages/nnet/index.html) - nnet: Feed-forward Neural Networks and Multinomial Log-Linear Models
* [oblique.tree](http://cran.r-project.org/web/packages/oblique.tree/index.html) - oblique.tree: Oblique Trees for Classification Data
* [pamr](http://cran.r-project.org/web/packages/pamr/index.html) - pamr: Pam: prediction analysis for microarrays
* [party](http://cran.r-project.org/web/packages/party/index.html) - party: A Laboratory for Recursive Partytioning
* [partykit](http://cran.r-project.org/web/packages/partykit/index.html) - partykit: A Toolkit for Recursive Partytioning
* [penalized](http://cran.r-project.org/web/packages/penalized/index.html) - penalized: L1 (lasso and fused lasso) and L2 (ridge) penalized estimation in GLMs and in the Cox model
* [penalizedLDA](http://cran.r-project.org/web/packages/penalizedLDA/index.html) - penalizedLDA: Penalized classification using Fisher's linear discriminant
* [penalizedSVM](http://cran.r-project.org/web/packages/penalizedSVM/index.html) - penalizedSVM: Feature Selection SVM using penalty functions
* [quantregForest](http://cran.r-project.org/web/packages/quantregForest/index.html) - quantregForest: Quantile Regression Forests
* [randomForest](http://cran.r-project.org/web/packages/randomForest/index.html) - randomForest: Breiman and Cutler's random forests for classification and regression
* [randomForestSRC](http://cran.r-project.org/web/packages/randomForestSRC/index.html) - randomForestSRC: Random Forests for Survival, Regression and Classification (RF-SRC)
* [rattle](http://cran.r-project.org/web/packages/rattle/index.html) - rattle: Graphical user interface for data mining in R
* [rda](http://cran.r-project.org/web/packages/rda/index.html) - rda: Shrunken Centroids Regularized Discriminant Analysis
* [rdetools](http://cran.r-project.org/web/packages/rdetools/index.html) - rdetools: Relevant Dimension Estimation (RDE) in Feature Spaces
* [REEMtree](http://cran.r-project.org/web/packages/REEMtree/index.html) - REEMtree: Regression Trees with Random Effects for Longitudinal (Panel) Data
* [relaxo](http://cran.r-project.org/web/packages/relaxo/index.html) - relaxo: Relaxed Lasso
* [rgenoud](http://cran.r-project.org/web/packages/rgenoud/index.html) - rgenoud: R version of GENetic Optimization Using Derivatives
* [rgp](http://cran.r-project.org/web/packages/rgp/index.html) - rgp: R genetic programming framework
* [Rmalschains](http://cran.r-project.org/web/packages/Rmalschains/index.html) - Rmalschains: Continuous Optimization using Memetic Algorithms with Local Search Chains (MA-LS-Chains) in R
* [rminer](http://cran.r-project.org/web/packages/rminer/index.html) - rminer: Simpler use of data mining methods (e.g. NN and SVM) in classification and regression
* [ROCR](http://cran.r-project.org/web/packages/ROCR/index.html) - ROCR: Visualizing the performance of scoring classifiers
* [RoughSets](http://cran.r-project.org/web/packages/RoughSets/index.html) - RoughSets: Data Analysis Using Rough Set and Fuzzy Rough Set Theories
* [rpart](http://cran.r-project.org/web/packages/rpart/index.html) - rpart: Recursive Partitioning and Regression Trees
* [RPMM](http://cran.r-project.org/web/packages/RPMM/index.html) - RPMM: Recursively Partitioned Mixture Model
* [RSNNS](http://cran.r-project.org/web/packages/RSNNS/index.html) - RSNNS: Neural Networks in R using the Stuttgart Neural Network Simulator (SNNS)
* [RWeka](http://cran.r-project.org/web/packages/RWeka/index.html) - RWeka: R/Weka interface
* [RXshrink](http://cran.r-project.org/web/packages/RXshrink/index.html) - RXshrink: Maximum Likelihood Shrinkage via Generalized Ridge or Least Angle Regression
* [sda](http://cran.r-project.org/web/packages/sda/index.html) - sda: Shrinkage Discriminant Analysis and CAT Score Variable Selection
* [SDDA](http://cran.r-project.org/web/packages/SDDA/index.html) - SDDA: Stepwise Diagonal Discriminant Analysis
* [SuperLearner](https://github.com/ecpolley/SuperLearner) and [subsemble](http://cran.r-project.org/web/packages/subsemble/index.html) - Multi-algorithm ensemble learning packages.
* [svmpath](http://cran.r-project.org/web/packages/svmpath/index.html) - svmpath: svmpath: the SVM Path algorithm
* [tgp](http://cran.r-project.org/web/packages/tgp/index.html) - tgp: Bayesian treed Gaussian process models
* [tree](http://cran.r-project.org/web/packages/tree/index.html) - tree: Classification and regression trees
* [varSelRF](http://cran.r-project.org/web/packages/varSelRF/index.html) - varSelRF: Variable selection using random forests
* [XGBoost.R](https://github.com/tqchen/xgboost/tree/master/R-package) - R binding for eXtreme Gradient Boosting (Tree) Library


<a name="r-data-analysis" />
#### Data Analysis / Data Visualization

* [Learning Statistics Using R](http://health.adelaide.edu.au/psychology/ccs/teaching/lsr/)
* [ggplot2](http://ggplot2.org/) - A data visualization package based on the grammar of graphics.

<a name="scala" />
## Scala

<a name="scala-nlp" />
#### Natural Language Processing

* [ScalaNLP](http://www.scalanlp.org/) - ScalaNLP is a suite of machine learning and numerical computing libraries.
* [Breeze](https://github.com/scalanlp/breeze) - Breeze is a numerical processing library for Scala.
* [Chalk](https://github.com/scalanlp/chalk) - Chalk is a natural language processing library.
* [FACTORIE](https://github.com/factorie/factorie) - FACTORIE is a toolkit for deployable probabilistic modeling, implemented as a software library in Scala. It provides its users with a succinct language for creating relational factor graphs, estimating parameters and performing inference.

<a name="scala-data-analysis" />
#### Data Analysis / Data Visualization

* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Scalding](https://github.com/twitter/scalding) - A Scala API for Cascading
* [Summing Bird](https://github.com/twitter/summingbird) - Streaming MapReduce with Scalding and Storm
* [Algebird](https://github.com/twitter/algebird) - Abstract Algebra for Scala
* [xerial](https://github.com/xerial/xerial) - Data management utilities for Scala
* [simmer](https://github.com/avibryant/simmer) - Reduce your data. A unix filter for algebird-powered aggregation.
* [PredictionIO](https://github.com/PredictionIO/PredictionIO) - PredictionIO, a machine learning server for software developers and data engineers.
* [BIDMat](https://github.com/BIDData/BIDMat) - CPU and GPU-accelerated matrix library intended to support large-scale exploratory data analysis.
* [Wolfe](http://www.wolfe.ml/) Declarative Machine Learning

<a name="scala-general-purpose" />
#### General-Purpose Machine Learning

* [Conjecture](https://github.com/etsy/Conjecture) - Scalable Machine Learning in Scalding
* [brushfire](https://github.com/avibryant/brushfire) - decision trees and random forests for scalding
* [ganitha](https://github.com/tresata/ganitha) - scalding powered machine learning
* [adam](https://github.com/bigdatagenomics/adam) - A genomics processing engine and specialized file format built using Apache Avro, Apache Spark and Parquet. Apache 2 licensed.
* [bioscala](https://github.com/bioscala/bioscala) - Bioinformatics for the Scala programming language
* [BIDMach](https://github.com/BIDData/BIDMach) - CPU and GPU-accelerated Machine Learning Library.
* [Figaro](https://github.com/p2t2/figaro) - a Scala library for constructing probabilistic models.
* [h2o-sparkling](https://github.com/0xdata/h2o-sparkling) - H2O and Spark interoperability.

<a name="swift" />
## Swift

<a name="swift-general-purpose" />
#### General-Purpose Machine Learning
* [swix](https://github.com/scottsievert/swix) - A bare bones library that
  includes a general matrix language and wraps some OpenCV for iOS development.

<a name="credits" />
## Credits

* Some of the python libraries were cut-and-pasted from [vinta](https://github.com/vinta/awesome-python)
* The few go reference I found where pulled from [this page](https://code.google.com/p/go-wiki/wiki/Projects#Machine_Learning)
