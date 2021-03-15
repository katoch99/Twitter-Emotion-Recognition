# Twitter Emotion Recognition using RNN

---

1.  **INTRODUCTION**

Emotions are considered of utmost importance as they have a key responsibility in human interaction. Nowadays, social media plays a pivotal role in the interaction of people all across the world. Such social media posts can be effectively analysed for emotions. Twitter is a microblogging service where worldwide users publish and share their feelings. However, sentiment analysis for Twitter messages (‘tweets’) is regarded as a challenging problem because tweets are short and informal. With the use of Recurrent Neural Networks, a model is created and trained to learn to recognize emotions in tweets. The dataset has thousands of tweets each classified in one of 6 emotions – love, fear, joy, sadness, surprise 
and anger. Using TensorFlow as the machine learning framework, this multi class classification problem of the natural language processing domain is solved.

Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.
Sentiment analysis is the process of retrieving information about a consumer’s perception of a product, service or brand. Social media sentiment analysis applies natural language processing (NLP) to analyse online mentions and determine the feelings behind the post. Social sentiment analysis will tell whether the post was positive, negative, or neutral.

A basic task in sentiment analysis is classifying the polarity of a given text at the document, sentence, or feature/aspect level—whether the expressed opinion in a document, a sentence or an entity feature/aspect is positive, negative, or neutral. Advanced, "beyond polarity" sentiment classification looks, for instance, at emotional states such as enjoyment, anger, disgust, sadness, fear, and surprise.

In this project with the use of Recurrent Neural Networks, a model is created and trained to learn to recognize emotions in tweets. The dataset has thousands of tweets each classified in one of 6 emotions – love, fear, joy, sadness, surprise and anger. Using TensorFlow as the machine learning framework, this multi class classification problem of the natural language processing domain is solved.

---

2.  **TOOLS USED**

  **TensorFlow**

TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google.

  **Keras**

Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible. It was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), and its primary author and maintainer is François Chollet, a Google engineer. Chollet also is the author of the XCeption deep neural network model. 

Keras is based on minimal structure that provides a clean and easy way to create deep learning models based on TensorFlow or Theano. Keras is designed to quickly define deep learning models. Well, Keras is an optimal choice for deep learning applications.


  **NumPy**

NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. The ancestor of NumPy, Numeric, was originally created by Jim Hugunin with contributions from several other developers. In 2005, Travis Oliphant created NumPy by incorporating features of the competing Numarray into Numeric, with extensive modifications. NumPy is open-source software and has many contributors.

NumPy targets the CPython reference implementation of Python, which is a non-optimizing bytecode interpreter. Mathematical algorithms written for this version of Python often run much slower than compiled equivalents. NumPy addresses the slowness problem partly by providing multidimensional arrays and functions and operators that operate efficiently on arrays, requiring rewriting some code, mostly inner loops, using NumPy.


  **Matplotlib**

Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+. There is also a procedural "pylab" interface based on a state machine (like OpenGL), designed to closely resemble that of MATLAB, though its use is discouraged. SciPy makes use of Matplotlib. Several toolkits are available which extend Matplotlib functionality. Some are separate downloads, others ship with the Matplotlib source code but have external dependencies.

Pyplot is a Matplotlib module which provides a MATLAB-like interface. Matplotlib is designed to be as usable as MATLAB, with the ability to use Python and the advantage of being free and open-source. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc. The various plots we can utilize using Pyplot are Line Plot, Histogram, Scatter, 3D Plot, Image, Contour, and Polar.


  **NLP**

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The result is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

HuggingFace/NLP is an open library of NLP datasets. the HuggingFace nlp library. HuggingFace nlp provides datasets for many NLP tasks like text classification, question answering, language modeling, etc and obviously these datasets can always be used to other tasks than their originally assigned task.


 **Jupyter**

Project Jupyter is a non-profit organization created to "develop open-source software, open-standards, and services for interactive computing across dozens of programming languages". Spun-off from IPython in 2014 by Fernando Pérez, Project Jupyter supports execution environments in several dozen languages. Project Jupyter's name is a reference to the three core programming languages supported by Jupyter, which are Julia, Python and R, and also a homage to Galileo's notebooks recording the discovery of the moons of Jupiter. Project Jupyter has developed and supported the interactive computing products Jupyter Notebook, JupyterHub, and JupyterLab, the next-generation version of Jupyter Notebook.

As a server-client application, the Jupyter Notebook App allows to edit and run notebooks via a web browser. The application can be executed on a PC without Internet access, or it can be installed on a remote server, where it can be accessed through the Internet.

---

3.  **DATASET**

The dataset consists of 20,000 tweets with their corresponding emotion. The dataset is already pre-processed and divided into the training, test and validation set. Each tweet is based on an emotion in one of the six categories – love, fear, joy, sadness, surprise and anger.

The training set consists of 16,000 tweets, the test set consists of 2,000 tweets and the validation set also consists of 2,000 tweets. The dataset is stored in a pickle file which takes 47.6 MB of space on disk.

This Emotion Dataset was prepared by Elvis Saravia and published on GitHub.

---

4.  **METHODOLOGY**

The RNN model is built using Google Collab environment which runs the Jupyter notebook in the cloud. It uses the TensorFlow as the machine learning framework. First all the necessary libraries are imported. Then the dataset is imported and assigned to the corresponding data object. The text pre-processing functions are done using the built-in tokenizer of TensorFlow and all the words in the dataset are assigned to a specific token. Next, the tokens are padded and truncated so that the model gets input of fixed shape. Then we create a dictionary for converting the name off the classes to their corresponding index. The text labels for the different classes are passes to get them as numeric representations. The 
sequential model is created using four different layers. The model is then trained and evaluated.

1.  **Installing Hugging Face’s nlp package**

The Hugging Face’s nlp package is installed using the following command: -

!pip install nlp 

2.  **Importing the libraries**

The following libraries are imported: -

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import nlp

import random

import warnings (To hide some unnecessary warnings shown by Jupyter)

Further some modules from the Tensorflow library are also imported.


3.  **Importing the Dataset**

The Emotion Dataset is imported using the nlp package. The dataset is already divided into test, training and validation sets. Each set has text and label features. There are 16,000 tweets in the training set, 2,000 tweets in the test set and 2,000 tweets in the validation set. Each tweet also has its corresponding emotion with it. These three sets are assigned to their respective objects. A function is also defined to get the text and label keys from the dataset. The first tweet and its corresponding label of the training set on which the model is going to be trained is displayed.

![image](https://user-images.githubusercontent.com/68529704/111146771-cffa7200-85af-11eb-9ce6-89be4bf2d79e.png)


4.  **Tokenizing the Tweets**

The TensorFlow comes with a built-in Tokenizer library which is imported from its text pre-processing module. Tokenization randomly generates a token value for plain text and stores the mapping in a database. The words of tweets need to be tokenized so that each word can be represented as a number to feed into the model and the model is able to train on the data. The tokenizer basically creates a corpus (collection) of all the words that exist in the dataset and give each unique word a unique corresponding token. A limit is also set to how many most frequently words are to be organized and the rest less commonly used words are given a common token called out of vocabulary which is basically an unknown word token.

An object tokenizer is created which tokenizes the most frequently used 10,000 words from the text corpus and assigns an unknown token (<UNK>) to the remaining words. Then the words  from the tweets from the training set are mapped to the numeric tokens using fit_on_texts function. Using the texts_to_sequences function we can see that the tweets have been tokenized.

![image](https://user-images.githubusercontent.com/68529704/111146918-fae4c600-85af-11eb-9fe9-0f06765d53a1.png)


5.  **Padding and Truncating Sequences**

The sequences generated from the Tokenizer need to be padded and truncated because the model requires a fixed input size. The tweets in the dataset are of different length of words and thus it is required for them to be padded or truncated. The length of the tweets is calculated by counting the number of words separated by a space. A histogram is plotted to get the most common lengths of tweets in the dataset.

![image](https://user-images.githubusercontent.com/68529704/111144796-8dd03100-85ad-11eb-93f5-c275b9c667e9.png)

Most of the tweets in the dataset are about 10 to 20 words long. There are very few tweets which are less than 4 words and also very few tweets of length 50 words or more.

A maximum length of 50 is set to truncate any tweets over the length of 50 words. Any tweet which has less than 50 words is padded with ‘0’ in its token sequence. This is done using the pad_sequences function from the TensorFlow library. Both truncating and padding is done ‘post’ which means that the function will remove or add words from the end of the token sequence to get the sequence length to 50. This will get all the tweets to a fixed input size.

![image](https://user-images.githubusercontent.com/68529704/111144827-99bbf300-85ad-11eb-879b-c6fce0be99d5.png)


6.  **Preparing the Labels**

There is a need of different numeric values for the different classes for multi class classifications. The classes are created using the labels from the training set. The six classes which represent the different emotions are - anger, joy, love, surprise, fear and sadness.

A histogram is plotted to see the number of tweets for the different classes.

![image](https://user-images.githubusercontent.com/68529704/111147117-28ca0a80-85b0-11eb-8174-8c1ee27b00a0.png)

Two dictionaries are created to convert the names of the classes to their corresponding numeric values. A lambda function is also created to convert the labels of the tweets of the training set to numeric representations.


7.  **Creating the Model**

A sequential model is created using keras. Recurrent Neural Network (RNN) is a deep learning algorithm that is specialized for sequential data. In a RNN the neural network gains information from the previous step in a loop. The output of one unit goes into the next one and the information is passed. 

But RNNs are not good for training large datasets. During the training of RNN, the information goes in loop again and again which results in very large updates to neural network model weights which lead to the accumulation of error gradients during the update and the network becomes unstable. At an extreme, the values of weights can become so large as to overflow and result in NaN values. The explosion occurs through exponential growth by repeatedly multiplying gradients through the network layers that have values larger than 1 or vanishing occurs if the values are less than 1.

To overcome this problem Long Short-Term Memory is used. LSTM can capture long-range dependencies. It can have memory about previous inputs for extended time durations. There are 3 
gates in an LSTM cell – Forget, Input and Output Gate.

•	Forget Gate: Forget gate removes the information that is no longer useful in the cell state.

•	Input Gate: Additional useful information to the cell state is added by input gate.

•	Output Gate: Additional useful information to the cell state is added by output gate.

Memory manipulations in LSTM are done using these gates. Long short-term memory (LSTM) utilizes gates to control the gradient propagation in the recurrent network’s memory. This gating mechanism of LSTM has allowed the network to learn the conditions for when to forget, ignore, or keep information in the memory cell.

![image](https://user-images.githubusercontent.com/68529704/111144967-c53edd80-85ad-11eb-9962-d30bfdb1c423.png)

The fist layer of the model is Embedding layer. Its input dimension is 10,000 (most commonly used words in the dataset) and output dimension is 16 which will be the size of the output vectors from this layer for each word. The input length of sequence is going to be the maximum length which is 50.

LSTM preserves information from inputs that has already passed through it using the hidden state. Unidirectional LSTM only preserves information of the past because the only inputs it has seen are from the past. Bidirectional LSTM will run the inputs in two ways, one from past to future and one from future to past and what differs this approach from unidirectional is that in the LSTM that runs backwards information from the future is preserved and using the two hidden states combined it is able in any point in time to preserve information from both past and future.

The second is a bidirectional LSTM layer. This means that the contents from the LSTM layer can go for both left to right and right to left. Its 20 cells (each cell has its own inputs, outputs and memory) are used and return sequence is set to true which means that every time there will be an output which will be fed into another bidirectional LSTM layer it is sent as a sequence rather than a single value of each input so that the subsequent LSTM layer can have the required input.

The final layer will be a Dense layer with 6 units for the six classes present and the activation is set to softmax which returns a probability distribution over the target classes.

The model is compiled with loss set to ‘sparse_categorical_crossentropy’ as it is used for multi class classification problems as the classes are not one-hot encoded (for binary classes). The optimizer used is ‘adam’ as it is really efficient for working with large datasets. The training metrics used is accuracy which calculates how often the predictions are equal to the actual labels. The model summary is generated.

![image](https://user-images.githubusercontent.com/68529704/111145017-d12a9f80-85ad-11eb-9d81-b94d62cd86bb.png)


8.  **Training the Model**

The validation set is prepared and its sequences are generated. Its labels are also converted to their corresponding numerical representation.

The model is then trained for 15 epochs. The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset. An early stopping callback is also set which stops the training if the model does not see any improvement in the validation accuracy for over 2 epochs. The model training to be completed takes only about 4 minutes as the Jupyter notebook service is hosted on Google Colab which is using a GPU for accelerated computation.

![image](https://user-images.githubusercontent.com/68529704/111147598-abeb6080-85b0-11eb-8dcb-536ae5250e73.png)


9.  **Evaluating the Model**

Plots are generated for the accuracy and loss for the training and validation set over epochs.

>Accuracy per epoch plot
![image](https://user-images.githubusercontent.com/68529704/111147770-da693b80-85b0-11eb-9fb1-310aa02d2bbc.png)

>Loss per epoch plot
![image](https://user-images.githubusercontent.com/68529704/111147807-e3f2a380-85b0-11eb-8f38-7de6e29dd647.png)

The training accuracy increased consistently and the validation accuracy plateaued and that’s when the training stopped. The training and validation loss are both decreasing gradually.

The test set is also prepared and the model is evaluated over it. Some predictions are also checked manually from the test set.

![image](https://user-images.githubusercontent.com/68529704/111147845-eead3880-85b0-11eb-9ae8-b594d72a20c4.png)

The model achieves 88.80% accuracy on the test set which is very similar to the accuracy achieved on the validation dataset.

---

5.  **OUTCOME**

Some predictions are checked manually from the test set against the actual class label. The tweet with its actual and predicted emotion are printed. About 9 out of every 10 predictions are correct which matches with the model accuracy rate.

![image](https://user-images.githubusercontent.com/68529704/111147912-04226280-85b1-11eb-96c2-d3030357b488.png)

![image](https://user-images.githubusercontent.com/68529704/111147919-07b5e980-85b1-11eb-8e48-02e985e4439e.png)

![image](https://user-images.githubusercontent.com/68529704/111147932-0b497080-85b1-11eb-8142-833b34085b98.png)

![image](https://user-images.githubusercontent.com/68529704/111147945-0edcf780-85b1-11eb-8a53-ca7050461077.png)

![image](https://user-images.githubusercontent.com/68529704/111147958-13a1ab80-85b1-11eb-8714-457631c20d1a.png)

---

6.  **IMPORTANCE OF SOCIAL MEDIA SENTIMENT ANALYSIS**

  **Improved customer satisfaction**
  
One should use sentiment analysis to improve the overall customer service. Thanks to analysing positive, negative, or neutral social mentions, one can identify the strong and weak points of any offering. Social listening will help you spot the customers pain points and solve their problems almost in real-time. Reaching out to people who may have a negative experience with the brand can help show how much one cares about them. Turning an unhappy customer into satisfied one helps the business thrive.

  **Understanding the audience**

A listening tool will help spot positive and negative mentions. A thorough social media monitoring and analytics will help better understanding of the audience. Social channels can be used to analyse the feelings of its followers towards a brand, product, or service. Ongoing sentiment analysis will make sure that the messaging of the brand is always in line with its followers needs. Analysing the sentiment during product launch will quickly tell whether the launch was a success or not.

  **Prevent social media crisis**
  
One negative mention can start an avalanche of complaints. In the era of Internet trolls, some users might be complaining even if they never had a chance to use the product. But if one is able to catch the original complaint early on and solve the problem, a social media crisis might be averted. Addressing complaints at the early stage will prevent the crisis from escalating and will protect the brand reputation.

  **Measure the results of a PR campaign**

Social media analytics is the most important part of every social media campaign. And social media sentiment analysis is a right addition to improve the social media marketing efforts. Sentiment analysis will tell what the target audience thinks about the campaign. Generating buzz and counting impressions is not the most crucial part of any campaign. Reaching the right audience with a positive message is.

  **Improving the product according to the customer needs**
  
With the use of sentiment analysis, one is able to spot the problem right at the source. The problem can be eradicated before it escalates and one can solve precisely the issues the customers want to be addressed. Negative sentiment can also give valuable insights into the products features. Taking a more in-depth look into all the negative mentions and finding out what the customers are complaining about the most is a very crucial task. Negative mentions will indicate the most important features that need to be improved and thus contribute in quickly improving and making the customer experience better.

---

7.  **CONCLUSION**

In this project, a RNN model is constructed to recognize the emotions in tweets. The Model produces an accuracy rate of about 89%.

![image](https://user-images.githubusercontent.com/68529704/111148123-451a7700-85b1-11eb-9e8e-831a5b388949.png)

All the predictions are also evaluated against all the ground truths using the test set. A confusion matrix is generated for the test labels in the against the actual classes.

![image](https://user-images.githubusercontent.com/68529704/111145728-95dca080-85ae-11eb-9be1-fc7319701797.png)

Mostly the model produces an accurate result but as observed from the confusion matrix the most common misclassification seems to be joy and love classes and also fear and surprise classes. This can be fixed by balancing the number of tweets in all the emotions.

For further enhancements in future, a much larger dataset with more epochs can be used to increase the accuracy.
