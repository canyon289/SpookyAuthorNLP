# Machine Learning Engineer Nanodegree
## Capstone Project - Spooky Author
Ravin Kumar
November 19th, 2017

## I. Definition
### Spooky Author Classification
Given a sentence is it possible to predict who wrote it? In one line that's the summarization of this project. Kaggle,
a data science platform, provided a corpus of text from three authors, HP Lovecraft, Edgar Allen Poe and
Mary Shelley, in two files, one with labels, and one without. Given this quote
"But a glance will show the fallacy of this idea.", the goal of the project is to correctly identify the author
that wrote it, in this case Edgar Allen Poe.

As well as provided the datasets Kaggle also provides a platform for sharing code and a leaderboard for rankings.
Although not required, after downloading the dataset and making predictions, the resultant predictions can be
uploaded back to Kaggle to see how they fare against others models.


### Problem Statement
The specific problem is "Given a string of words is it possible to classify the author of the text as Edgar Allen Poe,
Mary Shelley, or HP Lovecraft".

The problem will be split into approximately three chunks, Feature Engineering, Predictor Tuning, and Code Implementation

#### Feature Engineering
The dataset provided only comes with "base" feature column, which is a string of text. Fortunately the class of
problem is very well known, and is typically referred to as a Natural Language Processing problem. In this class
of problem, a raw string is translated into features using various transformations, such as Term Frequency Inverse
Document Frequency, Word Count Length, Word Stemmers etc.

#### Predictor Tuning
After features have been defined the next step is apply statistical predictors to the dataset. Numerous algorithms
exist but in particular the two that will be used here are Multinomial Gaussian Bayes and Xgboost. Both are models
that are known to work well in multiclass classification, and most importantly are able to provide class probability
which is necessary for th metric below.

#### Code Implementation
Often overlooked but a large part of machine learning problems is code implementation. While the steps may be
easy to discuss, in practice what ends up happening is "spaghetti code" ends up being the result a long running
machine learning project, and it becomes difficult for someone else to approach the codebase and continue working on it.

This occurs because unlike traditional programming, machine learning projects have a large iterative component to them.
The coder is typically trying things as they're going. What starts as a simple series of steps ends up getting convoluted
into a random assortment of predictions, data transformations, and matrix indexing, that at the end outputs
either a tabular dataset, or a picture.

Having being caught in that mess above numerous times, for me one large component of this project was to learn
how to structure a machine learning project so it is maintainable and so it would be more obvious to an outside
user what portion of the code does what function.
In this section, you will want to clearly define

### Metrics
Kaggle provides the metrics used to score predictions, in this case the metric chosen was the Log Loss metric. Log Loss
takes an array of probabilities, and essentially measures how separated the predictions for the classes are from
each other. Log Loss is a fairly common metric and used extensively in many areas, such as deep learning, as it
favors class separability, not just correct predictions.

Another softer metric I am using for this project though is "How maintainable is my code?". Again one of the objectives
I have for this project is to ensure I am continuing to write better libraries. While in a Kaggle competition
getting the top prize is all that's measured, in organizations it is necessary for the code to be maintainable for
a longer lifespan than just one prediction.

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
Taken from the project proposal.
**id26305, The surcingle hung in ribands from my body., EAP**
In this string the ID is labeled with id26305, and the author is labeled as Edgar Allen Poe.
In the string we can see some interesting words such as surgcingle and ribands.
The hops is by analyzing the writing styles, word usage, and phrasing, of thousands
of these strings we can id the authors without needing labels.

The dataset is roughly balanced with 7900 samples from Edgar Allen Poe,
6044 from Mary Shelley, and 5635 from HP Lovecraft. When making test
train splits the stratified shuffle split will be used to preserve
the class ratios from the train set.


In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
**Add more visualizations here later**
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
#### Techniques
For text feature extraction I chose to use two types of techniques, "off the shelf" text processors
such as "Term Frequency Inverse Document Frequency", Word Stemmers, and hand built features,
such as word count, string length etc.

The off the shelf features are chosen from prior research on Natural Language Processing. For exa
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem.
Many of these can be found in the aptly named "text" module of sklearn, which provides a series of tools for working
with textual data.

##### Term Frequency Inverse Document Frequency
Term Frequency Inverse Document Frequency is a method where each document is turned into a long vector, where each
column represents the number of times the term appears in that document, divided by the total words. Techniques like
these typically return what's called Sparse Matrices, as most of the columns have zero values. This technique was chosen
as it's easy to implement from sklearn and included as part of the package as it is commonly used in similar problems.

#### Word Stemmers
In natural language processing it is sometimes beneficial to unintuively made the text less readable. Take
for example the following sentences
* I like dogs
* I like a dog
* I like cats

To a human it is clear the first two sentences are more related than the third because they refer to dogs, but to a
computer the word dogs and dog are totally different. If the above string was passed into a TDIDF transformer
you would see that there would be one column generated for dog, and another for dogs. As is such another processing
step is used which is called word stemming. In this technique the strings would be preprocessed so that the word
"dogs" would be turned into the word "dog". Applying this technique helps computers determine relevant topics
more easily.

##### Hand Built Features
In addition to pre built algorithms, I also decided to hand build a couple of features, such as string length,
and word count. The reason for this was two fold. One is that these features would perhaps be predictive of author.
But more compelling to me was to learn how to build custom transformers using sklearns base Transformer classes.
As we will be discussed later, sklearn provides a number of well designed apis to handle data, such as the Transformer
Patten, and Predictor pattern, and these patterns are later abstracted using Pipelines and Feature Unions. As
part of this project I wanted to learn how to properly extend the sklearn package as I was hoping it would lead
to more maintainable code.


### Predictors
#### Naive Bayes
Naive Bayes family of algorithms are an easy to understand, easy to implement, classifier. In this particular
problem we will be using the MultiNominal Naive Bayes classifier as it can return probability of class
for all classes possible.

Naive Bayes Classifiers work by starting "naively" with an uninformed perspective of the world. During training
features are provided to the model, as well as the correct class label, and the model updates its prior distribution to
reflect the new information. For example in real life, if you went to a a restaurant for the first time
you would won't be sure whether it's good or not. The first time you eat there the dish might be really good, and
you'll update your opinion to "good!". The next time you eat there the dish might also be good, in which case
this reinforces your opinion on the establishment. Naive Bayes works similarly.

Naive Bayes models are user friendly as they have very few parameters, train efficiently, and have been shown
to have good performance in practice.

#### XGboost
XGBoost models are an implementation of boosted CART trees. In a quick summary the XGboost model is very popular
as they have been shown to have extremely good performance with classification tasks. Typically in Kaggle competitions
as well there are numerous tutorials that use XGboost models so to take advantage of other peoples experience,
it is helpful to use XGboost as well to compare to other folks performance.

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution.
 The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
