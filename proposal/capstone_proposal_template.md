# Machine Learning Engineer Nanodegree
## Capstone Proposal
Joe Udacity  
December 31st, 2050

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_
Every writer has a different style. Whether it's what words they use, or how
the words are put together, there is a difference in the way one author
writes in comparison to another.

Text analysis of authors is important **Because it is**. Here's research
proving it is



In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

This project will explore if it is possible to tell the works of three
horror authors apart by analyzing short strings of their writing. The project
is being hosted by Kaggle as part of their Kaggle kernel competitions.
Kaggle has provided a dataset of samples of writing from HP Lovecraft, **someone**,
**someone else**. Using a training set with labeled strings and authors
it is left to the machine learning engineer to train a model that is able
to correctly predict the author in the absence of labels. Kaggle is evaluating
the submissions based on the Log Loss criteria.

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_
Kaggle provides two datasets, a train dataset and a test dataset. The
train dataset has three columns, a random ID, a string of characters,
and a label indicating the author. The test dataset is similar but does
not contain labels.

Further details on the train and test datasets can be found in the
exploratory analysis folder

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

The metrics I will use to optimize the model is a log loss score, also called
entropy loss. Log Loss takes a vector of the probability of each class
and given the correct class label, measures the seperation of likelihood
between the correct class and the others. As it is a standard measure
of multiclass classification performance Kaggle will be evaluating
all its models using this criteria as well.

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_
I will be attempting to use 
natural language processing techniques such as Term Frequency Inverse 
Document Frequency analysis, or cosine similarity, to try and train
informed models. Analysis of the text will be required to determine what
preprocessing will be needed, such as stop words, or length of ngrams.

The project will follow a standard Data Science model pipeline.
**Insert picture here**. The training data will be loaded and split
into training and test examples immediately. After this step the training
data will be preprocessed and a undetermined model, or models, will be trained.
After model training, the model will be used to predict authors of 
the test split of the dataset. After adequete model performance has been
achieved, the model will be run on the provided test data to provide
a final set of predictions which will be uploaded to Kaggle.

The solution is clearly defined, especially as Kaggle
holds a validation set that will be used to judge model performance.


In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
