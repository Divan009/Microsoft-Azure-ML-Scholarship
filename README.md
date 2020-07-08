# Microsoft-Azure-ML-Scholarship

_Basically convert everything into numbers_

## Scaling Data
Let's consider an example. Imagine you have an image represented as a set of RGB values ranging from 0 to 255. We can scale the range of the values from 0–255 down to a range of 0–1. This scaling process will not affect the algorithm output since every value is scaled in the same way. But it can speed up the training process, because now the algorithm only needs to handle numbers less than or equal to 1.
There are 2 approaches to it: **standardization** and **normalization**.

**Standardization _rescales data so that it has a mean of 0 and a standard deviation of 1_** 
(𝑥 − 𝜇)/𝜎   We subtract the mean (𝜇) from each value (x) and then divide by the standard deviation (𝜎)

**Normalization _rescales the data into the range [0, 1]._**
(𝑥 −𝑥𝑚𝑖𝑛)/(𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛)    For each individual value, you subtract the minimum value (𝑥𝑚𝑖𝑛) for that input in the training dataset, and then divide by the range of the values in the training dataset. The range of the values is the difference between the maximum value (𝑥𝑚𝑎𝑥) and the minimum value (𝑥𝑚𝑖𝑛).



## Encoding Categorical Data
When we have categorical data, we need to encode it in some way so that it is represented numerically.

#### Ordinal Encoding
In ordinal encoding, we simply convert the categorical data into integer codes ranging from 0 to (number of categories – 1).

**drawbacks** 
This approach is that it implicitly assumes an order across the categories. In the above example, Blue (which is encoded with a value of 2) seems to be more than Red (which is encoded with a value of 1), even though this is in fact not a meaningful way of comparing those values. This is not necessarily a problem, but it is a reason to be cautious in terms of how the encoded data is used.

#### One-Hot Encoding
we transform each categorical value into a column. If there are n categorical values, n new columns are added. 

##  Image Data
If you zoom in on an image far enough, you can see that it consists of small tiles, called pixels.

- In **grayscale images**, each pixel can be represented by a single number, which typically ranges from 0 to 255. This value determines how dark the pixel appears (e.g., 0 is black, while 255 is bright white).

- In **colored images**, each pixel can be represented by a vector of three numbers (each ranging from 0 to 255) for the three primary color channels: red, green, and blue(RGB).

The _number of channels_ required to represent the color is known as the **color depth** or simply depth. With an RGB image, depth = 3, because there are three channels (Red, Green, and Blue). In contrast, a grayscale image has depth = 1, because there is only one channel.

#### Encoding an Image
We need to know the following three things about an image to reproduce it:

- Horizontal position of each pixel
- Vertical position of each pixel
- Color of each pixel

The size of the vector required for any given image would be the **height * width * depth** of that image.

#### Assumptions
We would want to ensure that the input images have a _uniform aspect ratio_ (e.g., by making sure all of the input images are square in shape) and are _normalized_ (e.g. subtract mean pixel value in a channel from each pixel value in that channel)

## Text Data

#### Normalization
One of the challenges that can come up in text analysis is that there are often multiple forms that mean the same thing. 
For example, the verb _to be_ may show up as _is, am, are_, and so on. Or a document may contain alternative spellings of a word, such as _behavior vs. behaviour_. So one step that you will sometimes conduct in processing text is normalization.

Text **normalization** is the _process of transforming a piece of text into a canonical (official) form._

**Lemmatization** is an example of normalization. A lemma is the dictionary form of a word and lemmatization is the process of reducing multiple inflections to that single dictionary form. For example, we can apply this to the is, am, are example we mentioned above:
| Original Word      | Lemmatized word |
| ----------- | ----------- |
| is     | be       |
| are   | be      |
| a |  be |

**Stop words** are high-frequency words that are unnecessary (or unwanted) during the analysis. So, remove them.

Here we have **tokenized** the text (i.e., split each string of text into a list of smaller parts or tokens), _removed_ stop words (the), and _standardized spelling_ (changing lazzy to lazy).

#### Vectorization

After we have normalized the text, we can identify the particular features of the text that will be relevant to us for the particular task we want to perform—and then get those features extracted in a numerical form

- The approach of **TF-IDF** is to give less importance to words that contain less information and are common in documents, such as "the" and "this"—and to give higher importance to words that contain relevant information and appear less frequently. 

Read more
[Term Frequency-Inverse Document Frequency (TF-IDF) vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
Word embedding, as done with [Word2vec](https://en.wikipedia.org/wiki/Word2vec) or [Global Vectors (GloVe)](https://nlp.stanford.edu/pubs/glove.pdf)

#### Pipeline for Text Data
- In summary, a typical pipeline for text data begins by pre-processing or normalizing the text. This step typically includes tasks such as breaking the text into sentence and word tokens, standardizing the spelling of words, and removing overly common words (called stop words).
- The next step is feature extraction and vectorization, which creates a numeric representation of the documents. Common approaches include TF-IDF vectorization, Word2vec, and Global Vectors (GloVe).
- Last, we will feed the vectorized document and labels into a model and start the training.

![preprocessing](https://video.udacity-data.com/topher/2020/February/5e361fb2_text/text.png)

## Computer science vs. Statistical perspective
A computer scientist might say something like:
_We are using input features to create a program that can generate the desired output._
For the **rows** in the table, we might call each row an entity or an observation about an entity. In our example above, each entity is simply a product, and when we speak of an observation, we are simply referring to the data collected about a given product. You'll also sometimes see a row of data referred to as an instance, in the sense that a row may be considered a single example (or instance) of data.

For the **columns** in the table, we might refer to each column as a feature or attribute which describes the property of an entity. In the above example, color and quantity are features (or attributes) of the products.

In contrast, someone with a background in statistics might be inclined to say something more like:
_We are trying to find a mathematical function that, given the values of the independent variables can predict the values of the dependent variables_.

## Models vs. Algorithms

**Models** are the _**specific representations** learned from data_

**Algorithms** are the _processes of learning_

We can think of the algorithm as a function—we give the algorithm data and it produces a model:

_Model=Algorithm(Data)_

We can think of an algorithm as a _mathematical tool_ that can usually be represented by an _equation_ as well as implemented in code. For example, y = Wx + b is an algorithm that can be used to calculate y from x if the values for W and b are known. But how do we get W and b?
This is the learning part of machine learning; That is, we can learn these values from training data.

## Linear Regression

**y = mx + b**

In algebraic terms, we may refer to **m** as the coefficient of x or simply the _slope of the line_, and we may call **b** the _y-intercept_. In machine learning, you will typically see the _y-intercept_ referred to as the **bias**. 

##### The Cost Function
Notice from our example of test scores earlier that the line we came up with did not perfectly fit the data. In fact, most of the data points were not on the line!
When we predict that a student who studies for 10 hours will get a score of 153, we do not expect their score to be exactly 153. Put another way, when we make a prediction using the line, we expect the prediction to have some error.

The **process of finding the best model is essentially a process of finding the coefficients and bias that minimize this error**. To calculate this error, we use a cost function. There are many cost functions you can choose from to train a model and the resulting error will be different depending one which cost function you choose. The most commonly used cost function for linear regression is the root mean squared error (RMSE)

We choose a cost function (like RMSE) to calculate the error and then _minimize that error_ in order to arrive at a line of best fit that models the training data and can be used to make predictions.

#### How to Prepare Data
- **Linear assumption**: linear regression describes variables using a line. So the relationship between the input variables and the output variable needs to be a linear relationship. If the raw data does not follow a linear relationship, you may be able to transform your data prior to using it with the linear regression algorithm. For example, if your data has an exponential relationship, you can use _log transformation_.
- **Remove collinearity**: When two variables are collinear, this means they can be modeled by the same line or are at least highly correlated; in other words, one input variable can be accurately predicted by the other. For example, suppose we want to predict education level using the input variables number of years studying at school, if an individual is male, and if an individual is female. In this case, we will see collinearity—the input variable if an individual is female can be perfectly predicted by if an individual is male, thus, we can say they are highly correlated. Having highly correlated input variables will make the model less consistent, so it's important to perform a correlation check among input variables and remove highly correlated input variables.
- **Gaussian (normal) distribution**: Linear regression assumes that the distance between output variables and real data (called residual) is normally distributed. If this is not the case in the raw data, you will need to first _transform the data_ so that the residual has a normal distribution.
- **Rescale data**: Linear regression is very sensitive to the distance among data points, so it's always a good idea to _normalize or standardize the data_.
- **Remove noise**: Linear regression is very sensitive to noise and outliers in the data. Outliers will significantly change the line learned. Thus, cleaning the data is a critical step prior to applying linear regression.

The training process is a process of minimizing the error
















