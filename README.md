# Microsoft-Azure-ML-Scholarship

#### Scaling Data
Let's consider an example. Imagine you have an image represented as a set of RGB values ranging from 0 to 255. We can scale the range of the values from 0–255 down to a range of 0–1. This scaling process will not affect the algorithm output since every value is scaled in the same way. But it can speed up the training process, because now the algorithm only needs to handle numbers less than or equal to 1.
There are 2 approaches to it: **standardization** and **normalization**.

**Standardization _rescales data so that it has a mean of 0 and a standard deviation of 1_** 
(𝑥 − 𝜇)/𝜎   We subtract the mean (𝜇) from each value (x) and then divide by the standard deviation (𝜎)

**Normalization _rescales the data into the range [0, 1]._**
(𝑥 −𝑥𝑚𝑖𝑛)/(𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛)    For each individual value, you subtract the minimum value (𝑥𝑚𝑖𝑛) for that input in the training dataset, and then divide by the range of the values in the training dataset. The range of the values is the difference between the maximum value (𝑥𝑚𝑎𝑥) and the minimum value (𝑥𝑚𝑖𝑛).



