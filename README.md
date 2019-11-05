# Naive-Bayes-for-Incremental-Data
Develop a Naive Bayes Classifier(Python), which works for Incremental Data, and uses running mean and variance to update Gaussian Probability Distributions

#### Description:
1. Build a Naive Bayes Classifier
2. Data is incremental, that is we don't have all the data beforehand, but it comes in batches
3. data_generator.pyc file added in 01 Data, which uses a secret function to generate blackbox data, which can be queried at intervals
4. We use Gaussian Distribution for posterior probability calculation
5. Dynamically update mean and variance as we see new data
6. Current code quereis dat from blackbox in intervals of 10 rows

![Naive Bayes Classifier](naive_bayes.jpeg)

#### References:
1. Image: https://chrisalbon.com/
2. https://math.stackexchange.com/questions/106700/incremental-averageing
3. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
