# Deepbow
A metric for automatic assessment of text simplification quality.

**Working principle:** multilayer perceptron with two hidden layers, receiving the original text and simplified output encoded with bag-of-words vectors (each followed with its own embedding layer)
and predicting the four outputs (grammaticality,
meaning preservation, level of simplification and an overall assessment) jointly with the same network.

The implementation uses Python 3.5 and [Keras](http://keras.io) as the neural networks library.

## Performance at [QATS](http://qats2016.github.io/)
The metric was submitted to the LREC 2016 Shared Task on Quality Assessment for Text Simplification. The results on the QATS test set:

### Accuracies

**Grammaticality:**

**Meaning:**

**Simplicity:**

**Overall:**

### Weighted F-scores

**Grammaticality:**

**Meaning:**

**Simplicity:**

**Overall:**

### Confusion Matrices

Horizontal: predicted, vertical: reference

**Grammaticality:**

         | Bad   | Ok    | Good
         | ----- | ----- | -----
**Bad**  |       |       |      
**Ok**   |       |       |      
**Good** |       |       |      

**Meaning:**

         | Bad   | Ok    | Good
         | ----- | ----- | -----
**Bad**  |       |       |      
**Ok**   |       |       |      
**Good** |       |       |      

**Simplicity:**

         | Bad   | Ok    | Good
         | ----- | ----- | -----
**Bad**  |       |       |      
**Ok**   |       |       |      
**Good** |       |       |      

**Overall:**

         | Bad   | Ok    | Good
         | ----- | ----- | -----
**Bad**  |       |       |      
**Ok**   |       |       |      
**Good** |       |       |      
