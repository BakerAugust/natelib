## Implementation

### Batch encoding
The sequences of training data are split into batches. Each batch is encoded into a `Y` and `X` tensor where `Y` is batch_size x sentence_length x tags and `X` is batch_size x sentence_length x vocab. The sentence length is set by the `max_sequence_length` parameter.

### Parameter estimation
Parameter tensors `t_feats` for transition features and `e_feats` for emission features are created for all possible transitions (tags x tags) and emissions (tags x vocab). Values are initiated at 0.

Parameter weights are updated by stochastic gradient descent. The loss is calculated as the `gold_logscore` less the `Z` from a forward pass. Here, I calculate the forward pass without a full construction of an alpha table. Instead, the forward pass simply accumulates values leading to a result equivalent to the final column of an alpha table.

The `loss.backward` function is used to calculate the gradients for the parameter tensors. Parameters are then updated by adding `param.grad * learning_rate`.

This implementation does not include any regularization term to limit overfitting. 

## Usage 

### Train

```
$ python hw3/crf.py train "/u/cs248/data/pos/train"
```
This will write an output file `my_weights` with all of the feature weights. 

### Test
```
$ python hw3/crf.py test "my_weights" "/u/cs248/data/test"
```

### Professor Gildea's test case
To run against the toy test case Professor Gildea shared, simply run

```
$ python3 hw3/crf.py toy
```