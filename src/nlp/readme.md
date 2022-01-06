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

## Results

### Parameter estimation
My implementation has no trouble getting to 100% accuracy on the "toy" example provided by Professor Gildea. It's worth noting that my logscores are considerable different than those shown in the example and fail to converge.

When training on the full `train` set, it is quickly stalling-out around 24-30% accuracy. Even when running dozens of epochs on a small subset of the training data (e.x. 100 sentences), the accuracy plateaus around the same place. I would expect a working model to easily overfit the data in that circumstance. 

Interestingly, the weights from training appear sensible relative to eachother. The attached `my_weights` contains weights from 2 epochs on the entire training set. A sample is shown below:

```
T_VBN_RP 0.005139059852808714
T_VBN_VBN 0.018533281981945038
T_VBN_VBP -0.000556912156753242
T_VBN_, 0.028536930680274963
T_VBN_JJR 0.00022771085787098855
T_VBN_NNPS -0.0008455130737274885
T_VBN_VBD 0.009713182225823402
T_VBN_EX -0.0008408243302255869
...
E_._spenders -3.7214832104837114e-08
E_._deference -3.705890350147456e-08
E_._. 1.0868686437606812
E_._UNR -3.639341628058901e-08
E_._swat -3.721224572927895e-08
```

Features which I would expect to have zero weight are instead push to very small negative values.

### Test performance
On the test set, `my_weights` produce an accuracy of 26.0% -- very similar to what is observed during the training process. 

Examples:
```
['No', ',', 'it', 'was', "n't", 'Black', 'Monday', '.']
['NN', ',', 'DT', 'NN', 'IN', 'DT', 'NN', '.']) 

['Some', '``', 'circuit', 'breakers', "''", 'installed', 'after', 'the', 'October', '1987', 'crash', 'failed', 'their', 'first', 'test', ',', 'traders', 'say', ',', 'unable', 'to', 'cool', 'the', 'selling', 'panic', 'in', 'both', 'stocks', 'and', 'futures', '.']
['IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', ',', 'DT', 'NN', ',', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', '.']
```

The predicted tags show a consistent pattern of correct labels on common punction (`,`, `.`) and most other tokens are assigned one of [`NN`, `DT` ,`IN`], despite the fact that emissions weights often are higher for other labels.

The weight for "Some", for example, is 0.0393 for `DT` (correct label) and -5.7092e-05 for `NN` (assigned label). The transition features from "START" to `DT` and `NN` are both ~-.010. 


## Thoughts on debugging
Given that my logscores are considerably different from those shown in the toy case, I suspect there is something wrong with how I am calculating both the `alpha` (`all_logscore`) and `gold_logscore`. Given several reviews of the high-level scoring logic, I believe root cause may actually be something to do with how the onehot encoded tensors are selecting features. The encodings themselves seem at least consistent as they pass a roundtrip test of encoding --> decoding... ¯\_(ツ)_/¯.
 