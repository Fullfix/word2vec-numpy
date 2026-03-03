# Word2Vec CBOW Implementation in Pure Numpy

This repository provides an implementation of the Word2Vec Continuous Bag-of-Words (CBOW) model using only NumPy. It is designed to train on the WikiText-103 dataset and includes evaluation for word similarity and analogy tasks.

## Training Details and Performance Analysis

My model **wikitext-256** was trained with the following settings:

- Hidden Size: 256
- Dataset: WikiText-103 (filtered to 34 million tokens after sub-sampling)
- Training Volume: 122 million tokens total (approximately 4 epochs)
- Method: CBOW with Negative Sampling

Evaluation consists of:
*   **Word Similarity:** `WordSim-353`, `SimLex-999`, and `MEN-3k`. These measure the Spearman correlation between the cosine similarity of model vectors and human assessment.
*   **Analogy Task:** The Google Analogy dataset, which tests the model's ability to solve "A is to B as C is to D" using vector arithmetic (v_B - v_A + v_C).

The following table compares **wikitext-256** model against standard pre-trained GloVe embeddings, original Word2Vec model and FastText. For WordSim-353, SimLex-999, and MEN, I report the Spearman correlation coefficient (with coverage in parentheses). For Google Analogies, I report Top-1 accuracy.

| Model | WordSim-353      | SimLex-999       | MEN Similarity (3k) | Google Analogies |
| :--- |:-----------------|:-----------------|:--------------------|:-----------------|
| glove-twitter-100 | 0.482 (94.6%)    | 0.120 (99.9%)    | 0.573 (100%)        | 0.442 (95.7%)    |
| glove-twitter-200 | 0.491 (94.6%)    | 0.128 (99.9%)    | 0.589 (100%)        | 0.549 (95.7%)    |
| glove-wiki-gigaword-100 | 0.487 (94.9%)    | 0.298 (100%)     | 0.693 (100%)        | 0.631 (100%)     |
| glove-wiki-gigaword-200 | 0.523 (94.9%)    | 0.340 (100%)     | 0.724 (100%)        | 0.698 (100%)     |
| glove-wiki-gigaword-300 | 0.550 (94.9%)    | 0.371 (100%)     | 0.749 (100%)        | 0.717 (100%)     |
| word2vec-google-news-300 | 0.688 (100%)     | **0.442** (100%) | 0.782 (98.2%)       | 0.736 (100%)     |
| fasttext-wiki-news-300 | **0.718** (100%) | 0.441 (100%)     | **0.803** (100%)    | **0.873** (100%) |
| **wikitext-256 (mine)** | 0.678 (94.9%)    | 0.361 (99.6%)    | 0.724 (100%)        | 0.361 (91.9%)    |

My model has competitive WordSim-353, SimLex-999 and MEN scores. Google Analogies accuracy is relatively low, likely due to short training duration.

### Nearest Neighbors Sanity Check

| Query Word | Top-5 Nearest Neighbors (Cosine Similarity) |
|:---|:---|
| **man** | woman (0.449), whom (0.350), soldier (0.339), bearded (0.331), manicured (0.325) |
| **physics** | physicists (0.479), mathematics (0.462), mesons (0.460), caltech (0.447), neutrino (0.445) |
| **boy** | blob (0.429), girl (0.420), kid (0.364), teenaged (0.363), young (0.362) |
| **london** | southwark (0.536), lambeth (0.490), piccadilly (0.468), islington (0.456), whitechapel (0.444) |
| **january** | december (0.776), february (0.740), march (0.733), april (0.714), july (0.682) |
| **computer** | computers (0.592), mainframe (0.464), software (0.462), computing (0.455), desktop (0.453) |

## Usage

### Requirements
```bash
pip install numpy scipy tqdm datasets
```

### Training
To train Word2Vec CBOW, run:

```bash
python train.py
```

### Using the Model
```python
from train import Word2VecInferenceModel

model = Word2VecInferenceModel.load('./data/model.npz')
vector = model.W_in[model.vocab['example']]
```

### Validating other Models
Validation of external models requires gensim dependency:
```bash
pip install gensim
```

To validate, run:
```bash
python validate.py
```

### Training Hyperparameters
The following are the main constants in the script that can be adjusted to change model behavior:
*   `W2V_HIDDEN_SIZE`: Size of the embedding vectors.
*   `TRAIN_BATCH_SIZE`: Number of center tokens per SGD step.
*   `TRAIN_WINDOW_SIZE`: Window size (number of tokens to the left and right from center).
*   `TRAIN_NEGATIVE_SAMPLES`: Number of tokens to sample as negatives per positive sample.
*   `TRAIN_LEARNING_RATE`: Initial learning rate for the linear decay scheduler.
*   `TRAIN_RANDOM_WINDOW_SIZE`: Whether to select random window size at each step from 1 to `TRAIN_WINDOW_SIZE`.
*  `TRAIN_STEPS`: Number of training steps.
*  `TRAIN_EVAL_EVERY_STEPS`: Evaluate on benchmarks every N training steps.

For more information, check `train.py`.

## Implementation Details

### Frequent Word Sub-sampling
In large corpora, the most frequent words (e.g., "the", "is", "in") provide less information than rare words. I perform sub-sampling to improve both training speed and the quality of the embeddings. For each word in the training set, I keep it with the following probability, and otherwise discard it:

$$P(\text{keep}) = \left( \sqrt{\frac{f(w)}{t}} + 1 \right) \cdot \frac{t}{f(w)}$$

Where:
*   **$f(w)$**: The frequency of the word in the corpus (number of occurrences divided by total tokens).
*   **$t$**: The `TRAIN_DISCARD_THRESHOLD` (e.g., $10^{-5}$).


### Negative Sampling Table
Calculating the full softmax over a large vocabulary is expensive, so typically *negative sampling* is preferred: instead of predicting the exact word, the model solves binary classification task, which consists of 1 positive example (center token) and sampled negative tokens.

The negative samples are drawn with the following probability:

$$P(w_i) = \frac{count(w_i)^{0.75}}{\sum count(w_j)^{0.75}}$$

To keep the sampling process cheap during training, I precompute a large `negative_sample_table` of `NEG_SAMPLE_TABLE_SIZE` (default: 10 million) entries, allowing for rapid index lookups.

### Dynamic Window Size

This implementation supports dynamic window size with flag `TRAIN_RANDOM_WINDOW_SIZE`. In dynamic mode, the window size is sampled uniformly from 1 to `TRAIN_WINDOW_SIZE` at each step. However, I found that it decreases the performance, so by default I use fixed window size.

### Technical details

- Training token IDs are precomputed to avoid string processing overhead during the training loop.
- Updates are performed using efficient `np.add.at` operations.
- Large batch sizes (default: 4096) are used to leverage efficient NumPy vectorized operations.
- Training is currently *single-threaded*, with a future intention to implement multi-threading.
