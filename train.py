from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import re
import os
import json
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset, Dataset


@dataclass
class Word2VecTrainingModel:
    W_in: np.ndarray  # V x D
    W_out: np.ndarray  # V x D

    @staticmethod
    def random_init(vocab_size: int, hidden_size: int, rng: np.random.Generator) -> Word2VecTrainingModel:
        W_in = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(hidden_size),
            size=(vocab_size, hidden_size)
        ).astype(np.float32)
        W_out = np.zeros((vocab_size, hidden_size), dtype=np.float32)
        return Word2VecTrainingModel(W_in, W_out)


@dataclass(frozen=True)
class Word2VecInferenceModel:
    W_in: np.ndarray
    vocab: dict[str, int]
    lowercase: bool

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        vocab_json = json.dumps(
            self.vocab,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False
        ).encode("utf-8")

        np.savez_compressed(
            path,
            W_in=self.W_in,
            vocab_json=np.frombuffer(vocab_json, dtype=np.uint8),
            lowercase=np.array([1 if self.lowercase else 0], dtype=np.uint8),
            format_version=np.array([1], dtype=np.int32),
        )

    @staticmethod
    def load(path: str) -> Word2VecInferenceModel:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            if "W_in" not in data or "vocab_json" not in data:
                raise ValueError(f"Invalid model file (missing keys): {path}")

            W_in = np.asarray(data["W_in"])
            vocab_bytes = data["vocab_json"].astype(np.uint8).tobytes()
            vocab = json.loads(vocab_bytes.decode("utf-8"))

            if "lowercase" in data:
                lowercase = bool(int(np.asarray(data["lowercase"]).reshape(-1)[0]))
            else:
                lowercase = False

        return Word2VecInferenceModel(W_in=W_in, vocab=vocab, lowercase=lowercase)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def word2vec_cbow_step(model: Word2VecTrainingModel, x: np.ndarray, y: np.ndarray, lr: float, negative_sample_table: np.ndarray, negative_samples: int) -> float:
    """
    :param model: Word2vec model
    :param x: context words ids: B x C
    :param y: target (middle) words ids: B
    :param lr: learning rate
    :param negative_sample_table: table of negative samples for negative sampling: N
    :param negative_samples: number of negative samples to compute (K)
    :return: mean loss value
    """

    V, D = model.W_in.shape
    B, C = x.shape
    K = negative_samples

    W_avg = model.W_in[x].mean(axis=1)  # B x D

    neg_ids = negative_sample_table[np.random.randint(0, negative_sample_table.shape[0], size=(B, K))].astype(y.dtype)  # B x K

    all_ids = np.concatenate([y.reshape(-1), neg_ids.reshape(-1)], axis=0)  # B(K+1)
    unique_ids, id_inv = np.unique(all_ids, return_inverse=True)
    W_out_snap = model.W_out[unique_ids].copy()

    inv_y = id_inv[:B]  # B
    inv_neg = id_inv[B:].reshape(B, K)  # B x K

    W_out_pos = W_out_snap[inv_y]  # B x D
    W_out_neg = W_out_snap[inv_neg]  # B x K x D

    p_pos = sigmoid(np.sum(W_out_pos * W_avg, axis=1)).astype(W_avg.dtype)  # B
    p_neg = sigmoid(np.einsum("bd,bkd->bk", W_avg, W_out_neg)).astype(W_avg.dtype)  # B x K

    upd_pos = (p_pos - 1.0)[:, None] * W_avg
    upd_neg = (p_neg[:, :, None] * W_avg[:, None, :]).reshape(-1, D)  # BK x D
    idx_neg = inv_neg.reshape(-1)  # BK

    upd_snap = np.zeros((unique_ids.shape[0], D), dtype=model.W_out.dtype)
    np.add.at(upd_snap, inv_y, upd_pos)
    np.add.at(upd_snap, idx_neg, upd_neg)

    np.add.at(model.W_out, unique_ids, -lr * upd_snap)

    dh = (p_pos - 1.0)[:, None] * W_out_pos + np.einsum("bk,bkd->bd", p_neg, W_out_neg)  # B x D

    flat_x = x.ravel()  # B * C
    flat_dh = np.repeat(dh, C, axis=0)
    np.add.at(model.W_in, flat_x, -(lr / C) * flat_dh)

    # Stable log
    eps = 1e-15
    loss = -(np.log(np.clip(p_pos, eps, 1.0 - eps)) + np.log(np.clip(1.0 - p_neg, eps, 1.0 - eps)).sum(axis=1)).mean()
    return loss


def word2vec_cbow_loss(model: Word2VecTrainingModel, x: np.ndarray, y: np.ndarray, negative_sample_table: np.ndarray, negative_samples: int) -> float:
    """
    :param model: Word2vec model
    :param x: context words ids: B x C
    :param y: target (middle) words ids: B
    :param negative_sample_table: table of negative samples for negative sampling: N
    :param negative_samples: number of negative samples to compute (K)
    :return: mean loss value
    """

    B, C = x.shape
    K = negative_samples

    W_avg = model.W_in[x].mean(axis=1)  # B x D
    W_out_pos = model.W_out[y]  # B x D

    neg_ids = negative_sample_table[np.random.randint(0, negative_sample_table.shape[0], size=(B, K))].astype(y.dtype)  # B x K
    W_out_neg = model.W_out[neg_ids]  # B x K x D

    s_pos = np.sum(W_out_pos * W_avg, axis=1)
    s_neg = np.einsum("bd,bkd->bk", W_avg, W_out_neg)

    log_pos = -np.logaddexp(0.0, -s_pos)  # B
    log_neg = -np.logaddexp(0.0, s_neg).sum(axis=1)  # B

    loss = -(log_pos + log_neg).mean()
    return loss


def _validate_similarity(model: Word2VecInferenceModel, triples: list[tuple[str, str, float]]) -> tuple[float, float]:
    """
    Validate Word2vec on similarity dataset
    :param model: Word2vec inference model
    :param triples: dataset of triples (word1, word2, similarity score)
    :return: (spearman correlation coefficient, coverage ratio)
    """

    W = model.W_in
    W_norms = np.linalg.norm(W, axis=1) + 1e-12

    human_scores = []
    model_scores = []

    covered_pairs = 0

    for w1, w2, score in triples:
        id1 = model.vocab.get(w1, None)
        id2 = model.vocab.get(w2, None)

        if id1 is None or id2 is None:
            continue

        cosine = np.dot(W[id1], W[id2]) / (W_norms[id1] * W_norms[id2])

        human_scores.append(score)
        model_scores.append(cosine)
        covered_pairs += 1

    coverage = covered_pairs / len(triples)
    rho, _ = spearmanr(np.asarray(human_scores), np.asarray(model_scores))

    return float(rho), float(coverage)


def word2vec_validate_wordsim353(model: Word2VecInferenceModel) -> tuple[float, float]:
    """
    Validate Word2vec on WordSim-353
    :param model: Word2vec inference model
    :return: (spearman correlation coefficient, coverage ratio)
    """

    ds = load_dataset("almogtavor/WordSim353")
    triples = []
    for row in ds['train']:
        triples.append((row['Word 1'], row['Word 2'], row['Human (Mean)']))

    return _validate_similarity(model, triples)


def word2vec_validate_simlex999(model: Word2VecInferenceModel) -> tuple[float, float]:
    """
    Validate Word2vec on SimLex-999
    :param model: Word2vec inference model
    :param vocab: vocabulary (map from word to id)
    :return: (spearman correlation coefficient, coverage ratio)
    """

    ds = load_dataset("tasksource/simlex")
    triples = []
    for row in ds['train']:
        triples.append((row['word1'], row['word2'], row['SimLex999']))

    return _validate_similarity(model, triples)


def word2vec_validate_men3k(model: Word2VecInferenceModel) -> tuple[float, float]:
    """
    Validate Word2vec on MEN-3k
    :param model: Word2vec inference model
    :return: (spearman correlation coefficient, coverage ratio)
    """

    ds = load_dataset("Yuti/MEN-word-similarity")
    triples = []
    for row in ds['train']:
        triples.append((row['word1'], row['word2'], row['score']))

    return _validate_similarity(model, triples)


def word2vec_validate_google_analogies(
    model: Word2VecInferenceModel,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """
    Validate Word2vec on Google-Analogy dataset
    :param model: Word2vec inference model
    :param batch_size: number of rows per batch
    :return:
    """

    ds = load_dataset("almogtavor/google-analogy-dataset")["train"]

    W = model.W_in
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    total = len(ds)
    covered = 0
    correct = 0

    def get_token_id(w: str) -> int:
        if model.lowercase:
            w = w.lower()
        return model.vocab.get(w, 0)

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        rows = ds.select(range(start, end))

        w1_ids = []
        w2_ids = []
        w3_ids = []
        w4_ids = []

        for row in rows:
            i1 = get_token_id(row["Word1"])
            i2 = get_token_id(row["Word2"])
            i3 = get_token_id(row["Word3"])
            i4 = get_token_id(row["Word4"])

            if i1 > 0 and i2 > 0 and i3 > 0 and i4 > 0:
                w1_ids.append(i1); w2_ids.append(i2); w3_ids.append(i3); w4_ids.append(i4)

        if len(w1_ids) > 0:
            w1_ids = np.asarray(w1_ids, dtype=np.int32)
            w2_ids = np.asarray(w2_ids, dtype=np.int32)
            w3_ids = np.asarray(w3_ids, dtype=np.int32)
            w4_ids = np.asarray(w4_ids, dtype=np.int32)

            B = w1_ids.shape[0]

            q = W_norm[w2_ids] - W_norm[w1_ids] + W_norm[w3_ids]  # B x D
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

            scores = q @ W_norm.T  # B x V
            scores[np.arange(B), w1_ids] = -np.inf
            scores[np.arange(B), w2_ids] = -np.inf
            scores[np.arange(B), w3_ids] = -np.inf

            pred = np.argmax(scores, axis=1).astype(np.int32)  # B

            covered += B
            correct += int(np.sum(pred == w4_ids))

    coverage = covered / total
    acc = correct / covered
    return float(acc), float(coverage)


def word2vec_validate_wikitext(
        ds: Dataset,
        model: Word2VecTrainingModel,
        vocab_regex: re.Pattern[str],
        vocab: dict[str, int],
        vocab_lowercase: bool,
        window_size: int,
        negative_sample_table: np.ndarray,
        negative_samples: int,
        ds_batch_size: int = 1000,
        verbose: bool = True,
):
    """
    Validate Word2Vec on WikiText 103
    :param ds: Wikitext ds
    :param model: Word2Vec inference model
    :param vocab_regex: Regex for token split
    :param vocab: Word2Vec vocabulary
    :param vocab_lowercase: whether to lowercase the words
    :param window_size: fixed window size for validation
    :param negative_sample_table: table of negative samples
    :param negative_samples: number of negative tokens to sample
    :param ds_batch_size: batch size for dataset iteration
    :param verbose: verbosity flag
    :return:
    """

    val_total_loss = 0.0
    val_total_batches = 0
    for batch in tqdm(
            ds.iter(batch_size=ds_batch_size),
            desc='Validating on WikiText',
            total=(len(ds) + ds_batch_size - 1) // ds_batch_size,
            disable=not verbose,
            leave=False
    ):
        for text in batch['text']:
            tokens = vocab_regex.findall(text.lower() if vocab_lowercase else text)
            token_ids = [idx for t in tokens if (idx := vocab.get(t, 0)) > 0]

            if len(token_ids) > 2 * window_size:
                x = []
                y = []
                for i in range(window_size, len(token_ids) - window_size):
                    center = token_ids[i]
                    if center > 0:
                        context = token_ids[i - window_size:i] + token_ids[i + 1:i + 1 + window_size]  # 2W
                        x.append(context)
                        y.append(center)

                if len(x) > 0 and len(y) > 0:
                    x = np.asarray(x, dtype=np.int32)  # B x C
                    y = np.asarray(y, dtype=np.int32)  # B
                    loss = word2vec_cbow_loss(model, x, y, negative_sample_table, negative_samples=negative_samples)
                    val_total_loss += loss
                    val_total_batches += 1

    val_loss = val_total_loss / val_total_batches
    return val_loss


def main():
    VERBOSE = True
    DS_BATCH_SIZE = 1000
    VOCAB_MIN_COUNT = 10
    VOCAB_LOWERCASE = True
    VOCAB_REGEX = re.compile(
        r"""
        ==+                             |  # wiki headings
        \.\.\.                          |  # ellipsis
        [a-z]+(?:'[a-z]+)?              |  # words
        \d+(?:\.\d+)?                   |  # numbers
        [^\s]                              # punctuation
        """,
        re.VERBOSE,
    )
    W2V_HIDDEN_SIZE = 256
    NEG_SAMPLE_TABLE_POWER = 0.75
    NEG_SAMPLE_TABLE_SIZE = 10_000_000
    TRAIN_BATCH_SIZE = 4096
    TRAIN_EVAL_EVERY_STEPS = 1000
    TRAIN_STEPS = 30_000
    TRAIN_LEARNING_RATE = 0.2
    TRAIN_WINDOW_SIZE = 5
    TRAIN_NEGATIVE_SAMPLES = 10
    TRAIN_DISCARD_THRESHOLD = 1e-5
    TRAIN_RANDOM_WINDOW_SIZE = False
    EVAL_BATCH_SIZE = 1024
    MODEL_SAVE_PATH = './data/model.npz'

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Build vocab
    counter = Counter()
    token_count = 0
    for batch in tqdm(
            ds['train'].iter(batch_size=1000),
            desc="Building vocabulary",
            total=(len(ds['train']) + DS_BATCH_SIZE - 1) // DS_BATCH_SIZE,
            disable=not VERBOSE
    ):
        for text in batch['text']:
            tokens = VOCAB_REGEX.findall(text.lower() if VOCAB_LOWERCASE else text)
            counter.update(tokens)
            token_count += len(tokens)

    items = [(word, count) for word, count in counter.items() if count >= VOCAB_MIN_COUNT]
    items.sort(key=lambda x: (-x[1], x[0]))

    vocab = {'<unk>': 0}
    for idx, (word, _) in enumerate(items, start=1):
        vocab[word] = idx

    print(f'Built vocabulary of length: {len(vocab)}')

    # Build negative sampling table
    table_counts = np.zeros(len(vocab), dtype=np.float32)
    keep_probs = np.ones(len(vocab), dtype=np.float32)

    for word, count in tqdm(items, desc='Iterating over counts', disable=not VERBOSE):
        table_counts[vocab[word]] = count

        freq = count / token_count
        p_keep = np.clip((np.sqrt(freq / TRAIN_DISCARD_THRESHOLD) + 1) * (TRAIN_DISCARD_THRESHOLD / freq), 0.0, 1.0)
        keep_probs[vocab[word]] = p_keep

    table_weights = table_counts ** NEG_SAMPLE_TABLE_POWER
    table_probs = table_weights / table_weights.sum()

    rng = np.random.default_rng(seed=69)
    table = rng.choice(
        np.arange(len(vocab), dtype=np.int32),
        size=NEG_SAMPLE_TABLE_SIZE,
        p=table_probs
    )

    train_ids = np.zeros(token_count, dtype=np.int32)
    train_ids_idx = 0
    for batch in tqdm(
            ds['train'].iter(batch_size=1000),
            desc='Precomputing token ids',
            total=(len(ds['train']) + DS_BATCH_SIZE - 1) // DS_BATCH_SIZE,
            disable=not VERBOSE
    ):
        for text in batch['text']:
            tokens = VOCAB_REGEX.findall(text.lower() if VOCAB_LOWERCASE else text)
            for token in tokens:
                train_ids[train_ids_idx] = vocab.get(token, 0)
                train_ids_idx += 1

    discard_mask = rng.random(len(train_ids)) > keep_probs[train_ids]
    train_ids[discard_mask] = 0

    train_ids = train_ids[train_ids > 0]
    print(f'Kept {len(train_ids)}/{token_count} ({len(train_ids) / token_count:5.1%}) tokens')

    train_valid_mask = (train_ids != 0)
    train_valid_mask[:TRAIN_WINDOW_SIZE] = False
    train_valid_mask[-TRAIN_WINDOW_SIZE:] = False

    if TRAIN_RANDOM_WINDOW_SIZE:
        all_offsets = [
            np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
            for w in range(1, TRAIN_WINDOW_SIZE + 1)
        ]
    else:
        all_offsets = [
            np.concatenate([np.arange(-TRAIN_WINDOW_SIZE, 0), np.arange(1, TRAIN_WINDOW_SIZE + 1)])
        ]

    train_valid_centers = np.nonzero(train_valid_mask)[0]
    rng.shuffle(train_valid_centers)

    # Train Word2Vec
    model = Word2VecTrainingModel.random_init(vocab_size=len(vocab), hidden_size=W2V_HIDDEN_SIZE, rng=rng)

    running_loss = 0.0
    running_loss_batches = 0

    center_step = 0

    for step in tqdm(range(TRAIN_STEPS), desc="Training", disable=not VERBOSE):
        if step % TRAIN_EVAL_EVERY_STEPS == 0:
            train_loss = running_loss / running_loss_batches if running_loss_batches > 0 else 0.0
            running_loss = 0.0
            val_loss = word2vec_validate_wikitext(
                ds=ds['validation'],
                model=model,
                vocab_regex=VOCAB_REGEX,
                vocab=vocab,
                vocab_lowercase=VOCAB_LOWERCASE,
                window_size=TRAIN_WINDOW_SIZE,
                negative_sample_table=table,
                negative_samples=TRAIN_NEGATIVE_SAMPLES,
                ds_batch_size=DS_BATCH_SIZE,
                verbose=VERBOSE
            )

            model_inf = Word2VecInferenceModel(W_in=model.W_in, vocab=vocab, lowercase=VOCAB_LOWERCASE)
            wordsim353_val = word2vec_validate_wordsim353(model_inf)
            simlex999_val = word2vec_validate_simlex999(model_inf)
            men3k_val = word2vec_validate_men3k(model_inf)
            google_analogies_val = word2vec_validate_google_analogies(model_inf, batch_size=EVAL_BATCH_SIZE)

            tqdm.write(
                f"[step {step:>7d}/{TRAIN_STEPS:<7d} | {(step/TRAIN_STEPS):6.2%}]  "
                f"train_loss {train_loss:8.4f}  |  "
                f"val_loss: {val_loss:8.4f}  |  "
                f"WS353: {wordsim353_val[0]:6.3f} (cov {wordsim353_val[1]:5.1%})  |  "
                f"SimLex: {simlex999_val[0]:6.3f} (cov {simlex999_val[1]:5.1%})  |  "
                f"MEN: {men3k_val[0]:6.3f} (cov {men3k_val[1]:5.1%})  |  "
                f"GA: {google_analogies_val[0]:6.3f} (cov {google_analogies_val[1]:5.1%})"
            )

        lr = TRAIN_LEARNING_RATE * (1 - step / TRAIN_STEPS)

        if (center_step + 1) * TRAIN_BATCH_SIZE > len(train_valid_centers):
            if VERBOSE:
                tqdm.write('Reshuffling centers')
            center_step = 0
            rng.shuffle(train_valid_centers)

        centers = train_valid_centers[center_step*TRAIN_BATCH_SIZE : (center_step+1)*TRAIN_BATCH_SIZE]  # B
        center_step += 1

        offsets_idx = rng.integers(0, len(all_offsets))
        offsets = all_offsets[offsets_idx]
        context_positions = centers[:, None] + offsets[None, :]  # B x 2W

        x = train_ids[context_positions]
        y = train_ids[centers]

        loss = word2vec_cbow_step(model, x, y, lr=lr, negative_sample_table=table, negative_samples=TRAIN_NEGATIVE_SAMPLES)

        running_loss += loss
        running_loss_batches += 1

    print('Finished training. Performing final validation')

    val_loss = word2vec_validate_wikitext(
        ds=ds['validation'],
        model=model,
        vocab_regex=VOCAB_REGEX,
        vocab=vocab,
        vocab_lowercase=VOCAB_LOWERCASE,
        window_size=TRAIN_WINDOW_SIZE,
        negative_sample_table=table,
        negative_samples=TRAIN_NEGATIVE_SAMPLES,
        ds_batch_size=DS_BATCH_SIZE,
        verbose=VERBOSE
    )
    test_loss = word2vec_validate_wikitext(
        ds=ds['test'],
        model=model,
        vocab_regex=VOCAB_REGEX,
        vocab=vocab,
        vocab_lowercase=VOCAB_LOWERCASE,
        window_size=TRAIN_WINDOW_SIZE,
        negative_sample_table=table,
        negative_samples=TRAIN_NEGATIVE_SAMPLES,
        ds_batch_size=DS_BATCH_SIZE,
        verbose=VERBOSE
    )

    model_inf = Word2VecInferenceModel(W_in=model.W_in, vocab=vocab, lowercase=VOCAB_LOWERCASE)
    wordsim353_val = word2vec_validate_wordsim353(model_inf)
    simlex999_val = word2vec_validate_simlex999(model_inf)
    men3k_val = word2vec_validate_men3k(model_inf)
    google_analogies_val = word2vec_validate_google_analogies(model_inf, batch_size=EVAL_BATCH_SIZE)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    print("\nWikiText-103:")
    print(f"  Validation Loss : {val_loss:8.4f}")
    print(f"  Test Loss       : {test_loss:8.4f}")

    print("\nWord Similarity (Spearman ρ | Coverage):")
    print(f"  WordSim-353     : {wordsim353_val[0]:6.3f}  |  {wordsim353_val[1]:5.1%}")
    print(f"  SimLex-999      : {simlex999_val[0]:6.3f}  |  {simlex999_val[1]:5.1%}")
    print(f"  MEN-3k          : {men3k_val[0]:6.3f}  |  {men3k_val[1]:5.1%}")

    print("\nGoogle Analogies:")
    print(f"  Accuracy        : {google_analogies_val[0]:6.3f}")
    print(f"  Coverage        : {google_analogies_val[1]:5.1%}")

    print("=" * 80 + "\n")

    model_inf.save(MODEL_SAVE_PATH)
    print(f'Saved final model to {MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()
