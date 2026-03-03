import numpy as np
from tqdm.auto import tqdm
from train import (
    Word2VecInferenceModel,
    word2vec_validate_simlex999,
    word2vec_validate_wordsim353,
    word2vec_validate_men3k,
    word2vec_validate_google_analogies
)
import gensim.downloader as api


def _kv_to_model_and_vocab(kv, lowercase: bool) -> Word2VecInferenceModel:
    W_in = np.asarray(kv.vectors, dtype=np.float32)  # (V, D)
    vocab = dict(kv.key_to_index)  # word -> row index
    return Word2VecInferenceModel(W_in, vocab, lowercase=lowercase)


def get_word2vec_models() -> dict[str, Word2VecInferenceModel]:
    """
    Get word2vec models and vocabulary
    :return: list of tuples (model, vocabulary, lowercase), where vocab is dict from words to indices
    """

    models_spec = {
        "glove-wiki-gigaword-50": {
            "id": "glove-wiki-gigaword-50",
            "lowercase": True,
        },
        "glove-wiki-gigaword-100": {
            "id": "glove-wiki-gigaword-100",
            "lowercase": True,
        },
        "glove-wiki-gigaword-200": {
            "id": "glove-wiki-gigaword-200",
            "lowercase": True,
        },
        "glove-wiki-gigaword-300": {
            "id": "glove-wiki-gigaword-300",
            "lowercase": True,
        },
        "glove-twitter-25": {
            "id": "glove-twitter-25",
            "lowercase": True,
        },
        "glove-twitter-50": {
            "id": "glove-twitter-50",
            "lowercase": True,
        },
        "glove-twitter-100": {
            "id": "glove-twitter-100",
            "lowercase": True,
        },
        "glove-twitter-200": {
            "id": "glove-twitter-200",
            "lowercase": True,
        },
    }

    out = {}
    for name, spec in models_spec.items():
        kv = api.load(spec["id"])
        out[name] = _kv_to_model_and_vocab(kv, spec['lowercase'])

    return out


if __name__ == '__main__':
    data = get_word2vec_models()

    for name, model in tqdm(data.items(), total=len(data)):
        wordsim353_val = word2vec_validate_wordsim353(model)
        simlex999_val = word2vec_validate_simlex999(model)
        men3k_val = word2vec_validate_men3k(model)
        google_analogies_val = word2vec_validate_google_analogies(model, batch_size=1000)

        tqdm.write(
            f"Model {name}  |  "
            f"WS353: {wordsim353_val[0]:6.3f} (cov {wordsim353_val[1]:5.1%})  |  "
            f"SimLex: {simlex999_val[0]:6.3f} (cov {simlex999_val[1]:5.1%})  |  "
            f"MEN: {men3k_val[0]:6.3f} (cov {men3k_val[1]:5.1%})  |  "
            f"GA: {google_analogies_val[0]:6.3f} (cov {google_analogies_val[1]:5.1%})"
        )