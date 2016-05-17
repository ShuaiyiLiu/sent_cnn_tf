config = {
    "webquestions_examples_file": "./data/webquestions.examples.train.e2e.top10.filter.patrel.sid.tsv",
    "word_embeddings_file": "./data/word_representations/glove.6B.100d.txt",
    "vocabulary_size": 400000,
    "embedding_size": 100,
    "num_classes": 6,
    "filter_sizes": [3, 4],
    "num_filters": 3,
    "dropout_keep_prob": 0.5,
    "embeddings_trainable": False,
    "total_iter": 5000,
    "batch_size": 500,
    "val_size": 500
}