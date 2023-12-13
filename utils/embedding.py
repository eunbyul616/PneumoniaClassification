import io
import numpy as np


class EmbeddingVector:
    def __init__(self, file):
        self.file = file

        vectors = []
        self.word2id = {}

        with io.open(self.file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)

            for i, line in enumerate(f):
                word, vec = line.rstrip().split(' ', 1)
                vec = np.fromstring(vec, sep=' ')

                assert word not in self.word2id, 'word found twice'

                vectors.append(vec)
                self.word2id[word] = len(self.word2id)

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embedding = np.vstack(vectors)

    def apply_transformation(self, transform):
        _transform = np.loadtxt(transform) if isinstance(transform, str) else transform
        self.embedding = np.matmul(self.embedding, _transform)

    def __getitem__(self, key):
        return self.embedding[self.word2id[key]]


class TransformEmbeddingVector:
    @classmethod
    def transformation(source_matrix, target_matrix, normalize=True):
        if normalize:
            source_matrix = np.linalg.norm(source_matrix)
            target_matrix = np.linalg.norm(target_matrix)

        product = np.matmul(source_matrix.transpose(), target_matrix)
        U, s, Vt = np.linalg.svd(product)

        return np.matmul(U, Vt)


class EmbeddingTransformation:
    def __init__(self, config):
        self.config = config

    def load_embedding_vector(self):
        self.source_vec = EmbeddingVector(file=self.config['embedding']['source_path'])
        self.target_vec = EmbeddingVector(file=self.config['embedding']['target_path'])

    def transform(self):
        source_words = self.source_vec.word2id.keys()
        target_words = self.target_vec.word2id.keys()
        dict = [(w, w) for w in list(source_words & target_words)]

        source_matrix = []
        target_matrix = []
        for (source, target) in dict:
            if source in self.source_vec and target in self.target_vec:
                source_matrix.append(self.source_vec[source])
                target_matrix.append(self.target_vec[target])

        source_matrix = np.array(source_matrix)
        target_matrix = np.array(target_matrix)

        transform = TransformEmbeddingVector.transformation(source_matrix, target_matrix)
        self.source_vec.apply_transformation(transform)

    def __call__(self, *args, **kwargs):
        self.load_embedding_vector()
        self.transform()

        return self.source_vec, self.target_vec
