from rank_bm25 import BM25Okapi

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from sentence_transformers import (
    SentenceTransformer
)

import numpy as np

import pickle
import os


class HybridRetriever:

    def __init__(self, chunks):

        self.chunks = chunks

        texts = [

            chunk["text"]

            for chunk in chunks
        ]

        # -----------------------------
        # BM25
        # -----------------------------

        tokenized = [

            text.lower().split()

            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

        # -----------------------------
        # TF-IDF
        # -----------------------------

        self.vectorizer = (
            TfidfVectorizer()
        )

        self.tfidf_matrix = (

            self.vectorizer.fit_transform(
                texts
            )
        )

        # -----------------------------
        # Embedding Model
        # -----------------------------

        self.embedding_model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

        # -----------------------------
        # Embedding Cache
        # -----------------------------

        cache_file = (
            "embeddings.pkl"
        )

        if (
            os.path.exists(cache_file)
            and len(chunks) > 0
        ):

            try:

                with open(
                    cache_file,
                    "rb"
                ) as f:

                    cached_embeddings = (
                        pickle.load(f)
                    )

                if len(
                    cached_embeddings
                ) == len(chunks):

                    self.chunk_embeddings = (
                        cached_embeddings
                    )

                else:

                    raise ValueError(
                        "Embedding cache size mismatch"
                    )

            except:

                self.chunk_embeddings = (
                    self.embedding_model.encode(
                        texts,
                        convert_to_numpy=True
                    )
                )

                with open(
                    cache_file,
                    "wb"
                ) as f:

                    pickle.dump(
                        self.chunk_embeddings,
                        f
                    )

        else:

            if len(texts) > 0:

                self.chunk_embeddings = (
                    self.embedding_model.encode(
                        texts,
                        convert_to_numpy=True
                    )
                )

                with open(
                    cache_file,
                    "wb"
                ) as f:

                    pickle.dump(
                        self.chunk_embeddings,
                        f
                    )

            else:

                self.chunk_embeddings = (
                    np.array([])
                )

    def add_chunks(self, new_chunks):

        self.chunks.extend(
            new_chunks
        )

        texts = [

         chunk["text"]

            for chunk in self.chunks
        ]

        tokenized = [

            text.lower().split()

            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

        self.vectorizer = (
            TfidfVectorizer()
        )

        self.tfidf_matrix = (

            self.vectorizer.fit_transform(
                texts
            )
        )

        self.chunk_embeddings = (
            self.embedding_model.encode(
                texts,
                convert_to_numpy=True
            )
        )

        with open(
            "embeddings.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.chunk_embeddings,
                f
            )