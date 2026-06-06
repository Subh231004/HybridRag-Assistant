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

        self.embedding_model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

        self.cache_file = (
            "embeddings.pkl"
        )

        self.build_indexes()

    # =====================================
    # BUILD INDEXES
    # =====================================

    def build_indexes(self):

        texts = [

            chunk["text"]

            for chunk in self.chunks
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
        # Embeddings
        # -----------------------------

        self.embedding_cache = {}

        if os.path.exists(
            self.cache_file
        ):

            try:

                with open(
                    self.cache_file,
                    "rb"
                ) as f:

                    self.embedding_cache = (
                        pickle.load(f)
                    )

            except:

                self.embedding_cache = {}

        embeddings = []

        cache_updated = False

        for idx, chunk in enumerate(
            self.chunks
        ):

            chunk_id = self.get_chunk_id(
                chunk
            )

            if chunk_id in self.embedding_cache:

                emb = (
                    self.embedding_cache[
                        chunk_id
                    ]
                )

            else:

                emb = (
                    self.embedding_model.encode(
                        chunk["text"],
                        convert_to_numpy=True
                    )
                )

                self.embedding_cache[
                    chunk_id
                ] = emb

                cache_updated = True

            embeddings.append(
                emb
            )

        self.chunk_embeddings = (
            np.array(
                embeddings
            )
        )

        if cache_updated:

            with open(
                self.cache_file,
                "wb"
            ) as f:

                pickle.dump(
                    self.embedding_cache,
                    f
                )

    # =====================================
    # UNIQUE CHUNK ID
    # =====================================

    def get_chunk_id(
        self,
        chunk
    ):

        source = chunk.get(
            "source",
            "unknown"
        )

        page = chunk.get(
            "page",
            1
        )

        text_hash = hash(
            chunk["text"][:200]
        )

        return (
            f"{source}_{page}_{text_hash}"
        )

    # =====================================
    # ADD NEW CHUNKS
    # =====================================

    def add_chunks(
        self,
        new_chunks
    ):

        self.chunks.extend(
            new_chunks
        )

        self.build_indexes()

    # =====================================
    # RETRIEVE
    # =====================================

    def retrieve(
        self,
        query,
        top_k=5,
        source_filter=None
    ):

        candidate_chunks = (
            self.chunks
        )

        candidate_indices = list(
            range(
                len(
                    self.chunks
                )
            )
        )

        # -----------------------------
        # Metadata Filtering
        # -----------------------------

        if source_filter:

            filtered_chunks = []
            filtered_indices = []

            for idx, chunk in enumerate(
                self.chunks
            ):

                if (
                    source_filter.lower()
                    in chunk["source"].lower()
                ):

                    filtered_chunks.append(
                        chunk
                    )

                    filtered_indices.append(
                        idx
                    )

            if len(
                filtered_chunks
            ) == 0:

                return []

            candidate_chunks = (
                filtered_chunks
            )

            candidate_indices = (
                filtered_indices
            )

        texts = [

            chunk["text"]

            for chunk in candidate_chunks
        ]

        tokenized = [

            text.lower().split()

            for text in texts
        ]

        bm25 = BM25Okapi(
            tokenized
        )

        vectorizer = (
            TfidfVectorizer()
        )

        tfidf_matrix = (
            vectorizer.fit_transform(
                texts
            )
        )

        chunk_embeddings = np.array(

            [
                self.chunk_embeddings[
                    idx
                ]

                for idx in candidate_indices
            ]
        )

        # -----------------------------
        # BM25
        # -----------------------------

        bm25_scores = (
            bm25.get_scores(
                query.lower().split()
            )
        )

        # -----------------------------
        # TF-IDF
        # -----------------------------

        q_vec = (
            vectorizer.transform(
                [query]
            )
        )

        tfidf_scores = (
            cosine_similarity(
                q_vec,
                tfidf_matrix
            )
            .flatten()
        )

        # -----------------------------
        # Semantic
        # -----------------------------

        q_emb = (
            self.embedding_model.encode(
                query,
                convert_to_numpy=True
            )
        )

        semantic_scores = np.dot(

            chunk_embeddings,

            q_emb
        )

        # -----------------------------
        # Hybrid Score
        # -----------------------------

        scores = (

            0.3 * bm25_scores +

            0.2 * tfidf_scores +

            0.5 * semantic_scores
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in top_indices:

            chunk = dict(
                candidate_chunks[idx]
            )

            chunk["score"] = float(
                scores[idx]
            )

            results.append(
                chunk
            )

        return results