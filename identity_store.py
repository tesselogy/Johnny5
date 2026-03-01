# identity_store.py

import psycopg2
import numpy as np

from config import (
    PG_HOST,
    PG_PORT,
    PG_DB,
    PG_USER,
    PG_PASSWORD,
)


class IdentityStore:

    def __init__(self):

        self.conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
        )

        self.conn.autocommit = True

        self._ensure_schema()

    # -------------------------------------------------
    # Schema
    # -------------------------------------------------
    def _ensure_schema(self):

        with self.conn.cursor() as cur:

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # person profile (one row per person)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            # multi-embedding gallery (many vectors per person)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS person_embeddings (
                    embedding_id BIGSERIAL PRIMARY KEY,
                    person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
                    embedding vector(512) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS person_embeddings_person_idx
                ON person_embeddings (person_id);
                """
            )

            # optional ANN index
            cur.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE indexname = 'person_embeddings_embedding_idx'
                    ) THEN
                        CREATE INDEX person_embeddings_embedding_idx
                        ON person_embeddings
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    END IF;
                END
                $$;
                """
            )

    # -------------------------------------------------
    # Insert
    # -------------------------------------------------
    def insert_person(self, person_id: str, embedding: np.ndarray):
        """Backward compatible API: ensure person exists and append embedding."""
        self.insert_person_embedding(person_id, embedding)

    def insert_person_embedding(self, person_id: str, embedding: np.ndarray):

        vector_str = self._to_pgvector(embedding)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO persons (person_id)
                VALUES (%s)
                ON CONFLICT (person_id) DO NOTHING
                """,
                (person_id,),
            )

            cur.execute(
                """
                INSERT INTO person_embeddings (person_id, embedding)
                VALUES (%s, %s)
                """,
                (person_id, vector_str),
            )

    # -------------------------------------------------
    # Load all embeddings
    # -------------------------------------------------
    def load_all(self):

        result = []

        # New schema: many vectors per person
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT person_id, embedding
                FROM person_embeddings
                ORDER BY embedding_id
                """
            )
            rows = cur.fetchall()

        for person_id, embedding in rows:
            result.append((person_id, self._embedding_to_np(embedding)))

        # Backward compatibility with legacy single-vector `persons.embedding`
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'persons' AND column_name = 'embedding'
                LIMIT 1
                """
            )
            has_legacy_col = cur.fetchone() is not None

        if has_legacy_col:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT person_id, embedding
                    FROM persons
                    WHERE embedding IS NOT NULL
                    """
                )
                legacy_rows = cur.fetchall()

            for person_id, embedding in legacy_rows:
                result.append((person_id, self._embedding_to_np(embedding)))

        return result

    # -------------------------------------------------
    # Helper
    # -------------------------------------------------
    def _embedding_to_np(self, embedding):
        # pgvector often returns string like "[0.123, -0.234, ...]"
        if isinstance(embedding, str):
            embedding = embedding.strip("[]")
            embedding = embedding.split(",")
            embedding = [float(x) for x in embedding]

        return np.array(embedding, dtype="float32")

    def _to_pgvector(self, embedding: np.ndarray) -> str:
        """
        Преобразует numpy array в формат pgvector:
        '[0.1,0.2,0.3,...]'
        """
        return "[" + ",".join(map(str, embedding.tolist())) + "]"

    # -------------------------------------------------
    # Optional: Close connection
    # -------------------------------------------------
    def close(self):
        if self.conn:
            self.conn.close()
