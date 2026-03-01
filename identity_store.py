# identity_store.py

import psycopg2
import psycopg2.extras
import numpy as np

from config import (
    PG_HOST,
    PG_PORT,
    PG_DB,
    PG_USER,
    PG_PASSWORD
)


class IdentityStore:

    def __init__(self):

        self.conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )

        self.conn.autocommit = True

        self._ensure_schema()

    # -------------------------------------------------
    # Schema
    # -------------------------------------------------
    def _ensure_schema(self):

        with self.conn.cursor() as cur:

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    embedding vector(512)
                );
            """)

            # optional ANN index
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE indexname = 'persons_embedding_idx'
                    ) THEN
                        CREATE INDEX persons_embedding_idx
                        ON persons
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    END IF;
                END
                $$;
            """)

    # -------------------------------------------------
    # Insert
    # -------------------------------------------------
    def insert_person(self, person_id: str, embedding: np.ndarray):

        vector_str = self._to_pgvector(embedding)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO persons (person_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (person_id) DO NOTHING
                """,
                (person_id, vector_str)
            )

    # -------------------------------------------------
    # Load all embeddings
    # -------------------------------------------------
    def load_all(self):

        with self.conn.cursor() as cur:
            cur.execute("SELECT person_id, embedding FROM persons")
            rows = cur.fetchall()

        result = []

        for person_id, embedding in rows:

            # pgvector возвращает строку вида:
            # "[0.123, -0.234, ...]"
            if isinstance(embedding, str):
                embedding = embedding.strip("[]")
                embedding = embedding.split(",")
                embedding = [float(x) for x in embedding]

            result.append(
                (person_id, np.array(embedding, dtype="float32"))
            )

        return result

    # -------------------------------------------------
    # Helper
    # -------------------------------------------------
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