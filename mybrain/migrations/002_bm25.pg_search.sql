-- ParadeDB pg_search: true BM25 with the ICU tokenizer (Persian + English in one index).
-- Only applied when the pg_search extension is available (ParadeDB image, pg_search >= 0.18 for the
-- ::pdb.* tokenizer casts). Otherwise the tsvector fallback in 001_init.sql is used.
-- Query side: WHERE lexical_text ||| 'terms' ORDER BY pdb.score(id) DESC, id
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON chunks
USING bm25 (id, (lexical_text::pdb.icu), index_version_id, document_id)
WITH (key_field = 'id');
