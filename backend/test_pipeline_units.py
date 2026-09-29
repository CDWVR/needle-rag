"""Gates and index registry behavior that do not call Jev or Gemini."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from index_store import IndexStore
from pipeline_logic import (
    filter_by_similarity,
    index_card,
    select_parents,
    tokenize,
    verdict_passes,
)


class PipelineLogicTests(unittest.TestCase):
    def test_tokenize_keeps_word_characters(self):
        self.assertEqual(tokenize("Hello, Mars-1!"), ["hello", "mars", "1"])

    def test_similarity_gate_uses_absolute_and_relative_floors(self):
        hits = [
            {"id": "a", "similarity": 0.80},
            {"id": "b", "similarity": 0.50},
            {"id": "c", "similarity": 0.20},
        ]
        kept = filter_by_similarity(hits, absolute_threshold=0.30, relative_floor=0.70)
        self.assertEqual([hit["id"] for hit in kept], ["a"])

    def test_parent_selection_keeps_the_strongest_jev_score(self):
        chosen = select_parents(
            [
                {"parent_id": "p1", "jev_score": 0.4},
                {"parent_id": "p1", "jev_score": 0.9},
                {"parent_id": "p2", "jev_score": 0.7},
            ],
            limit=1,
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["parent_id"], "p1")
        self.assertEqual(chosen[0]["jev_score"], 0.9)

    def test_verdict_requires_every_check(self):
        self.assertTrue(verdict_passes({"grounded": True, "safe": True, "relevant": True}))
        self.assertFalse(verdict_passes({"grounded": True, "safe": False, "relevant": True}))

    def test_index_card_keeps_heading_and_lead(self):
        card = index_card("Safety", "\n\nWear gloves.\nMore detail.")
        self.assertIn("Safety", card)
        self.assertIn("Wear gloves.", card)


class IndexStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = IndexStore(os.path.join(self.tmp.name, "index.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_bootstrap_and_publish_handoff(self):
        first = self.store.ensure_active_version("all-MiniLM-L6-v2")
        again = self.store.ensure_active_version("all-MiniLM-L6-v2")
        self.assertEqual(first["version_id"], again["version_id"])
        self.store.begin_version("v2", "all-MiniLM-L6-v2", "needle_v2")
        self.store.publish_version("v2")
        active = self.store.ensure_active_version("all-MiniLM-L6-v2")
        self.assertEqual(active["version_id"], "v2")
        self.assertEqual(active["status"], "active")
        retired = [row for row in self.store.list_versions() if row["version_id"] == first["version_id"]]
        self.assertEqual(retired[0]["status"], "retired")

    def test_delete_removes_keyword_rows(self):
        self.store.ensure_active_version("all-MiniLM-L6-v2")
        self.store.save_document(
            document_id="doc-1",
            name="notes.txt",
            num_pages=1,
            num_chunks=1,
            file_type="text",
            uploaded_at="2026-09-28T00:00:00+00:00",
            version_id="legacy-1",
            parents=[
                {
                    "parent_id": "doc-1_0",
                    "page_number": 1,
                    "chunk_index": 0,
                    "header_context": "Safety",
                    "parent_text": "Wear gloves in the lab.",
                    "summary": "Wear gloves.",
                }
            ],
            fts_rows=[("doc-1", "doc-1_0_0", "notes.txt", 1, "Safety", "Wear gloves in the lab.")],
        )
        self.assertEqual(len(self.store.keyword_search("gloves", 5)), 1)
        self.assertTrue(self.store.document_exists("doc-1"))
        self.store.mark_deleted("doc-1")
        self.assertFalse(self.store.document_exists("doc-1"))
        self.assertEqual(self.store.list_documents(), [])
        self.assertIsNone(self.store.fetch_parent("doc-1_0"))


if __name__ == "__main__":
    unittest.main()
