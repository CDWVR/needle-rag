"""Public-demo boundary checks. Not collected by discovery: test_demo.py runs this file in a
subprocess, because demo mode is read from the environment when the app is imported.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.update({
    "NEEDLE_DATA_DIR": tempfile.mkdtemp(prefix="needle-demo-test-"),
    "NEEDLE_DEMO_MODE": "true",
    "NEEDLE_ACCESS_TOKEN": "demo-owner-token-0123456789",
    "NEEDLE_SESSION_SECRET": "demo-session-secret-0123456789",
    "NEEDLE_ALLOWED_HOSTS": "testserver",
    "OPENROUTER_API_KEY": "",
})

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

WRITE = {"X-Needle-CSRF": "1"}
OWNER = {"Authorization": "Bearer demo-owner-token-0123456789"}


def visitor() -> TestClient:
    client = TestClient(main.app)
    client.get("/api/auth/session")  # receives the visitor cookie
    return client


class VisitorTests(unittest.TestCase):
    def setUp(self):
        main.security.limiter._hits.clear()
        main.security.DEMO_DAILY_QUESTIONS = 300

    def test_visitors_read_without_signing_in(self):
        client = visitor()
        session = client.get("/api/auth/session").json()
        self.assertEqual((session["authenticated"], session["role"], session["demo"]), (True, "visitor", True))
        for path in ("/api/settings", "/api/documents", "/api/index", "/api/conversations", "/api/eval/latest"):
            self.assertEqual(client.get(path).status_code, 200, path)

    def test_visitors_cannot_change_anything(self):
        client = visitor()
        blocked = [
            ("put", "/api/settings", {"json": {}}),
            ("post", "/api/upload", {"files": {"file": ("a.txt", b"hello world " * 10)}}),
            ("post", "/api/index/refresh", {}),
            ("post", "/api/index/rollback", {}),
            ("post", "/api/index/reconcile", {}),
            ("post", "/api/workspace/reset", {"json": {"confirm": "DELETE"}}),
            ("delete", "/api/documents/11111111-1111-4111-8111-111111111111", {}),
            ("patch", "/api/documents/11111111-1111-4111-8111-111111111111", {"json": {"included": False}}),
            ("get", "/api/documents/11111111-1111-4111-8111-111111111111/file", {}),
            ("get", "/api/analytics/export", {}),
        ]
        for method, path, kwargs in blocked:
            response = getattr(client, method)(path, headers=WRITE, **kwargs)
            self.assertEqual(response.status_code, 403, f"{method} {path}")

    def test_voice_route_is_open_to_visitors_but_off_without_a_key(self):
        client = visitor()
        files = {"file": ("voice", b"\x1a\x45\xdf\xa3" + b"\x00" * 50)}
        self.assertEqual(client.post("/api/transcribe", headers=WRITE, files=files).status_code, 503)
        self.assertFalse(client.get("/api/auth/session").json()["voice"])

    def test_visitor_writes_still_need_the_csrf_header(self):
        self.assertEqual(visitor().post("/api/conversations").status_code, 403)

    def test_conversations_are_private(self):
        alice, bob = visitor(), visitor()
        owner_conversation = TestClient(main.app).post("/api/conversations", headers=OWNER).json()["id"]
        mine = alice.post("/api/conversations", headers=WRITE).json()["id"]
        main.workspace.add_message(mine, "user", "alice's private question", original_query="x")
        listed = [item["id"] for item in alice.get("/api/conversations").json()["conversations"]]
        self.assertEqual(listed, [mine])
        self.assertEqual(bob.get("/api/conversations").json()["conversations"], [])
        self.assertEqual(bob.get(f"/api/conversations/{mine}").status_code, 404)
        self.assertEqual(alice.get(f"/api/conversations/{owner_conversation}").status_code, 404)
        message = main.workspace.add_message(mine, "assistant", "answer")
        response = bob.post(f"/api/messages/{message}/feedback", headers=WRITE, json={"rating": "helpful"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(alice.post(f"/api/messages/{message}/feedback", headers=WRITE, json={"rating": "helpful"}).status_code, 200)

    def test_analytics_hide_other_visitors_questions(self):
        main.workspace.record_event(query="somebody else's secret question", grounded=False, withheld=True,
                                    outcome="no_coverage", latency_ms=5, source_count=0, candidate_count=0)
        report = visitor().get("/api/analytics?days=7").json()
        self.assertEqual(report["gaps"], [])
        owner_report = TestClient(main.app).get("/api/analytics?days=7", headers=OWNER).json()
        self.assertTrue(owner_report["gaps"])

    def test_question_limits(self):
        client = visitor()
        long = client.post("/api/chat", headers=WRITE, json={"query": "x" * 600})
        self.assertEqual(long.status_code, 400)
        main.security.DEMO_DAILY_QUESTIONS = 0
        capped = client.post("/api/chat", headers=WRITE, json={"query": "What is the payload?"})
        self.assertEqual(capped.status_code, 429)
        self.assertIn("today's question limit", capped.json()["detail"])

    def test_visitor_rate_limit(self):
        client = visitor()
        statuses = [client.post("/api/chat", headers=WRITE, json={"query": f"question number {i}"}).status_code
                    for i in range(8)]
        self.assertEqual(statuses[:6], [200] * 6)
        self.assertEqual(statuses[6:], [429, 429])


class SeedTests(unittest.TestCase):
    def test_reseeding_neither_duplicates_nor_keeps_old_copies(self):
        import demo

        demo.seed_demo_corpus(main.workspace)  # the app already seeds in the background at startup; either order is fine
        self.assertEqual(demo.seed_demo_corpus(main.workspace), 0)  # a restart adds nothing, even for the rendered PDF
        names = [doc["name"] for doc in main.get_all_documents()]
        self.assertEqual(names.count("Attention Mechanism.pdf"), 1)  # rendered from its text source
        self.assertEqual(sorted(names), sorted({"Attention Mechanism.pdf", "Autoencoders.pdf", "rnn_basics.txt", "information_retrieval_basics.txt"}))

    def test_documents_from_an_earlier_demo_set_are_removed(self):
        """A redeploy keeps the volume, so a changed demo set must clear what the last release seeded."""
        import demo
        from ingest import process_document

        demo.seed_demo_corpus(main.workspace)
        old = process_document(b"Ottermere ships the Tern-3 cart. " * 20, "tern3_operator_manual.md", "text/plain", content_hash="old-demo-hash")
        main.workspace.set_policy(old.id, collection=demo.COLLECTION, byte_size=100, stored_name="", content_hash="old-demo-hash")
        self.assertIn("tern3_operator_manual.md", [doc["name"] for doc in main.get_all_documents()])
        demo.seed_demo_corpus(main.workspace)
        self.assertNotIn("tern3_operator_manual.md", [doc["name"] for doc in main.get_all_documents()])
        self.assertIn("rnn_basics.txt", [doc["name"] for doc in main.get_all_documents()])


class OwnerTests(unittest.TestCase):
    def test_owner_signs_in_and_changes_settings(self):
        client = TestClient(main.app)
        self.assertEqual(client.post("/api/auth/login", json={"token": "demo-owner-token-0123456789"}).status_code, 200)
        self.assertEqual(client.get("/api/auth/session").json()["role"], "owner")
        settings = client.get("/api/settings").json()
        payload = {key: settings[key] for key in ("workspace_name", "profile_name", "default_collection", "top_k")}
        self.assertEqual(client.put("/api/settings", headers=WRITE, json=payload).status_code, 200)

    def test_anonymous_never_exists_in_demo_but_bad_tokens_still_fail(self):
        self.assertEqual(TestClient(main.app).post("/api/auth/login", json={"token": "nope"}).status_code, 401)
        self.assertEqual(
            TestClient(main.app).get("/api/settings", headers={"Authorization": "Bearer wrong-token-value"}).status_code, 200
        )  # falls back to visitor read access, not owner access


if __name__ == "__main__":
    unittest.main()
