import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The splitting algorithm is deliberately testable without an installed
        # observability SDK; production injects the real logfire module.
        sys.modules.setdefault("logfire", types.SimpleNamespace(
            span=lambda *args, **kwargs: _NoopSpan(),
            info=lambda *args, **kwargs: None,
        ))
        path = Path(__file__).parents[1] / "app" / "ingestion" / "chunking" / "splitter.py"
        spec = importlib.util.spec_from_file_location("splitter_under_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_long_paragraph_is_split_without_data_loss(self):
        text = "word " * 100
        chunks = self.module.chunk_text(text, chunk_size=50)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())


if __name__ == "__main__":
    unittest.main()
