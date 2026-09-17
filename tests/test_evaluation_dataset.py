import json
import unittest
from pathlib import Path


DATASET_PATH = Path(__file__).parents[1] / "evals" / "golden_dataset.json"


class EvaluationDatasetTests(unittest.TestCase):
    def test_dataset_has_broad_rag_and_guardrail_coverage(self):
        dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(dataset["rag_samples"]), 25)
        self.assertGreaterEqual(len(dataset["guardrails_samples"]), 18)
        self.assertTrue(any(s["expected_blocked"] for s in dataset["guardrails_samples"]))
        self.assertTrue(any(not s["expected_blocked"] for s in dataset["guardrails_samples"]))

    def test_rag_samples_have_unique_ids_and_required_eval_fields(self):
        dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        samples = dataset["rag_samples"]
        self.assertEqual(len({sample["id"] for sample in samples}), len(samples))

        for sample in samples:
            self.assertTrue(sample["question"])
            self.assertTrue(sample["reference"])
            self.assertEqual(sample["expected_tools"], ["retrieve_documents"])


if __name__ == "__main__":
    unittest.main()
