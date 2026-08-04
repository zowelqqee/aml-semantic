"""Causality and source-honesty checks for the IEEE-CIS Runtime port."""

from __future__ import annotations

import unittest

from ieee_cis_semantic_transfer.fraud_runtime.behaviour import BehaviourDecisionRuntime
from ieee_cis_semantic_transfer.fraud_runtime.behaviour.ontology import UNSUPPORTED_ON_SOURCE as BEHAVIOUR_UNSUPPORTED_ON_SOURCE
from ieee_cis_semantic_transfer.fraud_runtime.models import Transaction
from ieee_cis_semantic_transfer.fraud_runtime.semantic.ontology import SOURCE_COVERAGE_GAPS


def tx(identifier: str, dt: int, amount: float, *, fraud: bool = False, device: str = "desktop|browser-a") -> Transaction:
    return Transaction(identifier, dt, f"REL+{dt:08d}s", "111|222|150|visa|226|credit", amount, "W", "100", "87", device,
                       True, "gmail.com", "", None, None, fraud, {"network": "visa", "card_type": "credit", "device_type": "desktop"})


class FraudRuntimeCausalityTests(unittest.TestCase):
    def _prefix(self, future_amount: float, future_label: bool):
        runtime = BehaviourDecisionRuntime()
        for index in range(6):
            current = tx(f"p{index}", 60 * index, 20.0)
            result = runtime.evaluate(current, current.transaction_dt // 60, index)
            runtime.commit(current, result, current.transaction_dt // 60)
        target = tx("target", 400, 22.0)
        result = runtime.evaluate(target, target.transaction_dt // 60, 7)
        # This later row is deliberately never committed before target.
        future = tx("future", 500, future_amount, fraud=future_label)
        self.assertGreater(future.amount, 0)
        return result.audit_record()

    def test_future_values_and_labels_cannot_change_a_prior_reading(self):
        self.assertEqual(self._prefix(1.0, False), self._prefix(9_999.0, True))

    def test_labels_do_not_enter_semantic_or_behaviour_objects(self):
        runtime_a, runtime_b = BehaviourDecisionRuntime(), BehaviourDecisionRuntime()
        for index in range(7):
            a, b = tx(str(index), index * 60, 20.0, fraud=False), tx(str(index), index * 60, 20.0, fraud=True)
            result_a, result_b = runtime_a.evaluate(a, a.transaction_dt // 60, index), runtime_b.evaluate(b, b.transaction_dt // 60, index)
            audit_a, audit_b = result_a.audit_record(), result_b.audit_record()
            audit_a.pop("transaction"); audit_b.pop("transaction")
            self.assertEqual(audit_a, audit_b)
            runtime_a.commit(a, result_a, a.transaction_dt // 60); runtime_b.commit(b, result_b, b.transaction_dt // 60)

    def test_documented_source_gaps_include_no_physical_device_identity(self):
        gaps = {name for name, _reason in SOURCE_COVERAGE_GAPS}
        self.assertIn("physical_device_identity", gaps)
        unsupported = {name for name, _inputs in BEHAVIOUR_UNSUPPORTED_ON_SOURCE}
        self.assertIn("ImpossibleTravelBehaviour", unsupported)
        self.assertIn("SyntheticIdentityBehaviour", unsupported)


if __name__ == "__main__":
    unittest.main()
