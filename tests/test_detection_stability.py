#!/usr/bin/env python

"""
Regression tests for reproducible injection-point verdicts.

Detection must not depend on fresh random draws, and a transport timeout must not
collapse into a negative observation. These tests use a mocked request seam, so they
are deterministic and network-free.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testutils import bootstrap
bootstrap()

import lib.controller.checks as checks
from lib.core.common import getCurrentThreadData
from lib.core.data import conf, kb
from lib.core.datatype import AttribDict, InjectionDict
from lib.core.enums import PAYLOAD, PLACE


class DetectionStabilityTest(unittest.TestCase):
    _CONF = ("level", "string", "notString", "hashDB")
    _KB = ("injection", "detectionSeed", "detectionIndeterminate", "falsePositiveCheckInconclusive")

    def setUp(self):
        self._conf = dict((k, conf.get(k)) for k in self._CONF)
        self._kb = dict((k, kb.get(k)) for k in self._KB)
        self._queryPage = checks.Request.queryPage
        self._checkBooleanExpression = checks.checkBooleanExpression
        self._checkSqlInjection = checks._checkSqlInjection
        self._threadData = getCurrentThreadData()
        self._timeout = getattr(self._threadData, "lastRequestTimedOut", False)
        self._comparisonPage = getattr(self._threadData, "lastComparisonPage", None)
        self._randomState = random.getstate()

        conf.level = 1
        conf.string = conf.notString = None
        conf.hashDB = None
        kb.detectionIndeterminate = False
        kb.falsePositiveCheckInconclusive = False
        self._threadData.lastRequestTimedOut = False
        self._threadData.lastComparisonPage = None

    def tearDown(self):
        for key, value in self._conf.items():
            conf[key] = value
        for key, value in self._kb.items():
            kb[key] = value
        checks.Request.queryPage = self._queryPage
        checks.checkBooleanExpression = self._checkBooleanExpression
        checks._checkSqlInjection = self._checkSqlInjection
        self._threadData.lastRequestTimedOut = self._timeout
        self._threadData.lastComparisonPage = self._comparisonPage
        random.setstate(self._randomState)

    def _injection(self):
        injection = InjectionDict()
        injection.place = PLACE.GET
        injection.parameter = "id"
        injection.data[PAYLOAD.TECHNIQUE.BOOLEAN] = AttribDict()
        injection.data[PAYLOAD.TECHNIQUE.BOOLEAN].title = "boolean"
        return injection

    def test_seed_is_stable_per_injection_point(self):
        conf.hostname = "example.test"
        conf.port = 80
        conf.path = "/item"
        conf.method = None

        first = checks._detectionSeed(PLACE.GET, "id")
        second = checks._detectionSeed(PLACE.GET, "id")
        other = checks._detectionSeed(PLACE.GET, "other")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_control_values_are_reproducible_and_ordered(self):
        first = checks._deterministicTestValues(12345, 0)
        second = checks._deterministicTestValues(12345, 0)

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 3)
        self.assertLess(first[0], first[1])
        self.assertLess(first[1], first[2])

    def test_detection_entry_point_replays_the_same_probe_sequence(self):
        calls = []

        def fake(place, parameter, value):
            calls.append((checks.randomInt(6), checks.randomStr(8)))

        checks._checkSqlInjection = fake
        checks.checkSqlInjection(PLACE.GET, "id", "1")
        first = calls[-1]
        checks.checkSqlInjection(PLACE.GET, "id", "1")
        second = calls[-1]

        self.assertEqual(first, second)

    def test_detection_entry_point_restores_rng_state(self):
        checks._checkSqlInjection = lambda place, parameter, value: None
        before = random.getstate()
        checks.checkSqlInjection(PLACE.GET, "id", "1")
        self.assertEqual(before, random.getstate())

    def test_clean_request_is_not_replayed(self):
        calls = []
        self._threadData.lastRequestTimedOut = False
        self._threadData.lastComparisonPage = "answered"

        def fake(payload, place, **kwargs):
            calls.append(payload)
            return False

        checks.Request.queryPage = staticmethod(fake)
        self.assertIs(checks._queryPageStable("p", PLACE.GET), False)
        self.assertEqual(calls, ["p"])
        self.assertFalse(kb.detectionIndeterminate)

    def test_timeout_replays_the_exact_payload_once(self):
        calls = []
        replies = iter([None, True])

        def fake(payload, place, **kwargs):
            calls.append(payload)
            result = next(replies)
            self._threadData.lastRequestTimedOut = result is None
            self._threadData.lastComparisonPage = None if result is None else "answered"
            return result

        checks.Request.queryPage = staticmethod(fake)
        self.assertIs(checks._queryPageStable("same-payload", PLACE.GET), True)
        self.assertEqual(calls, ["same-payload", "same-payload"])
        self.assertFalse(kb.detectionIndeterminate)

    def test_persistent_timeout_is_marked_inconclusive(self):
        calls = []

        def fake(payload, place, **kwargs):
            calls.append(payload)
            self._threadData.lastRequestTimedOut = True
            self._threadData.lastComparisonPage = None
            return None

        checks.Request.queryPage = staticmethod(fake)
        self.assertIsNone(checks._queryPageStable("same-payload", PLACE.GET))
        self.assertEqual(calls, ["same-payload", "same-payload"])
        self.assertTrue(kb.detectionIndeterminate)

    def test_definitive_none_from_a_real_page_is_not_retried(self):
        calls = []
        self._threadData.lastRequestTimedOut = False

        def fake(payload, place, **kwargs):
            calls.append(payload)
            self._threadData.lastComparisonPage = "error-page-received"
            return None

        checks.Request.queryPage = staticmethod(fake)
        self.assertIsNone(checks._queryPageStable("p", PLACE.GET))
        self.assertEqual(calls, ["p"])

    def test_false_positive_control_retries_timeout_with_same_expression(self):
        expressions = []
        replies = iter([None, True, False, False, True, False])

        def fake(expression):
            expressions.append(expression)
            return next(replies)

        checks.checkBooleanExpression = fake
        self.assertTrue(checks.checkFalsePositives(self._injection()))
        self.assertEqual(len(expressions), 6)
        self.assertEqual(expressions[0], expressions[1])
        self.assertFalse(kb.falsePositiveCheckInconclusive)

    def test_persistent_timeout_preserves_detected_point(self):
        calls = []

        def fake(expression):
            calls.append(expression)
            return None

        checks.checkBooleanExpression = fake
        self.assertTrue(checks.checkFalsePositives(self._injection()))
        self.assertEqual(len(calls), 2)
        self.assertTrue(kb.falsePositiveCheckInconclusive)

    def test_definitive_contradiction_still_rejects_false_positive(self):
        replies = iter([True, True])

        checks.checkBooleanExpression = lambda expression: next(replies)
        self.assertFalse(checks.checkFalsePositives(self._injection()))
        self.assertFalse(kb.falsePositiveCheckInconclusive)


if __name__ == "__main__":
    unittest.main(verbosity=2)
