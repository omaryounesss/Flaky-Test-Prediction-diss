from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flakeguard.cli import tokenize_java
from flakeguard.extract import compute_static_features, extract_test_methods

JAVA = """
public class ExampleTest {
    @Test
    public void flakySleeper() throws Exception {
        Thread.sleep(1000);
        HashMap<String, String> map = new HashMap<>();
        if (map.isEmpty()) {
            assertEquals(1, service.call());
        }
    }

    @Test
    public void cleanTest() {
        assertEquals(2, 1 + 1);
    }

    public void notATest() {
        Thread.sleep(50);
    }
}
"""


def test_extracts_only_annotated_methods():
    methods = extract_test_methods(JAVA, "ExampleTest.java")
    assert [m.name for m in methods] == ["flakySleeper", "cleanTest"]


def test_risky_api_detection():
    methods = [compute_static_features(m) for m in extract_test_methods(JAVA)]
    flaky, clean = methods
    assert flaky.features["uses_sleep"] == 1.0
    assert flaky.features["uses_collections_order"] == 1.0
    assert flaky.features["conditional-test-logic"] == 1.0
    assert any("wall-clock" in r for r in flaky.reasons)
    assert clean.features["uses_sleep"] == 0.0
    assert clean.features["numAsserts"] == 1.0


def test_body_brace_matching_stops_at_method_end():
    methods = extract_test_methods(JAVA)
    assert "cleanTest" not in methods[0].body


def test_tokenizer_splits_camel_case_and_drops_keywords():
    tokens = tokenize_java("assertEquals(fooBarBaz, HTTPClient)").split(",")
    assert "foo" in tokens and "bar" in tokens and "baz" in tokens
    assert "http" in tokens and "client" in tokens
    assert "if" not in tokens
