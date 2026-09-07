"""Static feature extraction from Java test source, for scoring unseen repos.

The research pipeline uses the FlakeFlagger dataset's precomputed features
(which include dynamic signals we cannot get without running the tests).
For the CLI we approximate the *static* families from source text alone:
test smells and size features, plus risky-API indicators.

This is deliberately regex-based rather than a full parser: the goal is a
zero-setup scorer that runs on any repo in seconds. Precision of individual
detectors is evaluated against the dataset labels in the dissertation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Risky-API patterns: each match is both a model feature and a user-facing reason.
RISKY_APIS: dict[str, tuple[re.Pattern[str], str]] = {
    "uses_sleep": (re.compile(r"Thread\.sleep|TimeUnit\.\w+\.sleep|await\(\)"),
                   "waits on wall-clock time (Thread.sleep / await)"),
    "uses_threads": (re.compile(r"new Thread|ExecutorService|CompletableFuture|Runnable|@Async|parallelStream"),
                     "starts or coordinates concurrent work"),
    "uses_network": (re.compile(r"HttpURLConnection|Socket\(|HttpClient|RestTemplate|WebClient|okhttp|localhost|127\.0\.0\.1|URL\("),
                     "talks to the network"),
    "uses_filesystem": (re.compile(r"new File\(|Files\.|FileReader|FileWriter|FileInputStream|FileOutputStream|createTempFile|Paths\.get"),
                        "reads or writes the filesystem"),
    "uses_random": (re.compile(r"new Random|Math\.random|UUID\.randomUUID|ThreadLocalRandom"),
                    "depends on randomness"),
    "uses_time": (re.compile(r"System\.currentTimeMillis|System\.nanoTime|new Date\(|LocalDate\.now|LocalDateTime\.now|Instant\.now|Calendar\.getInstance"),
                  "depends on the current time"),
    "uses_collections_order": (re.compile(r"HashMap|HashSet|\.hashCode\(\)"),
                               "relies on unordered collections or hash order"),
    "uses_static_state": (re.compile(r"static\s+(?!final)\w+[\w<>\[\], ]*\s+\w+\s*[=;]"),
                          "mutable static (shared) state"),
    "uses_timeout": (re.compile(r"@Test\s*\(\s*timeout|@Timeout|assertTimeout"),
                     "asserts on a timeout"),
}


@dataclass
class TestMethod:
    file: str
    name: str
    body: str
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# A @Test-annotated method and (approximately) its body. Brace matching is
# done separately since regex alone cannot balance braces.
TEST_ANNOTATION = re.compile(
    r"@(?:Test|ParameterizedTest|RepeatedTest)\b[^{]*?"
    r"(?:public\s+|protected\s+|private\s+)?[\w<>\[\], ]+\s+(\w+)\s*\([^)]*\)[^{]*\{",
    re.DOTALL,
)


def _method_body(source: str, open_brace_idx: int) -> str:
    depth, i = 1, open_brace_idx + 1
    while i < len(source) and depth:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    return source[open_brace_idx:i]


def extract_test_methods(java_source: str, filename: str = "<memory>") -> list[TestMethod]:
    methods = []
    for m in TEST_ANNOTATION.finditer(java_source):
        body = _method_body(java_source, m.end() - 1)
        methods.append(TestMethod(file=filename, name=m.group(1), body=body))
    return methods


_ASSERT_PATTERN = re.compile(r"\bassert\w*\s*\(|\bverify\s*\(")
_CONDITIONAL_PATTERN = re.compile(r"\b(if|for|while|switch)\s*\(")


def compute_static_features(test: TestMethod) -> TestMethod:
    body = test.body
    test.features["testLength"] = float(body.count("\n") + 1)
    test.features["numAsserts"] = float(len(_ASSERT_PATTERN.findall(body)))
    test.features["conditional-test-logic"] = float(bool(_CONDITIONAL_PATTERN.search(body)))
    for feat, (pattern, reason) in RISKY_APIS.items():
        hit = bool(pattern.search(body))
        test.features[feat] = float(hit)
        if hit:
            test.reasons.append(reason)
    return test


def scan_repo(root: Path) -> list[TestMethod]:
    """Extract features for every @Test method under root (test dirs first)."""
    tests: list[TestMethod] = []
    # "*Tests.java" files are already matched by "*Test*.java" (substring), so a
    # single rglob call covers both naming conventions with no duplicates.
    for path in sorted(root.rglob("*Test*.java")):
        try:
            source = path.read_text(errors="replace")
        except OSError:
            continue
        for t in extract_test_methods(source, str(path.relative_to(root))):
            tests.append(compute_static_features(t))
    return tests
