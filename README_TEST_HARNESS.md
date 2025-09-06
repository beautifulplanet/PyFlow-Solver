# Test Harness & Logging Governance

This repository enforces a stable test collection and auditable conversation log.

## Duplicate Test Discovery Prevention

Configured via `pytest.ini`:

* `norecursedirs` excludes legacy and salvage directories.
* `addopts` supplies explicit `--ignore` paths as a second guard.

CI runs `tools/assert_test_collection.py` to fail fast on accidental duplication or
unexpected shrinkage of the suite.

## Conversation Logging

Script: `scripts/log_conversation.py` appends prompt/response entries with a
filesystem lock. Set `CONVERSATION_LOG_DIR` to redirect output.

Usage example (PowerShell):

```powershell
python scripts/log_conversation.py --prompt "User question" --response "AI answer"
```

## Updating Expected Counts

If you add or remove tests intentionally, adjust `EXPECTED_MIN` in
`tools/assert_test_collection.py` accordingly.

## Slow Tests

Slow tests are marked with `@pytest.mark.slow` and executed only on one Python
version in CI to keep pipeline time predictable.
