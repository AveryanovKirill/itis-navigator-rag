from src.completion_reports import token_stats, label_metrics


def test_report_counts_real_attempts_and_keeps_unknown_labels():
    records = [{"id": "q01", "version": "v1", "answer": {"sources": []}, "origin": "validation_fallback",
                "usage": {"prompt_tokens": 120, "completion_tokens": 30}, "elapsed_seconds": 1.5,
                "calls": [{"response": {"model": "qwen3:4b"}, "raw_provider_response": {"done_reason": "length"}},
                          {"response": {"model": "qwen3:4b"}, "raw_provider_response": {"done_reason": "stop"}}]}]
    stats = token_stats(records)
    assert stats["api_calls"] == 2 and stats["fallback_count"] == 1
    assert stats["input_total"] == 120 and stats["seconds_mean"] == 1.5
    assert stats["done_by_length"] == 1
    assert label_metrics(records, [], ["v1"])["v1"]["correct_manual"] is None
    labels = [{"id": "q01", "version": "v1", "correct_manual": "0", "hallucinated_manual": "1"}]
    assert label_metrics(records, labels, ["v1"])["v1"]["hallucinated_manual"] == 1
    assert label_metrics(records, labels, ["v1"])["v1"]["source_support_manual"] is None
