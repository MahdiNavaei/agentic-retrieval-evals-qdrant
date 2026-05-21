from app.data import RetrievableItem
from app.evaluate import (
    SUPPORTED_STRATEGIES,
    StrategyBenchmarkResult,
    build_parser,
    format_gain,
    main,
    merge_candidate_ids,
    validate_candidate_k,
    validate_strategy,
)
from app.qdrant_store import (
    COLLECTIONS,
    SPARSE_COLLECTIONS,
    SPARSE_VECTOR_NAME,
    create_qdrant_client,
    deterministic_point_id,
)


def test_collection_names_are_explicit_and_prefixed() -> None:
    assert COLLECTIONS == {
        "tool": "tme_tools",
        "memory": "tme_memories",
        "evidence": "tme_evidence",
    }
    assert SPARSE_COLLECTIONS == {
        "tool": "tme_tools_sparse",
        "memory": "tme_memories_sparse",
        "evidence": "tme_evidence_sparse",
    }
    assert SPARSE_VECTOR_NAME == "text-sparse"


def test_deterministic_point_id_is_stable_and_distinct() -> None:
    item = RetrievableItem(id="tax_validator", text="Validate tax.", type="tool")
    same_item = RetrievableItem(id="tax_validator", text="Different text.", type="tool")
    other_type = RetrievableItem(id="tax_validator", text="Validate tax.", type="memory")

    assert deterministic_point_id(item) == deterministic_point_id(same_item)
    assert deterministic_point_id(item) != deterministic_point_id(other_type)


def test_parser_defaults_to_dense_baseline_settings() -> None:
    args = build_parser().parse_args([])

    assert args.strategy == "dense"
    assert args.data_dir == "data"
    assert args.qdrant_mode == "local-memory"
    assert args.qdrant_url == "http://localhost:6333"
    assert args.qdrant_path == ".qdrant_local"
    assert args.model == "BAAI/bge-small-en-v1.5"
    assert args.sparse_model == "Qdrant/bm25"
    assert args.reranker_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert args.top_k == 3
    assert args.candidate_k == 10
    assert args.recreate_collections is False


def test_local_memory_qdrant_client_creation() -> None:
    client = create_qdrant_client(
        mode="local-memory",
        url="http://localhost:6333",
        path=".qdrant_local",
    )

    try:
        assert client is not None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def test_invalid_qdrant_mode_fails_before_embedding(capsys) -> None:
    exit_code = main(["--qdrant-mode", "not-a-mode"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unsupported qdrant mode" in captured.err


def test_supported_strategy_set_includes_phase4_strategies() -> None:
    assert SUPPORTED_STRATEGIES == ("dense", "dense-rerank", "sparse", "hybrid-rerank")


def test_unsupported_strategy_fails_before_runtime_dependencies(capsys) -> None:
    exit_code = main(["--strategy", "not-a-strategy"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unsupported strategy 'not-a-strategy'" in captured.err


def test_validate_strategy_accepts_phase4_strategies() -> None:
    for strategy in SUPPORTED_STRATEGIES:
        validate_strategy(strategy)


def test_candidate_k_validation_for_rerank_strategies() -> None:
    validate_candidate_k("dense", top_k=3, candidate_k=1)
    validate_candidate_k("sparse", top_k=3, candidate_k=1)
    validate_candidate_k("dense-rerank", top_k=3, candidate_k=3)

    try:
        validate_candidate_k("hybrid-rerank", top_k=3, candidate_k=2)
    except ValueError as exc:
        assert "candidate_k must be >= top_k" in str(exc)
    else:
        raise AssertionError("expected candidate_k validation failure")


def test_candidate_merge_deduplicates_and_respects_limit() -> None:
    assert merge_candidate_ids(
        ["dense_a", "shared", "dense_b"],
        ["shared", "sparse_a", "sparse_b"],
        limit=4,
    ) == ["dense_a", "shared", "dense_b", "sparse_a"]


def test_gain_formatting() -> None:
    assert format_gain(None) == "n/a"
    assert format_gain(0.15) == "+0.150"
    assert format_gain(-0.05) == "-0.050"


def test_strategy_result_shape_is_stable() -> None:
    result = StrategyBenchmarkResult(
        strategy="dense",
        dense_model="dense-model",
        sparse_model=None,
        reranker_model=None,
        qdrant_mode="local-memory",
        qdrant_url="http://localhost:6333",
        qdrant_path=".qdrant_local",
        tools_count=10,
        memories_count=10,
        evidence_count=10,
        tasks_count=20,
        top_k=3,
        candidate_k=3,
        tool_at_1=0.7,
        tool_at_3=0.9,
        memory_at_1=0.65,
        evidence_at_3=0.55,
        mean_latency_ms=20.0,
        p95_latency_ms=30.0,
    )

    assert result.strategy == "dense"
    assert result.status == "runtime-validated"
    assert result.tool_at_1_gain_vs_dense is None


def test_phase4_does_not_add_phase5_strategy() -> None:
    assert "phase-5" not in SUPPORTED_STRATEGIES
    assert "discussion" not in SUPPORTED_STRATEGIES
