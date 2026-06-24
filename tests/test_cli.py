"""Tests for the CLI commands (offline, mock backend)."""

from __future__ import annotations

from pathlib import Path

from orchestra import cli


def _corpus() -> str:
    return str(Path(__file__).resolve().parent.parent / "data" / "sample_corpus")


def test_cli_ingest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OARAG_STORAGE_DIR", str(tmp_path / "store"))
    rc = cli.main(["ingest", _corpus()])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ingested" in out


def test_cli_ask_mock(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OARAG_STORAGE_DIR", str(tmp_path / "store2"))
    rc = cli.main(
        ["ask", "How much parental leave do employees get?", "--backend", "mock"]
    )
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "final answer" in out or "leave" in out


def test_cli_ask_hybrid(tmp_path, monkeypatch):
    monkeypatch.setenv("OARAG_STORAGE_DIR", str(tmp_path / "store3"))
    rc = cli.main(
        ["ask", "How long does the Atlas-7 battery last?", "--backend", "mock", "--hybrid"]
    )
    assert rc == 0


def test_cli_demo(monkeypatch, capsys):
    rc = cli.main(["demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q:" in out and "A:" in out


def test_cli_eval(capsys):
    rc = cli.main(["eval", "--k", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "precision@3" in out
    assert "recall@3" in out


def test_cli_eval_verbose(capsys):
    rc = cli.main(["eval", "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "grounded=" in out
