"""Tests for the rec-engine command line tool."""

from pathlib import Path

import pytest
import yaml

from rec_engine import cli
from rec_engine.data_loader import load_config

SAMPLE = "data/sample/ratings_sample.csv"


def test_packaged_default_config_matches_the_repo_config():
    """config.yaml is the source of truth; the copy inside the package must not drift from it."""
    assert Path("config.yaml").read_text(encoding="utf-8") == cli.DEFAULT_CONFIG.read_text(encoding="utf-8")


def test_packaged_page_and_artwork_exist():
    static = Path(cli.__file__).parent / "static"
    assert (static / "index.html").is_file()
    assert (static / "genres" / "Comedy.png").is_file()


def test_version_flag_prints_the_name(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("rec-engine ")


def test_init_writes_the_config_and_refuses_to_overwrite(tmp_path):
    target = tmp_path / "config.yaml"
    assert cli.main(["init", "--path", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == cli.DEFAULT_CONFIG.read_text(encoding="utf-8")

    target.write_text("mine", encoding="utf-8")
    assert cli.main(["init", "--path", str(target)]) == 1
    assert target.read_text(encoding="utf-8") == "mine"
    assert cli.main(["init", "--path", str(target), "--force"]) == 0


def test_config_lookup_prefers_a_named_file_then_the_local_one_then_the_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.resolve_config() == cli.DEFAULT_CONFIG
    (tmp_path / "config.yaml").write_text("a: 1", encoding="utf-8")
    assert cli.resolve_config() == Path("config.yaml")
    assert cli.resolve_config("other.yaml") == Path("other.yaml")


def test_serve_without_a_trained_model_says_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["serve"]) == 1
    assert "rec-engine train" in capsys.readouterr().err


def test_train_on_your_own_csv_creates_the_files_the_server_reads(tmp_path, monkeypatch):
    config = load_config("config.yaml")
    config["training"].update({"embedding_dim": 8, "bpr_epochs": 2, "batch_size": 32})
    config["evaluation"]["min_item_support"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    sample = str(Path(SAMPLE).resolve())

    monkeypatch.chdir(tmp_path)
    assert cli.main(["train", "--data", sample]) == 0
    assert (tmp_path / "models" / "mf_weights.npz").is_file()
    assert (tmp_path / "models" / "id_mappings.json").is_file()
