"""Static regression tests for known problems in the cleaned entry points."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clean_scripts_do_not_import_old_src_namespace():
    clean_files = list((ROOT / "morph_emd").glob("*.py")) + [
        ROOT / "finetune_pli.py",
        ROOT / "infer_pli.py",
        ROOT / "finetune_frac.py",
        ROOT / "infer_frac.py",
    ]
    for path in clean_files:
        text = path.read_text(encoding="utf-8")
        assert "from src." not in text
        assert "import src." not in text


def test_poseidon_files_are_preserved():
    expected = [
        ROOT / "dataloaders" / "dataloader_frac2d_poseidon.py",
        ROOT / "dataloaders" / "dataloader_heat2d_poseidon.py",
        ROOT / "dataloaders" / "dataloaderchaos_poseidon.py",
    ]
    for path in expected:
        assert path.is_file()
