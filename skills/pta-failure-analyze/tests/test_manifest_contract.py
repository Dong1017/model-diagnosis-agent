from pathlib import Path


def test_skill_yaml_manual_contract():
    skill_root = Path(__file__).resolve().parents[1]
    manifest = (skill_root / "skill.yaml").read_text(encoding="utf-8")

    assert 'schema_version: "1.0.0"' in manifest
    assert 'name: "pta-failure-analyze"' in manifest
    assert 'display_name: "PTA Failure Analyze"' in manifest
    assert 'type: "manual"' in manifest
    assert 'path: "SKILL.md"' in manifest
    assert 'report_schema: "skills/_shared/contract/report.schema.json"' in manifest
    assert 'out_dir_layout: "runs/<run_id>/out/"' in manifest
    assert 'network: "none"' in manifest
    assert 'filesystem: "workspace-write"' in manifest


def test_manifest_dependencies_are_honest_and_minimal():
    skill_root = Path(__file__).resolve().parents[1]
    manifest = (skill_root / "skill.yaml").read_text(encoding="utf-8")

    assert 'tools:' in manifest
    assert '- "bash"' in manifest
    assert '- "rg"' in manifest
    assert 'python: []' in manifest
