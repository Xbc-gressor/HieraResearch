from pathlib import Path
import pytest
from tools.mlebench_isolation import IsolationError, audit_artifact_paths, require_public_mount

def test_public_mount_is_soft(tmp_path):
    public = tmp_path / 'public'; public.mkdir()
    assert require_public_mount(public) == public

def test_artifact_private_path_rejected(tmp_path):
    artifact = tmp_path / 'artifact'; (artifact / 'private').mkdir(parents=True)
    with pytest.raises(IsolationError): audit_artifact_paths(artifact)
