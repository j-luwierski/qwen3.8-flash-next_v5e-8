import ast
import base64
import io
import json
import zipfile
from pathlib import Path

import pytest


def test_notebook_bundle_matches_sources_and_guards_glm():
    root = Path(__file__).resolve().parents[1]
    artifact = root / "notebooks/glm-qwen-flashnext-dflash2.ipynb"
    if not artifact.exists():
        pytest.skip("Notebook build artifact is not nested inside its own embedded source archive")
    notebook = json.loads(artifact.read_text())
    if (root / artifact.name).exists():
        assert (root / artifact.name).read_bytes() == artifact.read_bytes()
    legacy = False
    archive_bytes = None
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if source.startswith("## Oryginalny GLM"):
                legacy = True
            continue
        tree = ast.parse(source)
        if legacy:
            assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
            assert ast.unparse(tree.body[0].test) == "MODE == 'glm53_legacy'"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode":
                archive_bytes = base64.b64decode(ast.literal_eval(node.args[0]))
    assert archive_bytes
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for path in (root / "src/qwen_tpu").glob("*.py"):
            assert archive.read(str(path.relative_to(root))) == path.read_bytes()
        assert archive.read("third_party/SpecForge/SOURCE_REVISION").strip() == b"d80daf31aacfd2193283ae36f36864e4dde199bc"
