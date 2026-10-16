"""Build a self-contained Kaggle notebook, retaining the original GLM cells.

Source files, not generated notebook blobs, are the editable implementation.
The embedded SpecForge sources come from bootstrap_cpu.sh's pinned checkout.
"""

import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_REV = "d80daf31aacfd2193283ae36f36864e4dde199bc"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {},
            "source": source.splitlines(keepends=True)}


def payload():
    files = {}
    for folder in ("src", "tests", "configs"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ("pyproject.toml", "README.md", "scripts/cpu_smoke.sh", "scripts/benchmark_tpu.py"):
        files[name] = (ROOT / name).read_bytes()
    base = ROOT / "third_party/SpecForge"
    if not (base / "specforge").is_dir():
        raise SystemExit("Run scripts/bootstrap_cpu.sh first (pinned SpecForge source is required).")
    for path in (base / "specforge").rglob("*.py"):
        files[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ("pyproject.toml", "README.md", "version.txt", "LICENSE"):
        files[f"third_party/SpecForge/{name}"] = (base / name).read_bytes()
    files["third_party/SpecForge/SOURCE_REVISION"] = (SPEC_REV + "\n").encode()
    base = ROOT / "third_party/Torchax"
    for path in (base / "torchax").rglob("*.py"):
        files[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ("pyproject.toml", "README.md", "LICENSE", "SOURCE_REVISION"):
        files[f"third_party/Torchax/{name}"] = (base / name).read_bytes()
    for name in ("kaggle-tpu-lab.LICENSE", "transformers.LICENSE", "NOTICE.md"):
        files[f"third_party/{name}"] = (ROOT / f"third_party/{name}").read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return base64.b64encode(stream.getvalue()).decode()


def build():
    cells = [md("""# GLM + Qwen3.8-Flash-Next / SpecForge DFlash2

Domyślnie **CPU** i mały losowy Qwen4Exp. Pipeline: prompty → odpowiedzi
nauczyciela → tokeny/maski → cechy offline lub na bieżąco → trening DFlash2.
Źródła projektu, przypiętego SpecForge i Torchax są osadzone w notebooku.

Zbieranie danych i trening JAX obsługują wszystkie lokalne urządzenia TPU.
Generowanie używa cache DeltaNet/QSA/PLE, a QSA nie tworzy macierzy `[S,S]`.
Zoptymalizowany QSA wykonuje top-k i attention w JAX/XLA; MoE redukuje wyniki
przez kolektywy urządzeń. Cache wag ogranicza DMA, cache KV rośnie blokami,
a trening online zachowuje cechy nauczyciela na urządzeniach.
Dostępna jest też strumieniowa konwersja wag 4/8 bitów (`prepare-weights`).
**Testowano CPU, w tym osiem logicznych urządzeń. Nie testowano pełnych wag ani TPU.**

Wybierz tryb w następnej komórce. Instrukcja i konfiguracje zostaną rozpakowane
do `qwen_flashnext_project/README.md` oraz `qwen_flashnext_project/configs`.
"""), code('''MODE = "qwen_cpu_demo"
INSTALL_DEPS = True
BACKEND = "jax"
DEVICE = "cpu"             # wybierz tpu świadomie, po zamontowaniu pełnego nauczyciela
DEVICES = "all"            # wszystkie lokalne urządzenia JAX
WORKER_ID = 0              # 0, 1, 2, ...; inne ID na każdej instancji
JOB_CONFIG = "/kaggle/input/qwen-job-config/job.json"
ARTIFACTS = []              # przy treningu: ścieżki worker-0.zip, worker-1.zip, ...
MAX_RUN_MINUTES = 180       # własny budżet sesji; None wyłącza limit
SAVE_MARGIN_MINUTES = 10   # rezerwa na zapis, odejmowana od budżetu
CHECKPOINT_EVERY_MINUTES = 5
RESUME_FROM = None         # np. /kaggle/input/my-progress/qwen-job.resume-worker-0.zip
ARCHIVE_ON_PAUSE = True    # False: zachowaj i przenieś cały katalog zadania
import time
SESSION_STARTED = time.monotonic()
assert MODE in {"qwen_cpu_demo", "qwen_worker", "qwen_train", "glm53_legacy"}
if MODE == "qwen_cpu_demo":
    assert DEVICE == "cpu", "Demo uruchamia się na CPU"
'''), md("## Źródła i środowisko\n\nInstalacja nie pobiera wag nauczyciela. Demo tworzy mały model lokalnie.\n"), code(f'''import base64, io, os, pathlib, subprocess, sys, zipfile
PROJECT = pathlib.Path("qwen_flashnext_project").resolve()
if MODE != "glm53_legacy":
    PROJECT.mkdir(exist_ok=True)
    blob = base64.b64decode("{payload()}")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.infolist():
            if not (PROJECT / member.filename).resolve().is_relative_to(PROJECT):
                raise ValueError("Niepoprawna ścieżka archiwum")
        archive.extractall(PROJECT)
    PYTHON = PROJECT / ".venv/bin/python"
    if INSTALL_DEPS:
        subprocess.run([sys.executable, "-m", "venv", str(PROJECT / ".venv")], check=True)
        def pip_install(*items):
            subprocess.run([str(PYTHON), "-m", "pip", "install", *items], check=True)
        pip_install("--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.14.1+cpu")
        if DEVICE == "tpu":
            assert BACKEND == "jax"
            pip_install("jax[tpu]==0.8.3", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html")
        else:
            pip_install("jax[cpu]==0.8.3")
        pip_install("-e", str(PROJECT) + "[test,data]")
        pip_install("--no-deps", "-e", str(PROJECT / "third_party/SpecForge"))
        pip_install("--no-deps", "-e", str(PROJECT / "third_party/Torchax"))
    else:
        PYTHON = pathlib.Path(sys.executable)
    ENV = dict(os.environ)
    ENV["PYTHONPATH"] = os.pathsep.join(str(PROJECT / folder) for folder in
        ("src", "third_party/SpecForge", "third_party/Torchax"))
    if DEVICE == "cpu":
        ENV["JAX_PLATFORMS"] = "cpu"
    def command(*args):
        return [str(PYTHON), "-m", "qwen_tpu.cli", *map(str, args)]
    def run(*args):
        subprocess.run(command(*args), cwd=PROJECT, env=ENV, check=True)
    print("Projekt:", PROJECT)
'''), md("""## Pełny test CPU

Dwa równoległe procesy generują odpowiedzi do różnych promptów, zapisują tokeny,
maski i ZIP-y. Następnie scalane są korpusy treningowy i walidacyjny. Trzy kroki
treningu rzeczywistego SpecForge przez JAX kończą się eksportem draftera.
Nauczyciel jest losowy: test sprawdza kod, nie jakość językową.
"""), code('''if MODE == "qwen_cpu_demo":
    import json, tempfile
    (PROJECT / "runs").mkdir(exist_ok=True)
    DEMO = pathlib.Path(tempfile.mkdtemp(prefix="notebook-cpu-", dir=PROJECT / "runs"))
    run("tiny-model", "--out", DEMO / "target")
    config = json.loads((PROJECT / "configs/job.cpu.json").read_text())
    config.update(model=str(DEMO / "target"), out=str(DEMO / "job"), backend=BACKEND,
                  device="cpu", devices=DEVICES)
    config["data"]["data"] = str(PROJECT / "configs/demo-prompts.jsonl")
    config_path = DEMO / "job.json"
    config_path.write_text(json.dumps(config, indent=2))
    jobs = [subprocess.Popen(command("run", "--config", config_path, "--stage", "worker", "--worker", worker),
                             cwd=PROJECT, env=ENV) for worker in range(2)]
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(f"Błąd workera: {codes}")
    run("run", "--config", config_path, "--stage", "train", "--artifacts",
        DEMO / "job/worker-0.zip", DEMO / "job/worker-1.zip")
    assert (DEMO / "job/training/draft/model.safetensors").exists()
    print("Drafter:", DEMO / "job/training/draft")
'''), md("""## Praca z pełnym modelem

Limit ustawiasz w pierwszej komórce: `MAX_RUN_MINUTES`, `SAVE_MARGIN_MINUTES`
i `CHECKPOINT_EVERY_MINUTES`. Przykładowy budżet to 180 minut, z 10 minutami
rezerwy na zapis. Nie jest to odczyt limitu konta Kaggle. Czas instalacji od
wykonania komórki ustawień jest odejmowany od pozostałego budżetu pipeline'u.
Przy zatrzymaniu powstaje `session.json` i ZIP obok katalogu zadania:
`qwen-job.resume-worker-N.zip` lub `qwen-job.resume-train-0.zip`.

Zachowaj ZIP albo cały katalog zadania. W następnej sesji dołącz go jako Input,
ustaw `RESUME_FROM` i ten sam tryb/ID workera. `config.out` musi wskazywać
zapisywalny katalog pod `/kaggle/working`. Wejścia muszą mieć identyczną
zawartość, ale ich ścieżki mogą się zmienić. Podczas niedokończonego scalania
potrzebne są również oryginalne ZIP-y workerów.

Zapis następuje na bezpiecznych granicach (token, przykład, krok treningu,
przykład walidacyjny, fragment kopii). Długiego wywołania JAX ani zapisu dużego
checkpointu nie przerywamy w środku. Rezerwa musi je pomieścić. Nagłe odcięcie
procesu odzyskuje ostatni zapis. Sam zapis w `/kaggle/working` nie przenosi
plików do następnej sesji Kaggle.

Konfiguracja wzorcowa: `configs/job.kaggle-tpu.json`. Zamontuj kompletny model
oraz wspólny JSONL promptów. Wszystkie konta używają identycznych wejść i ustawień,
ale innego `WORKER_ID`. `DEVICES = "all"` dzieli próbki między lokalne rdzenie.
Domyślny `feature_mode = "tokens"` oszczędza dysk: podczas treningu nauczyciel
ponownie oblicza cechy. `hidden_states` zapisuje je wcześniej i pozwala trenować
bez pełnego nauczyciela, ale wymaga dużo większych artefaktów.

`qwen_worker` tworzy `worker-N.zip`. Przenieś komplet ZIP-ów na instancję
treningową, ustaw `ARTIFACTS`, wybierz `qwen_train`. Trening JAX dzieli macierze
i stany optymalizatora między lokalne rdzenie. Nie jest to trening synchronizowany
pomiędzy kontami Kaggle. Parametry pełnego modelu wymagają pomiarów jakości i zasobów.

Notebook nie loguje się na konta i nie publikuje wyników. Etapy można wznawiać.
"""), code('''if MODE in {"qwen_worker", "qwen_train"}:
    import json
    config = json.loads(pathlib.Path(JOB_CONFIG).read_text())
    config.update(backend=BACKEND, device=DEVICE, devices=DEVICES)
    config.setdefault("runtime", {}).update(
        max_run_seconds=None if MAX_RUN_MINUTES is None else max(0, MAX_RUN_MINUTES*60 - (time.monotonic()-SESSION_STARTED)),
        save_margin_seconds=SAVE_MARGIN_MINUTES*60,
        checkpoint_interval_seconds=CHECKPOINT_EVERY_MINUTES*60,
        resume_from=RESUME_FROM, archive_on_pause=ARCHIVE_ON_PAUSE)
    if MODE == "qwen_train":
        config.setdefault("training", {}).update(backend=BACKEND, device=DEVICE, devices=DEVICES)
    active = PROJECT / "active-job.json"
    active.write_text(json.dumps(config, indent=2))
    if MODE == "qwen_worker":
        run("run", "--config", active, "--stage", "worker", "--worker", WORKER_ID)
    else:
        args = ["run", "--config", active, "--stage", "train"]
        if ARTIFACTS:
            args += ["--artifacts", *ARTIFACTS]
        run(*args)
    status = pathlib.Path(config["out"]) / "session.json"
    if status.exists():
        print(status.read_text())
'''), md("""## Oryginalny GLM — tryb opcjonalny

Poniższe komórki pochodzą z dostarczonego notebooka. W trybach Qwen są pomijane.
`glm53_legacy` uruchamia oryginalny serwer GLM, jego instalację, tunel i keepalive.
Ta część nie jest testem CPU.
""")]
    original = json.loads((ROOT / "legacy/original_notebook.ipynb").read_text())
    for cell in original["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            cells.append(md(source))
            continue
        if source.startswith("%%writefile "):
            first, body = source.split("\n", 1)
            name = first.split(maxsplit=1)[1]
            source = f"from pathlib import Path\nPath({name!r}).write_text({body!r})\n"
        elif source.strip() == "!python serve_glm53.py":
            source = 'import subprocess, sys\nsubprocess.run([sys.executable, "serve_glm53.py"], check=True)\n'
        cells.append(code('if MODE == "glm53_legacy":\n' + textwrap.indent(source, "    ")))
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    for index, cell in enumerate(cells):
        cell["id"] = f"qwen-glm-{index:03d}"
    out = ROOT / "notebooks/glm-qwen-flashnext-dflash2.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    (ROOT / out.name).write_bytes(out.read_bytes())
    print(out)


if __name__ == "__main__":
    build()
