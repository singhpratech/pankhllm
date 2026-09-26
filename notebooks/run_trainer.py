#!/usr/bin/env python3
"""Train pankhllm on your skills, prompts and logs by running the trainer notebook headless.

    python notebooks/run_trainer.py --skills my-skills/ --prompts agent.yaml --logs app.jsonl --teacher ollama

Everything the notebook writes goes to --out (default ./pankh-work): the labelled data, the
mining report, the executed notebook, and bundle/ with the server, models.json, config and
skills. Keys are read from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY,
AZURE_OPENAI_API_KEY); they are never written anywhere.
"""
import argparse, os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills", required=True, help="folder of skill .md files")
    ap.add_argument("--prompts", nargs="*", default=[], help="agent or system prompt files to ship with the skills")
    ap.add_argument("--logs", nargs="*", default=[], help="log files: question JSONL, OpenAI request/response JSONL, CSV or TXT")
    ap.add_argument("--catalog", default="", help="optional operations catalog (router.yaml with planner.operations)")
    ap.add_argument("--teacher", default="ollama", choices=["anthropic", "openai", "azure", "ollama"])
    ap.add_argument("--teacher-model", default="", help="default: claude-sonnet-5, gpt-4o-mini, your Azure deployment, gemma4:12b")
    ap.add_argument("--azure-endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    ap.add_argument("--out", default="pankh-work")
    ap.add_argument("--per-op", type=int, default=30, help="teacher-written questions per operation (catalog only; 0 to skip)")
    ap.add_argument("--templates", type=int, default=20000, help="template questions (pharma example catalog only)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--bin", default="", help="an existing pankhllm binary; default builds from source")
    ap.add_argument("--port", type=int, default=4150, help="port for the final try-it step")
    a = ap.parse_args()

    for p in [a.skills, *a.prompts, *a.logs, *([a.catalog] if a.catalog else [])]:
        if not pathlib.Path(p).exists():
            sys.exit(f"not found: {p}")
    if a.teacher == "azure" and not a.azure_endpoint:
        sys.exit("--azure-endpoint (or AZURE_OPENAI_ENDPOINT) is required for the azure teacher")
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY"}.get(a.teacher)
    if key and not os.environ.get(key):
        sys.exit(f"set {key} in the environment first")

    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        import nbformat, yaml, ipykernel  # noqa: F401
        from nbclient import NotebookClient
    except ImportError:
        # Run inside a private virtual environment with the notebook tooling installed.
        venv = out / ".venv"
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if pathlib.Path(sys.prefix).resolve() == venv.resolve():
            raise  # already inside the environment and still missing: report it
        if not py.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", "nbclient", "nbformat", "ipykernel", "pyyaml", "openai"])
        os.execv(str(py), [str(py), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])
    abspath = lambda p: str(pathlib.Path(p).resolve())
    os.environ.update({
        "PANKH_TEACHER": a.teacher,
        "PANKH_TEACHER_MODEL": a.teacher_model,
        "PANKH_SKILLS": abspath(a.skills),
        "PANKH_PROMPTS": ",".join(abspath(p) for p in a.prompts),
        "PANKH_LOGS": ",".join(abspath(p) for p in a.logs),
        "PANKH_CATALOG": abspath(a.catalog) if a.catalog else "",
        "PANKH_WORK": str(out),
        "PANKH_PER_OP": str(a.per_op),
        "PANKH_N_TRAIN": str(a.templates),
        "PANKH_N_TEST": str(max(a.templates // 10, 0)),
        "PANKH_CONCURRENCY": str(a.concurrency),
        "PANKH_PORT": str(a.port),
    })
    if a.azure_endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = a.azure_endpoint
    if a.bin:
        os.environ["PANKH_BIN"] = abspath(a.bin)

    nb = nbformat.read(HERE / "train_pankhllm.ipynb", as_version=4)
    started = time.time()
    failed = None
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    except Exception as e:  # the executed notebook still shows where it stopped
        failed = e
    nbformat.write(nb, out / "trainer.executed.ipynb")

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for o in cell.get("outputs", []):
            text = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) or "\n".join(o.get("traceback", []))
            text = "\n".join(l for l in text.splitlines() if " INFO " not in l and "\x1b[32m INFO" not in l)
            if text.strip():
                print(text.rstrip())
    print(f"\n{'FAILED' if failed else 'done'} in {time.time() - started:.0f} s; executed notebook: {out / 'trainer.executed.ipynb'}")
    if not failed:
        print(f"bundle: {out / 'bundle'}  (zip: {out / 'pankhllm-bundle.zip'})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
