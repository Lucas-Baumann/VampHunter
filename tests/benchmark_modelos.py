#!/usr/bin/env python3
"""Benchmark de modelos de IA pro score de aderência vaga x currículo.

Duas etapas (nada é enviado, nada grava no jobs.db):
  1) python tests/benchmark_modelos.py coletar
       junta ~30 vagas reais (busca + descrição completa) em tests/_scratch/benchmark_vagas.json.
       Depois alguém (humano ou outra IA) dá a nota "certa" de cada vaga em
       tests/_scratch/benchmark_rotulos.json  ->  {"v01": 85, "v02": 10, ...}   (0-100, pro SEU currículo)
  2) python tests/benchmark_modelos.py rodar [--modelos a b c] [--url URL]
       roda cada modelo nas vagas rotuladas e dá a nota do modelo:
         30% formato   = % de respostas que vieram em JSON utilizável (senão o app cai no heurístico)
         50% acurácia  = 100 - erro médio absoluto contra as notas de referência
         20% velocidade= 100 até 5s/vaga, caindo linear até 0 em 30s
A chave NUNCA é passada na linha de comando: vem de CUSTOM_LLM_KEY no .env da pasta de dados.

--data-dir (padrão: %LOCALAPPDATA%\\VampHunter, onde o .exe guarda currículo e .env).
"""
import sys, os, json, time, argparse, pathlib, statistics, random

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
SCRATCH = BASE / "tests" / "_scratch"
DATASET = SCRATCH / "benchmark_vagas.json"
LABELS = SCRATCH / "benchmark_rotulos.json"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# candidatos de texto da lista pública da NIM (GET /v1/models) — nomes mudam, confira se der 404
DEFAULT_MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "google/gemma-4-31b-it",
    "mistralai/mistral-large-4-1t-a52b",
    "z-ai/glm-5.3-flash",
    "minimaxai/minimax-m3.1-flash",
    "moonshotai/kimi-k3.1-pro",
]


def _data_dir(argv):
    d = None
    for i, a in enumerate(argv):
        if a == "--data-dir" and i + 1 < len(argv):
            d = argv[i + 1]
    if d is None:
        la = os.environ.get("LOCALAPPDATA")
        cand = pathlib.Path(la) / "VampHunter" if la else None
        d = str(cand) if cand and cand.exists() else str(BASE)
    return pathlib.Path(d)


DATA = _data_dir(sys.argv)
from dotenv import load_dotenv
load_dotenv(DATA / ".env", override=True)   # ANTES de importar config (ele lê o ambiente na importação)

from config import Config
Config.CURRICULUM_PATH = DATA / "curriculum_base.json"


def coletar():
    from collector import collect_all_jobs, enrich_job_descriptions
    cfg = {}
    sp = DATA / "search_config.json"
    if sp.exists():
        cfg = json.loads(sp.read_text(encoding="utf-8"))
    kws = cfg.get("keywords") or ["Desenvolvedor Front-end", "React Native"]
    raw = collect_all_jobs(kws[:5], location="Brasil", limit_per_source=4, enable_linkedin_posts=False)
    seen, uniq = set(), []
    for j in raw:
        if j.get("url") and j["url"] not in seen:
            seen.add(j["url"]); uniq.append(j)
    random.Random(7).shuffle(uniq)
    enrich_job_descriptions(uniq[:60])
    pick = [j for j in uniq if len(j.get("description") or "") >= 300][:30]
    out = {f"v{i:02d}": {k: j.get(k, "") for k in ("title", "company", "location", "platform", "url")} | {"description": (j.get("description") or "")[:3500]}
           for i, j in enumerate(pick, 1)}
    SCRATCH.mkdir(exist_ok=True)
    DATASET.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[benchmark] {len(out)} vagas em {DATASET}")
    print(f"[benchmark] agora crie {LABELS} com a nota de referência de cada id (0-100).")


def rodar(models, url):
    from ats_optimizer import evaluate_match_score, load_base_curriculum
    if not Config.CUSTOM_LLM_KEY:
        sys.exit(f"CUSTOM_LLM_KEY vazio no .env de {DATA} — cole a chave lá (arquivo local, nunca no chat).")
    cv = load_base_curriculum()
    if not cv.get("skills"):
        sys.exit(f"Currículo vazio em {Config.CURRICULUM_PATH} — use --data-dir com a pasta que tem o seu curriculum_base.json.")
    jobs = json.loads(DATASET.read_text(encoding="utf-8"))
    labels = {k: v for k, v in json.loads(LABELS.read_text(encoding="utf-8")).items() if k in jobs}
    if not labels:
        sys.exit("Sem rótulos em benchmark_rotulos.json.")
    print(f"[benchmark] {len(labels)} vagas rotuladas x {len(models)} modelos | currículo: {len(cv['skills'])} skills | pasta: {DATA}")
    Config.LLM_PROVIDER, Config.CUSTOM_LLM_URL = "custom", url
    results = []
    for m in models:
        Config.CUSTOM_LLM_MODEL = m
        errs, lat, fmt_ok, rows = [], [], 0, []
        for jid, ref in labels.items():
            j = jobs[jid]
            t0 = time.time()
            score, reason, src = evaluate_match_score(j["title"], j["company"], j["description"], cv)
            dt = time.time() - t0
            lat.append(dt)
            if src == "llm":
                fmt_ok += 1; errs.append(abs(score - ref))
            rows.append({"id": jid, "ref": ref, "score": score if src == "llm" else None, "origem": src, "seg": round(dt, 1)})
            time.sleep(1.6)  # ~35 req/min: abaixo do limite de 40 RPM do plano gratuito
        n = len(labels)
        formato = 100 * fmt_ok / n
        mae = statistics.mean(errs) if errs else 100
        acur = max(0.0, 100 - mae)
        med = statistics.median(lat)
        vel = 100 if med <= 5 else max(0.0, 100 * (30 - med) / 25)
        nota = 0.3 * formato + 0.5 * acur + 0.2 * vel
        results.append({"modelo": m, "nota": round(nota, 1), "formato": round(formato), "erro_medio": round(mae, 1), "mediana_s": round(med, 1), "linhas": rows})
        print(f"  {nota:5.1f}  {m}   formato {formato:.0f}% | erro médio {mae:.1f} pts | mediana {med:.1f}s")
    results.sort(key=lambda r: -r["nota"])
    out = SCRATCH / f"benchmark_resultado_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nRANKING:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['nota']:5.1f}  {r['modelo']}  (formato {r['formato']}%, erro {r['erro_medio']} pts, {r['mediana_s']}s)")
    print(f"\ndetalhe por vaga: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["coletar", "rodar"])
    ap.add_argument("--modelos", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--url", default=NIM_URL)
    ap.add_argument("--data-dir")
    a = ap.parse_args()
    coletar() if a.etapa == "coletar" else rodar(a.modelos, a.url)
