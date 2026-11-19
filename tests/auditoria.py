#!/usr/bin/env python3
"""Auditoria de qualidade da busca e do score — NÃO envia nada e NÃO grava no jobs.db.

Roda coleta -> filtros -> descrição completa -> score da IA (a configurada no .env) e exporta
um JSON + CSV em tests/_scratch/ (gitignorado) pra você (ou outra pessoa/IA) conferir:
  - "aceitas": o que passou no filtro, com score, motivo, sinais de risco e trecho da vaga;
  - "rejeitadas": o que o filtro cortou e POR QUÊ (pra achar vaga boa cortada indevidamente).

Uso:  python tests/auditoria.py [--keywords "Dev Python" "Backend"] [--location Brasil]
                                 [--limit 6] [--max-score 12]
Gasta chamadas de IA (1 por vaga pontuada, até --max-score) — nada de candidatura.
"""
import sys, json, csv, argparse, pathlib
from datetime import datetime

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from config import Config
from collector import collect_all_jobs, enrich_job_descriptions
from filters import filter_jobs
from ats_optimizer import evaluate_match_score, load_base_curriculum
from sanitize import assess_job_risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", nargs="+", default=None)
    ap.add_argument("--location", default="Brasil")
    ap.add_argument("--limit", type=int, default=6, help="vagas por fonte")
    ap.add_argument("--max-score", type=int, default=12, help="máx. de vagas enviadas à IA")
    a = ap.parse_args()

    cfg = {}
    try:
        if Config.SEARCH_CONFIG_PATH.exists():
            cfg = json.loads(Config.SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    keywords = a.keywords or cfg.get("keywords") or ["Desenvolvedor Python"]
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    cv = load_base_curriculum()

    print(f"[auditoria] keywords={keywords} local={a.location} IA={Config.LLM_PROVIDER}")
    raw = collect_all_jobs(keywords, location=a.location, limit_per_source=a.limit, enable_linkedin_posts=False)
    print(f"[auditoria] coletadas: {len(raw)}")
    passed = filter_jobs(raw, cfg)  # devolve só as aprovadas e marca _filter_reason em cada item de `raw`
    n = enrich_job_descriptions(passed)
    print(f"[auditoria] passaram no 1º filtro: {len(passed)} (descrição completa buscada em {n})")
    final = filter_jobs(passed, cfg) if cfg else passed

    rows = []
    for j in final[: a.max_score]:
        score, reason, source = evaluate_match_score(j["title"], j["company"], j.get("description", ""), cv)
        rows.append({
            "titulo": j["title"], "empresa": j["company"], "local": j.get("location", ""), "fonte": j["platform"],
            "url": j.get("url", ""), "score": score, "origem_score": source, "motivo": reason,
            "risco": ",".join(assess_job_risk(j)), "tam_descricao": len(j.get("description") or ""),
            "trecho": (j.get("description") or "")[:600].replace("\n", " "),
        })
        print(f"  {score:>3}% [{source}] {j['title'][:50]} @ {j['company'][:25]}")
    cortadas = [{"titulo": j.get("title"), "empresa": j.get("company"), "fonte": j.get("platform"),
                 "url": j.get("url", ""), "motivo_do_corte": j.get("_filter_reason", "?"),
                 "trecho": (j.get("description") or "")[:300].replace("\n", " ")}
                for j in raw if j.get("_filter_reason") not in (None, "ok")]

    out_dir = BASE / "tests" / "_scratch"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"auditoria_{stamp}.json").write_text(
        json.dumps({"keywords": keywords, "aceitas": rows, "cortadas": cortadas}, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, data in (("aceitas", rows), ("cortadas", cortadas)):
        if data:
            with open(out_dir / f"auditoria_{stamp}_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()), delimiter=";")
                w.writeheader(); w.writerows(data)
    print(f"[auditoria] {len(rows)} pontuadas, {len(cortadas)} cortadas -> {out_dir}\\auditoria_{stamp}*")


if __name__ == "__main__":
    main()
