"""Checagem determinística do que a IA reescreve (resumo, skills, carta) contra o currículo
base — o prompt manda "não invente", mas isso nunca era verificado. Regras:
- skills: só ficam as que já existem no currículo (skills/experiências/formação/resumo);
- resumo: se trouxer número/percentual que não está no currículo nem na vaga, volta ao original;
- carta: frases com número/percentual sem lastro no currículo ou na vaga são removidas.
"""
import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def _cv_corpus(cv: dict) -> str:
    parts = [cv.get("summary", "")]
    parts += [str(x) for x in cv.get("skills", [])]
    parts += [str(x) for x in cv.get("languages", [])]
    for e in cv.get("experiences", []):
        parts += [e.get("company", ""), e.get("position", ""), e.get("period", "")] + [str(h) for h in e.get("highlights", [])]
    for e in cv.get("education", []):
        parts += [e.get("degree", ""), e.get("institution", ""), e.get("year", "")]
    return _norm(" ".join(str(p) for p in parts))


def _has_term(corpus: str, term: str) -> bool:
    t = _norm(term)
    return bool(t) and re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", corpus) is not None


_NUM = re.compile(r"\d[\d.,]*\d%?|\d%|\d{2,}")


def _numbers(text: str) -> set:
    return {re.sub(r"[.,]", "", m) for m in _NUM.findall(str(text or ""))}


def filter_skills(new_skills, base_cv: dict) -> list:
    corpus = _cv_corpus(base_cv)
    kept = [s for s in (new_skills or []) if isinstance(s, str) and _has_term(corpus, s)]
    return kept or list(base_cv.get("skills", []))


def check_generated(base_cv: dict, job_description: str, summary: str, skills, cover: str):
    """Retorna (summary, skills, cover, issues). issues = lista de avisos legíveis."""
    issues = []
    support = _numbers(_cv_corpus(base_cv)) | _numbers(job_description)

    skills_ok = filter_skills(skills, base_cv) if skills else []
    dropped = [s for s in (skills or []) if s not in skills_ok]
    if dropped:
        issues.append(f"skills fora do currículo removidas: {', '.join(map(str, dropped))[:120]}")

    if summary and (_numbers(summary) - support):
        issues.append("resumo com número sem lastro — mantido o resumo original")
        summary = ""

    if cover:
        sentences = re.split(r"(?<=[.!?])\s+", cover)
        kept = [s for s in sentences if not (_numbers(s) - support)]
        if len(kept) < len(sentences):
            issues.append(f"{len(sentences) - len(kept)} frase(s) da carta com número sem lastro removida(s)")
            cover = " ".join(kept)
    return summary, skills_ok, cover, issues
