"""Tratamento de texto NÃO CONFIÁVEL (vagas raspadas) antes de ir pra IA, e sinais de risco.

Duas frentes:
- sanitize_untrusted(): remove o que serve pra esconder instrução (caracteres invisíveis,
  HTML, controle) — a vaga vira texto puro e limitado.
- assess_job_risk(): sinais heurísticos de golpe/injeção/e-mail suspeito. Qualquer sinal
  bloqueia o ENVIO AUTOMÁTICO (a vaga vai pra revisão manual); não descarta a vaga.
"""
import re
import unicodedata

# zero-width, bidi override, BOM, soft hyphen — usados pra esconder texto de quem revisa
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿­]")
_TAGS = re.compile(r"<[^>]{0,200}>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_INJECTION = [
    r"ignor[ea]\s+(?:todas?\s+)?(?:as\s+)?(?:instru[çc][õo]es|regras)",
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
    r"\bsystem\s*prompt\b",
    r"voc[êe]\s+agora\s+[ée]\b",
    r"you\s+are\s+now\b",
    r"(?:novas?|new)\s+(?:instru[çc][õo]es|instructions)\s*:",
    r"(?:atribua|d[êe]|give|set)\s+(?:a\s+)?(?:nota|score|match)\s+(?:de\s+)?(?:100|m[áa]xim)",
    r"match[_ ]score\s*[\"']?\s*:\s*100",
    r"responda\s+(?:apenas|somente)\s+com",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION]

_SCAM = [
    (r"\b(?:taxa|pagamento)\s+(?:de\s+)?(?:inscri[çc][ãa]o|cadastro|treinamento|kit|material)", "pede_pagamento"),
    (r"\bpag(?:ar|ue)\s+(?:uma\s+)?taxa", "pede_pagamento"),
    (r"\b(?:pix|dep[óo]sito)\s+(?:antecipad|para\s+iniciar)", "pede_pagamento"),
    (r"\b(?:ganhe|ganhos?)\s+(?:at[ée]\s+)?r\$\s*[\d.]+\s*(?:por\s+(?:dia|hora|semana))", "ganho_irreal"),
    (r"\b(?:sem\s+experi[êe]ncia).{0,40}r\$\s*\d{1,3}\.?\d{3}", "ganho_irreal"),
    (r"\b(?:renda\s+extra|investimento\s+inicial|marketing\s+multin[ií]vel|pir[âa]mide)\b", "esquema_renda"),
    (r"(?:chame|contato|envie)\s+(?:apenas\s+|somente\s+)?(?:no\s+|pelo\s+)?(?:whatsapp|telegram)", "contato_so_mensageiro"),
    (r"\b(?:bit\.ly|tinyurl\.com|cutt\.ly|t\.me|wa\.me)/", "link_encurtado_ou_mensageiro"),
    (r"\b(?:cpf|rg|senha|dados\s+banc[áa]rios|n[úu]mero\s+do\s+cart[ãa]o)\b.{0,40}(?:envie|informe|mande)", "pede_dados_sensiveis"),
]
_SCAM_RE = [(re.compile(p, re.I), tag) for p, tag in _SCAM]

_FREE_MAIL = {"gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "yahoo.com.br", "live.com", "icloud.com", "bol.com.br", "uol.com.br", "proton.me", "protonmail.com"}


def sanitize_untrusted(text: str, max_len: int = 3000) -> str:
    """Texto puro, sem invisíveis/HTML/controle, espaços normalizados e limitado."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = _INVISIBLE.sub("", t)
    t = _CTRL.sub(" ", t)
    t = _TAGS.sub(" ", t)
    t = t.replace("```", "'''")  # não deixa a vaga fechar/abrir bloco de código do prompt
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t[:max_len]


def has_hidden_text(text: str) -> bool:
    return bool(_INVISIBLE.search(str(text or "")))


def looks_like_injection(text: str) -> bool:
    return any(p.search(text or "") for p in _INJECTION_RE)


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def email_domain_matches_company(email: str, company: str) -> bool:
    dom = (email.rsplit("@", 1)[-1] if "@" in email else "").lower()
    if not dom:
        return False
    base = _slug(dom.split(".")[0])
    comp = _slug(company)
    return bool(base and comp and (base in comp or comp in base))


def assess_job_risk(job: dict) -> list:
    """Lista de sinais de risco (vazia = nada suspeito). Qualquer sinal bloqueia auto-envio."""
    raw = f"{job.get('title') or ''}\n{job.get('description') or ''}"
    flags = []
    if has_hidden_text(raw):
        flags.append("texto_oculto")
    if looks_like_injection(raw):
        flags.append("possivel_injecao_prompt")
    for pat, tag in _SCAM_RE:
        if pat.search(raw) and tag not in flags:
            flags.append(tag)
    email = (job.get("contact_email") or "").strip().lower()
    if email:
        dom = email.rsplit("@", 1)[-1]
        if dom in _FREE_MAIL:
            flags.append("email_generico")
        elif not email_domain_matches_company(email, job.get("company") or ""):
            flags.append("email_dominio_diferente_da_empresa")
    return flags


def minimal_cv_for_llm(cv: dict) -> dict:
    """Currículo sem telefone/e-mail/endereço/links — o score e o texto da carta não precisam
    disso, e assim a IA/provedor recebe menos dado pessoal (e menos tokens)."""
    out = {k: v for k, v in (cv or {}).items() if k != "personal_info"}
    name = ((cv or {}).get("personal_info") or {}).get("name")
    if name:
        out["personal_info"] = {"name": name}
    return out


_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def scrub_generated_text(text: str, allowed: str = "", max_len: int = 2500) -> str:
    """Saída da IA (carta): remove URLs e e-mails que NÃO estão no currículo (`allowed`) —
    barra tentativa de injeção de induzir a carta a apontar pra link/endereço do atacante."""
    allowed_l = (allowed or "").lower()
    def _drop(m):
        return m.group(0) if m.group(0).lower().rstrip(".,;)") in allowed_l else "[removido]"
    t = _URL.sub(_drop, str(text or ""))
    t = _EMAIL.sub(_drop, t)
    return t[:max_len]
