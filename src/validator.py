from pathlib import Path
import subprocess
import re

def validate_ats_pdf(pdf_path: Path) -> dict:
    """Valida se PDF gerado é ATS-friendly (texto selecionável, não imagem)."""
    result = {"ok": False, "issues": [], "text_len": 0}
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        result["text_len"] = len(text)
        if len(text) < 200:
            result["issues"].append("Texto muito curto (<200 chars) — pode ser imagem")
        if "•" not in text and "-" not in text:
            result["issues"].append("Sem bullets detectados")
        # verifica se texto é selecionável (não imagem)
        if len(text.strip()) < 50:
            result["issues"].append("PDF sem texto extraível (imagem)")
        else:
            result["ok"] = len(result["issues"]) == 0
    except Exception as e:
        result["issues"].append(f"Erro leitura: {e}")
    return result

def check_pdf_with_pdftotext(pdf_path: Path) -> dict:
    """Opcional: usa pdftotext se disponível para validar."""
    try:
        r = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return {"ok": len(r.stdout) > 200, "text_len": len(r.stdout)}
    except: pass
    return {"ok": None}


def check_pdf_content(pdf_path, cv: dict) -> list:
    """Extrai o texto do PDF gerado (como um ATS faria) e devolve o que NÃO foi encontrado:
    nome, e-mail, telefone, seção de experiência e skills. Lista vazia = ok."""
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)
    flat = re.sub(r"\s+", " ", text).lower()
    digits = re.sub(r"\D", "", text)
    pi = cv.get("personal_info", {}) or {}
    missing = []
    if pi.get("name") and pi["name"].lower() not in flat:
        missing.append("nome")
    if pi.get("email") and pi["email"].lower() not in flat:
        missing.append("e-mail")
    ph = re.sub(r"\D", "", pi.get("phone") or "")
    if ph and ph not in digits:
        missing.append("telefone")
    if cv.get("experiences") and "experiência profissional" not in flat and "experiencia profissional" not in flat:
        missing.append("seção Experiência")
    if cv.get("skills") and not any(str(k).lower() in flat for k in cv["skills"][:5]):
        missing.append("skills")
    return missing
