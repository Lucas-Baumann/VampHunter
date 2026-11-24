"""Descobre e ranqueia os modelos de um provedor compatível com a API da OpenAI (provedor "custom").

Fluxo (botão "Descobrir melhores modelos" na aba IA & Conexões):
  1. list_models(): GET /models do provedor (com a sua chave) e descarta o que não é modelo de
     texto (embeddings, visão, segurança, áudio...);
  2. ping: 1 chamada mínima por modelo — descarta quem dá 404/403/timeout (sem acesso no seu plano);
  3. benchmark: os sobreviventes pontuam um conjunto FIXO de vagas fictícias contra um currículo
     fictício, cada uma com nota esperada conhecida. Nenhum dado seu sai do computador nesse teste;
  4. nota = 30% formato (JSON utilizável) + 50% acurácia (100 - erro médio) + 20% velocidade.
Os 5 melhores viram a cadeia de fallback do provedor (llm.call_llm tenta na ordem).

Limites: o conjunto é pequeno e genérico (mede se o modelo pontua com bom senso, não se ele
entende o SEU nicho) — pro teste pesado, com vagas e currículo reais, use tests/benchmark_modelos.py.
"""
import re
import time
import threading
import statistics
import requests

from config import Config

# --- classificação dos modelos listados -------------------------------------------------------
_NOT_TEXT = re.compile(
    r"embed|rerank|retriev|guard|safety|moderat|reward|clip|vlm|vision|[-_/]vl[-_/\d]|ocr|parse|"
    r"translate|riva|cosmos|detector|calibration|whisper|tts|stt|audio|speech|image|diffus|flux|"
    r"sdxl|video|starcoder|code-?llama|coder|bge|e5-|arctic-embed|nvclip|topic-control|nemoguard|pii|gliner|ner-", re.I)


def is_text_candidate(model_id: str) -> bool:
    return bool(model_id) and not _NOT_TEXT.search(model_id)


def models_url(chat_url: str) -> str:
    """https://x/v1/chat/completions -> https://x/v1/models"""
    return re.sub(r"/chat/completions/?$", "", (chat_url or "").rstrip("/")) + "/models"


def list_models(chat_url: str, api_key: str = "", timeout: int = 30) -> list:
    """IDs dos modelos de texto que o provedor anuncia. Levanta RuntimeError com o motivo se falhar."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(models_url(chat_url), headers=headers, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"não consegui falar com o provedor: {e}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} ao listar modelos: {r.text[:150]}")
    try:
        ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception:
        raise RuntimeError("resposta de /models em formato inesperado")
    return sorted({i for i in ids if is_text_candidate(i)})


# --- conjunto fixo de calibração (fictício) ----------------------------------------------------
DEMO_CV = {
    "personal_info": {"name": "Pessoa Exemplo"},
    "summary": "Desenvolvedor(a) júnior mobile e front-end, 2 anos de experiência, inglês intermediário.",
    "skills": ["React Native", "TypeScript", "JavaScript", "React", "Node.js", "SQL", "Git", "APIs REST"],
    "experiences": [{"company": "Empresa A", "position": "Desenvolvedor Júnior", "period": "2024 - Atual",
                     "highlights": ["Apps em React Native com TypeScript", "Integração com APIs REST"]}],
    "education": [{"degree": "Tecnólogo em Análise e Desenvolvimento de Sistemas", "institution": "Faculdade", "year": "2023"}],
}

# (título, empresa, descrição, nota esperada 0-100)
DEMO_JOBS = [
    ("Desenvolvedor(a) React Native Júnior", "Acme Apps",
     "Vaga remota, CLT. Buscamos dev júnior para evoluir nosso app em React Native com TypeScript. Requisitos: "
     "React Native, TypeScript, consumo de APIs REST, Git. Diferenciais: Node.js. Inglês intermediário.", 90),
    ("Desenvolvedor Front-end Pleno (React/Next.js)", "Loja Online",
     "Remoto. 3+ anos com React e TypeScript, Next.js, testes automatizados e performance web. Inglês intermediário desejável.", 70),
    ("Desenvolvedor Full Stack Júnior (Node + React)", "Software House Sul",
     "Presencial em Porto Alegre, 5 dias por semana. Node.js, React e SQL. Nível júnior. CLT.", 60),
    ("Analista de Design UI/UX Júnior", "Estúdio Criativo",
     "Criação de interfaces no Figma, protótipos e design system. Conhecimento básico de HTML e CSS é um diferencial.", 25),
    ("Engenheiro(a) de Software Sênior Java", "Banco Grande",
     "8+ anos com Java, Spring Boot, microsserviços, Kafka e arquitetura de alta disponibilidade. Presencial.", 10),
    ("Engenheiro(a) DevOps Especialista", "Cloud Corp",
     "6+ anos com Kubernetes, Terraform, AWS e pipelines CI/CD. Plantão. Inglês avançado obrigatório.", 8),
    ("Cientista de Dados Sênior", "Fintech X",
     "Mestrado em estatística, 7 anos com Python, modelos de machine learning em produção e Spark.", 8),
    ("Atendente de loja", "Mercado Bairro",
     "Atendimento ao cliente, reposição de produtos e operação de caixa. Escala 6x1.", 0),
]

_cfg_lock = threading.Lock()

# Limites do teste — um modelo lento/travado não pode segurar a descoberta por minutos:
PER_CALL_TIMEOUT = 25      # prazo RÍGIDO por chamada (o timeout do requests só mede o intervalo entre bytes)
MAX_CONSECUTIVE_FAILS = 2  # 2 falhas seguidas num modelo = pula o resto dele
MAX_TOTAL_SECONDS = 600    # teto da descoberta inteira; estourou, devolve o que já foi avaliado


def _guarded(fn, seconds):
    """Roda fn() com prazo rígido. Devolve (ok, resultado); estourou o prazo -> (False, None).
    A thread abandonada termina sozinha quando o servidor responder (é daemon)."""
    box = {}
    def run():
        try:
            box["v"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive() or "v" not in box:
        return False, None
    return True, box["v"]


def compute_nota(formato: float, erro_medio: float, mediana_s: float) -> float:
    acur = max(0.0, 100 - erro_medio)
    vel = 100 if mediana_s <= 5 else max(0.0, 100 * (30 - mediana_s) / 25)
    return round(0.3 * formato + 0.5 * acur + 0.2 * vel, 1)


class _CustomOverride:
    """Aponta Config.* pro provedor/modelo em teste e restaura no fim (um teste por vez)."""
    def __init__(self, url, key, model):
        self.new = ("custom", url, key, model, [])

    def __enter__(self):
        _cfg_lock.acquire()
        self.old = (Config.LLM_PROVIDER, Config.CUSTOM_LLM_URL, Config.CUSTOM_LLM_KEY, Config.CUSTOM_LLM_MODEL, Config.CUSTOM_LLM_MODELS)
        Config.LLM_PROVIDER, Config.CUSTOM_LLM_URL, Config.CUSTOM_LLM_KEY, Config.CUSTOM_LLM_MODEL, Config.CUSTOM_LLM_MODELS = self.new
        import llm
        self.old_net = (llm.REQUEST_TIMEOUT, llm.RETRIES_ENABLED)
        llm.REQUEST_TIMEOUT, llm.RETRIES_ENABLED = PER_CALL_TIMEOUT - 5, False
        return self

    def __exit__(self, *exc):
        import llm
        llm.REQUEST_TIMEOUT, llm.RETRIES_ENABLED = self.old_net
        Config.LLM_PROVIDER, Config.CUSTOM_LLM_URL, Config.CUSTOM_LLM_KEY, Config.CUSTOM_LLM_MODEL, Config.CUSTOM_LLM_MODELS = self.old
        _cfg_lock.release()


def rank_models(chat_url: str, api_key: str, models: list = None, top: int = 5, max_candidates: int = 10,
                progress=None, should_stop=None, pause: float = 1.6) -> list:
    """Retorna os `top` melhores: [{'model','nota','formato','erro_medio','mediana_s'}], melhor primeiro.
    `progress(msg)` recebe o andamento; `should_stop()` cancela. `pause` espaça as chamadas
    (planos gratuitos costumam limitar ~40 requisições/min)."""
    import llm
    from ats_optimizer import evaluate_match_score
    say = progress or (lambda m: None)
    stop = should_stop or (lambda: False)

    start = time.time()
    over = lambda: time.time() - start > MAX_TOTAL_SECONDS

    if models is None:
        say("Listando os modelos do provedor…")
        models = list_models(chat_url, api_key)
    if not models:
        return []

    # 2. ping — só segue com quem responde nesse plano (e rápido: PER_CALL_TIMEOUT)
    alive, lentos = [], []
    for i, m in enumerate(models, 1):
        if stop():
            return []
        if over():
            say("Tempo limite do teste atingido — usando o que já foi testado.")
            break
        say(f"Testando acesso {i}/{len(models)}: {m}")
        with _CustomOverride(chat_url, api_key, m):
            t0 = time.time()
            done, out = _guarded(lambda: llm.call_llm("Responda apenas: OK"), PER_CALL_TIMEOUT)
            dt = time.time() - t0
        if not done:
            lentos.append(m)
        elif out and out.strip():
            alive.append((dt, m))
        time.sleep(pause)
    alive.sort()  # os mais rápidos primeiro — o benchmark é limitado a max_candidates
    cands = [m for _, m in alive[:max_candidates]]
    if not cands:
        if lentos:
            say(f"{len(lentos)} modelo(s) não responderam em {PER_CALL_TIMEOUT}s e foram descartados.")
        return []

    # 3. benchmark
    results = []
    for mi, m in enumerate(cands, 1):
        if over():
            say("Tempo limite do teste atingido — usando o que já foi avaliado.")
            break
        errs, lat, ok, fails = [], [], 0, 0
        for ji, (title, company, desc, ref) in enumerate(DEMO_JOBS, 1):
            if stop():
                return []
            say(f"Avaliando {mi}/{len(cands)} ({m}) — vaga {ji}/{len(DEMO_JOBS)}")
            with _CustomOverride(chat_url, api_key, m):
                t0 = time.time()
                done, res = _guarded(lambda: evaluate_match_score(title, company, desc, DEMO_CV), PER_CALL_TIMEOUT)
                lat.append(time.time() - t0)
            if done and res and res[2] == "llm":
                ok += 1
                fails = 0
                errs.append(abs(res[0] - ref))
            else:
                fails += 1
                if fails >= MAX_CONSECUTIVE_FAILS:   # disjuntor: não insiste num modelo ruim
                    break
            time.sleep(pause)
        formato = 100 * ok / len(DEMO_JOBS)
        mae = statistics.mean(errs) if errs else 100.0
        med = statistics.median(lat)
        results.append({"model": m, "nota": compute_nota(formato, mae, med), "formato": round(formato),
                        "erro_medio": round(mae, 1), "mediana_s": round(med, 1)})
    results.sort(key=lambda r: -r["nota"])
    return results[:top]
