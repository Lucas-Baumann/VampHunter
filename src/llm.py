"""Chamada à IA configurada (gemini|ollama|openai|claude|groq|openrouter|custom) — todos opcionais.

Extraído de ats_optimizer.py. Novidades: nomes de modelo configuráveis pelo .env
(OPENAI_MODEL/CLAUDE_MODEL/GROQ_MODEL/CUSTOM_LLM_MODEL), retry com espera crescente em
429/5xx/timeout e diagnóstico do último erro (last_error / check_connection) — sem isso uma
chave inválida só aparecia como "IA falhou" no log e tudo caía em silêncio no score heurístico.

Como adicionar um provedor novo:
- compatível com a API da OpenAI (NVIDIA NIM, DeepSeek, Mistral, Together...): NÃO precisa de
  código — use o provedor "custom" (URL + chave + modelo) ou acrescente uma linha em PRESETS;
- API própria: um bloco em call_llm() + a chave em config.py/gui (ver gui_tabs/ia.py).
Todas as funções devolvem "" em caso de falha — quem chama decide o fallback.
"""
import time
import requests

from config import Config

_RETRY_STATUS = {429, 500, 502, 503, 504}
_BACKOFF = (2, 5)  # esperas antes da 2ª e da 3ª tentativa

# Provedores compatíveis com a API da OpenAI: nome -> (URL, atributo da chave, atributo do modelo)
OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "OPENAI_MODEL"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "GROQ_MODEL"),
}

# Atalhos da GUI (aba IA & Conexões) pro provedor "custom": nome -> URL do endpoint.
# O modelo e a chave continuam sendo do usuário (modelos gratuitos mudam sem aviso).
PRESETS = {
    "NVIDIA NIM (build.nvidia.com)": "https://integrate.api.nvidia.com/v1/chat/completions",
    "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
    "Mistral": "https://api.mistral.ai/v1/chat/completions",
    "Together AI": "https://api.together.xyz/v1/chat/completions",
}

# Último motivo de falha ("" se a última chamada deu certo) — lido por check_connection()
last_error = ""


def _fail(msg: str) -> str:
    global last_error
    last_error = msg
    print(f"[ATS AI] {msg}")
    return ""


def _explain_status(status: int, body: str) -> str:
    dica = {401: "chave inválida ou ausente", 403: "sem permissão (chave/plano)", 404: "modelo ou URL indisponível",
            429: "limite de uso atingido"}.get(status, "")
    return f"HTTP {status}" + (f" ({dica})" if dica else "") + f": {str(body)[:200]}"


def _post_with_retry(url: str, timeout: int = 45, **kwargs) -> requests.Response:
    """requests.post com até 3 tentativas em 429/5xx/erro de rede. Levanta a última exceção
    de rede se todas falharem; devolve a última resposta HTTP se o status continuar ruim."""
    for attempt in range(len(_BACKOFF) + 1):
        last = attempt == len(_BACKOFF)
        try:
            r = requests.post(url, timeout=timeout, **kwargs)
            if r.status_code not in _RETRY_STATUS or last:
                return r
        except (requests.Timeout, requests.ConnectionError):
            if last:
                raise
        time.sleep(_BACKOFF[attempt])


def _call_openai_compat(prompt: str, api_key: str, base_url: str, model: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
        r = _post_with_retry(base_url, headers=headers, json=payload)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return _fail("OpenAI-compat " + _explain_status(r.status_code, r.text))
    except Exception as e:
        return _fail(f"OpenAI-compat erro: {e}")


def call_openrouter_once(prompt: str, api_key: str, model: str, max_tokens: int = 2000) -> requests.Response:
    """Uma única chamada à API da OpenRouter, sem fallback de modelo — parametrizada (não lê
    Config) para servir tanto a call_llm() (vários modelos em sequência, chave salva) quanto ao
    botão "Testar Conexão" da GUI (aba 3), que testa a chave/modelo ainda digitados no formulário."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Lucas-Baumann/VampHunter",
        "X-Title": "VampHunter",
    }
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    return _post_with_retry("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)


def call_llm(prompt: str) -> str:
    global last_error
    last_error = ""
    provider = Config.LLM_PROVIDER

    if provider == "gemini" and Config.GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model = genai.GenerativeModel(Config.GEMINI_MODEL)
            # sem timeout a chamada poderia travar o pipeline indefinidamente
            response = model.generate_content(prompt, request_options={"timeout": 60})
            return response.text
        except Exception as e:
            return _fail(f"Erro Gemini: {e}. Usando heurístico como fallback.")
    if provider in OPENAI_COMPAT and getattr(Config, OPENAI_COMPAT[provider][1], ""):
        url, key_attr, model_attr = OPENAI_COMPAT[provider]
        return _call_openai_compat(prompt, getattr(Config, key_attr), url, getattr(Config, model_attr))
    if provider == "claude" and Config.CLAUDE_API_KEY:
        try:
            headers = {"x-api-key": Config.CLAUDE_API_KEY, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            payload = {"model": Config.CLAUDE_MODEL, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
            r = _post_with_retry("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            if r.status_code == 200:
                return "".join(c["text"] for c in r.json()["content"] if c["type"] == "text")
            return _fail("Claude " + _explain_status(r.status_code, r.text))
        except Exception as e:
            return _fail(f"Claude erro: {e}")
    if provider == "openrouter" and Config.OPENROUTER_API_KEY:
        # Modelos ":free" da OpenRouter mudam de disponibilidade sem aviso — a lista de reserva
        # vem de Config.OPENROUTER_FALLBACK_MODELS (fonte única, também usada pela GUI).
        models_to_try, seen = [], set()
        for m in [Config.OPENROUTER_MODEL] + Config.OPENROUTER_FALLBACK_MODELS:
            if m and m not in seen:
                seen.add(m)
                models_to_try.append(m)
        errors = []
        for model in models_to_try:
            try:
                r = call_openrouter_once(prompt, Config.OPENROUTER_API_KEY, model)
                if r.status_code == 200:
                    choices = r.json().get("choices")
                    if choices:
                        last_error = ""
                        return choices[0]["message"]["content"]
                    # HTTP 200 sem "choices": a OpenRouter às vezes devolve o erro dentro de um 200
                    errors.append(f"{model}: resposta 200 sem 'choices'")
                else:
                    errors.append(f"{model}: " + _explain_status(r.status_code, r.text))
            except Exception as e:
                errors.append(f"{model}: {e}")
            print(f"[ATS AI] OpenRouter {errors[-1]}")
            if "HTTP 401" in errors[-1] or "HTTP 403" in errors[-1]:
                break  # chave inválida/sem permissão vale pra TODOS os modelos — não adianta tentar os de reserva
        return _fail("OpenRouter: todos os modelos falharam — " + (errors[0] if errors else "sem detalhe"))
    if provider == "custom" and Config.CUSTOM_LLM_URL and Config.CUSTOM_LLM_KEY:
        return _call_openai_compat(prompt, Config.CUSTOM_LLM_KEY, Config.CUSTOM_LLM_URL, Config.CUSTOM_LLM_MODEL or "default")

    if provider == "ollama" or not any([Config.GEMINI_API_KEY, Config.OPENAI_API_KEY, Config.CLAUDE_API_KEY, Config.GROQ_API_KEY, Config.OPENROUTER_API_KEY, Config.CUSTOM_LLM_KEY]):
        try:
            res = requests.post(f"{Config.OLLAMA_HOST}/api/generate",
                                json={"model": Config.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=60)
            if res.status_code == 200:
                return res.json().get("response", "")
            return _fail("Ollama " + _explain_status(res.status_code, res.text))
        except Exception as e:
            return _fail(f"Ollama erro: {e}.")
    return _fail(f"provedor '{provider}' sem chave configurada")


def ai_configured() -> bool:
    """True se o provedor escolhido tem o que precisa pra ser chamado (não testa a conexão)."""
    p = Config.LLM_PROVIDER
    if p == "ollama":
        return True
    if p == "custom":
        return bool(Config.CUSTOM_LLM_URL and Config.CUSTOM_LLM_KEY)
    key = {"gemini": "GEMINI_API_KEY", "claude": "CLAUDE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(p)
    if key is None and p in OPENAI_COMPAT:
        key = OPENAI_COMPAT[p][1]
    return bool(key and getattr(Config, key, ""))


def check_connection() -> tuple:
    """Chamada mínima à IA configurada. Retorna (ok, detalhe) — detalhe explica a falha
    (chave inválida, limite, modelo fora do ar...) ou 'ok'/'não configurada'."""
    if not ai_configured():
        return False, f"nenhuma chave configurada para '{Config.LLM_PROVIDER}'"
    out = call_llm("Responda apenas: OK")
    if out and out.strip():
        return True, "ok"
    return False, last_error or "sem resposta"
