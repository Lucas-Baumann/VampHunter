"""Chamada à IA configurada (gemini|ollama|openai|claude|groq|openrouter|custom) — todos opcionais.

Extraído de ats_optimizer.py. Novidades: nomes de modelo configuráveis pelo .env
(OPENAI_MODEL/CLAUDE_MODEL/GROQ_MODEL) e retry com espera crescente em 429/5xx/timeout
(antes uma oscilação momentânea derrubava a vaga pro score heurístico).
Todas as funções devolvem "" em caso de falha — quem chama decide o fallback.
"""
import time
import requests

from config import Config

_RETRY_STATUS = {429, 500, 502, 503, 504}
_BACKOFF = (2, 5)  # esperas antes da 2ª e da 3ª tentativa


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
        print(f"[ATS AI] OpenAI-compat erro {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"[ATS AI] OpenAI-compat erro: {e}")
    return ""


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
            print(f"[ATS AI] Erro Gemini: {e}. Usando heurístico como fallback.")
            return ""
    if provider == "openai" and Config.OPENAI_API_KEY:
        return _call_openai_compat(prompt, Config.OPENAI_API_KEY, "https://api.openai.com/v1/chat/completions", Config.OPENAI_MODEL)
    if provider == "claude" and Config.CLAUDE_API_KEY:
        try:
            headers = {"x-api-key": Config.CLAUDE_API_KEY, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            payload = {"model": Config.CLAUDE_MODEL, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
            r = _post_with_retry("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            if r.status_code == 200:
                return "".join(c["text"] for c in r.json()["content"] if c["type"] == "text")
            print(f"[ATS AI] Claude erro {r.status_code}: {r.text[:300]}")
        except Exception as e:
            print(f"[ATS AI] Claude erro: {e}")
        return ""
    if provider == "groq" and Config.GROQ_API_KEY:
        return _call_openai_compat(prompt, Config.GROQ_API_KEY, "https://api.groq.com/openai/v1/chat/completions", Config.GROQ_MODEL)
    if provider == "openrouter" and Config.OPENROUTER_API_KEY:
        # Modelos ":free" da OpenRouter mudam de disponibilidade sem aviso — a lista de reserva
        # vem de Config.OPENROUTER_FALLBACK_MODELS (fonte única, também usada pela GUI).
        models_to_try, seen = [], set()
        for m in [Config.OPENROUTER_MODEL] + Config.OPENROUTER_FALLBACK_MODELS:
            if m and m not in seen:
                seen.add(m)
                models_to_try.append(m)
        for model in models_to_try:
            try:
                r = call_openrouter_once(prompt, Config.OPENROUTER_API_KEY, model)
                if r.status_code == 200:
                    choices = r.json().get("choices")
                    if choices:
                        return choices[0]["message"]["content"]
                    # HTTP 200 sem "choices": a OpenRouter às vezes devolve o erro dentro de um 200
                    print(f"[ATS AI] OpenRouter modelo '{model}': resposta 200 sem 'choices' — {r.text[:300]}")
                else:
                    print(f"[ATS AI] OpenRouter modelo '{model}' falhou: {r.status_code} - {r.text[:300]}")
            except Exception as e:
                print(f"[ATS AI] OpenRouter erro com modelo '{model}': {e}")
        print("[ATS AI] OpenRouter: todos os modelos falharam.")
        return ""
    if provider == "custom" and Config.CUSTOM_LLM_URL and Config.CUSTOM_LLM_KEY:
        return _call_openai_compat(prompt, Config.CUSTOM_LLM_KEY, Config.CUSTOM_LLM_URL, "default")

    if provider == "ollama" or not any([Config.GEMINI_API_KEY, Config.OPENAI_API_KEY, Config.CLAUDE_API_KEY, Config.GROQ_API_KEY, Config.OPENROUTER_API_KEY, Config.CUSTOM_LLM_KEY]):
        try:
            res = requests.post(f"{Config.OLLAMA_HOST}/api/generate",
                                json={"model": Config.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=60)
            if res.status_code == 200:
                return res.json().get("response", "")
        except Exception as e:
            print(f"[ATS AI] Ollama erro: {e}.")
    return ""
