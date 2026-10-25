"""
Praja Seva server. Serves the website and provides /api/analyze.

AI providers (set any of these as environment variables; tried in this order):
  GROQ_API_KEY       free, no card: https://console.groq.com/keys
  GEMINI_API_KEY     free tier:     https://aistudio.google.com/apikey
  ANTHROPIC_API_KEY  paid

Run locally:  python server.py   then open http://localhost:8000
Check which models your keys can use:  /api/models
"""
import json, os, re, time, itertools, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = next((os.path.join(HERE, f) for f in ("index.html", "prajaa-sevaa.html")
             if os.path.exists(os.path.join(HERE, f))), os.path.join(HERE, "prajaa-sevaa.html"))
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("PRAJA_MODEL", "claude-haiku-5-5")
GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL", ""), "gemini-3.8-flash", "gemini-3.5-flash"] if m]
GROQ_MODELS = [m for m in [os.environ.get("GROQ_MODEL", ""), "meta-llama/llama-4-scout-17b-16e-instruct"] if m]
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY = 12 * 1024 * 1024
UA = "praja-seva/1.0"  # some APIs block the default Python user agent


def build_prompt(text, has_img, lang, cats, sev, cat_names):
    reply_lang = {"te": "Telugu", "en": "English"}.get(
        lang, "the same language as the citizen's text (Telugu or English)")
    return (
        "You are Praja AI, a civic assistant for Andhra Pradesh, India. "
        f"A citizen reports a local civic problem{' with a photo' if has_img else ''}. "
        f'Citizen text: "{text}". Reply language: {reply_lang}. '
        'Return ONLY JSON: {"category":one of [' + ",".join(cats) + '],'
        '"severity":one of [' + ",".join(sev) + '],'
        '"title":"English title, max 8 words","description":"English, 2-3 factual sentences on what is visible or described",'
        '"reply":"2-3 short spoken sentences in the reply language: what the problem is, which department handles it, one safety tip if relevant",'
        '"lang":"te or en"}. '
        f"Categories: {', '.join(cat_names)}. If the photo shows no civic problem, pick the closest category and say so in the reply."
    )


def parse_image(data_url):
    m = re.match(r"data:(image/[a-z+]+);base64,(.*)$", data_url or "", re.S)
    return (m.group(1), m.group(2)) if m else None


def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        text = text[a:b + 1]
    return json.loads(text)


class AIError(Exception):
    def __init__(self, code, detail=""):
        super().__init__(f"HTTP {code}")
        self.code, self.detail = code, detail


def _http_json(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise AIError(e.code, e.read().decode(errors="replace")[:300])


# ---------- generic "race a few models at once" helper ----------
_bad = {}  # "provider:model" -> time until which we skip it


def _mark_bad(key, secs):
    _bad[key] = time.time() + secs


def _usable(provider, models):
    ok = [m for m in models if _bad.get(f"{provider}:{m}", 0) < time.time()]
    return ok or models


def _race(label, provider, models, fn, budget):
    """Try 3 models at the same time; the first good answer wins."""
    deadline = time.time() + budget
    last_err, i = None, 0
    while i < len(models) and deadline - time.time() > 5:
        wave, i = models[i:i + 3], i + 3
        ex = ThreadPoolExecutor(max_workers=len(wave))
        t = min(25, deadline - time.time())
        futs = {ex.submit(fn, m, t): m for m in wave}
        try:
            for f in as_completed(futs):
                m = futs[f]
                try:
                    result = f.result()
                    print(f"{label} OK with model {m}", flush=True)
                    return result
                except AIError as e:
                    print(f"{label} error {e.code} on model {m}: {e.detail}", flush=True)
                    if e.code in (401, 403):
                        raise  # key invalid / blocked: no model will work
                    _mark_bad(f"{provider}:{m}", 86400 if e.code == 404 else 600 if e.code in (400, 429) else 60)
                    last_err = e
                except Exception as e:
                    print(f"{label} timeout/network error on model {m}: {type(e).__name__}: {e}", flush=True)
                    _mark_bad(f"{provider}:{m}", 60)
                    last_err = e
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    raise last_err or RuntimeError(f"no {label} model available")


# ---------- Groq (OpenAI-compatible) ----------
_groq_cache = {"t": 0, "list": []}


def discover_groq_models():
    now = time.time()
    if _groq_cache["list"] and now - _groq_cache["t"] < 600:
        return _groq_cache["list"]
    if not GROQ_KEY:
        return []
    try:
        req = urllib.request.Request("https://api.groq.com/openai/v1/models",
                                     headers={"Authorization": f"Bearer {GROQ_KEY}", "User-Agent": UA})
        data = _http_json(req, 15)
        ids = [m.get("id", "") for m in data.get("data", []) if m.get("active", True)]
        good = [i for i in ids
                if re.search(r"scout|maverick|vision|-vl|qwen3\.[5-9]|llama-4|gemma-?[34]", i, re.I)
                and not re.search(r"whisper|guard|tts|orpheus|playai|embed|safeguard", i, re.I)]
        good.sort(key=lambda n: 0 if "scout" in n else 1 if "maverick" in n else 2)
        _groq_cache.update(t=now, list=good)
        print("Groq vision-capable models:", good, flush=True)
        return good
    except Exception as e:
        print(f"Groq model discovery failed: {type(e).__name__}: {e}", flush=True)
        return _groq_cache["list"]


def _groq_once(model, prompt, image_url, timeout):
    text = prompt + "\nReturn only the JSON object."
    content = [{"type": "text", "text": text},
               {"type": "image_url", "image_url": {"url": image_url}}] if image_url else text
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0.2, "max_tokens": 700}).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": UA,
                                          "Authorization": f"Bearer {GROQ_KEY}"})
    data = _http_json(req, timeout)
    return clean_json(data["choices"][0]["message"]["content"])


def call_groq(prompt, image_data_url, budget):
    image_url = image_data_url if parse_image(image_data_url) else None
    models = _usable("groq", list(dict.fromkeys(GROQ_MODELS + discover_groq_models())))
    return _race("Groq", "groq", models, lambda m, t: _groq_once(m, prompt, image_url, t), budget)


# ---------- Gemini ----------
_models_cache = {"t": 0, "list": []}


def discover_models():
    now = time.time()
    if _models_cache["list"] and now - _models_cache["t"] < 600:
        return _models_cache["list"]
    if not GEMINI_KEY:
        return []
    try:
        data = _http_json(f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key={GEMINI_KEY}", 15)
        names = []
        for m in data.get("models", []):
            n = m.get("name", "").replace("models/", "")
            if ("generateContent" in m.get("supportedGenerationMethods", [])
                    and "flash" in n and not any(x in n for x in ("image", "tts", "live", "audio", "embedding"))):
                names.append(n)
        names.sort(reverse=True)
        names.sort(key=lambda n: "lite" in n)
        _models_cache.update(t=now, list=names)
        print("Available Gemini flash models:", names, flush=True)
        return names
    except Exception as e:
        print(f"Gemini model discovery failed: {type(e).__name__}: {e}", flush=True)
        return _models_cache["list"]


def _gemini_models():
    pool = [m for m in dict.fromkeys(GEMINI_MODELS + discover_models()) if "omni" not in m]
    pool = _usable("gemini", pool)
    stable = [m for m in pool if "preview" not in m and "latest" not in m]
    rest = [m for m in pool if m not in stable]
    reg = [m for m in stable if "lite" not in m]
    lite = [m for m in stable if "lite" in m]
    return [x for pair in itertools.zip_longest(reg, lite) for x in pair if x] + rest


def _gemini_once(model, body, timeout):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    data = _http_json(req, timeout)
    return clean_json(data["candidates"][0]["content"]["parts"][0]["text"])


def call_gemini(prompt, image_data_url, budget):
    parts = [{"text": prompt + "\nReturn only the JSON object."}]
    img = parse_image(image_data_url)
    if img:
        parts.insert(0, {"inline_data": {"mime_type": img[0], "data": img[1]}})
    body = json.dumps({"contents": [{"parts": parts}],
                       "generationConfig": {"responseMimeType": "application/json"}}).encode()
    return _race("Gemini", "gemini", _gemini_models(), lambda m, t: _gemini_once(m, body, t), budget)


# ---------- Claude (paid, optional) ----------
def call_claude(prompt, image_data_url, budget):
    content = []
    img = parse_image(image_data_url)
    if img:
        content.append({"type": "image", "source": {"type": "base64", "media_type": img[0], "data": img[1]}})
    content.append({"type": "text", "text": prompt + "\nReturn only the JSON object."})
    body = json.dumps({"model": CLAUDE_MODEL, "max_tokens": 600,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"content-type": "application/json", "x-api-key": ANTHROPIC_KEY,
                                          "anthropic-version": "2023-06-01"})
    data = _http_json(req, min(40, budget))
    return clean_json("".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"))


def analyze(prompt, image):
    providers = []
    if GROQ_KEY:
        providers.append(("Groq", call_groq))
    if GEMINI_KEY:
        providers.append(("Gemini", call_gemini))
    if ANTHROPIC_KEY:
        providers.append(("Claude", call_claude))
    budget = 80 if len(providers) == 1 else 35
    last = None
    for name, fn in providers:
        try:
            return fn(prompt, image, budget)
        except Exception as e:
            print(f"{name} failed: {type(e).__name__}: {e}", flush=True)
            last = e
    raise last or RuntimeError("no AI provider configured")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/prajaa-sevaa.html"):
            with open(PAGE, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/models":
            self._send(200, json.dumps({
                "groq_key_set": bool(GROQ_KEY), "groq_models": discover_groq_models(),
                "gemini_key_set": bool(GEMINI_KEY), "gemini_models": discover_models(),
            }).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._send(404, b'{"error":"not found"}')
        if not (GROQ_KEY or GEMINI_KEY or ANTHROPIC_KEY):
            return self._send(503, json.dumps({"error": "no AI key set on server"}).encode())
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_BODY:
                return self._send(413, b'{"error":"too large"}')
            req = json.loads(self.rfile.read(n).decode())
            img = req.get("image")
            prompt = build_prompt(str(req.get("text", ""))[:1000], bool(img), req.get("lang", "auto"),
                                  req.get("cats", []), req.get("sev", []),
                                  req.get("catNames", req.get("cats", [])))
            self._send(200, json.dumps(analyze(prompt, img), ensure_ascii=False).encode())
        except Exception as e:
            print(f"AI request failed: {type(e).__name__}: {e}", flush=True)
            self._send(500, json.dumps({"error": str(e)[:200]}).encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Praja Seva running on http://localhost:{PORT}", flush=True)
    names = [n for n, k in (("Groq", GROQ_KEY), ("Gemini", GEMINI_KEY), ("Claude", ANTHROPIC_KEY)) if k]
    print("AI providers: " + (", ".join(names) if names else "NONE (set GROQ_API_KEY or GEMINI_API_KEY)"), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
