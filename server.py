"""
Praja Seva server. Serves prajaa-sevaa.html and provides /api/analyze.
Uses Google Gemini (free tier) if GEMINI_API_KEY is set,
otherwise Anthropic Claude if ANTHROPIC_API_KEY is set.

Get a free Gemini key: https://aistudio.google.com/apikey
Run locally:  python server.py   then open http://localhost:8000
"""
import json, os, re, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = next((os.path.join(HERE, f) for f in ("index.html", "prajaa-sevaa.html")
             if os.path.exists(os.path.join(HERE, f))), os.path.join(HERE, "prajaa-sevaa.html"))
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL", ""), "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"] if m]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("PRAJA_MODEL", "claude-haiku-5-5")
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY = 12 * 1024 * 1024


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
    return json.loads(re.sub(r"```json|```", "", text).strip())


def call_gemini(prompt, image_data_url):
    parts = [{"text": prompt + "\nReturn only the JSON object."}]
    img = parse_image(image_data_url)
    if img:
        parts.insert(0, {"inline_data": {"mime_type": img[0], "data": img[1]}})
    body = json.dumps({"contents": [{"parts": parts}],
                       "generationConfig": {"responseMimeType": "application/json"}}).encode()
    last_err = None
    for model in dict.fromkeys(GEMINI_MODELS):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read().decode())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"Gemini OK with model {model}", flush=True)
            return clean_json(text)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            print(f"Gemini error {e.code} on model {model}: {detail}", flush=True)
            last_err = e
            if e.code in (401, 403, 429):
                break  # key or quota problem, other models will fail too
    raise last_err


def call_claude(prompt, image_data_url):
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
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.loads(r.read().decode())
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return clean_json(text)


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
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._send(404, b'{"error":"not found"}')
        if not GEMINI_KEY and not ANTHROPIC_KEY:
            return self._send(503, json.dumps({"error": "no AI key set on server"}).encode())
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_BODY:
                return self._send(413, b'{"error":"too large"}')
            req = json.loads(self.rfile.read(n).decode())
            text = str(req.get("text", ""))[:1000]
            img = req.get("image")
            prompt = build_prompt(text, bool(img), req.get("lang", "auto"),
                                  req.get("cats", []), req.get("sev", []),
                                  req.get("catNames", req.get("cats", [])))
            result = call_gemini(prompt, img) if GEMINI_KEY else call_claude(prompt, img)
            self._send(200, json.dumps(result, ensure_ascii=False).encode())
        except urllib.error.HTTPError as e:
            print(f"AI request failed with HTTP {e.code}", flush=True)
            self._send(502, json.dumps({"error": "upstream", "status": e.code}).encode())
        except Exception as e:
            print(f"AI request failed: {type(e).__name__}: {e}", flush=True)
            self._send(500, json.dumps({"error": str(e)[:200]}).encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Praja Seva running on http://localhost:{PORT}")
    if GEMINI_KEY:
        print(f"AI: Gemini (models: {', '.join(dict.fromkeys(GEMINI_MODELS))})")
    elif ANTHROPIC_KEY:
        print(f"AI: Claude ({CLAUDE_MODEL})")
    else:
        print("WARNING: no GEMINI_API_KEY or ANTHROPIC_API_KEY set.")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
