import io, os, json, threading, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch, scipy.io.wavfile as w
from transformers import VitsModel, AutoTokenizer
MODEL = "UBC-NLP/Simba-TTS-sot"
torch.set_num_threads(2)
m = VitsModel.from_pretrained(MODEL).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
m.speaking_rate = 1.0; m.noise_scale = 0.667; m.noise_scale_duration = 0.2  # "C2 even rhythm"
lock = threading.Lock()
PAGE = open(os.path.join(os.path.dirname(__file__), "index.html"), "rb").read()

def synth(text):
    text = re.sub(r"\s+", " ", text).strip()[:400]
    with lock:
        torch.manual_seed(0)
        x = tok(text, return_tensors="pt")
        with torch.no_grad():
            o = m(**x).waveform[0].numpy()
    o = o / max(1e-6, np.abs(o).max()) * 0.8
    b = io.BytesIO(); w.write(b, m.config.sampling_rate, (o * 32767).astype(np.int16)); return b.getvalue()

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ct):
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(body)
    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "content-type"); self.end_headers()
    def do_GET(self):
        if self.path.startswith("/health"): return self._send(200, b"ok", "text/plain")
        return self._send(200, PAGE, "text/html; charset=utf-8")
    def do_POST(self):
        if self.path != "/api/tts": return self._send(404, b"no", "text/plain")
        n = int(self.headers.get("Content-Length", 0)); d = json.loads(self.rfile.read(n) or b"{}")
        t = (d.get("text") or "").strip()
        if not t: return self._send(400, b'{"error":"text required"}', "application/json")
        try: return self._send(200, synth(t), "audio/wav")
        except Exception as e: return self._send(500, json.dumps({"error": str(e)[:200]}).encode(), "application/json")

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8000))), H).serve_forever()
