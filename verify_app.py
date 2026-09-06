"""Smoke tests for the running application.

    python verify_app.py http://127.0.0.1:7860

Requires only the standard library. Works against the Flask dev server, the
gunicorn command from the Dockerfile, or a running container.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

if len(sys.argv) != 2:
    sys.exit("usage: python verify_app.py <base-url>")

BASE = sys.argv[1].rstrip('/')
results = []

def req(path, payload=None, raw=None, ct='application/json', method='POST'):
    if method == 'GET':
        r = urllib.request.Request(BASE + path)
    else:
        d = raw if raw is not None else json.dumps(payload).encode()
        r = urllib.request.Request(BASE + path, data=d, headers={'Content-Type': ct})
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            return f.status, f.headers.get('Content-Type', ''), f.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get('Content-Type', ''), e.read().decode()
    except urllib.error.URLError as e:
        # The server may reject an oversized body and close the connection
        # before the client has finished writing it.
        return 'closed', '', str(e.reason)

def check(name, cond, detail=''):
    results.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")

s, ct, b = req('/', method='GET')
check('GET / returns 200 HTML', s == 200 and 'text/html' in ct, f'status={s}')
check('GET / renders template', 'Supreme Court Opinion Classifier' in b)

s, ct, b = req('/predict', method='GET')
check('GET /predict returns 405 JSON', s == 405 and 'json' in ct, f'status={s} body={b.strip()[:60]}')

s, ct, b = req('/nope', method='GET')
check('unknown route returns 404 JSON', s == 404 and 'json' in ct, f'status={s}')

SHORT = "Fifth Amendment."
NORMAL = ("The petitioner was convicted of first-degree murder and sentenced to death. "
          "The defense argues that the confession was obtained through coercion and violated "
          "the defendant's Fifth Amendment rights against self-incrimination. The trial court "
          "failed to suppress the illegally obtained evidence.")
FIRSTAMD = ("The state legislature enacted a statute prohibiting the publication of certain "
            "political advertisements within 30 days of a general election. The plaintiff "
            "newspaper argues this constitutes a prior restraint on speech in violation of "
            "the First Amendment.")
DECOY_HEAD = ("The Court granted certiorari to consider a procedural question regarding the "
              "timeliness of the notice of appeal filed by the parties below. The docket "
              "reflects that the clerk entered judgment on the fourteenth day. The parties "
              "dispute the computation of the filing period. ")
DECOY_BODY = ("The employer refused to promote the plaintiff because of her race, in violation "
              "of Title VII of the Civil Rights Act of 1964. Evidence at trial established a "
              "pattern of systematic exclusion of Black applicants from supervisory positions "
              "over eleven years. " * 6)
LONG = DECOY_HEAD + DECOY_BODY

for name, txt, want in [('short input', SHORT, None),
                        ('normal input', NORMAL, 'Criminal Procedure'),
                        ('first amendment input', FIRSTAMD, 'First Amendment')]:
    s, ct, b = req('/predict', {'text': txt})
    d = json.loads(b)
    ok = s == 200 and len(d.get('results', [])) == 5
    top = d['results'][0] if ok else None
    if want:
        ok = ok and top['label'] == want
    check(f'{name} -> 200 with 5 labels', ok,
          f"top={top['label']} {top['score']:.4f}" if top else b[:80])

s, ct, b = req('/predict', {'text': LONG})
d = json.loads(b)
check('long input (topic hidden past first 512 chars)',
      s == 200 and d['results'][0]['label'] == 'Civil Rights',
      f"top={d['results'][0]['label']} {d['results'][0]['score']:.4f} chars={len(LONG)}")

s, _, b = req('/predict', {'text': '   '})
check('whitespace-only input -> 400', s == 400 and 'error' in json.loads(b), f'status={s}')
s, _, b = req('/predict', {})
check('missing text key -> 400', s == 400, f'status={s}')
s, _, b = req('/predict', {'text': 12345})
check('non-string text -> 400', s == 400, f'status={s}')
s, ct, b = req('/predict', raw=b'not json at all', ct='text/plain')
check('wrong content type -> 415 JSON', s == 415 and 'json' in ct, f'status={s}')
s, ct, b = req('/predict', raw=b'{"text": broken', ct='application/json')
check('malformed JSON -> 415 JSON', s == 415 and 'json' in ct, f'status={s}')
s, ct, b = req('/predict', raw=b'[1,2,3]', ct='application/json')
check('JSON array body -> 415', s == 415, f'status={s}')
s, ct, b = req('/predict', {'text': 'x' * (2 * 1024 * 1024)})
check('oversized body rejected without processing',
      (s == 413 and 'json' in ct) or s == 'closed', f'result={s} {b[:40]}')
check('server healthy after oversized body', req('/', method='GET')[0] == 200)

probs = json.loads(req('/predict', {'text': NORMAL})[2])['results']
check('scores descending', all(probs[i]['score'] >= probs[i+1]['score'] for i in range(4)))
check('top-5 scores sum <= 1', sum(p['score'] for p in probs) <= 1.0 + 1e-6,
      f"sum={sum(p['score'] for p in probs):.4f}")

a = json.loads(req('/predict', {'text': NORMAL})[2])['results'][0]['score']
b2 = json.loads(req('/predict', {'text': NORMAL})[2])['results'][0]['score']
check('inference is deterministic across calls', a == b2, f'{a} vs {b2}')

errs, oks = [], []
def worker():
    try:
        s, _, b = req('/predict', {'text': NORMAL})
        (oks if s == 200 and json.loads(b)['results'][0]['label'] == 'Criminal Procedure' else errs).append(s)
    except Exception as e:
        errs.append(repr(e))
t0 = time.time()
ths = [threading.Thread(target=worker) for _ in range(12)]
[t.start() for t in ths]; [t.join() for t in ths]
check('12 concurrent requests all correct', len(oks) == 12 and not errs,
      f'ok={len(oks)} err={errs[:2]} elapsed={time.time()-t0:.1f}s')

failed = [n for n, c, _ in results if not c]
print(f"\n{len(results)-len(failed)}/{len(results)} passed")
if failed:
    print("FAILED:", failed); sys.exit(1)
