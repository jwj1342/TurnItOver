"""Local Chromium CDP transport; uses the repository's prebuilt rendering harness."""
from pathlib import Path
import functools,http.server,threading,subprocess,json,time,urllib.request,os,shutil
import websocket

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args): pass

class Browser:
    def __init__(self,root): self.root=Path(root);self.seq=0
    def __enter__(self):
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(self.root)))
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        profile=self.root/'browser-profile';profile.mkdir(exist_ok=True)
        portfile=profile/'DevToolsActivePort'
        portfile.unlink(missing_ok=True)
        exe=os.environ.get('CHROMIUM_EXECUTABLE')
        if not exe and os.name=='nt':
            candidate=Path(os.environ.get('LOCALAPPDATA',''))/'ms-playwright/chromium-1187/chrome-win/chrome.exe'
            if candidate.is_file():exe=str(candidate)
        if not exe:exe=shutil.which('chromium') or shutil.which('chromium-browser') or shutil.which('google-chrome')
        if not exe:
            raise RuntimeError('Set CHROMIUM_EXECUTABLE to an installed Chromium/Chrome binary')
        self.process=subprocess.Popen([str(exe),'--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1',f'--user-data-dir={profile}', '--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(100):
            if portfile.exists():
                try:
                    port=int(portfile.read_text().splitlines()[0]);break
                except (ValueError,IndexError): pass
            time.sleep(.1)
        else:raise RuntimeError('Chromium startup timeout')
        request_client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        targets=json.load(request_client.open(f'http://127.0.0.1:{port}/json'))
        target=next(t for t in targets if t['type']=='page')
        self.ws=websocket.create_connection(target['webSocketDebuggerUrl'],origin='http://127.0.0.1',http_no_proxy=['127.0.0.1','localhost'],timeout=300)
        self.call('Page.navigate',{'url':f'http://127.0.0.1:{self.server.server_port}/index.html'})
        for _ in range(100):
            if self.evaluate('Boolean(window.harnessReady)'):break
            time.sleep(.1)
        else:raise RuntimeError('Harness did not initialize')
        return self
    def call(self,method,params):
        self.seq+=1;seq=self.seq
        self.ws.send(json.dumps({'id':seq,'method':method,'params':params}))
        while True:
            r=json.loads(self.ws.recv())
            if r.get('id')==seq:
                if 'error' in r:raise RuntimeError(r['error'])
                return r.get('result',{})
    def evaluate(self,expression):
        r=self.call('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True})
        if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
        return r['result'].get('value')
    def __exit__(self,*exc):
        try:self.call('Browser.close',{})
        except Exception:pass
        self.ws.close();self.process.wait(timeout=15);self.server.shutdown();self.server.server_close()
