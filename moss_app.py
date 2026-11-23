# MOSS LoRA 학습 UI 를 앱 창으로 띄운다 (2026-11-10, Fizgig 처럼 pywebview + WebView2).
# 실행: moss_app.bat (콘솔 창 없음). 브라우저로 쓰려면 run.bat.
# - 같은 프로세스에서 UI 서버(moss_ui)를 스레드로 돌리고, 창을 닫으면 서버도 끝난다. 학습은 별도 프로세스라 계속되고
#   다시 켜면 붙는다(moss_ui.TrainRunner). 스튜디오 렌더 프로세스는 함께 내린다.
# - 포트에 이미 MOSS UI 서버가 있으면(브라우저용 run.bat 등) 그 서버에 창만 연다.
# - 창 위치·크기는 .moss_app.json, 브라우저 저장소(마지막 설정·탭·스튜디오 설정)는 .app_profile/ 에 남는다.
# - 저장 안 된 변경이 있으면 닫기 전에 묻는다(화면이 1초마다 알려 준다. 닫는 중에 화면 JS 를 부르면 멈출 수 있어서).
import argparse
import ctypes
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# moss_app.bat 은 콘솔 창이 뜨지 않게 venv 의 바탕 파이썬(pythonw.exe)으로 실행한다(venv 의 pythonw.exe 는 uv 런처라
# 바탕의 python.exe 를 콘솔과 함께 띄운다). 그래서 venv 의 패키지(pywebview·PIL 등)를 직접 더한다. 버전은 같다(pyvenv.cfg)
VENV_SITE = os.path.join(HERE, "sd-scripts", "venv", "Lib", "site-packages")
if sys.prefix == sys.base_prefix and os.path.isdir(VENV_SITE):
    import site
    site.addsitedir(VENV_SITE)
LOG_DIR = os.path.join(HERE, "ui_logs")
STATE_FILE = os.path.join(HERE, ".moss_app.json")
PROFILE_DIR = os.path.join(HERE, ".app_profile")
TITLE = "MOSS LoRA 학습"
# 작업 표시줄에서 하나의 앱으로 묶이게(2026-11-14 사용자 요청): 프로세스의 앱 ID 를 정하고, 고정용 바로 가기
# (make_shortcut.ps1 이 만드는 'MOSS LoRA 학습.lnk')에도 같은 ID 를 넣는다. 둘이 같아야 고정한 단추와 실행 중인 창이 한 칸이 된다.
# 예전에는 .bat 바로 가기(cmd) 단추와 pythonw 창이 따로 떠서 파이썬 아이콘이 하나 더 생겼다.
APP_ID = "LoraTrainingTool.MossTrainer"
ICON = os.path.join(HERE, "moss_app.ico")


def set_app_id():
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception as e:
        print("app id:", e)


def setup_output():
    """pythonw 에는 표준 출력이 없다 → 로그 파일로 돌린다(moss_ui 의 print 가 깨지지 않게)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if sys.stdout is None or sys.stderr is None or not getattr(sys.stdout, "isatty", lambda: False)():
        log = open(os.path.join(LOG_DIR, "app.log"), "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log


def message_box(text, title=TITLE):
    try:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except Exception:
        print(text)


def ui_alive(port):
    """그 포트에 MOSS UI 서버가 떠 있나."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/studio/events?since=999999999", timeout=3) as r:
            return r.headers.get("Server", "").startswith("MossUI")
    except Exception:
        return False


def load_geom():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            g = json.load(f)
    except (OSError, ValueError):
        return {}
    # 최소화한 채 닫으면 Windows 는 -32000 같은 좌표를 준다. 모니터 밖이면 버린다
    try:
        u = ctypes.windll.user32
        vx, vy, vw, vh = u.GetSystemMetrics(76), u.GetSystemMetrics(77), u.GetSystemMetrics(78), u.GetSystemMetrics(79)
        x, y, w, h = int(g["x"]), int(g["y"]), int(g["width"]), int(g["height"])
        if w < 400 or h < 300 or x + w < vx + 80 or y < vy - 20 or x > vx + vw - 80 or y > vy + vh - 80:
            return {"maximized": bool(g.get("maximized"))}
    except Exception:
        return {}
    return g


def save_geom(g):
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(g, f)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


class Api:
    """화면이 부르는 함수(window.pywebview.api.*)."""

    def __init__(self):
        self.dirty = False

    def set_dirty(self, v):
        self.dirty = bool(v)
        return True


def main():
    set_app_id()  # 창을 만들기 전에(창이 생긴 뒤에는 이미 다른 앱으로 묶인다)
    setup_output()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7865)
    a = ap.parse_args()
    port = a.port
    url = f"http://127.0.0.1:{port}/"

    try:
        import webview
    except ImportError:
        message_box("pywebview 가 설치되어 있지 않습니다.\nvenv 에서 pip install pywebview 를 하거나, 브라우저용 run.bat 을 쓰세요.")
        return 1

    sys.path.insert(0, HERE)
    import moss_ui
    srv = None
    moss_ui.PORT = port
    moss_ui.ThreadingHTTPServer.allow_reuse_address = False  # WIN-002: 같은 포트 이중 바인딩 방지
    try:
        srv = moss_ui.ThreadingHTTPServer(("127.0.0.1", port), moss_ui.Handler)
    except OSError:
        if not ui_alive(port):
            message_box(f"포트 {port} 를 다른 프로그램이 쓰고 있습니다.\n그 프로그램을 닫거나 moss_app.bat --port 다른번호 로 실행하세요.")
            return 1
        print(f"이미 켜진 UI 서버에 창만 엽니다: {url}")
    if srv is not None:
        moss_ui.runner = moss_ui.TrainRunner()
        moss_ui.queue = moss_ui.TrainQueue(moss_ui.runner)
        moss_ui.sysmon = moss_ui.SystemMonitor()
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"MOSS 학습 UI(앱 창): {url}")

    geom = load_geom()
    api = Api()
    window = webview.create_window(
        TITLE, url, js_api=api, width=int(geom.get("width") or 1500), height=int(geom.get("height") or 940),
        x=geom.get("x"), y=geom.get("y"), min_size=(960, 600), text_select=True, confirm_close=False,
        maximized=bool(geom.get("maximized")), background_color="#121211")
    state = {"maximized": bool(geom.get("maximized")), "geom": {k: geom[k] for k in ("x", "y", "width", "height") if k in geom}}

    def remember():
        if state["maximized"]:
            return
        try:
            g = {"x": window.x, "y": window.y, "width": window.width, "height": window.height}
        except Exception:
            return
        # 최소화 중에는 Windows 가 (-32000 ÷ DPI, 작은 크기)를 준다→ 기억하지 않는다
        if g["x"] > -10000 and g["y"] > -10000 and g["width"] >= 400 and g["height"] >= 300:
            state["geom"] = g

    def on_max():
        state["maximized"] = True

    def on_restore():
        state["maximized"] = False
        remember()

    window.events.maximized += on_max
    window.events.restored += on_restore
    window.events.moved += lambda *a: remember()
    window.events.resized += lambda *a: remember()

    # 실수로 닫지 않게 학습 중지처럼 두 번 눌러야 닫힌다(확인 창 대신, 2026-11-12 사용자 요청).
    # 첫 번째는 닫지 않고 화면 오른쪽 위에 알림만 띄운다. CLOSE_WINDOW 초 안에 다시 누르면 닫힌다(Alt+F4 도 같다)
    CLOSE_WINDOW = 3.0
    close_at = {"t": -1e9}

    def notify(msg):
        # closing 처리 중(GUI 스레드)에 화면 JS 를 부르면 서로 기다리다 멈출 수 있어, 핸들러가 끝난 뒤 다른 스레드에서 띄운다
        def run():
            time.sleep(0.05)
            try:
                # 알림은 다시 누를 수 있는 시간만큼만 보인다(빨간 알림 기본 6초보다 짧게)
                window.evaluate_js(f"if (typeof toast === 'function') {{ toast({json.dumps(msg)}, true, document.querySelector('#btnStop'));"
                                   f" clearTimeout(toastTimer); toastTimer = setTimeout(() => document.querySelector('#toast').style.display = 'none', {int(CLOSE_WINDOW * 1000)}); }}")
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    def closing():
        now = time.monotonic()
        if now - close_at["t"] > CLOSE_WINDOW:
            close_at["t"] = now
            try:
                training = srv is not None and moss_ui.runner.running()
                q = moss_ui.queue  # 연속 학습의 남은 수(서버가 다음을 시작하므로 창을 닫으면 이어지지 않는다)
                queued = q.pending() if srv is not None and q is not None and q.active else 0
            except Exception:
                training, queued = False, 0
            msg = f"닫으려면 {CLOSE_WINDOW:g}초 안에 창 닫기(✕)를 한 번 더 누르세요"
            if api.dirty:
                msg += " — 저장 안 된 변경은 사라집니다"
            if training:
                msg += " · 학습은 계속됩니다"
            if queued:
                msg += f" · 연속 학습의 남은 {queued}개는 시작되지 않습니다"
            notify(msg)
            return False
        save_geom({**state["geom"], "maximized": state["maximized"]})
        return True

    window.events.closing += closing
    webview.settings["ALLOW_DOWNLOADS"] = True                # 에포크 비교의 비교표 저장
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    try:
        webview.start(private_mode=False, storage_path=PROFILE_DIR, gui="edgechromium",
                      icon=ICON if os.path.isfile(ICON) else None)  # 창·작업 표시줄 아이콘(winforms 도 .ico 를 받는다)
    except Exception as e:
        message_box(f"앱 창을 열 수 없습니다 ({e}).\nMicrosoft Edge WebView2 Runtime 이 필요합니다. 브라우저용 run.bat 을 쓰세요.")
        return 1
    finally:
        if srv is not None:
            try:
                moss_ui.studio.stop()  # 스튜디오 렌더 프로세스(GPU)를 내린다. 학습은 그대로 둔다
            except Exception:
                pass
            srv.shutdown()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)  # 미리보기 작업 프로세스 읽기 스레드 등이 남아 있어도 끝낸다(자식은 입력이 닫혀 스스로 끝남)
