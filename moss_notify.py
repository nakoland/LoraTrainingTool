# Windows 알림(토스트) — 학습이 끝났을 때·오류로 끝났을 때(2026-11-13 사용자 요청).
# 패키지 없이 Windows PowerShell 5.1 의 WinRT ToastNotificationManager 를 쓴다. 보내는 앱 이름은 Windows PowerShell 로 나온다
# (따로 앱 ID 를 등록하려면 레지스트리·시작 메뉴 바로 가기가 필요해 쓰지 않음). 표준 라이브러리만 씀(학습 파일·UI 서버 공통).
# 학습 파일(moss_train.py)이 끝날 때 직접 보내고 로그에 'moss_notified <종류>' 를 남긴다 → UI 서버는 그 줄이 없을 때만(강제 종료 등) 대신 보낸다.
# MOSS_NO_NOTIFY=1 이면 보내지 않는다(시험용).
import base64
import os
import re
import subprocess
from xml.sax.saxutils import escape

POWERSHELL_AUMID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
NOTIFIED_RE = re.compile(r"^moss_notified (\w+)")
_PS = (
    "$ErrorActionPreference = 'Stop';"
    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
    "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null;"
    "$x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml($env:MOSS_TOAST_XML);"
    "$t = [Windows.UI.Notifications.ToastNotification]::new($x);"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:MOSS_TOAST_APP).Show($t)"
)


def notify(title, body, error=False, wait=False):
    """토스트 하나를 띄운다. 기본은 기다리지 않음(Popen 을 돌려줌), wait=True 면 종료 코드(시험용). 끈 상태면 None."""
    if os.environ.get("MOSS_NO_NOTIFY"):
        return None
    body = re.sub(r"\s+", " ", str(body)).strip()
    if len(body) > 300:
        body = body[:297] + "..."
    # 오류는 오래 남게(약 25초), 텍스트는 XML 이스케이프. 환경 변수로 넘겨 따옴표·한글 문제를 피한다
    attr = ' duration="long"' if error else ""  # 파이썬 3.11: f-string 식 안에 역슬래시를 못 씀
    xml = (f'<toast{attr}><visual><binding template="ToastGeneric">'
           f"<text>{escape(str(title))}</text><text>{escape(body)}</text></binding></visual></toast>")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "__PYVENV_LAUNCHER__"}
    env.update(MOSS_TOAST_XML=xml, MOSS_TOAST_APP=POWERSHELL_AUMID)
    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
           "-EncodedCommand", base64.b64encode(_PS.encode("utf-16-le")).decode("ascii")]
    try:
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE if wait else subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    except OSError:
        return None
    if wait:
        _out, err = p.communicate(timeout=60)
        return p.returncode, err.decode("utf-8", "replace") if err else ""
    return p


# 학습이 끝나면 PC 끄기(2026-11-14 사용자 요청). 화면 스위치가 이 파일을 만들고 지운다. 학습 파일이 정상 완료 때만 보고
# 'shutdown /s /t 60' 을 건다(앱 창을 닫아도 동작). 한 번 쓰면 파일을 지운다(다음 학습이 저절로 PC 를 끄지 않게).
# MOSS_SHUTDOWN_DRY=1 이면 명령을 실행하지 않고 로그에만 남긴다(시험용).
SHUTDOWN_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".moss_shutdown_after")
SHUTDOWN_DELAY = 60
# 연속 학습(2026-11-15): 대기 중인 학습이 남아 있는 동안 UI 서버가 만들어 두는 파일. 있으면 학습 파일은 PC 를 끄지 않는다.
# 마지막 학습이면 파일이 없어 학습 파일이 직접 끈다(앱 창을 닫아 두어도). 중간 학습이 오류로 끝나 대기열이 비면 서버가 끈다
QUEUE_MORE_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".moss_queue_more")
SHUTDOWN_RE = re.compile(r"^moss_shutdown (\d+)")


def shutdown_requested():
    return os.path.exists(SHUTDOWN_FLAG)


def shutdown_if_requested(name=""):
    """플래그가 있으면 SHUTDOWN_DELAY 초 뒤 PC 끄기를 걸고 True. 로그에 'moss_shutdown <예정 시각(초)>' 를 남긴다.
    연속 학습에 남은 학습이 있으면(QUEUE_MORE_FLAG, UI 서버가 관리) 끄지 않는다 — 마지막 학습 뒤에 끈다."""
    if not shutdown_requested():
        return False
    if os.path.exists(QUEUE_MORE_FLAG):
        print("moss_shutdown_skipped 연속 학습에 남은 학습이 있어 끄지 않습니다", flush=True)
        return False
    try:
        os.remove(SHUTDOWN_FLAG)
    except OSError:
        pass
    import time
    at = int(time.time()) + SHUTDOWN_DELAY
    msg = f"MOSS 학습({name or '학습'})이 끝나 {SHUTDOWN_DELAY}초 뒤 PC 를 끕니다. 취소: 학습 화면의 '취소' 또는 shutdown /a"
    if os.environ.get("MOSS_SHUTDOWN_DRY"):
        print(f"[dry] shutdown /s /t {SHUTDOWN_DELAY}", flush=True)
    else:
        try:
            subprocess.run(["shutdown", "/s", "/t", str(SHUTDOWN_DELAY), "/c", msg[:500]], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"moss_shutdown_failed {e}", flush=True)
            return False
    notify("MOSS: 곧 PC 를 끕니다", msg, error=True)
    print(f"moss_shutdown {at}", flush=True)
    return True


def cancel_shutdown():
    """걸어 둔 PC 끄기를 취소한다. 걸린 것이 없으면 Windows 가 1116 을 돌려준다(그대로 False)."""
    if os.environ.get("MOSS_SHUTDOWN_DRY"):
        print("[dry] shutdown /a", flush=True)
        return True
    r = subprocess.run(["shutdown", "/a"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
    return r.returncode == 0


def training_done(name):
    notify("MOSS 학습 완료", f"{name or '학습'} — 학습이 끝났습니다")
    print("moss_notified done", flush=True)


def training_failed(name, message):
    notify("MOSS 학습 오류", f"{name or '학습'} — {message or '오류로 끝났습니다'}", error=True)
    print("moss_notified error", flush=True)


# 예외 없이 죽었을 때의 종료 코드 풀이(accelerate 가 'returned non-zero exit status N' 으로 남긴다)
EXIT_REASONS = {
    0x40010004: "Windows 종료·다시 시작·로그오프로 학습이 중단되었습니다(예약 종료 shutdown /s /t 포함)",
    0xC000013A: "콘솔 창을 닫거나 Ctrl+C 로 학습이 중단되었습니다",
    0xC0000005: "학습 프로세스가 메모리 접근 오류로 죽었습니다(드라이버·확장 모듈 충돌 가능)",
    0xC0000409: "학습 프로세스가 내부 오류로 강제 종료되었습니다(0xC0000409)",
    0xC00000FD: "학습 프로세스가 스택 넘침으로 죽었습니다",
    0xC0000142: "학습 프로세스를 시작하지 못했습니다(Windows 종료 중이었을 수 있음)",
}
EXIT_STATUS_RE = re.compile(r"returned non-zero exit status (\d+)")


def exit_reason(lines):
    """로그 줄에서 학습 파이썬의 종료 코드를 찾아 풀이한 글. 모르는 코드면 ''(1 같은 일반 코드는 풀이하지 않는다)."""
    for line in reversed(list(lines)):
        m = EXIT_STATUS_RE.search(line)
        if m:
            code = int(m.group(1)) & 0xFFFFFFFF
            return EXIT_REASONS.get(code, f"학습 프로세스가 예외 없이 끝났습니다(종료 코드 0x{code:08X})" if code > 255 else "")
    return ""


def error_summary(lines):
    """로그 줄에서 마지막 예외 한 줄(예: 'AssertionError: caption file is empty ...'). accelerate 의 CalledProcessError 는 뺀다."""
    for line in reversed(list(lines)):
        t = line.strip()
        m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)): ?(.*)$", t)
        if m and "CalledProcessError" not in m.group(1):
            return t
    return ""
