# MOSS LoRA 학습 앱의 바로 가기를 만든다(2026-11-14). 작업 표시줄에 고정하면 실행 중인 창과 한 칸으로 묶인다.
# 실행: powershell -NoProfile -ExecutionPolicy Bypass -File make_shortcut.ps1 [-Dest <폴더>]
# - 대상은 .bat 이 아니라 venv 의 바탕 pythonw.exe(pyvenv.cfg 의 home): .bat 바로 가기는 cmd 로 고정돼 앱 창과 따로 뜬다.
# - 바로 가기의 앱 ID(System.AppUserModel.ID)를 moss_app.py 의 APP_ID 와 같게 넣는다. 아이콘은 moss_app.ico.
# - 이 파일은 UTF-8(BOM) 이어야 PowerShell 5.1 이 한글을 바르게 읽는다.
param([string]$Dest = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$appId = 'LoraTrainingTool.MossTrainer'
$name = 'MOSS LoRA 학습'

$cfg = Join-Path $here 'sd-scripts\venv\pyvenv.cfg'
$home_ = (Get-Content $cfg | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1) -replace '^\s*home\s*=\s*', ''
$pyw = Join-Path $home_.Trim() 'pythonw.exe'
if (-not (Test-Path $pyw)) { throw "pythonw.exe 를 찾지 못했습니다: $pyw" }
$app = Join-Path $here 'moss_app.py'
$ico = Join-Path $here 'moss_app.ico'

Add-Type -TypeDefinition @"
using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
namespace MossLnk {
  [ComImport, Guid("00021401-0000-0000-C000-000000000046")] class CShellLink {}
  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
  interface IShellLinkW {
    void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder f, int cch, IntPtr fd, int flags);
    void GetIDList(out IntPtr p); void SetIDList(IntPtr p);
    void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch);
    void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch);
    void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch);
    void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string s);
    void GetHotkey(out short k); void SetHotkey(short k);
    void GetShowCmd(out int c); void SetShowCmd(int c);
    void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder s, int cch, out int i);
    void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string s, int i);
    void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string s, int r);
    void Resolve(IntPtr hwnd, int flags);
    void SetPath([MarshalAs(UnmanagedType.LPWStr)] string s);
  }
  [StructLayout(LayoutKind.Sequential, Pack = 4)] public struct PropertyKey { public Guid fmtid; public uint pid; }
  [StructLayout(LayoutKind.Explicit, Size = 16)] public struct PropVariant { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; }
  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
  interface IPropertyStore {
    void GetCount(out uint c); void GetAt(uint i, out PropertyKey k);
    void GetValue(ref PropertyKey k, out PropVariant v); void SetValue(ref PropertyKey k, ref PropVariant v); void Commit();
  }
  public static class Maker {
    public static void Make(string lnk, string target, string args, string dir, string icon, string desc, string appId) {
      var link = (IShellLinkW)new CShellLink();
      link.SetPath(target); link.SetArguments(args); link.SetWorkingDirectory(dir);
      link.SetIconLocation(icon, 0); link.SetDescription(desc);
      var store = (IPropertyStore)link;
      var key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };  // PKEY_AppUserModel_ID
      var v = new PropVariant { vt = 31, p = Marshal.StringToCoTaskMemUni(appId) };  // VT_LPWSTR
      store.SetValue(ref key, ref v); store.Commit();
      Marshal.FreeCoTaskMem(v.p);
      ((IPersistFile)link).Save(lnk, true);
    }
  }
}
"@

$lnk = Join-Path $Dest ($name + '.lnk')
[MossLnk.Maker]::Make($lnk, $pyw, ('-X utf8 "' + $app + '"'), $here, $ico, 'MOSS-Image LoRA 학습 도구', $appId)
"만들었습니다: $lnk"
"  대상: $pyw -X utf8 `"$app`""
$item = (New-Object -ComObject Shell.Application).NameSpace($Dest).ParseName($name + '.lnk')
"  앱 ID: " + $item.ExtendedProperty('System.AppUserModel.ID')
