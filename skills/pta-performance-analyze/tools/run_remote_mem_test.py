#!/usr/bin/env python3
"""
远程显存测试工具 - 在 Ascend/GPU 服务器上并行运行显存基准测试。

使用 SSH_ASKPASS + subprocess（与 remote_deploy_build.py 相同方式），
无需 paramiko 等第三方依赖。

功能:
  - 并行连接 Ascend (NPU) 和 GPU 服务器
  - 上传测试脚本并执行，捕获显存统计 JSON
  - [Ascend] 开启 plog，记录 PID，执行后用 filter_plog_memory.py 过滤并回传
  - 生成对比结果 mem_results.md

用法:
    python run_remote_mem_test.py <npu_script> <gpu_script> [选项]

示例:
    python run_remote_mem_test.py torchapi_id0299_nanmean.py torchapi_id0299_nanmean_gpu.py
    python run_remote_mem_test.py npu.py gpu.py --skip-gpu --gpu-json '{"target_api":"torch.nanmean","total_driver_GB":4.27}'
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SERVERS_JSON = SCRIPT_DIR.parent / "servers.json"
FILTER_NPU_SCRIPT = SCRIPT_DIR / "filter_plog_memory.py"

# Prefer system OpenSSH; allow override via env on Windows
SSH_BIN = os.environ.get("SSH_BIN", "ssh")
SCP_BIN = os.environ.get("SCP_BIN", "scp")

_print_lock = threading.Lock()


def log(tag, msg):
    with _print_lock:
        print(f"{tag} {msg}", flush=True)


# ─── config helpers ────────────────────────────────────────────────────────

def load_servers(config_path):
    """Load servers.json and return the 'servers' dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("servers", {})


def extract_api_name(script_path):
    """从脚本中提取 TARGET_API = '...' 的值。"""
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r'^TARGET_API\s*=\s*["\'](.+?)["\']', line)
            if m:
                return m.group(1)
    return None


def extract_key_code_lines(script_path, api_name):
    """从测试脚本中提取 API 调用的关键代码行。
    
    提取逻辑：在 calculate_xxx_non32aligned() 函数内，
    从 device = torch.device(...) 的下一行非空行开始，
    到 output = torch.<api>... 结束。
    
    返回: (key_lines, start_line, end_line) 或 (None, None, None)
    """
    if not script_path or not os.path.isfile(script_path):
        return None, None, None
    
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    in_non32aligned_func = False
    device_line_idx = None
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if re.match(r'^def\s+calculate_\w+_non32aligned\s*\(', stripped):
            in_non32aligned_func = True
            continue
        
        if in_non32aligned_func:
            if stripped.startswith("def ") and "non32aligned" not in stripped:
                break
            
            if device_line_idx is None and re.match(r'^device\s*=\s*torch\.device\s*\(', stripped):
                device_line_idx = i
                continue
            
            if device_line_idx is not None and start_idx is None and stripped:
                start_idx = i
            
            if start_idx is not None and re.match(r'^output\s*=\s*torch\.\w+', stripped):
                end_idx = i
                break
    
    if start_idx is not None and end_idx is not None:
        key_lines = "".join(lines[start_idx:end_idx + 1]).rstrip()
        return key_lines, start_idx + 1, end_idx + 1
    
    return None, None, None


# ─── SSH helpers (SSH_ASKPASS, same as remote_deploy_build.py) ─────────────

def _make_askpass(password: str) -> str:
    """Create a .bat helper that echoes the password (Windows SSH_ASKPASS)."""
    bat = os.path.join(tempfile.gettempdir(), "mem_test_askpass.bat")
    # Use an environment variable to avoid special characters in the password
    # being interpreted by cmd (e.g. &, |, >). We escape % to avoid expansion.
    safe = password.replace("%", "%%")
    with open(bat, "w") as f:
        f.write("@echo off\n")
        f.write(f"set P={safe}\n")
        f.write("echo %P%\n")
    return bat


def _ssh_env(askpass_bat: str) -> dict:
    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass_bat
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = ":0"
    return env


def ssh_run(target, cmd, env, timeout=600):
    """通过 SSH 执行远程命令，返回 (stdout, stderr, returncode)。"""
    r = subprocess.run(
        [SSH_BIN, "-o", "StrictHostKeyChecking=no", target, cmd],
        env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout, r.stderr, r.returncode


def scp_upload(target, local_path, remote_path, env, timeout=120):
    """SCP 上传文件到远程服务器。"""
    r = subprocess.run(
        [SCP_BIN, "-o", "StrictHostKeyChecking=no",
         str(local_path), f"{target}:{remote_path}"],
        env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode == 0, r.stderr


def scp_download(target, remote_path, local_path, env, timeout=120):
    """SCP 从远程服务器下载文件。"""
    r = subprocess.run(
        [SCP_BIN, "-o", "StrictHostKeyChecking=no",
         f"{target}:{remote_path}", str(local_path)],
        env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode == 0, r.stderr


# ─── Ascend (NPU) test ────────────────────────────────────────────────────

def run_ascend_test(cfg, npu_script, api_name, out_dir, results):
    T = "[Ascend]"
    host, user, pw = cfg["host"], cfg["user"], cfg["password"]
    base = cfg["remote_dir"]
    wdir = f"{base}/{api_name}"
    target = f"{user}@{host}"

    askpass = _make_askpass(pw)
    env = _ssh_env(askpass)

    try:
        log(T, f"连接 {host} ...")

        # 创建工作目录
        ssh_run(target, f"mkdir -p {wdir}", env)
        log(T, f"远程工作目录: {wdir}")

        # 上传文件
        sn = os.path.basename(npu_script)
        ok, err = scp_upload(target, npu_script, f"{wdir}/{sn}", env)
        if not ok:
            log(T, f"上传 {sn} 失败: {err}")
            results["ascend_error"] = f"SCP upload failed: {err}"
            return

        fn = FILTER_NPU_SCRIPT.name
        scp_upload(target, str(FILTER_NPU_SCRIPT), f"{wdir}/{fn}", env)
        log(T, f"已上传: {sn}, {fn}")

        # 构建远程执行命令
        remote_cmd = (
            f"cd {wdir} && "
            f"{{ [ -f {base}/env.sh ] && source {base}/env.sh; true; }} && "
            f"export ASCEND_GLOBAL_LOG_LEVEL=0 && "
            f"export ASCEND_PROCESS_LOG_PATH={wdir} && "
            f"python {sn} > {wdir}/_stdout.txt 2>&1 & "
            f"SCRIPT_PID=$! && "
            f"echo SCRIPT_PID=$SCRIPT_PID && "
            f"wait $SCRIPT_PID; "
            f"echo EXIT_CODE=$?; "
            f"cat {wdir}/_stdout.txt"
        )

        log(T, "执行 NPU 测试脚本 ...")
        out, err, _ = ssh_run(target, remote_cmd, env, timeout=600)
        lines = out.strip().split("\n")

        pid, exit_code, json_result = None, -1, None
        for l in lines:
            l = l.strip()
            m = re.match(r"SCRIPT_PID=(\d+)", l)
            if m:
                pid = m.group(1)
            m2 = re.match(r"EXIT_CODE=(\d+)", l)
            if m2:
                exit_code = int(m2.group(1))
            if l.startswith("{") and "target_api" in l:
                try:
                    json_result = json.loads(l)
                except json.JSONDecodeError:
                    pass

        log(T, f"PID={pid}, exit_code={exit_code}")

        if exit_code != 0:
            log(T, f"执行失败!\nstdout:\n{out[-1000:]}\nstderr:\n{err[-500:]}")
            results["ascend_error"] = f"exit_code={exit_code}"
            return

        if json_result:
            results["ascend"] = json_result
            log(T, f"显存结果: {json.dumps(json_result, ensure_ascii=False)}")
        else:
            log(T, f"警告: 未捕获到 JSON 输出\n{out}")

        # ── 查找 plog 日志 ──
        log(T, "查找 plog 日志 ...")
        if pid:
            find_cmd = f"find {wdir} -name '*plog*{pid}*' -type f 2>/dev/null"
        else:
            find_cmd = f"find {wdir} -path '*/plog*' -type f 2>/dev/null"
        plog_out, _, _ = ssh_run(target, find_cmd, env)
        plog_files = [p.strip() for p in plog_out.strip().split("\n") if p.strip()]

        if not plog_files:
            find_cmd2 = f"find {wdir} -name 'plog*' 2>/dev/null"
            plog_out2, _, _ = ssh_run(target, find_cmd2, env)
            candidates = [p.strip() for p in plog_out2.strip().split("\n") if p.strip()]
            for cand in candidates:
                chk, _, _ = ssh_run(target, f"[ -f '{cand}' ] && echo FILE || echo DIR", env)
                if "FILE" in chk:
                    plog_files.append(cand)
                elif "DIR" in chk:
                    sub_out, _, _ = ssh_run(target, f"find '{cand}' -type f 2>/dev/null", env)
                    plog_files.extend(
                        s.strip() for s in sub_out.strip().split("\n") if s.strip()
                    )

        if plog_files:
            # 选最大的 plog 文件
            best, best_sz = plog_files[0], 0
            for pf in plog_files:
                sz_out, _, _ = ssh_run(target, f"wc -c < '{pf}' 2>/dev/null", env)
                try:
                    sz = int(sz_out.strip())
                    if sz > best_sz:
                        best_sz, best = sz, pf
                except ValueError:
                    pass

            log(T, f"plog 文件: {best} ({best_sz:,} bytes)")

            filt_name = f"filtered_plog_{api_name.replace('.', '_')}.log"
            remote_filt = f"{wdir}/{filt_name}"
            filter_cmd = (
                f"cd {wdir} && "
                f"{{ [ -f {base}/env.sh ] && source {base}/env.sh; true; }} && "
                f"python {fn} '{best}' -o '{remote_filt}' 2>&1"
            )
            fo, fe, frc = ssh_run(target, filter_cmd, env)

            if frc == 0:
                local_filt = os.path.join(out_dir, filt_name)
                ok2, err2 = scp_download(target, remote_filt, local_filt, env)
                if ok2:
                    results["ascend_plog"] = local_filt
                    log(T, f"过滤后 plog 已下载: {local_filt}")
                else:
                    log(T, f"下载 plog 失败: {err2}")
            else:
                log(T, f"filter 执行失败:\n{fo}")
        else:
            log(T, "未找到 plog 文件")

    except Exception as e:
        log(T, f"异常: {e}")
        results["ascend_error"] = str(e)
    finally:
        try:
            os.remove(askpass)
        except OSError:
            pass
        log(T, "完成")


# ─── GPU test ──────────────────────────────────────────────────────────────

def run_gpu_test(cfg, gpu_script, api_name, out_dir, results):
    T = "[GPU]"
    host, user, pw = cfg["host"], cfg["user"], cfg["password"]
    base = cfg["remote_dir"]
    wdir = f"{base}/{api_name}"
    target = f"{user}@{host}"

    askpass = _make_askpass(pw)
    env = _ssh_env(askpass)

    try:
        log(T, f"连接 {host} ...")
        ssh_run(target, f"mkdir -p {wdir}", env)
        log(T, f"远程工作目录: {wdir}")

        sn = os.path.basename(gpu_script)
        ok, err = scp_upload(target, gpu_script, f"{wdir}/{sn}", env)
        if not ok:
            log(T, f"上传 {sn} 失败: {err}")
            results["gpu_error"] = f"SCP upload failed: {err}"
            return
        log(T, f"已上传: {sn}")

        remote_cmd = (
            f"cd {wdir} && "
            f"{{ [ -f {base}/env.sh ] && source {base}/env.sh; true; }} && "
            f"python {sn} 2>&1; echo EXIT_CODE=$?"
        )

        log(T, "执行 GPU 测试脚本 ...")
        out, err, _ = ssh_run(target, remote_cmd, env, timeout=600)
        lines = out.strip().split("\n")

        exit_code, json_result = -1, None
        for l in lines:
            l = l.strip()
            m = re.match(r"EXIT_CODE=(\d+)", l)
            if m:
                exit_code = int(m.group(1))
            if l.startswith("{") and "target_api" in l:
                try:
                    json_result = json.loads(l)
                except json.JSONDecodeError:
                    pass

        log(T, f"exit_code={exit_code}")

        if exit_code != 0:
            log(T, f"执行失败!\nstdout:\n{out[-800:]}\nstderr:\n{err[-500:]}")
            results["gpu_error"] = f"exit_code={exit_code}"
            return

        if json_result:
            results["gpu"] = json_result
            log(T, f"显存结果: {json.dumps(json_result, ensure_ascii=False)}")
        else:
            log(T, f"警告: 未捕获到 JSON 输出\n{out}")

    except Exception as e:
        log(T, f"异常: {e}")
        results["gpu_error"] = str(e)
    finally:
        try:
            os.remove(askpass)
        except OSError:
            pass
        log(T, "完成")


# ─── results ───────────────────────────────────────────────────────────────

def write_results(results, path, key_code=None, script_path=None, line_range=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    api = (results.get("ascend") or results.get("gpu") or {}).get(
        "target_api", "unknown"
    )
    L = [f"# Memory Benchmark: {api}", f"Time: {ts}\n"]
    
    if key_code:
        L.append("## Key Code")
        L.append("```python")
        L.append(key_code)
        L.append("```\n")

    if "ascend" in results:
        r = results["ascend"]
        L += [
            "## Ascend (NPU)",
            f"- target_api: {r['target_api']}",
            f"- 32aligned: {r.get('32aligned')}",
            f"- total_driver_GB: {r.get('total_driver_GB')}",
            f"- pta_reserved_GB: {r.get('pta_reserved_GB')}",
            f"- pta_activated_GB: {r.get('pta_activated_GB')}",
            "",
        ]
    if "ascend_error" in results:
        L += ["## Ascend (NPU) - ERROR", results["ascend_error"], ""]

    if "gpu" in results:
        r = results["gpu"]
        L += [
            "## GPU (CUDA)",
            f"- target_api: {r['target_api']}",
            f"- 32aligned: {r.get('32aligned')}",
            f"- total_driver_GB: {r.get('total_driver_GB')}",
            f"- gpu_reserved_GB: {r.get('gpu_reserved_GB')}",
            f"- gpu_activated_GB: {r.get('gpu_activated_GB')}",
            "",
        ]
    if "gpu_error" in results:
        L += ["## GPU (CUDA) - ERROR", results["gpu_error"], ""]

    if "ascend" in results and "gpu" in results:
        a, g = results["ascend"], results["gpu"]
        L += [
            "## Comparison (NPU vs GPU)",
            "| Metric | NPU | GPU | Delta | Ratio |",
            "|--------|-----|-----|-------|-------|",
        ]
        pairs = [
            ("total_driver_GB", a.get("total_driver_GB", 0), g.get("total_driver_GB", 0)),
            ("reserved_GB", a.get("pta_reserved_GB", 0), g.get("gpu_reserved_GB", 0)),
            ("activated_GB", a.get("pta_activated_GB", 0), g.get("gpu_activated_GB", 0)),
        ]
        for name, nv, gv in pairs:
            d = nv - gv
            ratio = nv / gv if gv else float("inf")
            L.append(
                f"| {name} | {nv:.4f} | {gv:.4f} | {d:+.4f} | {ratio:.2f}x |"
            )
        L.append("")

    if "ascend_plog" in results:
        L += ["## Plog", f"- Filtered plog: {results['ascend_plog']}", ""]

    content = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n{'=' * 60}")
    print(content)
    print(f"{'=' * 60}")
    print(f"结果已保存: {path}")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="远程显存测试工具 – NPU vs GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python run_remote_mem_test.py npu_test.py gpu_test.py
              python run_remote_mem_test.py npu_test.py gpu_test.py --api-name torch.nanmean
              python run_remote_mem_test.py npu_test.py gpu_test.py --skip-gpu
              python run_remote_mem_test.py npu.py gpu.py --skip-gpu --gpu-json '{"target_api":"torch.x","total_driver_GB":1.0,"gpu_reserved_GB":1.0,"gpu_activated_GB":1.0}'
        """),
    )
    ap.add_argument("npu_script", help="NPU 测试脚本路径")
    ap.add_argument("gpu_script", help="GPU 测试脚本路径")
    ap.add_argument("--api-name", help="算子 API 名 (默认从脚本中提取 TARGET_API)")
    ap.add_argument(
        "--servers",
        default=str(SERVERS_JSON),
        help=f"servers.json 路径 (默认: {SERVERS_JSON})",
    )
    ap.add_argument("--output-dir", help="本地输出目录 (默认: NPU 脚本同目录)")
    ap.add_argument("--ascend-key", default="910b", help="Ascend 服务器 key (默认: 910b)")
    ap.add_argument("--gpu-key", default="gpu", help="GPU 服务器 key (默认: gpu)")
    ap.add_argument("--skip-ascend", action="store_true", help="跳过 Ascend 测试")
    ap.add_argument("--skip-gpu", action="store_true", help="跳过 GPU 测试")
    ap.add_argument(
        "--gpu-json",
        help="手动提供 GPU 结果 JSON（跳过 GPU 远程测试时使用）",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.npu_script):
        sys.exit(f"错误: 文件不存在 {args.npu_script}")
    if not args.skip_gpu and not os.path.isfile(args.gpu_script):
        sys.exit(f"错误: 文件不存在 {args.gpu_script}")
    if not os.path.isfile(args.servers):
        sys.exit(f"错误: servers.json 不存在 {args.servers}")
    if not FILTER_NPU_SCRIPT.is_file():
        sys.exit(f"错误: filter 脚本不存在 {FILTER_NPU_SCRIPT}")

    api_name = args.api_name or extract_api_name(args.npu_script)
    if not api_name:
        sys.exit("错误: 无法提取 API 名，请用 --api-name 指定")
    print(f"目标 API: {api_name}")

    servers = load_servers(args.servers)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.npu_script))
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    threads = []

    if not args.skip_ascend:
        if args.ascend_key not in servers:
            sys.exit(f"错误: 未找到服务器 '{args.ascend_key}'")
        threads.append(
            threading.Thread(
                target=run_ascend_test,
                args=(servers[args.ascend_key], args.npu_script, api_name, out_dir, results),
                name="ascend",
            )
        )

    if not args.skip_gpu:
        if args.gpu_key not in servers:
            sys.exit(f"错误: 未找到服务器 '{args.gpu_key}'")
        threads.append(
            threading.Thread(
                target=run_gpu_test,
                args=(servers[args.gpu_key], args.gpu_script, api_name, out_dir, results),
                name="gpu",
            )
        )
    elif args.gpu_json:
        gpu_json_str = args.gpu_json
        if os.path.isfile(gpu_json_str):
            with open(gpu_json_str, "r", encoding="utf-8") as f:
                gpu_json_str = f.read().strip()
        try:
            results["gpu"] = json.loads(gpu_json_str)
            print(f"使用手动提供的 GPU 结果: {gpu_json_str}")
        except json.JSONDecodeError as e:
            sys.exit(f"错误: --gpu-json 格式不正确: {e}")

    if not threads and "gpu" not in results:
        sys.exit("错误: 没有可运行的测试")

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")

    result_file = os.path.join(out_dir, "mem_results.md")
    
    key_code, line_start, line_end = extract_key_code_lines(args.npu_script, api_name)
    if not key_code and not args.skip_gpu and os.path.isfile(args.gpu_script):
        key_code, line_start, line_end = extract_key_code_lines(args.gpu_script, api_name)
    
    script_src = args.npu_script if key_code else None
    line_range = (line_start, line_end) if key_code else None
    
    write_results(results, result_file, key_code, script_src, line_range)


if __name__ == "__main__":
    main()
