"""Put a student on the MaixCAM2's edgepilot for a road test, check it there, switch to it and back.

The student is a CPU-heads core: its axmodel ends in out_hidden and the hx_<head> features, and edgepilot's modeld
(branch student-cpu-heads, src/model/cpu_heads.h) computes each head's last linear layer from <name>.heads next to it.
That modeld still runs the shipped split-output model, so it replaces the installed one; the model in use is chosen
by EDGEPILOT_MODEL in a systemd drop-in, and params/driving_model.json says which (recordd copies params/*.json into
every route, so each recorded drive keeps the model it ran).

  package <name> --ckpt C --build B --modeld M   deploy/<name>/: <name>.axmodel (B/m.axmodel), <name>.heads (from C,
                                                 export_cpu_heads.py), modeld (M), SHA256SUMS
  install <name>      upload: modeld (the installed one kept once as modeld.orig), models/<name>.axmodel/.heads
  check <name>        service stopped: each model loaded by the new modeld (contract message), then a rehearsal of a
                      clip on the board with the student (EDGEPILOT_PROFILE timings, restarts); service started again
  select <name>       params/ copied once to params.before_student/ (the learners and calibration save there while
                      the student drives), drop-in EDGEPILOT_MODEL=<name> (--shadow adds EDGEPILOT_PANDA_TX=0: nothing
                      sent on CAN), params/driving_model.json, restart, then checks modeld's log
  rollback            removes the drop-in (the shipped model again), params/driving_model.json says so;
                      --binary also puts modeld.orig back, --params puts params.before_student/ back
  status              the drop-in, the models, modeld's version and its first log lines

The board password comes from SSHPASS (never a file), as for board_eval.py.
usage: SSHPASS=<board password> .venv/bin/python scripts/deploy_student.py <command> [name] [--board root@IP]"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DEST = "/root/edgepilot"
DROPIN = "/etc/systemd/system/edgepilot.service.d/driving-model.conf"
VERSION_TAG = "student-cpu-heads"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def package(name: str, ckpt: Path, build: Path, modeld: Path):
    d = ROOT / "deploy" / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "m.axmodel", d / f"{name}.axmodel")
    subprocess.run([sys.executable, str(ROOT / "scripts/export_cpu_heads.py"), str(ckpt), str(d / f"{name}.heads")], check=True)
    shutil.copy2(modeld, d / "modeld")
    ver = subprocess.run(["grep", "-ao", "EDGEPILOT_VERSION=[A-Za-z0-9._+-]*", str(d / "modeld")], capture_output=True,
                         text=True).stdout.split()[0]
    if VERSION_TAG not in ver:
        sys.exit(f"{modeld} is {ver}, not a {VERSION_TAG} build")
    files = [f"{name}.axmodel", f"{name}.heads", "modeld"]
    (d / "SHA256SUMS").write_text("".join(f"{sha256(d / f)}  {f}\n" for f in files))
    (d / "source.json").write_text(json.dumps({"ckpt": str(ckpt), "build": str(build), "modeld": ver.split("=")[1]}, indent=1))
    print((d / "SHA256SUMS").read_text() + f"modeld {ver}")


def board(args):
    from board_eval import Board
    b = Board(args.board)
    b.ssh("true")
    return b


def install(b, name: str):
    d = ROOT / "deploy" / name
    sums = dict(l.split()[::-1] for l in (d / "SHA256SUMS").read_text().splitlines())
    print(b.ssh(f"grep -ao 'EDGEPILOT_VERSION=[A-Za-z0-9._+-]*' {DEST}/modeld | head -1"), end="")
    b.ssh(f"mkdir -p {DEST}/.upload_student")
    for f in sums:
        b.put(d / f, f"{DEST}/.upload_student/{f}")
    got = b.ssh(f"cd {DEST}/.upload_student && sha256sum {' '.join(sums)}")
    for l in got.splitlines():
        h, f = l.split()
        if sums[f] != h:
            sys.exit(f"upload of {f} differs: {h} != {sums[f]}")
    # 돌고 있는 바이너리는 덮어쓸 수 없어 mv로 바꾼다. 처음 설치 전의 modeld는 한 번만 modeld.orig로 남긴다.
    b.ssh(f"cd {DEST} && ([ -e modeld.orig ] || cp -p modeld modeld.orig) && mv .upload_student/modeld modeld && "
          f"mv .upload_student/{name}.axmodel .upload_student/{name}.heads models/ && rmdir .upload_student")
    print(b.ssh(f"cd {DEST} && ls -la modeld modeld.orig models/{name}.* && grep -ao 'EDGEPILOT_VERSION=[A-Za-z0-9._+-]*' modeld | head -1"))


def check(b, name: str, clip: str, seconds: int):
    """서비스를 멈춘 동안: 두 모델을 새 modeld로 열어 보고(계약 메시지), 클립 리허설을 돈다. 끝나면(실패해도) 다시 켠다."""
    script = f"""
set -u
trap 'systemctl start edgepilot.service' EXIT
systemctl stop edgepilot.service; sleep 2
cd {DEST}
for m in models/supercombo.axmodel models/{name}.axmodel; do
  echo "== load $m"; timeout 6 ./modeld $m 2>&1 | grep -a -E "axmodel|error|Error|lacks|expected|terminate" | head -5
done
if [ -x ./replayd ] && [ -d /root/rehearsal/{clip} ]; then
  W=$(mktemp -d /run/edgepilot-rehearsal.XXXXXX)
  cp -a params $W/params && cp -a params.defaults $W/params.defaults
  sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
  EDGEPILOT_LOG_DIR=$W EDGEPILOT_PARAMS_DIR=$W/params EDGEPILOT_MODEL={DEST}/models/{name}.axmodel EDGEPILOT_PROFILE=1 \\
  EDGEPILOT_REPLAY_ROUTE=/root/rehearsal/{clip} EDGEPILOT_REPLAY_START=0 EDGEPILOT_REPLAY_DURATION={seconds} \\
  EDGEPILOT_FORCE_ENGAGED=1 timeout -s TERM {seconds + 15} python3 {DEST}/manager.py > $W/manager.out 2>&1
  echo "== rehearsal {clip} {seconds} s with {name} (logs $W)"
  grep -a -E "axmodel|profile ms" $W/modeld.log | head -3; echo ...; grep -a "profile ms" $W/modeld.log | tail -3
  echo "modeld starts: $(grep -a -c 'CPU-heads axmodel' $W/modeld.log)"
  grep -a -i -E "restart|exited|died|error" $W/manager.out | head -5
  grep -a -i -E "error|terminate|abort" $W/modeld.log | head -5
else
  echo "== no ./replayd or /root/rehearsal/{clip}: rehearsal skipped"
fi
"""
    print(b.ssh(script, check=False))
    print(b.ssh("systemctl is-active edgepilot.service", check=False), end="")


def select(b, name: str, shadow: bool):
    d = ROOT / "deploy" / name
    sums = dict(l.split()[::-1] for l in (d / "SHA256SUMS").read_text().splitlines())
    env = [f"EDGEPILOT_MODEL={DEST}/models/{name}.axmodel"] + (["EDGEPILOT_PANDA_TX=0"] if shadow else [])
    conf = "# deploy_student.py select: 주행 모델(CPU 헤드 코어 학생). 되돌리기: deploy_student.py rollback\n[Service]\n" + \
           "".join(f"Environment={e}\n" for e in env)
    info = {"model": f"models/{name}.axmodel", "heads": f"models/{name}.heads", "axmodel_sha256": sums[f"{name}.axmodel"],
            "heads_sha256": sums[f"{name}.heads"], "shadow": shadow, "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **json.loads((d / "source.json").read_text())}
    b.ssh(f"cd {DEST} && ([ -e params.before_student ] || cp -a params params.before_student) && "
          f"mkdir -p {Path(DROPIN).parent} && cat > {DROPIN} <<'EOF'\n{conf}EOF\n"
          f"cat > {DEST}/params/driving_model.json <<'EOF'\n{json.dumps(info, indent=1)}\nEOF\n"
          "systemctl daemon-reload && systemctl restart edgepilot.service")
    after_restart(b)


def rollback(b, binary: bool, params: bool):
    info = {"model": "models/supercombo.axmodel", "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rollback": True}
    cmd = f"rm -f {DROPIN} && cat > {DEST}/params/driving_model.json <<'EOF'\n{json.dumps(info, indent=1)}\nEOF\n"
    if binary:
        cmd += f"[ -e {DEST}/modeld.orig ] && cp -p {DEST}/modeld.orig {DEST}/.modeld.tmp && mv {DEST}/.modeld.tmp {DEST}/modeld\n"
    if params:   # 서비스를 멈춘 뒤 바꾼다(학습기가 끝날 때 저장한다). 학생으로 달린 동안의 params는 params.student/로 남긴다.
        cmd += (f"systemctl stop edgepilot.service && cd {DEST} && rm -rf params.student && mv params params.student && "
                f"cp -a params.before_student params && cp params.student/driving_model.json params/\n")
    b.ssh(cmd + "systemctl daemon-reload && systemctl restart edgepilot.service")
    after_restart(b)


def after_restart(b):
    time.sleep(15)
    print(b.ssh("systemctl is-active edgepilot.service; systemctl show edgepilot.service -p Environment | tr ' ' '\\n' | "
                "grep -E 'EDGEPILOT_(MODEL|PANDA_TX)'; grep -a -E 'axmodel|lacks|expected|error' /run/edgepilot/modeld.log | "
                "tail -3; pgrep -x modeld > /dev/null && echo modeld running", check=False))


def status(b):
    print(b.ssh(f"cat {DROPIN} 2>/dev/null || echo 'no drop-in: shipped model'; cat {DEST}/params/driving_model.json 2>/dev/null; "
                f"ls -d {DEST}/params.before_student 2>/dev/null; "
                f"ls -la {DEST}/modeld {DEST}/modeld.orig {DEST}/models/; grep -ao 'EDGEPILOT_VERSION=[A-Za-z0-9._+-]*' {DEST}/modeld | head -1; "
                "grep -a -E 'axmodel' /run/edgepilot/modeld.log | tail -2", check=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["package", "install", "check", "select", "rollback", "status"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--board", default="root@192.168.219.107")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--build", type=Path)
    ap.add_argument("--modeld", type=Path)
    ap.add_argument("--clip", default="clip34", help="check: the rehearsal clip under /root/rehearsal")
    ap.add_argument("--seconds", type=int, default=40, help="check: rehearsal length")
    ap.add_argument("--shadow", action="store_true", help="select: also EDGEPILOT_PANDA_TX=0 (nothing sent on CAN)")
    ap.add_argument("--binary", action="store_true", help="rollback: also put modeld.orig back")
    ap.add_argument("--params", action="store_true", help="rollback: also put params.before_student/ back")
    args = ap.parse_args()
    if args.command == "package":
        return package(args.name, args.ckpt, args.build, args.modeld)
    b = board(args)
    if args.command == "install":
        install(b, args.name)
    elif args.command == "check":
        check(b, args.name, args.clip, args.seconds)
    elif args.command == "select":
        select(b, args.name, args.shadow)
    elif args.command == "rollback":
        rollback(b, args.binary, args.params)
    else:
        status(b)


if __name__ == "__main__":
    main()
