"""Minimal HTTPS CONNECT relay for a login node, so compute jobs without internet can reach ONE generator API host.

TLS stays end to end (the relay only copies bytes after CONNECT). Only host:443 of the allowed host is accepted, and
every request must carry Proxy-Authorization Basic with the token, so other users on the cluster cannot use it.
Standard library only. Run it in tmux on one login node, and stop it when the experiment ends.

    python scripts/exp900/connect_proxy.py --allow-host api.example.com --bind 0.0.0.0 --port 0 --token-file ~/.tio-proxy-token
    # prints:  GEN_PROXY=http://tio:<token>@<login-host>:<port>
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import secrets
import socket
import sys
import time
from pathlib import Path


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


def make_handler(allow_host: str, expected: str):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        lines = head.decode("latin-1").split("\r\n")
        method, target, *_ = (lines[0].split(" ") + ["", ""])[:3]
        headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
        if headers.get("proxy-authorization", "") != expected:
            log("reject auth", peer)
            writer.write(b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=\"tio\"\r\nContent-Length: 0\r\n\r\n")
            await writer.drain(); writer.close()
            return
        if method != "CONNECT" or target.lower() != f"{allow_host}:443":
            log("reject target", peer, method, target)
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await writer.drain(); writer.close()
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(allow_host, 443), timeout=20)
        except (OSError, asyncio.TimeoutError) as exc:
            log("upstream failed", type(exc).__name__)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            await writer.drain(); writer.close()
            return
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        log("tunnel", peer)
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
    return handle


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-host", required=True, help="the single API host, e.g. the host part of ANTHROPIC_BASE_URL")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--token-file", type=Path, required=True, help="created with a random token if missing (mode 600)")
    args = parser.parse_args()
    if not args.token_file.exists():
        args.token_file.write_text(secrets.token_urlsafe(24))
        args.token_file.chmod(0o600)
    token = args.token_file.read_text().strip()
    expected = "Basic " + base64.b64encode(f"tio:{token}".encode()).decode()
    server = await asyncio.start_server(make_handler(args.allow_host.lower(), expected), args.bind, args.port)
    port = server.sockets[0].getsockname()[1]
    print(f"GEN_PROXY=http://tio:{token}@{socket.getfqdn()}:{port}", flush=True)
    log(f"relaying CONNECT {args.allow_host}:443 on {args.bind}:{port}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
