"""Local trusted CLI. A web adapter must obtain the Session from authenticated middleware."""
import argparse
import asyncio
import getpass
import json
import os
from pathlib import Path

from .security import Security, Session

ROOT = Path(__file__).resolve().parents[1]


def local_session():
    # These are host configuration, not fields accepted from a chatbot request.
    return Session(getpass.getuser(), os.getenv('ITIS_TRUSTED_ROLE', 'student'),
                   os.getenv('ITIS_TRUSTED_SESSION', 'local-cli'))


async def ask_live(question):
    from .agent import create_llm
    from .mcp_client import connect_mcp
    from .secure_agent import ask
    security = Security(ROOT / '.cache/security')
    try:
        async with connect_mcp() as mcp:
            return await ask(question, local_session(), security, create_llm(), mcp)
    finally:
        security.close()


def main():
    parser = argparse.ArgumentParser(description='Защищённый локальный интерфейс Практики 8')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('ask')
    p.add_argument('question')
    p = sub.add_parser('remember')
    p.add_argument('key', choices=['group','course'])
    p.add_argument('value')
    p = sub.add_parser('confirm')
    p.add_argument('token')
    p.add_argument('--cancel', action='store_true')
    sub.add_parser('memory')
    sub.add_parser('forget')
    args = parser.parse_args()
    if args.command == 'ask':
        try:
            result = asyncio.run(ask_live(args.question))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print('Не удалось выполнить живой запрос:', type(exc).__name__, str(exc))
            return 1
    else:
        security = Security(ROOT / '.cache/security')
        ctx = local_session()
        try:
            if not security.admit(ctx):
                result = {'status':'limited'}
            elif args.command == 'remember':
                result = asyncio.run(security.call_tool('memory_propose', {'key':args.key,'value':args.value}, ctx))
            elif args.command == 'confirm':
                result = security.confirm(ctx, args.token, not args.cancel)
            elif args.command == 'memory':
                result = asyncio.run(security.call_tool('memory_read', {}, ctx))
            else:
                security.forget(ctx)
                result = {'status':'forgotten'}
        finally:
            security.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
