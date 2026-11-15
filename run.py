import argparse
import asyncio
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Практика 8: безопасность ИТИС Навигатора')
    parser.add_argument('--attacks', action='store_true', required=True)
    parser.add_argument('--version', choices=['v4','v5','both'], default='both')
    parser.add_argument('--backend', choices=['contracts','live'], default='live')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from src.security_eval import evaluate
    try:
        directory = asyncio.run(evaluate(args.output, args.backend, ('v4','v5') if args.version == 'both' else (args.version,)))
        import json
        complete = True
        for version in (('v4','v5') if args.version == 'both' else (args.version,)):
            metrics = json.loads((directory / f'results_{version}.json').read_text(encoding='utf-8'))['metrics']
            print(version, json.dumps(metrics, ensure_ascii=False, indent=2))
            complete &= metrics['complete']
        return 0 if complete else 2
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
