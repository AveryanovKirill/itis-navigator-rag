"""Server-side trust boundary. No identity or permissions are taken from LLM text."""
import hashlib
import html
import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROLES = {
    'student': frozenset({'search_docs', 'get_group_schedule', 'faq', 'memory_read', 'memory_propose'}),
    'teacher': frozenset({'search_docs', 'get_group_schedule', 'faq', 'deadlines', 'memory_read', 'memory_propose'}),
}
FIELDS = {'group': r'11-\d{3}', 'course': r'[1-6]'}


@dataclass(frozen=True)
class Session:
    """Construct only in trusted entrypoint after authentication, never from request JSON."""
    user_id: str
    role: str = 'student'
    session_id: str = 'local'

    def __post_init__(self):
        if not isinstance(self.role, str) or self.role not in ROLES or not isinstance(self.user_id, str) or not self.user_id or not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError('Invalid trusted session')


def wrap(tag, value, **attributes):
    if tag not in {'doc', 'memory', 'tool_result', 'question'}:
        raise ValueError('Unknown data tag')
    attrs = ''.join(f' {k}="{html.escape(str(v), quote=True)}"' for k, v in attributes.items())
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return f'<{tag}{attrs}>\n{html.escape(text, quote=True)}\n</{tag}>'


def mask(value):
    """Deny-by-default logging: arbitrary query/result/free text never reaches disk."""
    if isinstance(value, dict):
        return {str(k) if str(k) in {'query', 'group', 'day', 'key', 'value', 'user_id', 'role', 'token', 'email', 'password', 'api_key'} else '[field]': mask(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask(v) for v in value[:20]]
    return '[redacted]'


class Security:
    def __init__(self, directory, rpm=30, denial_threshold=3, cooldown=60, clock=time.time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if rpm < 1 or denial_threshold < 1 or cooldown <= 0:
            raise ValueError('Limits must be positive')
        self.rpm, self.denial_threshold, self.cooldown, self.clock = rpm, denial_threshold, cooldown, clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / 'state.sqlite3', check_same_thread=False)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS facts(user TEXT, key TEXT, value TEXT, expires REAL, PRIMARY KEY(user,key));
        CREATE TABLE IF NOT EXISTS pending(user TEXT, session TEXT, token TEXT PRIMARY KEY, key TEXT, value TEXT, expires REAL);
        CREATE TABLE IF NOT EXISTS requests(user TEXT, ts REAL);
        CREATE INDEX IF NOT EXISTS request_user_ts ON requests(user,ts);
        CREATE TABLE IF NOT EXISTS limits(user TEXT PRIMARY KEY, denials INTEGER, blocked_until REAL);
        ''')

    def close(self):
        self.db.close()

    def audit(self, ctx, tool, args, decision, reason):
        record = {'ts': datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
                  'user_id': hashlib.sha256(ctx.user_id.encode()).hexdigest()[:20], 'role': ctx.role,
                  'tool': tool if isinstance(tool, str) and tool in set().union(*ROLES.values()) | {'request', 'memory_confirm'} else 'unknown',
                  'args': mask(args), 'decision': decision, 'reason': reason}
        with self.lock, (self.directory / 'audit.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')

    def deny(self, ctx, tool, args, reason):
        with self.lock, self.db:
            old = self.db.execute('SELECT denials, blocked_until FROM limits WHERE user=?', (ctx.user_id,)).fetchone()
            count = (old[0] if old else 0) + 1
            blocked = max(old[1] if old else 0, self.clock() + self.cooldown if count >= self.denial_threshold else 0)
            self.db.execute('INSERT OR REPLACE INTO limits VALUES(?,?,?)', (ctx.user_id, count, blocked))
            self.audit(ctx, tool, args, 'deny', reason)
            if count == self.denial_threshold:
                self.audit(ctx, 'request', {}, 'alert', 'denial_threshold')
        return {'status': 'denied', 'message': 'Нет доступа', 'reason': reason}

    def admit(self, ctx):
        with self.lock, self.db:
            current = self.clock()
            self.db.execute('DELETE FROM requests WHERE ts <= ?', (current - 60,))
            state = self.db.execute('SELECT denials,blocked_until FROM limits WHERE user=?', (ctx.user_id,)).fetchone()
            if state and state[1] > current:
                self.audit(ctx, 'request', {}, 'deny', 'cooldown')
                return False
            if state and state[1] and state[1] <= current:
                self.db.execute('DELETE FROM limits WHERE user=?', (ctx.user_id,))
            count = self.db.execute('SELECT count(*) FROM requests WHERE user=?', (ctx.user_id,)).fetchone()[0]
            if count >= self.rpm:
                self.audit(ctx, 'request', {}, 'deny', 'rate_limit')
                return False
            self.db.execute('INSERT INTO requests VALUES(?,?)', (ctx.user_id, current))
            self.audit(ctx, 'request', {}, 'allow', 'admitted')
            return True

    def read_memory(self, ctx):
        with self.lock, self.db:
            self.db.execute('DELETE FROM facts WHERE expires <= ?', (self.clock(),))
            return dict(self.db.execute('SELECT key,value FROM facts WHERE user=?', (ctx.user_id,)))

    def propose(self, ctx, key, value):
        if not isinstance(key, str) or key not in FIELDS or not isinstance(value, str) or not re.fullmatch(FIELDS[key], value):
            return self.deny(ctx, 'memory_propose', {'key': key, 'value': value}, 'invalid_fact')
        token = secrets.token_urlsafe(24)
        with self.lock, self.db:
            self.db.execute('DELETE FROM pending WHERE expires <= ? OR (user=? AND session=?)', (self.clock(), ctx.user_id, ctx.session_id))
            self.db.execute('INSERT INTO pending VALUES(?,?,?,?,?,?)', (ctx.user_id, ctx.session_id, token, key, value, self.clock() + 600))
        return {'status': 'confirmation_required', 'token': token, 'key': key, 'value': value}

    def confirm(self, ctx, token, approved):
        """Call only from explicit UI/CLI confirmation, never register as an LLM tool."""
        if not isinstance(token, str) or not 1 <= len(token) <= 200:
            return self.deny(ctx, 'memory_confirm', {}, 'invalid_confirmation')
        with self.lock, self.db:
            row = self.db.execute('SELECT key,value FROM pending WHERE token=? AND user=? AND session=? AND expires>?',
                                  (token, ctx.user_id, ctx.session_id, self.clock())).fetchone()
            if not row or type(approved) is not bool:
                return self.deny(ctx, 'memory_confirm', {}, 'invalid_confirmation')
            self.db.execute('DELETE FROM pending WHERE token=?', (token,))
            if approved:
                self.db.execute('INSERT OR REPLACE INTO facts VALUES(?,?,?,?)', (ctx.user_id, *row, self.clock() + 180 * 86400))
            self.audit(ctx, 'memory_confirm', {}, 'allow', 'confirmed' if approved else 'cancelled')
            return {'status': 'saved' if approved else 'cancelled'}

    def forget(self, ctx):
        with self.lock, self.db:
            self.db.execute('DELETE FROM facts WHERE user=?', (ctx.user_id,))
            self.db.execute('DELETE FROM pending WHERE user=?', (ctx.user_id,))
        self.audit(ctx, 'memory_confirm', {}, 'allow', 'forgotten')

    async def call_tool(self, name, args, ctx, mcp=None):
        if not isinstance(args, dict):
            return self.deny(ctx, name, {}, 'invalid_arguments')
        if not isinstance(name, str) or name not in ROLES[ctx.role]:
            return self.deny(ctx, name, args, 'role')
        # Ignore attempts to spoof identity. All local SQL queries bind ctx.user_id.
        clean = {k: v for k, v in args.items() if k not in {'user_id', 'role', 'session_id', 'confirmed'}}
        allowed = {'search_docs': {'query'}, 'get_group_schedule': {'group', 'day'},
                   'faq': set(), 'deadlines': set(), 'memory_read': set(), 'memory_propose': {'key', 'value'}}[name]
        if set(clean) - allowed:
            return self.deny(ctx, name, args, 'invalid_arguments')
        if name == 'memory_read':
            data = self.read_memory(ctx)
        elif name == 'memory_propose':
            data = self.propose(ctx, clean.get('key'), clean.get('value'))
            if data['status'] == 'denied':
                return data
        elif name == 'faq':
            data = {'message': 'Справки и правила доступны через поиск документов ИТИС.'}
        elif name == 'deadlines':
            # Staff-only read. No invented dates if a source has not been configured.
            data = {'status': 'not_found', 'message': 'Источник преподавательских сроков не подключён.'}
        else:
            valid = (name == 'search_docs' and isinstance(clean.get('query'), str) and 1 <= len(clean['query'].strip()) <= 2000)
            valid |= (name == 'get_group_schedule' and all(isinstance(clean.get(k, ''), str) and len(clean.get(k, '')) <= 100 for k in ('group', 'day')))
            if not valid:
                return self.deny(ctx, name, args, 'invalid_arguments')
            if mcp is None:
                raise RuntimeError('MCP is required')
            result = await mcp.call_tool(name, clean)
            if result.isError:
                self.audit(ctx, name, args, 'deny', 'tool_error')
                return {'status': 'error', 'message': 'Ошибка инструмента'}
            data = result.structuredContent or json.loads('\n'.join(c.text for c in result.content if c.type == 'text'))
        self.audit(ctx, name, args, 'allow', 'role_and_scope_checked')
        return data
