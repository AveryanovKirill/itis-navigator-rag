"""Practice 8 production entrypoint. Baseline v4 exists only in isolated evaluation."""
import json
from pathlib import Path

from .security import wrap


def messages(question, fragments, memory, tool_results=(), version='v5'):
    root = Path(__file__).resolve().parents[1]
    if version not in {'v4', 'v5'}:
        raise ValueError('Expected v4/v5')
    if version == 'v5':
        body = '\n'.join([wrap('question', question),
                          *(wrap('doc', c['text'], id=c.get('id', ''), number=c['number']) for c in fragments),
                          wrap('memory', memory), *(wrap('tool_result', t) for t in tool_results)])
    else:
        body = json.dumps({'question': question, 'documents': fragments, 'memory': memory, 'tools': tool_results}, ensure_ascii=False)
    return [{'role': 'system', 'content': (root / f'prompts/system_{version}.md').read_text(encoding='utf-8')},
            {'role': 'user', 'content': body}]


async def generate(question, fragments, memory, llm, version='v5', tool_results=()):
    from .schemas import validate_answer, fallback
    prompt = messages(question, fragments, memory, tool_results, version)
    record = await llm.complete(prompt)
    raw = record['response']['choices'][0]['message'].get('content', '')
    try:
        answer = validate_answer(raw, fragments, 'v2').model_dump()
        valid = True
    except (ValueError, TypeError):
        answer, valid = fallback().model_dump(), False
    # Raw output is for isolated evaluation only, never returned by production ask().
    return {'answer': answer, 'json_valid': valid, 'raw': raw,
            'usage': record['response'].get('usage', {}),
            'model': record['response'].get('model'), 'temperature': 0}


async def ask(question, ctx, security, llm, mcp):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        return security.deny(ctx, 'request', {}, 'invalid_question')
    if not security.admit(ctx):
        return {'status': 'limited', 'message': 'Превышен лимит. Повторите позже.'}
    # Commands are parsed before the model. The model cannot confirm its own write.
    if question.strip().casefold() == 'забудь':
        security.forget(ctx)
        return {'status': 'forgotten'}
    from .schedule import schedule_intent, schedule_args
    memory = await security.call_tool('memory_read', {}, ctx)
    if schedule_intent(question):
        group, day = schedule_args(question)
        group = group or memory.get('group', '')
        data = await security.call_tool('get_group_schedule', {'group': group, 'day': day}, ctx, mcp)
        return {'route': 'schedule', 'result': data, 'usage': {'cost': 0}}
    retrieved = await security.call_tool('search_docs', {'query': question}, ctx, mcp)
    if retrieved.get('status') in {'denied', 'error'}:
        return retrieved
    result = await generate(question, retrieved.get('fragments', []), memory, llm, tool_results=[retrieved])
    security.audit(ctx, 'request', {}, 'allow' if result['answer']['found'] else 'deny',
                   'answered' if result['answer']['found'] else 'insufficient_evidence')
    return {k: result[k] for k in ('answer', 'usage', 'json_valid')}
