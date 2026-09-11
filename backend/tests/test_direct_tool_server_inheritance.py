import asyncio
import copy
import importlib
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(scope='module', autouse=True)
def load_open_webui_modules():
    global main, middleware, subagents

    existing_modules = set(sys.modules)
    with tempfile.TemporaryDirectory() as runtime:
        test_env = {
            'DATA_DIR': runtime,
            'STATIC_DIR': f'{runtime}/static',
            'USE_SLIM_DOCKER': 'true',
            'WEBUI_SECRET_KEY': 'test-secret-key-test-secret-key-123',
        }
        original_env = {key: os.environ.get(key) for key in test_env}
        try:
            os.makedirs(test_env['STATIC_DIR'])
            os.environ.update(test_env)
            middleware = importlib.import_module('open_webui.utils.middleware')
            subagents = importlib.import_module('open_webui.utils.subagents')
            main = importlib.import_module('open_webui.main')
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        yield

    for name in set(sys.modules) - existing_modules:
        if name == 'open_webui' or name.startswith('open_webui.'):
            sys.modules.pop(name, None)


DIRECT_TOOL_SERVERS = [
    {
        'url': 'http://localhost:8000',
        'headers': {'Authorization': 'Bearer test'},
        'system_prompt': 'Use OpenTerminal when needed.',
        'specs': [
            {
                'name': 'run_command',
                'description': 'Run a command.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'list_files',
                'description': 'List files.',
                'parameters': {'type': 'object', 'properties': {}},
            },
        ],
    },
    {
        'url': 'http://localhost:8001',
        'specs': [
            {
                'name': 'read_file',
                'description': 'Read a file.',
                'parameters': {'type': 'object', 'properties': {}},
            }
        ],
    },
]


def _mock_payload_dependencies(monkeypatch):
    async def passthrough(_request, form_data, *_args, **_kwargs):
        return form_data

    async def passthrough_form_data(form_data, **_kwargs):
        return form_data

    async def no_events(_metadata):
        async def emit(_event):
            pass

        return emit

    monkeypatch.setattr(middleware, 'apply_params_to_form_data', lambda form_data, _model: form_data)
    monkeypatch.setattr(middleware, 'process_messages_with_output', lambda messages, **_kwargs: messages)
    monkeypatch.setattr(middleware, 'sanitize_tool_pairs', lambda messages: messages)
    monkeypatch.setattr(middleware, 'apply_system_prompt_to_body', passthrough)
    monkeypatch.setattr(middleware, 'convert_url_images_to_base64', passthrough_form_data)
    monkeypatch.setattr(middleware, 'get_event_emitter', no_events)
    monkeypatch.setattr(middleware, 'get_event_call', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware, 'get_system_oauth_token', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware.Config, 'get', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware, 'get_task_model_id', lambda model_id, *_args: model_id)
    monkeypatch.setattr(middleware, 'process_pipeline_inlet_filter', passthrough)
    monkeypatch.setattr(middleware, 'extract_skill_ids_from_messages', lambda _messages: set())
    monkeypatch.setattr(
        middleware,
        'chat_completion_files_handler',
        AsyncMock(side_effect=lambda _request, form_data, *_args: (form_data, {})),
    )
    monkeypatch.setattr(middleware, 'resolve_system_prompt', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware, 'normalize_messages_for_model', lambda form_data: form_data)
    monkeypatch.setattr(middleware, 'ENABLE_PLUGINS', False)


@pytest.mark.asyncio
async def test_process_chat_payload_preserves_direct_tool_server_metadata(monkeypatch):
    _mock_payload_dependencies(monkeypatch)

    model = {'id': 'test-model', 'info': {'meta': {}}}
    request = SimpleNamespace(
        state=SimpleNamespace(direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'test-model': model})),
    )
    metadata = {'tool_servers': copy.deepcopy(DIRECT_TOOL_SERVERS)}
    original = copy.deepcopy(metadata['tool_servers'])

    form_data, metadata, _events = await middleware.process_chat_payload(
        request,
        {
            'model': 'test-model',
            'messages': [
                {
                    'role': 'system',
                    'content': 'Do not follow this example:\nUse OpenTerminal when needed.',
                },
                {'role': 'user', 'content': 'Use the terminal.'},
            ],
        },
        SimpleNamespace(id='user-id', role='user'),
        metadata,
        model,
    )

    assert metadata['tool_servers'] == original
    assert {tool['function']['name'] for tool in form_data['tools']} == {
        'run_command',
        'list_files',
        'read_file',
    }
    assert metadata['tool_servers'][0]['system_prompt'] == 'Use OpenTerminal when needed.'
    assert metadata['direct_tool_server_prompts_applied'] is True
    assert metadata['system_prompt'].count('Use OpenTerminal when needed.') == 2
    assert metadata['system_prompt'].endswith('\nUse OpenTerminal when needed.')


async def _capture_subagent_run(monkeypatch, *, background, metadata=None, handler=None, request=None):
    captured = {}

    monkeypatch.setattr(
        subagents.Config,
        'get_many',
        AsyncMock(
            return_value={
                'subagents.background_enabled': True,
                'subagents.max_concurrent': -1,
                'subagents.max_async': -1,
                'subagents.max_iterations': 30,
                'subagents.max_output': 30_000,
                'subagents.system_prompt': '',
            }
        ),
    )
    monkeypatch.setattr(subagents, 'UserModel', lambda **data: SimpleNamespace(**data))
    monkeypatch.setattr(subagents, 'ChatForm', lambda **data: data)
    monkeypatch.setattr(subagents, '_background_active', set())
    monkeypatch.setattr(subagents.Chats, 'insert_new_chat', AsyncMock(return_value=object()))
    monkeypatch.setattr(
        subagents.Chats,
        'get_message_by_id_and_message_id',
        AsyncMock(return_value={'content': 'done'}),
    )

    async def capture_handler(_request, form_data, *, user):
        captured['form_data'] = copy.deepcopy(form_data)
        captured['user'] = user

        if handler:
            await handler(_request, form_data, user=user)

    async def create_task(_redis, coroutine, *, id):
        captured['run'] = copy.deepcopy(coroutine.cr_frame.f_locals['run'])
        if background:
            coroutine.close()

            async def done():
                return {'status': 'completed', 'summary': 'done', 'error': None}

            child_task = asyncio.create_task(done())
        else:
            child_task = asyncio.create_task(coroutine)
        return id, child_task

    monkeypatch.setattr(subagents, 'create_task', create_task)

    if request is None:
        request = SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace()))
    request.app.state.redis = object()
    request.app.state.CHAT_COMPLETION_HANDLER = capture_handler
    request.app.state.MODELS = {'test-model': {'id': 'test-model', 'info': {'meta': {}}}}
    await subagents.delegate(
        'Use the terminal.',
        '',
        background,
        request=request,
        user_data={'id': 'user-id'},
        metadata=metadata
        or {
            'model_id': 'test-model',
            'tool_servers': copy.deepcopy(DIRECT_TOOL_SERVERS),
        },
        parent_chat_id='parent-chat-id',
        parent_message_id='parent-message-id',
    )
    return captured


@pytest.mark.asyncio
async def test_foreground_subagent_inherits_direct_tool_servers(monkeypatch):
    _mock_payload_dependencies(monkeypatch)
    model = {'id': 'test-model', 'info': {'meta': {}}}
    request = SimpleNamespace(
        state=SimpleNamespace(direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'test-model': model})),
    )
    _, parent_metadata, _ = await middleware.process_chat_payload(
        request,
        {
            'model': 'test-model',
            'messages': [{'role': 'user', 'content': 'Use the terminal.'}],
        },
        SimpleNamespace(id='user-id', role='user'),
        {'tool_servers': copy.deepcopy(DIRECT_TOOL_SERVERS)},
        model,
    )
    child = {}

    async def process_child(child_request, form_data, *, user):
        child_metadata = {
            'chat_id': form_data.pop('chat_id'),
            'message_id': form_data.pop('id'),
            'session_id': form_data.pop('session_id'),
            'tool_servers': form_data.pop('tool_servers'),
            'direct_tool_server_prompts_applied': form_data.pop('direct_tool_server_prompts_applied'),
        }
        child['form_data'], child['metadata'], _ = await middleware.process_chat_payload(
            child_request,
            form_data,
            user,
            child_metadata,
            model,
        )

    captured = await _capture_subagent_run(
        monkeypatch,
        background=False,
        metadata=parent_metadata,
        handler=process_child,
        request=request,
    )

    assert captured['run']['tool_servers'] == DIRECT_TOOL_SERVERS
    assert captured['form_data']['tool_servers'] == DIRECT_TOOL_SERVERS
    assert child['metadata']['tool_servers'] == DIRECT_TOOL_SERVERS
    assert {
        'run_command',
        'list_files',
        'read_file',
    } <= {tool['function']['name'] for tool in child['form_data']['tools']}
    assert child['metadata']['system_prompt'].count('Use OpenTerminal when needed.') == 1


@pytest.mark.asyncio
async def test_foreground_subagent_adds_direct_prompt_when_parent_used_explicit_tools(monkeypatch):
    _mock_payload_dependencies(monkeypatch)
    model = {'id': 'test-model', 'info': {'meta': {}}}
    request = SimpleNamespace(
        state=SimpleNamespace(direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'test-model': model})),
    )
    explicit_tools = [
        {
            'type': 'function',
            'function': {
                'name': 'caller_tool',
                'description': 'Caller-provided tool.',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }
    ]
    parent_form_data, parent_metadata, _ = await middleware.process_chat_payload(
        request,
        {
            'model': 'test-model',
            'messages': [
                {'role': 'system', 'content': 'Parent instructions.'},
                {'role': 'user', 'content': 'Use the terminal.'},
            ],
            'tools': copy.deepcopy(explicit_tools),
        },
        SimpleNamespace(id='user-id', role='user'),
        {'tool_servers': copy.deepcopy(DIRECT_TOOL_SERVERS)},
        model,
    )
    child = {}

    async def process_child(child_request, form_data, *, user):
        child_metadata = {
            'chat_id': form_data.pop('chat_id'),
            'message_id': form_data.pop('id'),
            'session_id': form_data.pop('session_id'),
            'tool_servers': form_data.pop('tool_servers'),
            'direct_tool_server_prompts_applied': form_data.pop('direct_tool_server_prompts_applied'),
        }
        child['form_data'], child['metadata'], _ = await middleware.process_chat_payload(
            child_request,
            form_data,
            user,
            child_metadata,
            model,
        )

    captured = await _capture_subagent_run(
        monkeypatch,
        background=False,
        metadata=parent_metadata,
        handler=process_child,
        request=request,
    )

    assert parent_form_data['tools'] == explicit_tools
    assert not parent_metadata.get('direct_tool_server_prompts_applied', False)
    assert captured['run']['tool_servers'] == DIRECT_TOOL_SERVERS
    assert {
        'run_command',
        'list_files',
        'read_file',
    } <= {tool['function']['name'] for tool in child['form_data']['tools']}
    assert child['metadata']['system_prompt'].count('Use OpenTerminal when needed.') == 1


@pytest.mark.asyncio
async def test_each_model_gets_direct_prompt_when_request_is_shared(monkeypatch):
    _mock_payload_dependencies(monkeypatch)
    model = {'id': 'test-model', 'info': {'meta': {}}}
    request = SimpleNamespace(
        state=SimpleNamespace(direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'test-model': model})),
    )

    async def process_model():
        _, metadata, _ = await middleware.process_chat_payload(
            request,
            {
                'model': 'test-model',
                'messages': [{'role': 'user', 'content': 'Use the terminal.'}],
            },
            SimpleNamespace(id='user-id', role='user'),
            {'tool_servers': copy.deepcopy(DIRECT_TOOL_SERVERS)},
            model,
        )
        return metadata

    results = await asyncio.gather(process_model(), process_model())

    assert all(
        metadata['system_prompt'].count('Use OpenTerminal when needed.') == 1
        and metadata['direct_tool_server_prompts_applied'] is True
        for metadata in results
    )


@pytest.mark.asyncio
async def test_background_subagent_excludes_direct_tool_servers(monkeypatch):
    captured = await _capture_subagent_run(monkeypatch, background=True)

    assert captured['run']['tool_servers'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('allowed', 'internal', 'expected_prompts_applied'),
    [(False, True, True), (True, True, True), (True, False, False)],
)
async def test_chat_completion_rechecks_direct_tool_server_permission(
    monkeypatch, allowed, internal, expected_prompts_applied
):
    model = {'id': 'test-model', 'info': {'meta': {}}}
    request = SimpleNamespace(
        state=SimpleNamespace(internal=internal),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'test-model': model}, redis=object())),
        headers={},
    )
    user = SimpleNamespace(id='user-id', role='user')
    captured = {}

    async def config_get(key, default=None):
        if key == 'user.permissions':
            return {'features': {'direct_tool_servers': False}}
        return default

    async def capture_payload(_request, form_data, _user, metadata, _model):
        captured['tool_servers'] = copy.deepcopy(metadata['tool_servers'])
        captured['direct_tool_server_prompts_applied'] = metadata['direct_tool_server_prompts_applied']
        return form_data, metadata, []

    monkeypatch.setattr(main, 'BYPASS_MODEL_ACCESS_CONTROL', True)
    monkeypatch.setattr(main.Models, 'get_model_by_id', AsyncMock(return_value=None))
    monkeypatch.setattr(main.Config, 'get', config_get)
    permission_check = AsyncMock(return_value=allowed)
    monkeypatch.setattr(main, 'has_permission', permission_check)
    monkeypatch.setattr(main, 'process_chat_payload', capture_payload)
    monkeypatch.setattr(main, 'drain_approved_tool_calls', AsyncMock(return_value=False))
    monkeypatch.setattr(main, 'chat_completion_handler', AsyncMock(return_value={'ok': True}))
    monkeypatch.setattr(main, 'build_chat_response_context', AsyncMock(return_value={}))
    monkeypatch.setattr(main, 'process_chat_response', AsyncMock(return_value={'ok': True}))

    response = await main.chat_completion(
        request,
        {
            'model': 'test-model',
            'messages': [{'role': 'user', 'content': 'Use the terminal.'}],
            'tool_servers': copy.deepcopy(DIRECT_TOOL_SERVERS),
            'direct_tool_server_prompts_applied': True,
        },
        user,
    )

    assert response == {'ok': True}
    assert captured['tool_servers'] == (DIRECT_TOOL_SERVERS if allowed else None)
    assert captured['direct_tool_server_prompts_applied'] is expected_prompts_applied
    permission_check.assert_awaited_once()
