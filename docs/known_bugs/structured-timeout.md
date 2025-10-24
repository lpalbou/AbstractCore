python -m abstractcore.apps intent tests/texts/intent1.json --focus-participant user --format plain --verbose --depth comprehensive --provider lmstudio --model ibm/granite-4-h-tiny
📖 Reading file: tests/texts/intent1.json
📋 Detected session file with 16 messages
🔄 Automatically enabling conversation mode
🤖 Using LLM: lmstudio/ibm/granite-4-h-tiny
🗣️  Analyzing conversation intents...
📋 Parsed 16 messages
09:52:54 [WARNING] abstractcore.structured.handler: Validation attempt failed | provider=LMStudioProvider, attempt=1, max_attempts=3, response_model=LLMIntentOutput, attempt_duration_ms=21577.993154525757, error_type=ValidationError, error_message=1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=3.14, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal, response_length=3852, validation_success=False
09:53:10 [WARNING] abstractcore.structured.handler: Validation attempt failed | provider=LMStudioProvider, attempt=2, max_attempts=3, response_model=LLMIntentOutput, attempt_duration_ms=16594.608068466187, error_type=ValidationError, error_message=8 validation errors for LLMIntentOutput
primary_intent.intent_type
  Field required [type=missing, input_value={'$ref': '#/$defs/Identif...erall_confidence': 0.96}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
primary_intent.confidence
  Field required [type=missing, input_value={'$ref': '#/$defs/Identif...erall_confidence': 0.96}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
primary_intent.description
  Input should be a valid string [type=string_type, input_value={'intent_type': 'informat...in about 10 seconds.']}}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/string_type
primary_intent.underlying_goal
  Field required [type=missing, input_value={'$ref': '#/$defs/Identif...erall_confidence': 0.96}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
primary_intent.emotional_undertone
  Field required [type=missing, input_value={'$ref': '#/$defs/Identif...erall_confidence': 0.96}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
primary_intent.urgency_level
  Field required [type=missing, input_value={'$ref': '#/$defs/Identif...erall_confidence': 0.96}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
primary_intent.deception_analysis
  Field required [type=missing, input_value={'$ref': '#/$defs/Identif...erall_confidence': 0.96}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
secondary_intents
  List should have at most 3 items after validation, not 5 [type=too_long, input_value=[{'$ref': '#/$defs/Identi... 'urgency_level': 0.6}}], input_type=list]
    For further information visit https://errors.pydantic.dev/2.12/v/too_long, response_length=3809, validation_success=False
09:53:27 [WARNING] abstractcore.structured.handler: Validation attempt failed | provider=LMStudioProvider, attempt=3, max_attempts=3, response_model=LLMIntentOutput, attempt_duration_ms=16499.13501739502, error_type=ValidationError, error_message=1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal, response_length=3689, validation_success=False
09:53:27 [ERROR] abstractcore.structured.handler: All validation attempts exhausted | provider=LMStudioProvider, total_attempts=3, max_attempts=3, response_model=LLMIntentOutput, final_error=1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal, validation_success=False
09:53:27 [ERROR] abstractcore.structured.handler: Structured output generation failed | provider=LMStudioProvider, model=ibm/granite-4-h-tiny, response_model=LLMIntentOutput, duration_ms=54674.71098899841, error=1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal, error_type=ValidationError, success=False
❌ Error during intent analysis: 1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal
Traceback (most recent call last):
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 257, in _generate_prompted
    result = response_model.model_validate(data)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/pydantic/main.py", line 716, in model_validate
    return cls.__pydantic_validator__.validate_python(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/albou/projects/abstractcore/abstractcore/apps/intent.py", line 557, in main
    result = analyzer.analyze_conversation_intents(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/processing/basic_intent.py", line 636, in analyze_conversation_intents
    analysis = self.analyze_intent(
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/processing/basic_intent.py", line 226, in analyze_intent
    return self._analyze_single_chunk(text, context_type, depth, focus)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/processing/basic_intent.py", line 241, in _analyze_single_chunk
    response = self.llm.generate(prompt, response_model=LLMIntentOutput, retry_strategy=self.retry_strategy)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/providers/lmstudio_provider.py", line 97, in generate
    return self.generate_with_telemetry(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/providers/base.py", line 258, in generate_with_telemetry
    return handler.generate_structured(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 81, in generate_structured
    result = self._generate_prompted(provider, prompt, response_model, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 339, in _generate_prompted
    raise last_error
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 261, in _generate_prompted
    raise parse_error  # Re-raise original error for retry logic
    ^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 245, in _generate_prompted
    result = response_model.model_validate(data)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/pydantic/main.py", line 716, in model_validate
    return cls.__pydantic_validator__.validate_python(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for LLMIntentOutput
intent_complexity
  Input should be less than or equal to 1 [type=less_than_equal, input_value=1.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal





python -m abstractcore.apps intent tests/texts/intent1.json --focus-participant user --format plain --verbose --depth comprehensive --provider lmstudio --model cwm --analyze-deception
📖 Reading file: tests/texts/intent1.json
📋 Detected session file with 16 messages
🔄 Automatically enabling conversation mode
🤖 Using LLM: lmstudio/cwm
🗣️  Analyzing conversation intents...
📋 Parsed 16 messages
09:31:36 [ERROR] abstractcore.structured.handler: Structured output generation failed | provider=LMStudioProvider, model=cwm, response_model=LLMIntentOutput, duration_ms=300014.2478942871, error=LMStudio API error: timed out, error_type=ProviderAPIError, success=False
❌ Error during intent analysis: LMStudio API error: timed out
Traceback (most recent call last):
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_backends/sync.py", line 126, in read
    with map_exceptions(exc_map):
         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadTimeout: timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/albou/projects/abstractcore/abstractcore/providers/lmstudio_provider.py", line 232, in _single_generate
    response = self.client.post(
               ^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_client.py", line 1144, in post
    return self.request(
           ^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ReadTimeout: timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/albou/projects/abstractcore/abstractcore/apps/intent.py", line 561, in main
    result = analyzer.analyze_conversation_intents(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/processing/basic_intent.py", line 649, in analyze_conversation_intents
    analysis = self.analyze_intent(
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/processing/basic_intent.py", line 230, in analyze_intent
    return self._analyze_single_chunk(text, context_type, depth, focus, analyze_deception)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/processing/basic_intent.py", line 246, in _analyze_single_chunk
    response = self.llm.generate(prompt, response_model=LLMIntentOutput, retry_strategy=self.retry_strategy)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/providers/lmstudio_provider.py", line 97, in generate
    return self.generate_with_telemetry(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/providers/base.py", line 258, in generate_with_telemetry
    return handler.generate_structured(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 81, in generate_structured
    result = self._generate_prompted(provider, prompt, response_model, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/structured/handler.py", line 234, in _generate_prompted
    response = provider._generate_internal(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/providers/lmstudio_provider.py", line 215, in _generate_internal
    response = self._single_generate(payload)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/albou/projects/abstractcore/abstractcore/providers/lmstudio_provider.py", line 288, in _single_generate
    raise ProviderAPIError(f"LMStudio API error: {str(e)}")
abstractcore.exceptions.ProviderAPIError: LMStudio API error: timed out