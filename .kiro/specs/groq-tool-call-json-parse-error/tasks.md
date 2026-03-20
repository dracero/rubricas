# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Large Payload Tool Call JSON Parse Error
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to the concrete failing case — `guardar_ontologia_en_qdrant` called with 15 entidades + 45 relaciones (serialized JSON > GROQ_TOOL_CALL_LIMIT)
  - From Bug Condition in design: `isBugCondition(tool_call_args)` is true when `len(JSON.serialize(tool_call_args)) > GROQ_TOOL_CALL_LIMIT` AND `tool_name == "guardar_ontologia_en_qdrant"`
  - Test that calling `guardar_ontologia_en_qdrant` with a large list of entidades/relaciones raises `litellm.BadRequestError` with "Failed to parse tool call arguments as JSON" (or produces a truncated JSON payload)
  - Simulate the ADK tool call by constructing the args dict and serializing it, then verifying the payload size exceeds the Groq limit
  - Run test on UNFIXED code — expect FAILURE (confirms the bug exists)
  - Document counterexamples found (e.g., "15 entidades + 45 relaciones → JSON ~6KB → Groq truncates → BadRequestError")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Small Payload Behavior Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: `guardar_ontologia_en_qdrant(entidades=[3 entidades simples], relaciones=[5 relaciones])` returns `"✅ Ontología guardada exitosamente..."` on unfixed code
  - Observe: `buscar_contexto_qdrant(query)` returns results with `score` field on unfixed code
  - Observe: `buscar_contexto_para_evaluacion(consulta)` returns results without errors on unfixed code
  - Write property-based test: for all inputs where `NOT isBugCondition(tool_call_args)` (i.e., small payloads with ≤3 entidades), `guardar_ontologia_en_qdrant` returns a string starting with `"✅"` and entities are persisted with correct fields (nombre, tipo, contexto, propiedades, relaciones_salientes)
  - Write property-based test: for any query string, `buscar_contexto_qdrant` and `buscar_contexto_para_evaluacion` return results without raising exceptions
  - Verify tests pass on UNFIXED code
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 3. Fix for guardar_ontologia_en_qdrant large payload bug

  - [ ] 3.1 Implement the fix in `agents/generator/app/adk_agents.py`
    - Change `guardar_ontologia_en_qdrant` signature to accept `ontologia_json: str` (JSON-as-string) instead of `entidades: List[Dict]` and `relaciones: List[Dict]`
    - The LLM passes a serialized JSON string as a single argument — this keeps the tool call argument small and avoids Groq's structured-object parsing limit
    - Internally deserialize `ontologia_json` with `json.loads()` to extract `entidades` and `relaciones` lists
    - Update the docstring to reflect the new signature and explain the JSON string format expected
    - Update the instruction of the `ontologo_agent` to tell the model to serialize the ontology as a JSON string and pass it as the `ontologia_json` argument
    - _Bug_Condition: isBugCondition(tool_call_args) where len(JSON.serialize(tool_call_args)) > GROQ_TOOL_CALL_LIMIT AND tool_name == "guardar_ontologia_en_qdrant"_
    - _Expected_Behavior: guardar_ontologia_en_qdrant returns "✅ Ontología guardada exitosamente..." for any number of entidades/relaciones without raising litellm.BadRequestError_
    - _Preservation: buscar_contexto_qdrant and buscar_contexto_para_evaluacion unchanged; entity fields (nombre, tipo, contexto, propiedades, relaciones_salientes) persisted correctly; small-payload pipeline still works_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [ ] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Large Payload Tool Call JSON Parse Error
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior: `guardar_ontologia_en_qdrant` with large payload must succeed (return `"✅"`) without raising `litellm.BadRequestError`
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2_

  - [ ] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Small Payload Behavior Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in small-payload behavior, search tools, and evaluator agent)
    - Confirm all tests still pass after fix (no regressions)

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
