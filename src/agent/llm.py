"""LLM adapters: direct google-genai Client (Gemini Developer API / Vertex AI) + offline MockPursuitLLM."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Protocol

from google import genai
from google.genai import types
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agent.models import (
    FreshnessDiscrepancy,
    RequirementEvidence,
    RequirementItem,
    TurnRecord,
    WhatIfPatch,
)
from src.config import Settings

logger = logging.getLogger(__name__)

_TOOL_ALLOWED_KEYS: dict[str, tuple[str, ...]] = {
    "query_projects": (
        "subsector",
        "state",
        "role",
        "min_sqft",
        "min_fee_usd",
        "min_year",
        "leed_min",
        "search_text",
    ),
    "check_staff_credentials": ("credential_type", "state", "active_only"),
    "get_client_history": ("client_name",),
    "search_project_narratives": ("query", "subsector", "include_drafts", "top_k"),
}


class _ExtractedList(BaseModel):
    requirements: list[RequirementItem] = Field(default_factory=list)


class _ToolChoice(BaseModel):
    """Flat, Vertex-Gemini-compatible tool call schema (zero untyped dict fields)."""

    tool_name: Literal[
        "query_projects",
        "check_staff_credentials",
        "get_client_history",
        "search_project_narratives",
    ]
    subsector: str | None = None
    state: str | None = None
    role: str | None = None
    min_sqft: int | None = None
    min_fee_usd: float | None = None
    min_year: int | None = None
    leed_min: str | None = None
    search_text: str | None = None
    credential_type: str | None = None
    active_only: bool | None = None
    client_name: str | None = None
    query: str | None = None
    include_drafts: bool | None = None
    top_k: int | None = None

    def to_tool_args(self) -> dict[str, Any]:
        """Return only non-None arguments allowlisted for the chosen MCP tool."""
        allowed = _TOOL_ALLOWED_KEYS[self.tool_name]
        return {key: getattr(self, key) for key in allowed if getattr(self, key) is not None}


class PursuitLLM(Protocol):
    """Protocol implemented by both GeminiPursuitLLM and MockPursuitLLM."""

    def extract_requirements(
        self, rfp_text: str, config: RunnableConfig | None = None
    ) -> tuple[list[RequirementItem], int, int]: ...

    def plan_mcp_call(
        self, req: RequirementItem, retry_attempt: int, config: RunnableConfig | None = None
    ) -> tuple[str, dict[str, Any], int, int]: ...

    def parse_what_if(
        self,
        followup_text: str,
        pinned_requirements: list[RequirementItem],
        config: RunnableConfig | None = None,
    ) -> tuple[WhatIfPatch, int, int]: ...

    def summarize_turns(
        self,
        existing_summary: str,
        dropped_turns: list[TurnRecord],
        config: RunnableConfig | None = None,
    ) -> tuple[str, int, int]: ...

    def draft_scorecard_summary(
        self,
        requirements: list[RequirementItem],
        evidence_by_id: dict[str, RequirementEvidence],
        discrepancies: list[FreshnessDiscrepancy],
        what_if_assumptions: list[str],
        config: RunnableConfig | None = None,
    ) -> tuple[str, int, int]: ...


def _estimate_tokens(text: str) -> int:
    return max(20, len(text) // 4)


def _extract_genai_usage(response: Any, prompt_text: str, response_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        in_tok = getattr(usage, "prompt_token_count", None)
        out_tok = getattr(usage, "candidates_token_count", None)
        if in_tok is not None or out_tok is not None:
            return int(in_tok or 0), int(out_tok or 0)
    return _estimate_tokens(prompt_text), _estimate_tokens(response_text)


class MockPursuitLLM:
    """Deterministic zero-cost LLM used for offline pytest runs and CI evaluation."""

    def extract_requirements(
        self, rfp_text: str, config: RunnableConfig | None = None
    ) -> tuple[list[RequirementItem], int, int]:
        _ = config
        start = time.perf_counter()
        lower = rfp_text.lower()
        state = None
        if "texas" in lower or "(tx)" in lower or "austin" in lower or "houston" in lower:
            state = "TX"
        elif "california" in lower or "(ca)" in lower or "san francisco" in lower:
            state = "CA"

        subsector = None
        if "acute-care" in lower or "acute care" in lower or "hospital" in lower or "icu" in lower:
            subsector = "acute_care"
        elif "water" in lower or "pccp" in lower or "pump station" in lower:
            subsector = "water_conveyance"
        elif "seismic" in lower or "courthouse" in lower or "base isolation" in lower:
            subsector = "civic_seismic"

        role = "prime" if "prime" in lower else None
        leed_min = "Gold" if "leed gold" in lower else None

        sqft_match = re.search(r"(\d{2,3}),?000\s*sq\s*ft", lower)
        min_sqft = int(sqft_match.group(1)) * 1000 if sqft_match else None

        year_match = re.search(r"(202\d)\s*or\s*later|since\s*(202\d)|>=\s*(202\d)", lower)
        min_year = None
        if year_match:
            min_year = int(next(g for g in year_match.groups() if g is not None))

        count_match = re.search(r"at least\s*\*?\*?(\d+)\s+completed", lower)
        min_count = int(count_match.group(1)) if count_match else 2

        fee_match = re.search(r"\$(\d+(?:,\d+){2,})", rfp_text)
        min_total_fee_usd = None
        if "combined contract value" in lower and fee_match:
            min_total_fee_usd = float(fee_match.group(1).replace(",", ""))

        items: list[RequirementItem] = []
        if subsector:
            items.append(
                RequirementItem(
                    req_id="req_projects",
                    title=f"Past {subsector.replace('_', ' ')} projects ({state or 'US'})",
                    description=f"Minimum {min_count} completed {subsector} projects",
                    kind="crm_projects",
                    is_mandatory=True,
                    subsector=subsector,
                    state=state,
                    role=role,
                    min_count=min_count,
                    min_sqft=min_sqft,
                    min_total_fee_usd=min_total_fee_usd,
                    min_year=min_year,
                    leed_min=leed_min,
                )
            )

        if "structural engineer" in lower or " se " in lower or "(se)" in lower:
            items.append(
                RequirementItem(
                    req_id="req_credential",
                    title=f"Active {state or ''} SE license".strip(),
                    description=f"Mandatory active Structural Engineer (SE) license in {state or 'target state'}.",
                    kind="staff_credential",
                    is_mandatory=True,
                    state=state,
                    credential_type="SE",
                )
            )
        elif "professional engineer" in lower or "(pe)" in lower or " pe " in lower:
            items.append(
                RequirementItem(
                    req_id="req_credential",
                    title=f"Active {state or ''} PE license".strip(),
                    description=f"Mandatory active Professional Engineer (PE) license in {state or 'target state'}.",
                    kind="staff_credential",
                    is_mandatory=True,
                    state=state,
                    credential_type="PE",
                )
            )

        if "telemetry" in lower:
            items.append(
                RequirementItem(
                    req_id="req_scope",
                    title="Telemetry / Acute ICU phased scope",
                    description="Proven phased acute-care telemetry or ICU unit renovation.",
                    kind="narrative_scope",
                    is_mandatory=True,
                    subsector=subsector,
                    narrative_query="telemetry stepdown ward",
                )
            )
        elif "occupied icu" in lower or "icra" in lower:
            items.append(
                RequirementItem(
                    req_id="req_scope",
                    title="Occupied ICU / ICRA Class IV scope",
                    description="Proven phased occupied ICU renovation with ICRA Class IV containment.",
                    kind="narrative_scope",
                    is_mandatory=True,
                    subsector=subsector,
                    narrative_query="occupied ICU ICRA Class IV renovation",
                )
            )
        elif "pccp" in lower or "without service shutdown" in lower:
            items.append(
                RequirementItem(
                    req_id="req_scope",
                    title="Live-flow PCCP rehabilitation scope",
                    description="Large-diameter PCCP pipeline rehabilitation without service shutdown.",
                    kind="narrative_scope",
                    is_mandatory=True,
                    subsector=subsector,
                    narrative_query="PCCP rehabilitation without service shutdown",
                )
            )
        elif "base isolation" in lower or "historic landmark" in lower:
            items.append(
                RequirementItem(
                    req_id="req_scope",
                    title="Historic base-isolation seismic scope",
                    description="Base isolation seismic retrofit inside occupied historic landmark.",
                    kind="narrative_scope",
                    is_mandatory=True,
                    subsector=subsector,
                    narrative_query="base isolation historic landmark seismic",
                )
            )

        if "iso 19650" in lower or "iso-19650" in lower or "kitemark" in lower:
            items.append(
                RequirementItem(
                    req_id="req_unmapped_iso",
                    title="ISO 19650-2 BIM Kitemark Certification",
                    description="Corporate ISO 19650-2 BIM audit certificate (not tracked in CRM).",
                    kind="unmapped",
                    is_mandatory=True,
                )
            )

        for client_candidate in ("VA VISN 17", "Texas Water Development Authority", "GSA Region 9"):
            if client_candidate.lower() in lower:
                items.append(
                    RequirementItem(
                        req_id="req_client_history",
                        title=f"Client pursuit history ({client_candidate})",
                        description=f"Relationship track record with {client_candidate}.",
                        kind="client_history",
                        is_mandatory=False,
                        client_name=client_candidate,
                    )
                )
                break

        in_tokens = _estimate_tokens(rfp_text)
        out_tokens = max(80, len(items) * 45)
        logger.info(
            "LLM provider=mock method=extract_requirements items=%d in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            len(items),
            in_tokens,
            out_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return items, in_tokens, out_tokens

    def plan_mcp_call(
        self, req: RequirementItem, retry_attempt: int, config: RunnableConfig | None = None
    ) -> tuple[str, dict[str, Any], int, int]:
        _ = config
        start = time.perf_counter()

        if req.kind == "crm_projects":
            args = {
                "subsector": req.subsector,
                "state": req.state,
                "role": req.role,
                "min_sqft": req.min_sqft,
                "min_year": req.min_year,
                "leed_min": req.leed_min,
            }
            tool_name = "query_projects"
            tool_args = {k: v for k, v in args.items() if v is not None}
            in_tokens, out_tokens = 95, 45
        elif req.kind == "staff_credential":
            tool_name = "check_staff_credentials"
            tool_args = {
                "credential_type": req.credential_type,
                "state": req.state,
                "active_only": True,
            }
            in_tokens, out_tokens = 70, 35
        elif req.kind == "client_history":
            tool_name = "get_client_history"
            tool_args = {"client_name": req.client_name or ""}
            in_tokens, out_tokens = 65, 30
        else:
            # Narrative scope with bounded synonym broadening on retry_attempt == 1
            query = req.narrative_query or req.description
            if retry_attempt > 0:
                if "icu" in query.lower() or "acute" in req.title.lower():
                    query = "ICU ICRA critical care acute"
                elif "pccp" in query.lower():
                    query = "PCCP relining transmission water"
                else:
                    query = "seismic isolation courthouse retrofit"
            args = {"query": query, "subsector": req.subsector, "include_drafts": False, "top_k": 5}
            tool_name = "search_project_narratives"
            tool_args = {k: v for k, v in args.items() if v is not None}
            in_tokens, out_tokens = 85, 40

        logger.info(
            "LLM provider=mock method=plan_mcp_call req_id=%s retry=%d tool=%s in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            req.req_id,
            retry_attempt,
            tool_name,
            in_tokens,
            out_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return tool_name, tool_args, in_tokens, out_tokens

    def parse_what_if(
        self,
        followup_text: str,
        pinned_requirements: list[RequirementItem],
        config: RunnableConfig | None = None,
    ) -> tuple[WhatIfPatch, int, int]:
        _ = config
        start = time.perf_counter()
        lower = followup_text.lower()
        updated: list[RequirementItem] = []
        target_ids: list[str] = []

        sqft_match = re.search(r"(\d{2,3}),?000\s*sq\s*ft", lower)
        new_sqft = int(sqft_match.group(1)) * 1000 if sqft_match else None

        if new_sqft is not None or "threshold" in lower or "exclude" in lower:
            for req in pinned_requirements:
                if req.kind == "crm_projects":
                    copy = req.model_copy(deep=True)
                    if new_sqft is not None:
                        copy.min_sqft = new_sqft
                    updated.append(copy)
                    target_ids.append(copy.req_id)

        if "jv" in lower or "partner" in lower or "waive" in lower:
            for req in pinned_requirements:
                waive_cred = ("se" in lower or "license" in lower or "credential" in lower) and req.kind == "staff_credential"
                waive_iso = ("iso" in lower or "19650" in lower) and req.kind == "unmapped"
                if waive_cred or waive_iso:
                    copy = req.model_copy(deep=True)
                    copy.waived_by_what_if = True
                    updated.append(copy)
                    target_ids.append(copy.req_id)

        if not target_ids and pinned_requirements:
            target_ids = [pinned_requirements[0].req_id]

        patch = WhatIfPatch(
            summary=f"Applied What-If sensitivity adjustment: {followup_text}",
            target_req_ids=list(dict.fromkeys(target_ids)),
            updated_requirements=updated,
            assumption_note=followup_text,
        )
        in_tokens = _estimate_tokens(followup_text) + 120
        out_tokens = 75
        logger.info(
            "LLM provider=mock method=parse_what_if targets=%s in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            patch.target_req_ids,
            in_tokens,
            out_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return patch, in_tokens, out_tokens

    def summarize_turns(
        self,
        existing_summary: str,
        dropped_turns: list[TurnRecord],
        config: RunnableConfig | None = None,
    ) -> tuple[str, int, int]:
        _ = config
        start = time.perf_counter()
        snippets = [f"T{t.turn_index}({t.verdict}): {t.user_input[:60]}" for t in dropped_turns]
        joined = " | ".join(snippets)
        combined = f"{existing_summary}; {joined}".strip("; ")[:350]
        in_tokens = _estimate_tokens(combined) + 60
        out_tokens = 45
        logger.info(
            "LLM provider=mock method=summarize_turns dropped=%d in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            len(dropped_turns),
            in_tokens,
            out_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return combined, in_tokens, out_tokens

    def draft_scorecard_summary(
        self,
        requirements: list[RequirementItem],
        evidence_by_id: dict[str, RequirementEvidence],
        discrepancies: list[FreshnessDiscrepancy],
        what_if_assumptions: list[str],
        config: RunnableConfig | None = None,
    ) -> tuple[str, int, int]:
        _ = config
        start = time.perf_counter()
        passed = sum(1 for r in requirements if evidence_by_id.get(r.req_id) and evidence_by_id[r.req_id].status == "PASSED")
        failed = sum(1 for r in requirements if evidence_by_id.get(r.req_id) and evidence_by_id[r.req_id].status == "FAILED")
        parts = [f"Evaluated {len(requirements)} criteria ({passed} passed, {failed} failed, {len(discrepancies)} freshness flags)."]
        if what_if_assumptions:
            parts.append(f"What-If assumptions active: {'; '.join(what_if_assumptions)}.")
        summary_text = " ".join(parts)
        in_tokens, out_tokens = 140, 65
        logger.info(
            "LLM provider=mock method=draft_scorecard_summary criteria=%d in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            len(requirements),
            in_tokens,
            out_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return summary_text, in_tokens, out_tokens


@contextmanager
def _temporarily_unset_env(*keys: str) -> Iterator[None]:
    """Temporarily pop environment variables during SDK init and restore them afterward."""
    saved: dict[str, str] = {}
    for key in keys:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ[key] = value


def _build_http_options() -> types.HttpOptions:
    """Configure bounded timeout and retries for transient SSL/network/5xx errors."""
    return types.HttpOptions(
        timeout=60_000,
        retry_options=types.HttpRetryOptions(
            attempts=4,
            initial_delay=0.5,
            max_delay=4.0,
        ),
    )


def _build_genai_client(settings: Settings) -> Any:
    """Build google.genai.Client for Gemini Developer API (default live) or Vertex AI."""
    api_key = settings.resolved_google_api_key
    project = settings.google_cloud_project.strip() or None
    location = settings.google_cloud_location.strip() or "global"
    http_options = _build_http_options()

    if not settings.use_vertexai:
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required when LLM_PROVIDER=gemini.")
        with _temporarily_unset_env(
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        ):
            client = genai.Client(api_key=api_key, vertexai=False, http_options=http_options)
        backend = "gemini_api"
        auth_mode = "api_key"
        endpoint_mode = "gemini_developer_api"
        project_label = "<gemini_api>"
        location_label = "<gemini_api>"
    elif api_key:
        backend = "vertex_ai"
        auth_mode = "api_key"
        if project:
            endpoint_mode = "project_bound"
            project_label = project
            location_label = location
            client = genai.Client(
                vertexai=True,
                api_key=api_key,
                project=project,
                location=location,
                http_options=http_options,
            )
        else:
            endpoint_mode = "express_projectless"
            project_label = "<express/projectless>"
            location_label = "<express>"
            with _temporarily_unset_env("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
                client = genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
    else:
        backend = "vertex_ai"
        auth_mode = "adc"
        endpoint_mode = "project_bound"
        project_label = project or "<unset>"
        location_label = location
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=http_options,
        )

    api_client = getattr(client, "_api_client", None)
    base_url = getattr(getattr(api_client, "_http_options", None), "base_url", None) or "default"
    logger.info(
        "Initialized Google GenAI client: backend=%s auth_mode=%s endpoint_mode=%s "
        "primary_model=%s light_model=%s project=%s location=%s base_url=%s api_key_configured=%s",
        backend,
        auth_mode,
        endpoint_mode,
        settings.gemini_primary_model,
        settings.gemini_light_model,
        project_label,
        location_label,
        base_url,
        bool(api_key),
    )
    return client


class GeminiPursuitLLM:
    """Live Gemini adapter using direct `google.genai.Client` (`models.generate_content`)."""

    def __init__(self, settings: Settings) -> None:
        self._provider = "vertex_gemini" if settings.use_vertexai else "gemini"
        self._auth_mode = settings.vertex_auth_mode
        self._primary_model = settings.gemini_primary_model
        self._light_model = settings.gemini_light_model
        self.client = _build_genai_client(settings)

    def close(self) -> None:
        """Close the underlying HTTP/GenAI client if supported."""
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def _invoke_structured[T: BaseModel](
        self,
        schema: type[T],
        prompt: str,
        *,
        method: str,
        config: RunnableConfig | None,
        err_ctx: str = "",
    ) -> tuple[T, int, int, float]:
        _ = config
        start = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self._primary_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            parsed = resp.parsed
            if not isinstance(parsed, schema):
                parsed = schema.model_validate_json(resp.text or "{}")
        except Exception:
            logger.exception(
                "Gemini %s failed (provider=%s, model=%s, auth_mode=%s%s)",
                method,
                self._provider,
                self._primary_model,
                self._auth_mode,
                err_ctx,
            )
            raise
        in_tok, out_tok = _extract_genai_usage(resp, prompt, parsed.model_dump_json())
        return parsed, in_tok, out_tok, (time.perf_counter() - start) * 1000

    def _invoke_light(
        self,
        prompt: str,
        *,
        method: str,
        config: RunnableConfig | None,
    ) -> tuple[str, int, int, float]:
        _ = config
        start = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self._light_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
        except Exception:
            logger.exception(
                "Gemini %s failed (provider=%s, model=%s, auth_mode=%s)",
                method,
                self._provider,
                self._light_model,
                self._auth_mode,
            )
            raise
        text = (resp.text or "").strip()
        in_tok, out_tok = _extract_genai_usage(resp, prompt, text)
        return text, in_tok, out_tok, (time.perf_counter() - start) * 1000

    def extract_requirements(
        self, rfp_text: str, config: RunnableConfig | None = None
    ) -> tuple[list[RequirementItem], int, int]:
        prompt = (
            "Extract the AEC Go/No-Go qualification requirements from this solicitation into "
            "RequirementItem entries. Classify each as crm_projects, staff_credential, "
            "narrative_scope, client_history, or unmapped (for certifications not in CRM like ISO 19650).\n\n"
            f"{rfp_text}"
        )
        parsed, in_tok, out_tok, elapsed_ms = self._invoke_structured(
            _ExtractedList, prompt, method="extract_requirements", config=config
        )
        logger.info(
            "LLM provider=%s method=extract_requirements model=%s auth_mode=%s "
            "items=%d in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            self._provider,
            self._primary_model,
            self._auth_mode,
            len(parsed.requirements),
            in_tok,
            out_tok,
            elapsed_ms,
        )
        return parsed.requirements, in_tok, out_tok

    def plan_mcp_call(
        self, req: RequirementItem, retry_attempt: int, config: RunnableConfig | None = None
    ) -> tuple[str, dict[str, Any], int, int]:
        prompt = (
            "Select the single best read-only MCP tool (`query_projects`, `check_staff_credentials`, "
            "`get_client_history`, or `search_project_narratives`) and typed arguments to verify this "
            f"requirement (retry_attempt={retry_attempt}; if >0 broaden narrative synonyms):\n"
            f"{req.model_dump_json()}"
        )
        parsed, in_tok, out_tok, elapsed_ms = self._invoke_structured(
            _ToolChoice,
            prompt,
            method="plan_mcp_call",
            config=config,
            err_ctx=f", req_id={req.req_id}",
        )
        logger.info(
            "LLM provider=%s method=plan_mcp_call model=%s auth_mode=%s "
            "req_id=%s retry=%d tool=%s in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            self._provider,
            self._primary_model,
            self._auth_mode,
            req.req_id,
            retry_attempt,
            parsed.tool_name,
            in_tok,
            out_tok,
            elapsed_ms,
        )
        return parsed.tool_name, parsed.to_tool_args(), in_tok, out_tok

    def parse_what_if(
        self,
        followup_text: str,
        pinned_requirements: list[RequirementItem],
        config: RunnableConfig | None = None,
    ) -> tuple[WhatIfPatch, int, int]:
        reqs_json = "\n".join(r.model_dump_json() for r in pinned_requirements)
        prompt = (
            "Given these pinned RFP requirements and a What-If follow-up scenario, return a WhatIfPatch "
            "updating ONLY affected requirements (e.g. min_sqft change or waived_by_what_if=True):\n"
            f"Pinned:\n{reqs_json}\n\nFollow-up: {followup_text}"
        )
        parsed, in_tok, out_tok, elapsed_ms = self._invoke_structured(
            WhatIfPatch, prompt, method="parse_what_if", config=config
        )
        logger.info(
            "LLM provider=%s method=parse_what_if model=%s auth_mode=%s "
            "targets=%s in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            self._provider,
            self._primary_model,
            self._auth_mode,
            parsed.target_req_ids,
            in_tok,
            out_tok,
            elapsed_ms,
        )
        return parsed, in_tok, out_tok

    def summarize_turns(
        self,
        existing_summary: str,
        dropped_turns: list[TurnRecord],
        config: RunnableConfig | None = None,
    ) -> tuple[str, int, int]:
        turns_text = "\n".join(t.model_dump_json() for t in dropped_turns)
        prompt = (
            f"Compress these older What-If pursuit turns into <=250 chars preserving decisions.\n"
            f"Prior summary: {existing_summary}\nTurns:\n{turns_text}"
        )
        text, in_tok, out_tok, elapsed_ms = self._invoke_light(
            prompt, method="summarize_turns", config=config
        )
        logger.info(
            "LLM provider=%s method=summarize_turns model=%s auth_mode=%s "
            "dropped=%d in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            self._provider,
            self._light_model,
            self._auth_mode,
            len(dropped_turns),
            in_tok,
            out_tok,
            elapsed_ms,
        )
        return text[:350], in_tok, out_tok

    def draft_scorecard_summary(
        self,
        requirements: list[RequirementItem],
        evidence_by_id: dict[str, RequirementEvidence],
        discrepancies: list[FreshnessDiscrepancy],
        what_if_assumptions: list[str],
        config: RunnableConfig | None = None,
    ) -> tuple[str, int, int]:
        prompt = (
            "Write a concise 2-sentence AEC Go/No-Go executive summary grounded strictly in these results:\n"
            f"Requirements: {[r.title for r in requirements]}\n"
            f"Evidence: {[e.model_dump() for e in evidence_by_id.values()]}\n"
            f"Discrepancies: {[d.details for d in discrepancies]}\n"
            f"What-If assumptions: {what_if_assumptions}"
        )
        text, in_tok, out_tok, elapsed_ms = self._invoke_light(
            prompt, method="draft_scorecard_summary", config=config
        )
        logger.info(
            "LLM provider=%s method=draft_scorecard_summary model=%s auth_mode=%s "
            "criteria=%d in_tokens=%d out_tokens=%d elapsed_ms=%.1f",
            self._provider,
            self._light_model,
            self._auth_mode,
            len(requirements),
            in_tok,
            out_tok,
            elapsed_ms,
        )
        return text, in_tok, out_tok


VertexGeminiPursuitLLM = GeminiPursuitLLM


def create_llm(settings: Settings) -> PursuitLLM:
    """Return either GeminiPursuitLLM (Gemini Developer API / Vertex AI) or MockPursuitLLM."""
    if settings.llm_provider in {"gemini", "vertex_gemini"}:
        logger.info(
            "Creating LLM adapter: GeminiPursuitLLM (provider=%s use_vertexai=%s auth_mode=%s "
            "primary_model=%s light_model=%s)",
            settings.llm_provider,
            settings.use_vertexai,
            settings.vertex_auth_mode,
            settings.gemini_primary_model,
            settings.gemini_light_model,
        )
        return GeminiPursuitLLM(settings)
    logger.info(
        "Creating LLM adapter: MockPursuitLLM (auth_mode=%s primary_model=mock light_model=mock)",
        settings.vertex_auth_mode,
    )
    return MockPursuitLLM()
