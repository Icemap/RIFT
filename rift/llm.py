"""Language-model factory helpers."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import openai
from concordia.language_model import language_model as concordia_language_model
from concordia.language_model.google_aistudio_model import (
    GoogleAIStudioLanguageModel,
)
from concordia.language_model.no_language_model import (
    BiasedMedianChoiceLanguageModel,
    NoLanguageModel,
    RandomChoiceLanguageModel,
)
from concordia.language_model.retry_wrapper import RetryLanguageModel


class MissingAPIKeyError(RuntimeError):
    pass


class DeterministicStubLanguageModel(BiasedMedianChoiceLanguageModel):
    """Stub LM that keeps episodes running and picks the last option."""

    def __init__(self):
        super().__init__(median_probability=1.0)

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 0,
        terminators=None,
        temperature: float = 0.0,
        timeout: float = 0.0,
        seed: int | None = None,
    ) -> str:
        lower = prompt.lower()
        if "game/simulation finished" in lower or "terminate" in lower:
            return "No"
        if "action spec" in lower:
            return "prompt: what will the agent do?;;type: free"
        return "Continue."

    def sample_choice(self, prompt: str, responses, *, seed=None):
        if not responses:
            return 0, "", {}
        lower_prompt = prompt.lower()
        if "game/simulation finished" in lower_prompt or "terminate" in lower_prompt:
            for idx, letter in enumerate(responses):
                token = f"({letter.lower()})"
                for line in prompt.splitlines():
                    if token in line.lower() and "no" in line.lower():
                        return idx, responses[idx], {}
        idx = len(responses) - 1
        return idx, responses[idx], {}


class OpenRouterLanguageModel(concordia_language_model.LanguageModel):
    """OpenRouter-backed chat model using the OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.7,
    ):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY missing; unable to call OpenRouter."
            )
        # OpenRouter exposes an OpenAI-compatible endpoint.
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )
        self._model_name = model_name
        self._temperature = temperature

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = concordia_language_model.DEFAULT_MAX_TOKENS,
        terminators=concordia_language_model.DEFAULT_TERMINATORS,
        temperature: float = concordia_language_model.DEFAULT_TEMPERATURE,
        timeout: float = concordia_language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        stop = list(terminators) if terminators else None
        attempt = 0
        base_delay = 10.0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=(
                        temperature if temperature is not None else self._temperature
                    ),
                    max_tokens=max_tokens,
                    stop=stop,
                    timeout=timeout,
                    seed=seed,
                )
                return response.choices[0].message.content
            except openai.RateLimitError as exc:  # type: ignore[attr-defined]
                attempt += 1
                if attempt > 5:
                    raise
                wait = base_delay * (2 ** (attempt - 1))
                print(
                    f"[OpenRouterLanguageModel] Rate limit encountered, "
                    f"retry {attempt}/5 in {wait:.0f}s: {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait)

    def sample_choice(
        self,
        prompt: str,
        responses,
        *,
        seed: int | None = None,
    ):
        choice_prompt = (
            f"{prompt}\n\nRespond with exactly one of: {', '.join(responses)}"
        )
        text = self.sample_text(
            choice_prompt,
            max_tokens=16,
            temperature=0.0,
        ).strip()
        lowered = text.lower().strip(" ()")
        for idx, option in enumerate(responses):
            opt_norm = option.lower().strip(" ()")
            if lowered.startswith(opt_norm):
                return idx, option, {}
        # Fallback to first option.
        return 0, responses[0], {}


class OpenAIChatLanguageModel(concordia_language_model.LanguageModel):
    """Minimal OpenAI chat wrapper compatible with Concordia's interface."""

    @staticmethod
    def _use_responses_api(model_name: str) -> bool:
        # Newer OpenAI models (e.g., gpt-5-*) may not reliably return text via
        # Chat Completions in some SDK/model combinations. Use Responses API.
        lowered = model_name.lower()
        return (
            lowered.startswith("gpt-5")
            or lowered.startswith("o1")
            or lowered.startswith("o3")
            or lowered.startswith("o4")
        )

    @staticmethod
    def _supports_temperature(model_name: str) -> bool:
        # Some newer OpenAI models reject sampling parameters like `temperature`.
        # Keep this conservative; we can expand if we hit additional models.
        lowered = model_name.lower()
        return not (
            lowered.startswith("gpt-5")
            or lowered.startswith("o1")
            or lowered.startswith("o3")
            or lowered.startswith("o4")
        )

    @staticmethod
    def _supports_stop(model_name: str) -> bool:
        # Some models reject `stop`. Keep this conservative.
        lowered = model_name.lower()
        return not (
            lowered.startswith("gpt-5")
            or lowered.startswith("o1")
            or lowered.startswith("o3")
            or lowered.startswith("o4")
        )

    @staticmethod
    def _response_format(model_name: str):
        # Some models are stricter about structured outputs; default to explicit
        # text to keep behavior stable across SDK/model changes.
        lowered = model_name.lower()
        if lowered.startswith("gpt-5") or lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("o4"):
            return {"type": "text"}
        return openai.omit

    def _responses_text(self, prompt: str, *, max_output_tokens: int, temperature):
        temperature_param = (
            temperature if self._supports_temperature(self._model_name) else openai.omit
        )
        # For these models, ensure we leave enough budget for visible output.
        # Some requests may consume output budget on reasoning before emitting text.
        min_budget = 128
        max_output_tokens = max(max_output_tokens, min_budget)
        response = self._client.responses.create(
            model=self._model_name,
            input=prompt,
            max_output_tokens=max_output_tokens,
            service_tier=self._service_tier if self._service_tier is not None else openai.omit,
            temperature=temperature_param,
            reasoning={"effort": "minimal"},
            text={"format": {"type": "text"}, "verbosity": "low"},
        )
        return response.output_text

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        temperature: float = 0.7,
        service_tier: str | None = None,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY missing; unable to call OpenAI chat API."
            )
        self._client = openai.OpenAI(api_key=self._api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._service_tier = service_tier

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = concordia_language_model.DEFAULT_MAX_TOKENS,
        terminators=concordia_language_model.DEFAULT_TERMINATORS,
        temperature: float = concordia_language_model.DEFAULT_TEMPERATURE,
        timeout: float = concordia_language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        if self._use_responses_api(self._model_name):
            return self._responses_text(
                prompt,
                max_output_tokens=max_tokens,
                temperature=temperature if temperature is not None else self._temperature,
            )

        stop_param = (
            (list(terminators) if terminators else None)
            if self._supports_stop(self._model_name)
            else openai.omit
        )
        temperature_param = (
            temperature if self._supports_temperature(self._model_name) else openai.omit
        )
        attempt = 0
        base_delay = 5.0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature_param,
                    # Some models (e.g., gpt-5-nano) reject `max_tokens`; use
                    # `max_completion_tokens` instead.
                    max_completion_tokens=max_tokens,
                    stop=stop_param,
                    timeout=timeout,
                    seed=seed,
                    service_tier=(
                        self._service_tier if self._service_tier is not None else openai.omit
                    ),
                    response_format=self._response_format(self._model_name),
                )
                return response.choices[0].message.content
            except openai.RateLimitError as exc:  # type: ignore[attr-defined]
                attempt += 1
                if attempt > 5:
                    raise
                wait = base_delay * (2 ** (attempt - 1))
                print(
                    f"[OpenAIChatLanguageModel] Rate limit encountered, "
                    f"retry {attempt}/5 in {wait:.0f}s: {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait)

    def sample_choice(
        self,
        prompt: str,
        responses,
        *,
        seed: int | None = None,
    ):
        choice_prompt = (
            f"{prompt}\n\nRespond with exactly one of: {', '.join(responses)}"
        )
        text = self.sample_text(
            choice_prompt,
            max_tokens=16,
            temperature=0.0,
            seed=seed,
        ).strip()
        lowered = text.lower().strip(" ()")
        for idx, option in enumerate(responses):
            opt_norm = option.lower().strip(" ()")
            if lowered.startswith(opt_norm):
                return idx, option, {}
        # If parsing fails, default to a conservative non-terminating option
        # when available (many Concordia prompts include Yes/No choices).
        for idx, option in enumerate(responses):
            if option.strip().lower() in {"no", "(no)"}:
                return idx, option, {}
        # Fallback to first option.
        return 0, responses[0], {}


@dataclass
class ModelConfig:
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    service_tier: str | None = None


def _wrap_with_retry(model):
    """Add basic retry/backoff around Concordia LMs."""
    return RetryLanguageModel(model=model, retry_tries=3, backoff_factor=2.0)


def build_model(cfg: ModelConfig):
    provider = cfg.provider.lower()
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY/GOOGLE_API_KEY missing; unable to call Gemini."
            )
        model = GoogleAIStudioLanguageModel(
            model_name=cfg.model_name,
            api_key=api_key,
        )
        return _wrap_with_retry(model)
    if provider == "openai":
        model = OpenAIChatLanguageModel(
            model_name=cfg.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=cfg.temperature,
            service_tier=cfg.service_tier,
        )
        return _wrap_with_retry(model)
    if provider == "openrouter":
        model = OpenRouterLanguageModel(
            model_name=cfg.model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=cfg.temperature,
        )
        return _wrap_with_retry(model)
    if provider == "stub":
        return DeterministicStubLanguageModel()
    if provider == "stub-random":
        return RandomChoiceLanguageModel()
    if provider == "offline-null":
        return NoLanguageModel()
    raise ValueError(f"Unknown model provider: {cfg.provider}")
