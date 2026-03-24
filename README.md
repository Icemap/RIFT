# RIFT: LLM Polarization Simulation (Concordia + OpenAI)

- Google DeepMind Concordia (`gdm-concordia`) as the social-simulation kernel
- OpenAI API (via `OPENAI_API_KEY`), Gemini API (via `GEMINI_API_KEY`), or offline stubs
- A lightweight embedder model for semantic distances
- NetworkX metrics for modularity / echo-chamber style measurements

## Quickstart

1. Put your key in `.env`:

    ```properities
    OPENAI_API_KEY=xxxxx
    # Optional, Gemini Key
    # GEMINI_API_KEY=xxxxx
    ```

2. Smoke test without network (checks the pipeline end-to-end):

    ```shell
    uv run python main.py --model stub --steps 1 --agents-per-side 1
    ```

3. Minimal OpenAI run (default is `gpt-4o-mini`, short to avoid long calls):

    ```shell
    uv run python main.py --steps 2 --agents-per-side 2
    ```

4. Optional Gemini run:

    ```shell
    uv run python main.py --steps 2 --agents-per-side 2 --model gemini
    ```

Artifacts land in `artifacts.*/`:

- `metrics.json` — semantic centroid distance, echo-chamber score, modularity, in-group exposure ratio, bimodality coefficient
- `raw_log.json` — Concordia raw interaction log (for paper figures or ablations)
- `actions.json` — flattened per-step actions (actor/text)
- `graph_edges.json` — network used for structural metrics

## Scenario knobs

- `--topology` (`small_world` | `scale_free`)
- `--homophily` (0–1 rewiring probability toward in-group)
- `--recommender` (`homophily` | `random`) — guides the game-master narration style
- `--agents-per-side` — number of liberal/conservative personas (pre-built with media diet + identity cues from the plan)

## Notes for the paper

- Personas encode Social Identity Theory cues and media diets; goals are injected into Concordia's `basic.Entity` prefab.
- The game master runs in fixed order to surface backfire vs. convergence dynamics without short-circuiting conflict.
- Metrics align with the plan: semantic centroid distance, echo chamber delta, modularity, in-group exposure ratio, and a bimodality coefficient from projected opinions.
