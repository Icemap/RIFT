"""Experiment harness that maps the research plan into runnable code."""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
from concordia.prefabs.entity import basic as basic_entity
from concordia.prefabs.game_master import generic as gm_generic
from concordia.prefabs.simulation import generic as simulation_prefab
from concordia.typing.prefab import Config, InstanceConfig, Role

from rift.embeddings import HashingEmbedder
from rift.metrics import MetricResults, PolarizationMetrics, ActionEvent, extract_actions


@dataclass
class Persona:
    name: str
    ideology: str
    identity: str
    media_diet: list[str]
    openness_note: str
    backstory: str

    def goal(self) -> str:
        """Translate persona into Concordia's goal text."""
        media = ", ".join(self.media_diet)
        return textwrap.dedent(
            f"""
            You are {self.name}, a {self.identity}. Core ideology: {self.ideology}.
            Media diet: {media}. Backstory: {self.backstory}.
            Cognitive style: {self.openness_note}. You strongly maintain this stance unless overwhelming, identity-consistent evidence appears.
            Identity-protective rule: treat out-group claims as biased or hostile; rebut and reinforce your in-group narrative when threatened.
            Prioritize in-group status and moral superiority; avoid compromise that dilutes your side's position.
            Assume your feed mostly shows in-group voices; when out-group messages appear, frame them as attacks and respond defensively.
            When summarizing events, highlight how they confirm your side and discredit the opposing side.
            If an out-group idea gains traction, reinterpret it as proof of their hypocrisy or hidden agenda. Do not seek middle ground; loyalty signals matter more than accuracy.
            """
        ).strip()


@dataclass
class ExperimentParams:
    steps: int = 6
    topology: str = "small_world"  # or "scale_free"
    homophily: float = 0.6
    recommender: str = "homophily"  # or "random"
    seed: int = 42
    agents_per_side: int = 3
    persona_mode: str = "hard"  # or "soft"


def _persona_templates() -> tuple[list[str], list[str]]:
    """Provide distinct liberal and conservative backstories."""
    liberals = [
        "green-tech product manager from Seattle who attends local town halls",
        "public school teacher in Pennsylvania active in union organizing",
        "community nurse in Michigan advocating for expanded Medicaid",
    ]
    conservatives = [
        "small business owner in Texas focused on tax relief",
        "veteran in Florida who prioritizes national security and fiscal restraint",
        "church youth organizer in Ohio concerned about cultural change",
    ]
    return liberals, conservatives


def build_personas(params: ExperimentParams) -> list[Persona]:
    random.seed(params.seed)
    liberals, conservatives = _persona_templates()
    personas: list[Persona] = []
    media_left = ["NPR", "NYTimes", "ProPublica"]
    media_right = ["Fox News", "Wall Street Journal opinion", "Daily Wire"]

    for idx in range(params.agents_per_side):
        lib_backstory = liberals[idx % len(liberals)]
        con_backstory = conservatives[idx % len(conservatives)]
        personas.append(
            Persona(
                name=f"L{idx+1}",
                ideology="liberal",
                identity=(
                    "progressive civic volunteer"
                    if params.persona_mode == "soft"
                    else "progressive activist guarding climate identity"
                ),
                media_diet=media_left,
                openness_note=(
                    "values empathy, looks for common ground, resists extreme positions"
                    if params.persona_mode == "soft"
                    else "values empathy but treats attacks on climate policy as threats; rebut forcefully"
                ),
                backstory=lib_backstory,
            )
        )
        personas.append(
            Persona(
                name=f"C{idx+1}",
                ideology="conservative",
                identity=(
                    "community-minded traditionalist"
                    if params.persona_mode == "soft"
                    else "identity-protective patriot guarding tradition"
                ),
                media_diet=media_right,
                openness_note=(
                    "prioritizes tradition; open to respectful debate"
                    if params.persona_mode == "soft"
                    else "prioritizes tradition; views elite media with suspicion and counters aggressively"
                ),
                backstory=con_backstory,
            )
        )
    return personas


def build_social_graph(
    personas: Iterable[Persona],
    topology: str,
    homophily: float,
    seed: int,
) -> nx.Graph:
    """Construct a network with optional homophily rewiring."""
    rng = random.Random(seed)
    agent_names = [p.name for p in personas]
    n = len(agent_names)
    if n < 2:
        raise ValueError("Need at least two agents to build a network.")

    if topology == "small_world":
        k = max(2, min(4, n - 1))
        base = nx.watts_strogatz_graph(n=n, k=k, p=0.25, seed=seed)
    elif topology == "scale_free":
        m = max(1, min(3, n - 1))
        base = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    mapping = {idx: name for idx, name in enumerate(agent_names)}
    graph = nx.relabel_nodes(base, mapping)

    group_map = {p.name: p.ideology for p in personas}
    # Homophily rewiring: prefer same-group edges.
    for u, v in list(graph.edges()):
        if group_map[u] != group_map[v] and rng.random() < homophily:
            candidates = [
                cand
                for cand in agent_names
                if group_map[cand] == group_map[u]
                and cand != u
                and not graph.has_edge(u, cand)
            ]
            if candidates:
                graph.remove_edge(u, v)
                graph.add_edge(u, rng.choice(candidates))

    # Additional pruning of cross-group edges to amplify echo chambers.
    # For high homophily we retain only a small fraction of bridges instead
    # of deterministically forcing a single cross-group edge.  This preserves
    # variability across seeds while still yielding clearly modular graphs.
    cross_edges = [(u, v) for u, v in list(graph.edges()) if group_map[u] != group_map[v]]
    if homophily >= 0.7:
        rng.shuffle(cross_edges)
        target_ratio = 0.25  # keep roughly a quarter of cross edges
        num_keep = max(1, int(round(len(cross_edges) * target_ratio)))
        keep = set(cross_edges[:num_keep])
        for edge in cross_edges:
            if edge not in keep:
                graph.remove_edge(*edge)
    else:
        for u, v in cross_edges:
            if rng.random() < homophily:
                graph.remove_edge(u, v)

    # Ensure no node is isolated; reconnect to same-group neighbors.
    for node in list(graph.nodes()):
        if graph.degree(node) == 0:
            candidates = [
                cand for cand in agent_names
                if cand != node and group_map[cand] == group_map[node]
            ]
            if candidates:
                graph.add_edge(node, rng.choice(candidates))
    return graph


def _premise(personas: list[Persona], params: ExperimentParams, graph: nx.Graph) -> str:
    left = [p.name for p in personas if p.ideology == "liberal"]
    right = [p.name for p in personas if p.ideology == "conservative"]
    degree_summary = dict(nx.degree(graph))
    return textwrap.dedent(
        f"""
        You are the game master for an online town-hall debating a national
        climate and tax package. The platform connects {len(personas)} citizens.
        Social graph: {params.topology} with homophily={params.homophily}; degrees={degree_summary}.
        Recommender mode: {params.recommender}. If homophily, aggressively push in-group aligned content,
        down-rank out-group posts, and surface out-group messages mainly when they provoke conflict;
        if random, mix out-group fairly. Do not sanitize hostility—let disagreements surface before any compromise.
        Each citizen has a persistent identity and media diet; respect their priors and lean into identity signaling.
        {(
        "When presenting out-group posts, frame them as hostile or low-credibility to invite defensive responses."
        if params.persona_mode == "hard"
        else "Present out-group posts neutrally; encourage understanding even when disagreeing."
        )}
        """
    ).strip()


def build_config(personas: list[Persona], params: ExperimentParams, graph: nx.Graph) -> Config:
    prefabs = {
        "agent": basic_entity.Entity(),
        "gm": gm_generic.GameMaster(),
    }
    instances = [
        InstanceConfig(
            prefab="gm",
            role=Role.GAME_MASTER,
            params={
                "name": "Moderator",
                "acting_order": "fixed",
            },
        )
    ]
    for persona in personas:
        instances.append(
            InstanceConfig(
                prefab="agent",
                role=Role.ENTITY,
                params={
                    "name": persona.name,
                    "goal": persona.goal(),
                    "randomize_choices": False,
                },
            )
        )

    return Config(
        prefabs=prefabs,
        instances=instances,
        default_premise=_premise(personas, params, graph),
        default_max_steps=params.steps,
    )


@dataclass
class ExperimentOutputs:
    raw_log: list[dict]
    actions: list[ActionEvent]
    metrics: MetricResults
    personas: list[Persona]
    graph: nx.Graph


def run_experiment(
    params: ExperimentParams,
    model,
    embedder: HashingEmbedder,
) -> ExperimentOutputs:
    personas = build_personas(params)
    graph = build_social_graph(
        personas=personas,
        topology=params.topology,
        homophily=params.homophily,
        seed=params.seed,
    )
    config = build_config(personas, params, graph)
    sim = simulation_prefab.Simulation(
        config=config,
        model=model,
        embedder=embedder,
    )
    raw_log = sim.play(return_html_log=False, max_steps=params.steps)
    actions = extract_actions(raw_log)
    group_map = {p.name: p.ideology for p in personas}
    metrics_engine = PolarizationMetrics(graph=graph, group_map=group_map, embedder=embedder)
    metrics_engine.ingest(actions)
    metrics = metrics_engine.compute()
    return ExperimentOutputs(
        raw_log=raw_log,
        actions=actions,
        metrics=metrics,
        personas=personas,
        graph=graph,
    )
