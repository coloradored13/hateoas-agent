"""Tests for async_runner — parallel agent execution and async workflow runner."""

import asyncio

import pytest

from hateoas_agent.agent_slot import AgentSlot, AgentStatus
from hateoas_agent.async_runner import AsyncRunner
from hateoas_agent.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def review_orchestrator():
    """Three-phase orchestrator for async tests."""
    orch = Orchestrator(
        name="review",
        agents=[
            AgentSlot("tech-architect", role="Architecture"),
            AgentSlot("product-strategist", role="Market"),
            AgentSlot("devils-advocate", role="Adversarial", join_phase="challenge"),
        ],
    )
    orch.phase("research", parallel=True)
    orch.phase("challenge", parallel=True)
    orch.phase("synthesis", terminal=True)

    orch.transition(
        "research",
        "challenge",
        guard=lambda ctx: ctx.get("converged", False),
    )
    orch.transition(
        "challenge",
        "synthesis",
        guard=lambda ctx: ctx.get("exit_gate") == "PASS",
    )
    orch.transition(
        "challenge",
        "challenge",
        guard=lambda ctx: ctx.get("exit_gate") == "FAIL",
    )
    return orch


# ---------------------------------------------------------------------------
# run_agents_parallel
# ---------------------------------------------------------------------------


class TestRunAgentsParallel:
    @pytest.mark.asyncio
    async def test_parallel_sync_executors(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("a1"), AgentSlot("a2"), AgentSlot("a3")],
        )
        orch.set_executor(lambda a, t: f"{a.name}:{t}")

        agents = list(orch.agents.values())
        results = await orch.run_agents_parallel(agents, task="analyze")

        assert len(results) == 3
        outputs = {r.agent_name: r.output for r in results}
        assert outputs["a1"] == "a1:analyze"
        assert outputs["a2"] == "a2:analyze"
        assert outputs["a3"] == "a3:analyze"

    @pytest.mark.asyncio
    async def test_parallel_async_executors(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("a1"), AgentSlot("a2")],
        )

        async def async_exec(agent, task):
            await asyncio.sleep(0.01)
            return f"async:{agent.name}"

        orch.set_executor(async_exec)
        agents = list(orch.agents.values())
        results = await orch.run_agents_parallel(agents, task="work")

        assert all(r.status == AgentStatus.CONVERGED for r in results)
        assert {r.output for r in results} == {"async:a1", "async:a2"}

    @pytest.mark.asyncio
    async def test_parallel_mixed_success_error(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("good"), AgentSlot("bad")],
        )

        def executor(agent, task):
            if agent.name == "bad":
                raise RuntimeError("agent failed")
            return "ok"

        orch.set_executor(executor)
        agents = list(orch.agents.values())
        results = await orch.run_agents_parallel(agents, task="work")

        by_name = {r.agent_name: r for r in results}
        assert by_name["good"].status == AgentStatus.CONVERGED
        assert by_name["good"].output == "ok"
        assert by_name["bad"].status == AgentStatus.ERROR
        assert "agent failed" in by_name["bad"].error

    @pytest.mark.asyncio
    async def test_parallel_timeout(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("fast"), AgentSlot("slow")],
        )

        async def executor(agent, task):
            if agent.name == "slow":
                await asyncio.sleep(10)
            return "done"

        orch.set_executor(executor)
        agents = list(orch.agents.values())
        results = await orch.run_agents_parallel(agents, task="work", timeout=0.1)

        by_name = {r.agent_name: r for r in results}
        assert by_name["fast"].status == AgentStatus.CONVERGED
        assert by_name["slow"].status == AgentStatus.ERROR
        assert "Timeout" in by_name["slow"].error

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("c"), AgentSlot("a"), AgentSlot("b")],
        )
        orch.set_executor(lambda a, t: a.name)

        agents = [orch.get_agent("c"), orch.get_agent("a"), orch.get_agent("b")]
        results = await orch.run_agents_parallel(agents, task="x")
        assert [r.agent_name for r in results] == ["c", "a", "b"]

    @pytest.mark.asyncio
    async def test_parallel_inline_executor(self):
        orch = Orchestrator(name="test", agents=[AgentSlot("a1")])

        results = await orch.run_agents_parallel(
            list(orch.agents.values()),
            task="data",
            executor=lambda a, t: f"{a.name}+{t}",
        )
        assert results[0].output == "a1+data"

    @pytest.mark.asyncio
    async def test_parallel_no_executor_returns_error(self):
        orch = Orchestrator(name="test", agents=[AgentSlot("a1")])
        results = await orch.run_agents_parallel(list(orch.agents.values()), task="work")
        assert results[0].status == AgentStatus.ERROR
        assert "No executor" in results[0].error

    @pytest.mark.asyncio
    async def test_parallel_sets_agent_statuses(self):
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot("a1"), AgentSlot("a2")],
        )
        orch.set_executor(lambda a, t: "ok")

        agents = list(orch.agents.values())
        await orch.run_agents_parallel(agents, task="x")

        assert all(a.status == AgentStatus.CONVERGED for a in agents)

    @pytest.mark.asyncio
    async def test_parallel_truly_concurrent(self):
        """Verify agents actually run in parallel, not sequentially."""
        orch = Orchestrator(
            name="test",
            agents=[AgentSlot(f"a{i}") for i in range(5)],
        )

        async def slow_exec(agent, task):
            await asyncio.sleep(0.05)
            return "done"

        orch.set_executor(slow_exec)
        agents = list(orch.agents.values())

        import time

        start = time.monotonic()
        results = await orch.run_agents_parallel(agents, task="x")
        elapsed = time.monotonic() - start

        assert all(r.status == AgentStatus.CONVERGED for r in results)
        # 5 agents x 0.05s each = 0.25s sequential. Parallel should be ~0.05s.
        assert elapsed < 0.2


# ---------------------------------------------------------------------------
# AsyncRunner.run_orchestrated
# ---------------------------------------------------------------------------


class TestRunOrchestrated:
    @pytest.mark.asyncio
    async def test_simple_to_terminal(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b", terminal=True)
        orch.transition("a", "b")

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated()

        assert state.current_phase == "b"
        assert state.is_terminal is True
        assert state.phase_history == ["a", "b"]

    @pytest.mark.asyncio
    async def test_guarded_workflow(self, review_orchestrator):
        orch = review_orchestrator
        orch.set_executor(lambda a, t: f"{a.name}: analyzed")

        @orch.on_phase("research")
        def handle_research(orchestrator, agents, context):
            return {"converged": True}

        @orch.on_phase("challenge")
        def handle_challenge(orchestrator, agents, context):
            round_num = context.get("challenge_round", 0) + 1
            if round_num < 2:
                return {"challenge_round": round_num, "exit_gate": "FAIL"}
            return {"challenge_round": round_num, "exit_gate": "PASS"}

        @orch.on_phase("synthesis")
        def handle_synthesis(orchestrator, agents, context):
            return {"final": True}

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated()

        assert state.is_terminal is True
        assert state.current_phase == "synthesis"
        assert state.context["challenge_round"] == 2
        assert state.phase_history == ["research", "challenge", "challenge", "synthesis"]

    @pytest.mark.asyncio
    async def test_async_phase_handler(self):
        orch = Orchestrator(name="test")
        orch.phase("init")
        orch.phase("done", terminal=True)
        orch.transition("init", "done")

        @orch.on_phase("init")
        async def async_handler(orchestrator, agents, context):
            await asyncio.sleep(0.01)
            return {"async_ran": True}

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated()

        assert state.context["async_ran"] is True
        assert state.is_terminal is True

    @pytest.mark.asyncio
    async def test_parallel_agents_in_async_handler(self, review_orchestrator):
        orch = review_orchestrator

        async def async_exec(agent, task):
            await asyncio.sleep(0.01)
            return f"{agent.name}: {task}"

        orch.set_executor(async_exec)

        @orch.on_phase("research")
        async def handle_research(orchestrator, agents, context):
            results = await orchestrator.run_agents_parallel(
                agents, task=context.get("task", "default")
            )
            return {
                "findings": [r.output for r in results],
                "converged": True,
            }

        @orch.on_phase("challenge")
        def handle_challenge(orchestrator, agents, context):
            return {"exit_gate": "PASS"}

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated(context={"task": "review API"})

        assert state.is_terminal is True
        assert len(state.context["findings"]) == 2  # TA + PS (DA joins at challenge)

    @pytest.mark.asyncio
    async def test_stalls_when_no_guard_matches(self):
        orch = Orchestrator(name="test")
        orch.phase("stuck")
        orch.phase("unreachable", terminal=True)
        orch.transition("stuck", "unreachable", guard=lambda ctx: False)

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated()

        assert state.current_phase == "stuck"
        assert not state.is_terminal

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        orch = Orchestrator(name="test")
        orch.phase("loop")
        orch.transition("loop", "loop")  # infinite unguarded loop

        counter = {"n": 0}

        @orch.on_phase("loop")
        def handle(orchestrator, agents, context):
            counter["n"] += 1
            return {}

        runner = AsyncRunner(orch, max_iterations=5)
        state = await runner.run_orchestrated()

        # 1 from start + 5 from iterations = 6 total handler calls
        assert counter["n"] == 6
        assert len(state.phase_history) == 6

    @pytest.mark.asyncio
    async def test_with_context(self):
        orch = Orchestrator(name="test")
        orch.phase("a")
        orch.phase("b", terminal=True)
        orch.transition("a", "b", guard=lambda ctx: ctx.get("ready"))

        @orch.on_phase("a")
        def handle(orchestrator, agents, context):
            return {"ready": True}

        runner = AsyncRunner(orch)
        state = await runner.run_orchestrated(context={"input": "data"})

        assert state.current_phase == "b"
        assert state.context["input"] == "data"
        assert state.context["ready"] is True

    @pytest.mark.asyncio
    async def test_orchestrator_property(self, review_orchestrator):
        runner = AsyncRunner(review_orchestrator)
        assert runner.orchestrator is review_orchestrator
