import sqlite3

import pytest

from agentbench_frame.generals.macro_action_space import PrimitiveTransition


class FakePrefixGraph:
    def __init__(self, graph, spec_id="fake-spec-v1"):
        self.graph = graph
        self.spec_id = spec_id

    def legal_transitions(self, state):
        yield from self.graph.get(state["id"], ())


def state(identifier):
    return {"id": identifier, "actor": 0}


def edge(command, target, terminal=False):
    return PrimitiveTransition(
        command=(command,),
        state_after=state(target),
        terminal=terminal,
    )


def test_exact_counter_counts_distinct_prefixes_with_shared_successor(
    tmp_path,
):
    from agentbench_frame.generals.macro_counter import ExactMacroCounter

    graph = FakePrefixGraph(
        {
            "root": [
                edge(1, "shared"),
                edge(2, "shared"),
            ],
            "shared": [edge(3, "terminal", terminal=True)],
        }
    )

    result = ExactMacroCounter(
        graph,
        tmp_path / "cache.sqlite",
    ).count(state("root"))

    assert result.support_size == 5
    assert result.status == "complete"


def test_exact_counter_counts_immediate_end_and_terminal_branch(tmp_path):
    from agentbench_frame.generals.macro_counter import ExactMacroCounter

    empty = ExactMacroCounter(
        FakePrefixGraph({}),
        tmp_path / "empty.sqlite",
    ).count(state("root"))
    terminal = ExactMacroCounter(
        FakePrefixGraph({"root": [edge(1, "done", terminal=True)]}),
        tmp_path / "terminal.sqlite",
    ).count(state("root"))

    assert empty.support_size == 1
    assert terminal.support_size == 2


def test_counter_resumes_only_exact_completed_subtrees(tmp_path):
    from agentbench_frame.generals.macro_counter import (
        ExactCountIncomplete,
        ExactMacroCounter,
    )

    graph = FakePrefixGraph(
        {
            "root": [edge(1, "short"), edge(2, "long")],
            "short": [edge(3, "done", terminal=True)],
            "long": [edge(4, "tail")],
            "tail": [edge(5, "done", terminal=True)],
        }
    )
    cache = tmp_path / "cache.sqlite"

    with pytest.raises(ExactCountIncomplete) as stopped:
        ExactMacroCounter(
            graph,
            cache,
            max_expanded_states=3,
        ).count(state("root"))

    assert stopped.value.result.support_size is None
    resumed = ExactMacroCounter(
        graph,
        cache,
        max_expanded_states=100,
    ).count(state("root"))
    assert resumed.status == "complete"
    assert resumed.support_size == 6
    assert resumed.cache_hits > 0


def test_counter_persists_arbitrary_precision_decimal_strings(tmp_path):
    from agentbench_frame.generals.macro_counter import ExactMacroCounter

    graph = {}
    for index in range(60):
        graph[str(index)] = [
            edge(1, str(index + 1)),
            edge(2, str(index + 1)),
        ]
    cache = tmp_path / "cache.sqlite"
    result = ExactMacroCounter(
        FakePrefixGraph(graph),
        cache,
    ).count(state("0"))

    assert result.support_size is not None
    assert result.support_size > 2**53
    with sqlite3.connect(cache) as connection:
        stored = connection.execute(
            "SELECT count_decimal FROM exact_counts "
            "WHERE state_id = ?",
            (result.root_state_id,),
        ).fetchone()[0]
    assert stored == str(result.support_size)


def test_counter_detects_cycles(tmp_path):
    from agentbench_frame.generals.macro_counter import (
        ExactCounterError,
        ExactMacroCounter,
    )

    with pytest.raises(ExactCounterError, match="cycle"):
        ExactMacroCounter(
            FakePrefixGraph({"root": [edge(1, "root")]}),
            tmp_path / "cache.sqlite",
        ).count(state("root"))


def test_counter_cache_isolated_by_action_space_spec(tmp_path):
    from agentbench_frame.generals.macro_counter import ExactMacroCounter

    cache = tmp_path / "cache.sqlite"
    one = ExactMacroCounter(
        FakePrefixGraph({}, spec_id="one"),
        cache,
    ).count(state("root"))
    two = ExactMacroCounter(
        FakePrefixGraph(
            {"root": [edge(1, "done", terminal=True)]},
            spec_id="two",
        ),
        cache,
    ).count(state("root"))

    assert one.support_size == 1
    assert two.support_size == 2


def test_counter_rejects_corrupted_cached_count(tmp_path):
    from agentbench_frame.generals.macro_counter import (
        ExactCounterError,
        ExactMacroCounter,
    )

    cache = tmp_path / "cache.sqlite"
    counter = ExactMacroCounter(FakePrefixGraph({}), cache)
    result = counter.count(state("root"))
    with sqlite3.connect(cache) as connection:
        connection.execute(
            "UPDATE exact_counts SET count_decimal = 'corrupt' "
            "WHERE state_id = ?",
            (result.root_state_id,),
        )
        connection.commit()

    with pytest.raises(ExactCounterError, match="corrupt"):
        ExactMacroCounter(
            FakePrefixGraph({}),
            cache,
        ).count(state("root"))


def test_counter_reports_wall_time_guard_without_partial_root(tmp_path):
    from agentbench_frame.generals.macro_counter import (
        ExactCountIncomplete,
        ExactMacroCounter,
    )

    with pytest.raises(ExactCountIncomplete) as stopped:
        ExactMacroCounter(
            FakePrefixGraph({}),
            tmp_path / "cache.sqlite",
            max_wall_time_s=0,
        ).count(state("root"))

    assert stopped.value.result.status == "incomplete_wall_time"
    assert stopped.value.result.support_size is None
