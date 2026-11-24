"""The report of a cache measurement, as the command prints it.

It is made from the saved figures of a run (``summary.json``) and from nothing else, so
it cannot say anything the run did not save.
"""

from collections.abc import Mapping
from typing import Any

from handbook_rag.evals.cache_pairs import (
    CHOOSING,
    COSINE,
    DIFFERENT,
    ENTAILS_BACKWARD,
    ENTAILS_FORWARD,
    MISSED,
    NOT_SERVED,
    POLICIES,
    REPORTING,
    RIGHT,
    SAME,
    THRESHOLDS,
    WRONG,
)


def report_lines(summary: Mapping[str, Any]) -> list[str]:
    """A cache measurement as lines of text, made from its saved figures alone.

    What the command prints is then what the run saved, and the report of a committed
    run can be printed again without running a model.
    """
    return [
        *_models(summary),
        "",
        *_pair_view(summary),
        "",
        *_chosen(summary),
        "",
        *_reported(summary),
        "",
        *_tested(summary),
        "",
        *_described(summary),
        *_disputed(summary),
    ]


def _threshold(value: float | None) -> str:
    return "none" if value is None else f"{value:.4f}"


def _share(estimate: Mapping[str, Any] | None) -> str:
    if estimate is None:
        return "no pairs"
    return f"{estimate['mean']:.3f} ({estimate['low']:.3f} to {estimate['high']:.3f})"


def _whose(serve: Mapping[str, Any]) -> str:
    whose = "its own golden question" if serve["own"] else "another stored question"
    # What a cache holds for a question the handbook does not answer is a decline.
    return f"{whose} {serve['served']}{', a stored decline' if serve['stored_unanswered'] else ''}"


def _serve(serve: Mapping[str, Any] | None) -> str:
    """A wrong serve in one phrase, or ``none`` where a half holds no such question."""
    if serve is None:
        return "none"
    return f"{serve['pair']} ({serve['group']}) at {serve['score']:.4f}, {_whose(serve)}"


def _models(summary: Mapping[str, Any]) -> list[str]:
    counted, config, timing = summary["pairs"], summary["config"], summary["timing"]
    return [
        f"Cache measurement of {summary['measured_on']}: {counted['agreed']} agreed pairs "
        f"({counted[SAME]} same, {counted[DIFFERENT]} different). Disputed and listed apart: "
        f"{', '.join(counted['disputed']) or 'none'}",
        "  requests     none to any model API: both models run on this machine",
        f"  embedding    {config['embedding_model']} on {config['embedding_device']}: "
        f"{timing['questions_embedded']} questions in {timing['embedding_seconds']:.1f} s, "
        f"one at a time",
        f"  verifier     {config['verifier_model']} on {config['verifier_device']}: "
        f"{timing['pairs_verified']} pairs, each read both ways, in "
        f"{timing['verifying_seconds']:.1f} s, in batches",
        f"               entailment is output {config['verifier_entailment_output']} of "
        f"{', '.join(config['verifier_outputs'])}, read from the model's configuration",
    ]


def _pair_view(summary: Mapping[str, Any]) -> list[str]:
    counted, view = summary["pairs"], summary["pair_view"]["scores"]
    lines = [
        "Pair view: each pair against its own golden question",
        "  All agreed pairs. A description of the scores, not a held-out result.",
        f"  {'score':<18}{'area':>6}   {'same: min':>9}{'median':>8}{'max':>7}"
        f"   {'different: min':>14}{'median':>8}{'max':>7}",
    ]
    for score, figures in view.items():
        same, different = figures["by_kind"][SAME], figures["by_kind"][DIFFERENT]
        lines.append(
            f"  {score:<18}{figures['area_under_curve']:>6.3f}   {same['min']:>9.3f}"
            f"{same['median']:>8.3f}{same['max']:>7.3f}   {different['min']:>14.3f}"
            f"{different['median']:>8.3f}{different['max']:>7.3f}"
        )
    lines.append(f"  median by group     {'pairs':>5}" + "".join(f"{p:>14}" for p in POLICIES))
    for kind, by in ((SAME, "by_how"), (DIFFERENT, "by_what_differs"), (DIFFERENT, "by_source")):
        for group, spread in view[COSINE][by].items():
            medians = "".join(f"{view[policy][by][group]['median']:>14.3f}" for policy in POLICIES)
            lines.append(f"  {kind + ', ' + group:<20}{spread['pairs']:>5}{medians}")
    lines.append(
        f"  pairs at or above a threshold: same of {counted[SAME]}, then different of "
        f"{counted[DIFFERENT]}"
    )
    lines.append("  threshold " + "".join(f"{policy:>16}" for policy in POLICIES))
    for place, threshold in enumerate(THRESHOLDS):
        cells = [view[policy]["at_or_above"][place] for policy in POLICIES]
        counts = "".join(f"{f'{cell[SAME]}, {cell[DIFFERENT]}':>16}" for cell in cells)
        lines.append(f"  at {threshold:<7.3f}{counts}")
    return lines


def _chosen(summary: Mapping[str, Any]) -> list[str]:
    cache = summary["cache_view"]
    own, split = cache["own_question"], cache["split"]
    lines = [
        f"Cache view: {cache['stored_questions']} stored questions, "
        f"{cache['stored_unanswered']} of them ones the handbook does not answer. Each question "
        f"meets its {cache['candidates']} nearest above a cosine of {cache['floor']}",
    ]
    declines = cache.get("stored_declines")
    if declines is not None:
        lines.append(
            f"  a stored decline is no candidate: the {declines['left_out']} are left out, and "
            f"each question meets the nearest of the other {declines['candidates_from']}"
        )
    above = cache["stored_above_floor"]
    lines.append(
        f"  the floor left between {above['min']:.0f} and {above['max']:.0f} stored questions "
        f"to choose the nearest from, and {cache['questions_with_fewer_candidates']} questions "
        f"with fewer than {cache['candidates']} candidates"
    )
    for label, key in (("is the nearest", "nearest"), ("is a candidate", "among_candidates")):
        lines.append(
            f"  own golden question {label}: same {own[SAME][key]} of {own[SAME]['pairs']}, "
            f"different {own[DIFFERENT][key]} of {own[DIFFERENT]['pairs']}"
        )
    lines.append(
        f"  split        {split['groups']} groups of pairs that name the same golden question "
        f"(the largest has {split['largest_group']} pairs), never divided"
    )
    for half in (CHOOSING, REPORTING):
        of_half = split[half]
        groups, sources = of_half["groups_of"], of_half["different_by_source"]
        lines.append(
            f"               {half}: {of_half['pairs']} pairs. {of_half[SAME]} same from "
            f"{groups[SAME]['groups']} groups, {of_half[DIFFERENT]} different from "
            f"{groups[DIFFERENT]['groups']} groups ({groups[DIFFERENT]['largest']} from the "
            f"largest), {sources['golden']} of them two golden questions"
        )
        lines.append(
            f"               {half}: {of_half['candidates']['named_by_the_other_half']} of its "
            f"{of_half['candidates']['not_own']} candidates that are not the pair's own golden "
            "question are named by a pair of the other half"
        )
    lines += [
        "",
        "  Chosen on the choosing half: the loosest threshold from which upwards nothing "
        "wrong was served",
        f"  {'score':<14}{'threshold':>10}   {'same served right':<19}set by",
    ]
    for policy, figures in cache["policies"].items():
        chose, by = figures[CHOOSING][SAME], figures["decided_by"]
        set_by = "nothing wrong at any threshold" if by is None else _serve(by)
        if figures["halfway_to"] is not None:
            set_by += f"; halfway to the next score, {figures['halfway_to']:.4f}"
        right = f"{chose[RIGHT]} of {chose['pairs']}"
        lines.append(f"  {policy:<14}{_threshold(figures['threshold']):>10}   {right:<19}{set_by}")
    lines.append(
        f"  the rule takes: {cache['chosen_policy'] or 'none, so a cache would serve nothing'}"
    )
    return lines


def _reported(summary: Mapping[str, Any]) -> list[str]:
    cache = summary["cache_view"]
    lines = [
        "  On the reporting half, which chose nothing (share with its 95% interval)",
        "  Pairs share golden questions, so each interval is narrower than it should be.",
    ]
    for policy, figures in cache["policies"].items():
        same, different = figures[REPORTING][SAME], figures[REPORTING][DIFFERENT]
        served_nothing = figures["threshold"] is None
        lines.append(
            f"  {policy} at {_threshold(figures['threshold'])}"
            + (": it serves nothing by the rule, so no rate is estimated" if served_nothing else "")
        )
        for label, name in (
            ("same: served right", RIGHT),
            ("same: served another answer", WRONG),
            ("same: missed", MISSED),
        ):
            share = "" if served_nothing else _share(same[f"{name}_share"])
            lines.append(f"    {label:<30}{same[name]:>3} of {same['pairs']:<4}{share}".rstrip())
        share = "" if served_nothing else _share(different["wrong_share"])
        lines.append(
            f"    {'different: served wrongly':<30}{different[WRONG]:>3} of "
            f"{different['pairs']:<4}{share}".rstrip()
        )
        for wrong in figures["wrong_on_reporting"]:
            lines.append(f"      wrong: {_serve(wrong)}")
        for right in figures["right_only_by_the_margin"]:
            lines.append(
                f"      right only because the threshold is halfway to the next score: "
                f"{right['pair']} ({right['group']}) at {right['score']:.4f}"
            )
    allows = cache["zero_wrong_allows"]
    if allows["up_to"] is not None:
        lines.append(
            f"  What this half cannot show: none wrong of {allows['near_misses']} near-misses "
            f"would still allow a true rate of up to {allows['up_to']:.3f}."
        )
    return lines


def _tested(summary: Mapping[str, Any]) -> list[str]:
    """What each half holds to test a threshold with.

    Nothing wrong on the reporting half says little when that half holds nothing that
    comes near the threshold, so the hardest question of each half is printed.
    """
    cache = summary["cache_view"]
    above = cache["near_misses_above"]
    lines = [
        "  What each half holds to test a threshold with: the highest score at which a "
        "question of it would be served something wrong",
    ]
    for policy, figures in cache["policies"].items():
        for half in (CHOOSING, REPORTING):
            highest = figures["highest_wrong"][half]
            lines.append(f"  {policy}, {half}, a near-miss: {_serve(highest[DIFFERENT])}")
            lines.append(
                f"  {policy}, {half}, a rewording served another answer: {_serve(highest[SAME])}"
            )
    for policy, found in above["policies"].items():
        named = ", ".join(
            f"{pair['pair']} at {pair['score']:.4f} ({pair['half']})" for pair in found["pairs"]
        )
        line = f"  {policy}, near-misses it would serve at {above['score']}: {named or 'none'}"
        if found["pairs"]:
            line += (
                ". All of them are on the choosing half in "
                f"{found['other_splits_with_all_on_the_choosing_half']} of the "
                f"{cache['other_splits']['splits']} other splits"
            )
        lines.append(line)
    return lines


def _described(summary: Mapping[str, Any]) -> list[str]:
    whole, others = summary["cache_view"]["whole_set"], summary["cache_view"]["other_splits"]
    lines = [
        "  All agreed pairs at fixed thresholds",
        "  A description: these thresholds are scored on the pairs they are read from.",
        f"  {'score':<14}{'threshold':>10}   {'same: right':>11}{'wrong':>7}{'missed':>8}"
        f"   {'different: own':>14}{'other':>7}{'none':>6}",
    ]
    for policy, curve in whole["curves"].items():
        for row in [*curve, {**whole["chosen_on_all_pairs"][policy], "rule": True}]:
            same, different = row[SAME], row[DIFFERENT]
            mark = "  the rule, with every pair in view" if row.get("rule") else ""
            lines.append(
                f"  {policy:<14}{_threshold(row['threshold']):>10}   {same[RIGHT]:>11}"
                f"{same[WRONG]:>7}{same[MISSED]:>8}   {different['wrong_own']:>14}"
                f"{different['wrong_other']:>7}{different[NOT_SERVED]:>6}{mark}"
            )
    lines += [
        "",
        f"  The same rules on {others['splits']} other splits of the same pairs: how far the "
        "result depends on the split",
        f"  {'score':<18}{'taken':>6}{'no threshold':>14}   {'same right: min':>15}{'median':>8}"
        f"{'max':>7}   {'different wrong: min':>20}{'median':>8}{'max':>7}{'clean':>7}",
    ]
    every = {**others["policies"], "the rule's choice": others["chosen_policy"]}
    for name, figures in every.items():
        if "chosen" in figures:
            taken = f"{figures['chosen']:>6}{figures['splits_without_a_threshold']:>14}"
        else:
            taken = f"{others['splits'] - figures['splits_without_a_choice']:>6}"
        lines.append(f"  {name:<18}{taken:<20}   {_over_splits(figures)}")
    lines.append("  clean: the splits whose reporting half was served nothing wrong")
    lines.append(
        "  The same, apart by where one pair fell: the pair whose wrong serve scores highest "
        "with every pair in view"
    )
    for policy, figures in others["policies"].items():
        fell = figures["by_where_the_hardest_pair_fell"]
        if fell is None:
            continue
        for half in (CHOOSING, REPORTING):
            where = f"{policy}, {fell['pair']} on the {half} half"
            lines.append(
                f"  {where:<44}{fell[half]['splits']:>4} splits   {_over_splits(fell[half])}"
            )
    return lines


def _over_splits(figures: Mapping[str, Any]) -> str:
    """The shares over some splits, and in how many of them nothing wrong was served."""
    return (
        f"{_across(figures['right_share'], 15)}   "
        f"{_across(figures['wrong_share_of_different'], 20)}"
        f"{figures['splits_with_nothing_wrong']:>7}"
    )


def _across(spread: Mapping[str, float] | None, width: int) -> str:
    """The lowest, middle and highest value over the splits, or blanks where no split had one."""
    if spread is None:
        return " " * (width + 15)
    return f"{spread['min']:>{width}.3f}{spread['median']:>8.3f}{spread['max']:>7.3f}"


def _disputed(summary: Mapping[str, Any]) -> list[str]:
    lines = []
    for pair in summary["disputed"]:
        gave = ", ".join(
            f"{policy}: {answer or 'nothing'}" for policy, answer in pair["served"].items()
        )
        lines += [
            "",
            f"  Disputed {pair['pair']} (maker: {pair['kind']}, second: {pair['second_label']}), "
            f"in no figure: cosine {pair[COSINE]:.3f}, entailment {pair[ENTAILS_FORWARD]:.3f} "
            f"forward and {pair[ENTAILS_BACKWARD]:.3f} backward. Served: {gave}",
        ]
    return lines
