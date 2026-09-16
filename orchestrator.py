"""
Language orchestrator for the dinner-table task.

Turns a plain-English instruction plus the current scene state into a validated plan of
sub-skills, and re-plans when the scene changes underneath it (an object moved, a grasp
slipped, an arm is already holding something).

Two planners, same interface:
  * LLMPlanner     - asks a hosted model, then validates its JSON against the real vocabulary.
  * ScriptedPlanner - deterministic, no network, used as the fallback so a dead key or a rate
                      limit degrades the demo instead of ending it.

The key is read from the environment only. This module never opens a secrets file.

    python orchestrator.py            # runs the offline self-test, no network, no API key
"""
import json
import os
import re

ARMS = ("left", "right")
SKILLS = ("pick", "place", "handoff")
OBJECTS = ("plate", "cup", "spoon")

SYSTEM = """You plan for two 6-DOF robot arms, "left" and "right", facing each other across a table.
Available skills, and nothing else:
  {"skill": "pick",    "arm": <arm>, "object": <object>}
  {"skill": "place",   "arm": <arm>, "object": <object>, "target": <"table"|"plate">}
  {"skill": "handoff", "arm": <arm>, "object": <object>, "to_arm": <arm>}
Objects: plate, cup, spoon.
An arm can hold only one object at a time. An arm can only pick an object on its own side of
the table; to move an object across, hand it off. Reply with a JSON array of steps and no prose."""


class PlanError(ValueError):
    pass


def validate(plan, state=None):
    """Reject anything the executor could not actually run. Raises PlanError with a reason."""
    if not isinstance(plan, list) or not plan:
        raise PlanError("plan must be a non-empty list")
    holding = dict(state.get("holding", {})) if state else {}
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            raise PlanError("step %d is not an object" % i)
        skill, arm, obj = step.get("skill"), step.get("arm"), step.get("object")
        if skill not in SKILLS:
            raise PlanError("step %d: unknown skill %r" % (i, skill))
        if arm not in ARMS:
            raise PlanError("step %d: unknown arm %r" % (i, arm))
        if obj not in OBJECTS:
            raise PlanError("step %d: unknown object %r" % (i, obj))
        if skill == "pick":
            if holding.get(arm):
                raise PlanError("step %d: %s arm already holds %s" % (i, arm, holding[arm]))
            holding[arm] = obj
        elif skill == "place":
            if holding.get(arm) != obj:
                raise PlanError("step %d: %s arm is not holding %s" % (i, arm, obj))
            if step.get("target") not in ("table", "plate"):
                raise PlanError("step %d: bad place target %r" % (i, step.get("target")))
            holding[arm] = None
        elif skill == "handoff":
            to_arm = step.get("to_arm")
            if to_arm not in ARMS or to_arm == arm:
                raise PlanError("step %d: bad handoff target %r" % (i, to_arm))
            if holding.get(arm) != obj:
                raise PlanError("step %d: %s arm is not holding %s" % (i, arm, obj))
            if holding.get(to_arm):
                raise PlanError("step %d: %s arm is not free" % (i, to_arm))
            holding[arm] = None
            holding[to_arm] = obj
    return plan


class ScriptedPlanner:
    """Keyword planner. Covers the declared task and degrades honestly on anything else."""

    name = "scripted"

    def plan(self, instruction, state):
        text = instruction.lower()
        side = state.get("side", {}) if state else {}
        steps = []
        for obj in OBJECTS:
            if obj not in text:
                continue
            owner = side.get(obj, "right")
            target = "plate" if obj == "cup" else "table"
            receiver = "left" if ("hand" in text and obj == "cup") else owner
            steps.append({"skill": "pick", "arm": owner, "object": obj})
            if receiver != owner:
                steps.append({"skill": "handoff", "arm": owner, "object": obj, "to_arm": receiver})
            steps.append({"skill": "place", "arm": receiver, "object": obj, "target": target})
        if not steps:
            raise PlanError("no known object named in: %r" % instruction)
        return validate(steps, state)


class LLMPlanner:
    """Hosted planner. Falls back to the scripted one rather than failing the run."""

    name = "llm"

    def __init__(self, model="gpt-5.6", fallback=None):
        self.model = model
        self.fallback = fallback or ScriptedPlanner()

    def plan(self, instruction, state):
        try:
            raw = self._ask(instruction, state)
            return validate(self._extract(raw), state)
        except Exception as exc:
            print("[orchestrator] LLM planner unusable (%s: %s); using %s"
                  % (type(exc).__name__, str(exc)[:90], self.fallback.name))
            return self.fallback.plan(instruction, state)

    def _ask(self, instruction, state):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment")
        from openai import OpenAI
        client = OpenAI(api_key=key)
        reply = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": "Scene: %s\nInstruction: %s"
                       % (json.dumps(state), instruction)}],
        )
        return reply.choices[0].message.content

    @staticmethod
    def _extract(raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*\]", raw or "", re.S)     # strip prose or code fences
        if not match:
            raise PlanError("no JSON array in the reply")
        return json.loads(match.group(0))


def replan_needed(planned_state, observed_state):
    """True when the world no longer matches what the plan assumed."""
    if planned_state.get("holding") != observed_state.get("holding"):
        return True
    for obj, xy in observed_state.get("pos", {}).items():
        was = planned_state.get("pos", {}).get(obj)
        if was and (abs(was[0] - xy[0]) > 0.03 or abs(was[1] - xy[1]) > 0.03):
            return True
    return False


def _selftest():
    state = {"holding": {"left": None, "right": None},
             "side": {"plate": "left", "cup": "right", "spoon": "right"},
             "pos": {"plate": (0.0, 0.1), "cup": (0.2, -0.1)}}
    declared = ("Pick up the plate and place it on the table with the left arm. "
                "The right arm picks up the cup and hands it to the left arm, "
                "which places it on the plate.")

    plan = ScriptedPlanner().plan(declared, state)
    print("scripted plan:")
    for step in plan:
        print("   ", json.dumps(step))

    for bad, why in [
        ([{"skill": "fly", "arm": "left", "object": "cup"}], "unknown skill"),
        ([{"skill": "place", "arm": "left", "object": "cup", "target": "table"}], "not holding"),
        ([{"skill": "pick", "arm": "left", "object": "cup"},
          {"skill": "pick", "arm": "left", "object": "plate"}], "already holding"),
        ([{"skill": "pick", "arm": "right", "object": "cup"},
          {"skill": "handoff", "arm": "right", "object": "cup", "to_arm": "right"}], "same arm"),
    ]:
        try:
            validate(bad, state)
            print("REJECT FAILED, accepted a bad plan:", why)
        except PlanError as exc:
            print("rejected (%s): %s" % (why, exc))

    moved = dict(state, pos={"plate": (0.0, 0.1), "cup": (0.2, 0.4)})
    print("replan on moved cup:", replan_needed(state, moved))
    print("replan on unchanged:", replan_needed(state, state))
    print("LLM planner with no key falls back:",
          len(LLMPlanner().plan(declared, state)), "steps")


if __name__ == "__main__":
    _selftest()
