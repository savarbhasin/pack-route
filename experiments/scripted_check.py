"""Quality gates without a real model: drive the taskset through Verifiers' Env API (the calls
vf-eval makes) with a scripted OpenAI-compatible model.

Policies:
  oracle    get_orders -> check_plan(optimum) -> submit_plan(optimum)   expect 1.0
  ffd       same, with the first-fit-decreasing baseline plan          expect the experiment's grade
  invalid   submit everything in one van                                expect < 0.2
  nop       answer in text, never submit                                expect 0.0

usage: uv run python experiments/scripted_check.py [--policies oracle ffd invalid nop]
"""

import argparse
import asyncio
import json
import os
import socket
from pathlib import Path

from aiohttp import web

from pack_route.plan import PublicInstance, VanPlan, evaluate, grade
from pack_route.taskset import DEFAULT_TASKS
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.loaders import harness_config_type, load_environment

ROWS = [json.loads(line) for line in DEFAULT_TASKS.read_text().splitlines()]


def match_row(orders: dict) -> dict:
    """Identify the task from get_orders' answer, as an agent with the answer key would."""
    want = sorted((p["id"], p["volume_l"]) for s in orders["stops"] for p in s["parcels"])
    for row in ROWS:
        inst = PublicInstance.model_validate(row["instance"])
        if sorted((p.id, p.volume_dl / 10) for p in inst.parcels) == want:
            return row
    raise LookupError("no task matches these orders")


def policy_plan(policy: str, row: dict) -> list[dict]:
    if policy == "invalid":
        inst = PublicInstance.model_validate(row["instance"])
        return [{"route": inst.stops, "parcels": []}]
    return row["reference"]["plans"]["optimum" if policy == "oracle" else policy]


def tool_result_json(content) -> dict:
    text = content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content if isinstance(part, dict))
    return json.loads(text[text.index("{"):])


def scripted_reply(policy: str, body: dict) -> dict:
    tools = {t["function"]["name"] for t in body.get("tools", [])}
    name = lambda short: next(t for t in tools if t.endswith(short))  # noqa: E731
    msgs = body["messages"]
    calls = [c for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    message: dict = {"role": "assistant", "content": None}
    if policy == "nop" or len(calls) >= 3:
        message["content"] = "Done."
    elif not calls:
        message["tool_calls"] = [_call(0, name("get_orders"), {})]
    else:
        orders_msg = next(m for m in msgs if m["role"] == "tool"
                          and m.get("tool_call_id") == calls[0]["id"])
        plan = policy_plan(policy, match_row(tool_result_json(orders_msg["content"])))
        step = "check_plan" if len(calls) == 1 else "submit_plan"
        message["tool_calls"] = [_call(len(calls), name(step), {"vans": plan})]
    return {"id": "chatcmpl-scripted", "object": "chat.completion", "created": 0,
            "model": body.get("model", "scripted"),
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


async def serve_model(policy: str) -> tuple[web.AppRunner, str]:
    dialect = ChatDialect()

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if not request.path.endswith("/chat/completions"):
            raise web.HTTPNotImplemented(text=f"scripted model only speaks chat completions, got {request.path}")
        raw = scripted_reply(policy, body)
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, f"http://127.0.0.1:{port}/v1"


async def run_policy(policy: str) -> list[tuple[str, float, dict]]:
    runner, url = await serve_model(policy)
    os.environ["SCRIPTED_API_KEY"] = "scripted"
    try:
        config = EvalConfig(
            env={"taskset": {"id": "pack_route"},
                 "agent": {"harness": harness_config_type("null")(id="null"),
                           "runtime": {"type": "subprocess"}, "max_turns": 10}},
            select={"limit": len(ROWS)}, num_rollouts=1, model="scripted",
            sampling=SamplingConfig(max_tokens=1024),
            client=EvalClientConfig(base_url=url, api_key_var="SCRIPTED_API_KEY"),
        )
        env = load_environment(config.env)
        tasks = list(env.taskset.select(config.select))
        ctx = ModelContext(client=config.client, model=config.model, sampling=config.sampling)
        async with env.serving():
            episodes = await asyncio.gather(*(env.run_slot(slot, ctx) for task in tasks
                                              for slot in env.slots(task, 1)))
    finally:
        await runner.cleanup()
    out = []
    for ep in episodes:
        for tr in ep.traces:
            for err in tr.errors:
                print(f"  ERROR {tr.task.data.id}: {err}")
            out.append((tr.task.data.id, tr.reward, dict(tr.metrics)))
    return out


def expected(policy: str, row: dict) -> float:
    if policy == "nop":
        return 0.0
    inst = PublicInstance.model_validate(row["instance"])
    plan = [VanPlan.model_validate(v) for v in policy_plan(policy, row)]
    ref = row["reference"]
    return grade(evaluate(inst, plan), ref["optimum_s"], ref["floor_s"])[0]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["oracle", "ffd", "invalid", "nop"])
    args = ap.parse_args()
    by_id = {r["id"]: r for r in ROWS}
    ok = True
    for policy in args.policies:
        results = await run_policy(policy)
        print(f"== {policy}: {len(results)} episodes")
        for tid, reward, metrics in sorted(results):
            want = expected(policy, by_id[tid])
            good = reward is not None and abs(reward - want) < 1e-9
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} {tid}: reward {reward} (expected {want:.4f}) "
                  f"checks_used={metrics.get('checks_used')} valid={metrics.get('valid')}")
        ok &= len(results) == len(ROWS)
    print("ALL GATES PASS" if ok else "SOME GATES FAILED")


if __name__ == "__main__":
    asyncio.run(main())
