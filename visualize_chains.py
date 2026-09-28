#!/usr/bin/env python3
"""Draw the extract / recheck chains of hw1.py as Mermaid diagrams (and optionally PNG).

The chain construction code below is copied unchanged from hw1.py: MODEL_NAME, the ReceiptData /
ReceiptLine schema, LINE_RULES / RECHECK_RULES, both prompt templates, _structured_chain() and the
ChatDeepSeek instance. Nothing in this file calls the model - it only builds the runnables and
inspects their get_graph() - so it spends no tokens and needs no real API key.

Usage:
    python visualize_chains.py           # chain_graphs/chains.html + chains.md + 10 *.mmd
    python visualize_chains.py --png     # also renders every diagram to *.png via mermaid.ink

Why the diagrams are built this way: langchain draws a chain made with `with_fallbacks([...])` as a
single collapsed node, so the three fallback branches are drawn separately here, and the overall
call logic (which branch is tried first) is written out by hand as one more diagram.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ChatDeepSeek validates at construction time that DEEPSEEK_API_KEY is non-empty
# (langchain_deepseek/chat_models.py:300-307). No request is ever sent from this script, so a
# placeholder is enough and a machine that has no key can still draw the diagrams.
os.environ.setdefault("DEEPSEEK_API_KEY", "dummy-key-only-used-to-build-the-diagrams")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Copied from hw1.py: build_chain() schema, prompts and model (hw1.py:88-93, 95-131, 132-205, 235-247)
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-v4-flash-vision-exp"
API_KEY_ENV = "DEEPSEEK_API_KEY"


class ReceiptLine(BaseModel):
    """One printed line of the item block (a product line or a discount line)."""

    description: str = Field(
        description="Text printed on the line, e.g. '084213 韭菜豬肉雲吞20粒裝' or '5% OFF (CU)'."
    )
    amount: float = Field(
        description="Line amount in HKD as a positive number (use 12.4 for a '-$12.40' discount)."
    )


class ReceiptData(BaseModel):
    """Structured content of one supermarket receipt."""

    store_name: str = Field(description="Banner / store printed at the top, e.g. 'fusion'.")
    items: list[ReceiptLine] = Field(
        description="Every positive product line of the item block, in printed order."
    )
    discounts: list[ReceiptLine] = Field(
        description="Every discount line of the item block, stored as positive magnitudes."
    )
    subtotal_after_discounts: float = Field(
        description="Amount on the SUBTOTAL / 小計 line (already after all discounts)."
    )
    rounding: float = Field(
        description="Signed value on the ROUNDING line; 0.0 when the receipt has no rounding."
    )
    amount_paid: float = Field(
        description=(
            "Amount actually charged, i.e. the payment line right after ROUNDING "
            "(OCTOPUS / VISA / CASH / EPS ...), NOT the CHANGE / 找續 line."
        )
    )
    payment_method: str = Field(description="Label of that payment line, e.g. 'OCTOPUS'.")
    unclear_lines: str = Field(
        default="",
        description="Parts of the receipt that could not be read confidently; empty if all clear.",
    )


LINE_RULES = """\
You read exactly one photo of a Hong Kong PARKnSHOP / fusion (PNS) supermarket receipt and answer
with structured JSON. Use only what is printed: never invent a line and never guess a number.

Reading order:
HEADER - never an item and never a discount: card number (Card no. / 卡號碼), MoneyBack point
balance (易賞錢積分結餘), points earned in this purchase (今次賺取積分), phone / address / store
header, column separators.
ITEM BLOCK - from the first product line down to the SUBTOTAL / 小計 line:
  * product line = "<code> DESCRIPTION" together with an amount at the right margin. Put that
    right-margin amount into `items` exactly once. A "數量: 2" / "QTY: 2" line and a separate
    "SP: $43.00 QTY: 2" unit-price line describe the product above them: never add them as extra
    money and never multiply anything yourself, the receipt already prints the extended amount.
  * discount line = an amount printed with a minus sign, or a discount label. Put the absolute
    value into `discounts`. Labels used by this shop: 包裝變形, Buy 2 Save $x, Buy 3 Save $x,
    5% OFF (CU), 5% OFF (CU-SCO), MB PRICE $x, MB $200get 5%off, MB APP UPGRADE,
    App Upgrade$350->$20_C, App upgrade -$30_B, OVER$40 ENJOY 10%, COUPON.
  * real charges still count as items: PLASTIC BAG CHARGIN / 膠袋收費 / 膠袋稅 go to `items`, and a
    "$0.00" VCODE / COUPON line goes to `items` with amount 0.0.
TOTALS:
  * `subtotal_after_discounts` = the amount on the SUBTOTAL / 小計 line.
  * `rounding` = the signed amount on the ROUNDING line (0.0 if the receipt has none).
  * `amount_paid` = the amount charged on the payment line printed right after ROUNDING
    (OCTOPUS / 八達通 / VISA / MASTER / CASH / EPS ...). CHANGE / 找續 / 找零 is change, not a
    payment, and is never an item.
FOOTER - never an item and never a discount, even when printed with a minus sign: 機號 / Device
No., 八達通卡號碼 / Octopus Card No., 扣除金額 / Amount Deducted, 餘額 / Remaining Value, GP.VISA,
APP Label, Ref / Auth / RRN / TID / MID / trace / PAN / exp numbers, 收銀員 / CASHIER / STORE /
TICKET# / 單號, barcode and the refund-policy text. Known trap: lines like "餘額 -$15.90" or
"扣除金額 $394.70" must NOT be counted as discounts.

Arithmetic self-check that must hold before you answer (tolerance 0.05):
  sum(items) - sum(discounts) == subtotal_after_discounts
  subtotal_after_discounts + rounding == amount_paid
If a check fails, re-read the photo, look for a missed line or a misread digit, and fix it.
Report every amount in HKD with two decimals and describe unreadable parts in `unclear_lines`.
"""

RECHECK_RULES = (
    "You are auditing an earlier extraction of ONE receipt photo. Read that image again line "
    "by line and return the corrected JSON for the same receipt.\n\n" + LINE_RULES
)

extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", LINE_RULES),
        (
            "human",
            [
                {"type": "text", "text": "Extract this receipt as JSON: {receipt_name}"},
                {"type": "image_url", "image_url": {"url": "{image_data_url}"}},
            ],
        ),
    ]
)
recheck_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RECHECK_RULES),
        (
            "human",
            [
                {
                    "type": "text",
                    "text": (
                        "Receipt file: {receipt_name}\n"
                        "Earlier extraction as JSON:\n{previous_json}\n"
                        "What does not add up: {mismatch_note}\n"
                        "Re-read the photo and return the corrected JSON."
                    ),
                },
                {"type": "image_url", "image_url": {"url": "{image_data_url}"}},
            ],
        ),
    ]
)

model = ChatDeepSeek(model=MODEL_NAME, temperature=0, max_retries=3, timeout=180)


def _structured_chain(prompt: ChatPromptTemplate) -> Any:
    """Tool-calling structured output first, then JSON mode, then text JSON."""
    return (
        prompt | model.with_structured_output(ReceiptData, method="function_calling")
    ).with_fallbacks(
        [
            prompt | model.with_structured_output(ReceiptData, method="json_mode"),
            prompt | model | JsonOutputParser(pydantic_object=ReceiptData),
        ]
    )



# ---------------------------------------------------------------------------
# Diagram building
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "chain_graphs"
FALLBACK_LABELS = (
    "with_structured_output(ReceiptData,<br/>method='function_calling')",
    "with_structured_output(ReceiptData,<br/>method='json_mode')",
    "plain text answer<br/>+ JsonOutputParser(pydantic_object=ReceiptData)",
)


def build_chains() -> dict[str, dict[str, Any]]:
    """Rebuild both chains of build_chain() plus the three fallback branches of each."""
    chains: dict[str, dict[str, Any]] = {}
    for name, prompt, rules in (
        ("extract", extract_prompt, "LINE_RULES"),
        ("recheck", recheck_prompt, "RECHECK_RULES"),
    ):
        chains[name] = {
            "chain": _structured_chain(prompt),
            "prompt": prompt,
            "rules": rules,
            "branches": {
                "function_calling": prompt
                | model.with_structured_output(ReceiptData, method="function_calling"),
                "json_mode": prompt
                | model.with_structured_output(ReceiptData, method="json_mode"),
                "json_parser": prompt | model | JsonOutputParser(pydantic_object=ReceiptData),
            },
        }
    return chains


def strip_front_matter(source: str) -> str:
    """Drop langchain's `--- config: ... ---` preamble; the renderers only need the bare graph."""
    lines = source.splitlines()
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[index + 1 :]).strip()
    return source.strip()


def steps_of(runnable: Any) -> list[str]:
    """Class names of a RunnableSequence, or just the class for anything else."""
    steps = [type(step).__name__ for step in getattr(runnable, "steps", [])]
    return steps or [type(runnable).__name__]


def logical_diagram(prompt: ChatPromptTemplate, rules: str) -> str:
    """Hand-written graph, because get_graph() collapses with_fallbacks() into one node.

    Laid out top-down so the PNG stays narrow: a left-to-right version is 1784x203 px and gets
    unreadably small when a README scales it down to the content width.
    """
    variables = ", ".join(prompt.input_variables)
    lines = [
        "graph TD",
        "\tclassDef prompt fill:#e3f2fd,stroke:#1565c0,stroke-width:1px",
        "\tclassDef model fill:#fff3e0,stroke:#ef6c00,stroke-width:1px",
        "\tclassDef result fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px",
        f'\tPROMPT["ChatPromptTemplate<br/>system = {rules}<br/>variables = {variables}"]:::prompt',
        f'\tCALL1["ChatDeepSeek<br/>{FALLBACK_LABELS[0]}"]:::model',
        f'\tCALL2["ChatDeepSeek<br/>{FALLBACK_LABELS[1]}"]:::model',
        f'\tCALL3["ChatDeepSeek<br/>{FALLBACK_LABELS[2]}"]:::model',
        '\tRESULT(["ReceiptData"]):::result',
        "\tPROMPT --> CALL1",
        "\tCALL1 -->|valid JSON| RESULT",
        "\tCALL1 -.->|fallback 1| CALL2",
        "\tCALL2 -.->|fallback 2| CALL3",
        "\tCALL3 --> RESULT",
    ]
    return "\n".join(lines)


def collect_diagrams(chains: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every diagram, most informative first: call logic, top-level graph, then each branch."""
    diagrams: list[dict[str, Any]] = []
    for name, info in chains.items():
        variables = ", ".join(info["prompt"].input_variables)
        diagrams.append(
            {
                "name": f"{name}_logical",
                "title": f"{name} chain: call logic and fallback order (hand written)",
                "source": logical_diagram(info["prompt"], info["rules"]),
                "detail": f"prompt variables: {variables}",
            }
        )
        diagrams.append(
            {
                "name": f"{name}_overview",
                "title": f"{name} chain: langchain get_graph() of the whole chain",
                "source": info["chain"].get_graph().draw_mermaid(),
                "detail": "RunnableWithFallbacks collapses into one node - see the logical diagram",
            }
        )
        for method, branch in info["branches"].items():
            diagrams.append(
                {
                    "name": f"{name}_branch_{method}",
                    "title": f"{name} fallback branch: {method}",
                    "source": branch.get_graph().draw_mermaid(),
                    "detail": " -> ".join(steps_of(branch)),
                }
            )
    return diagrams


def prompt_text(prompt: ChatPromptTemplate) -> str:
    """Flatten a ChatPromptTemplate back to readable text for the HTML appendix."""
    chunks: list[str] = []
    try:
        for message in prompt.messages:
            inner = getattr(message, "prompt", None)
            if isinstance(inner, str):
                chunks.append(inner)
            else:
                for block in getattr(inner, "messages", []):
                    if isinstance(block, dict):
                        text = block.get("text")
                    else:
                        content = getattr(block, "content", None)
                        text = content if isinstance(content, str) else None
                    if text:
                        chunks.append(text)
    except Exception as exc:  # never let a rendering nicety break the run
        chunks.append(f"<prompt text unavailable: {type(exc).__name__}: {exc}>")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _escape(text: str) -> str:
    """Escape text for HTML: `&` first, then `<` (mermaid reads the decoded text back)."""
    return text.replace("&", "&amp;").replace("<", "&lt;")


def write_mmd(diagrams: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """One raw Mermaid file per diagram, ready for https://mermaid.live."""
    written: list[Path] = []
    for diagram in diagrams:
        target = out_dir / f"{diagram['name']}.mmd"
        target.write_text(diagram["source"].rstrip() + "\n", encoding="utf-8")
        written.append(target)
    return written


def write_markdown(diagrams: list[dict[str, Any]], out_dir: Path) -> Path:
    """All diagrams as ```mermaid blocks, readable in VS Code / GitHub."""
    parts = [
        "# hw1.py chains: extract / recheck",
        "",
        f"Generated by `visualize_chains.py` (model: `{MODEL_NAME}`). Every graph below comes from",
        "`get_graph()` on the real chain copied out of `hw1.py`; no model call is made anywhere.",
        "The `logical` diagram is hand written because `with_fallbacks()` is drawn as a single node.",
        "",
    ]
    for index, diagram in enumerate(diagrams, start=1):
        parts += [
            f"## {index}. {diagram['title']}",
            "",
            "```mermaid",
            strip_front_matter(diagram["source"]),
            "```",
            "",
            f"`{diagram['detail']}`",
            "",
        ]
    target = out_dir / "chains.md"
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>hw1.py chains: extract / recheck</title>
<!-- Rendering uses the Mermaid CDN, so this page needs internet the first time it is opened. -->
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "default", flowchart: { curve: "linear" } });
</script>
<style>
  :root { color-scheme: light; }
  body { margin: 0; padding: 32px; background: #f6f7fb; color: #1f2430;
         font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }
  h1 { margin: 0 0 4px; font-size: 24px; }
  h2 { margin: 28px 0 12px; font-size: 19px; color: #2b3446; }
  .lead { margin: 0 0 28px; color: #5b6474; max-width: 78ch; }
  .card { background: #fff; border: 1px solid #dfe3ec; border-radius: 10px;
          padding: 18px 22px 10px; margin: 0 0 22px; box-shadow: 0 1px 2px rgba(20, 28, 45, .06); }
  .card h2 { margin: 0 0 12px; font-size: 17px; color: #2b3446; }
  .mermaid { display: flex; justify-content: center; background: #fff; }
  .meta { margin: 10px 0 0; padding-top: 8px; border-top: 1px dashed #e6e9f0;
          color: #6b7385; font: 12.5px/1.45 Consolas, "Courier New", monospace; }
  details { background: #fff; border: 1px solid #dfe3ec; border-radius: 10px;
            padding: 12px 18px; margin: 0 0 14px; }
  summary { cursor: pointer; font-weight: 600; color: #2b3446; }
  pre.prompt { white-space: pre-wrap; color: #333c4d; font: 12.5px/1.5 Consolas, monospace; }
  footer { color: #6b7385; font-size: 12.5px; margin-top: 26px; }
</style>
</head>
<body>
<h1>hw1.py chains: extract / recheck</h1>
<p class="lead">
  __COUNT__ diagrams generated by <code>visualize_chains.py</code> from the chain that
  <code>hw1.py</code> builds with <code>__MODEL__</code>. Nothing here calls the model: the graphs
  come from <code>get_graph()</code> and the prompt templates at the bottom are the real ones.
</p>
__CARDS__
<h2>Prompt templates</h2>
__APPENDIX__
<footer>langchain-core's <code>RunnableWithFallbacks</code> has no graph of its own, so its three
branches are drawn one by one in addition to the hand written call-logic diagram.</footer>
</body>
</html>
"""

# --- more writers ---


def write_html(chains: dict[str, dict[str, Any]], diagrams: list[dict[str, Any]], out_dir: Path) -> Path:
    """One self-contained page: every diagram plus the real prompt templates."""
    cards = [
        "\n".join(
            [
                '<section class="card">',
                f"<h2>{index}. {_escape(diagram['title'])}</h2>",
                '<pre class="mermaid">',
                _escape(strip_front_matter(diagram["source"])),
                "</pre>",
                f'<p class="meta">{_escape(diagram["detail"])}</p>',
                "</section>",
            ]
        )
        for index, diagram in enumerate(diagrams, start=1)
    ]
    appendix = [
        "\n".join(
            [
                f"<details><summary>{name} prompt (system + human template)</summary>",
                f'<pre class="prompt">{_escape(prompt_text(info["prompt"]))}</pre>',
                "</details>",
            ]
        )
        for name, info in chains.items()
    ]
    html = (
        HTML_TEMPLATE.replace("__CARDS__", "\n".join(cards))
        .replace("__APPENDIX__", "\n".join(appendix))
        .replace("__MODEL__", MODEL_NAME)
        .replace("__COUNT__", str(len(diagrams)))
    )
    target = out_dir / "chains.html"
    target.write_text(html, encoding="utf-8")
    return target


def write_ascii(chains: dict[str, dict[str, Any]], out_dir: Path) -> None:
    """Bonus text drawing, only when grandalf happens to be installed (never auto-installed)."""
    try:
        import grandalf  # noqa: F401
    except ImportError:
        print("[viz] ASCII skipped (pip install grandalf to enable draw_ascii())", file=sys.stderr)
        return
    blocks: list[str] = []
    for name, info in chains.items():
        blocks.append(
            f"=== {name} chain (with_fallbacks is one collapsed node) ===\n"
            f"{info['chain'].get_graph().draw_ascii()}"
        )
        for method, branch in info["branches"].items():
            blocks.append(f"\n=== {name} branch: {method} ===\n{branch.get_graph().draw_ascii()}")
    (out_dir / "chains.ascii.txt").write_text("\n".join(blocks), encoding="utf-8")


def write_png(diagrams: list[dict[str, Any]], out_dir: Path) -> None:
    """Render every diagram to PNG through the public mermaid.ink service (needs internet)."""
    try:
        from langchain_core.runnables.graph_mermaid import MermaidDrawMethod, draw_mermaid_png
    except ImportError as exc:  # keeps the run alive on an older langchain-core
        print(f"[viz] PNG skipped: {exc}", file=sys.stderr)
        return
    for diagram in diagrams:
        try:
            png = draw_mermaid_png(
                mermaid_syntax=strip_front_matter(diagram["source"]),
                draw_method=MermaidDrawMethod.API,
                background_color="white",
                max_retries=2,
                retry_delay=2.0,
            )
        except Exception as exc:  # no network / service hiccup must not kill the whole run
            print(
                f"[viz] PNG failed for {diagram['name']}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        target = out_dir / f"{diagram['name']}.png"
        target.write_bytes(png)
        print(f"[viz] wrote {target.name} ({len(png)} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the extract / recheck chains of hw1.py.")
    parser.add_argument(
        "--png",
        action="store_true",
        help="also render every diagram to PNG via mermaid.ink (first run needs internet)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="output folder (default: chain_graphs next to this file)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chains = build_chains()
    diagrams = collect_diagrams(chains)

    write_mmd(diagrams, out_dir)
    write_markdown(diagrams, out_dir)
    write_html(chains, diagrams, out_dir)
    write_ascii(chains, out_dir)
    if args.png:
        write_png(diagrams, out_dir)

    print(f"[viz] {len(diagrams)} diagrams of {MODEL_NAME} -> {out_dir}")
    for index, diagram in enumerate(diagrams, start=1):
        print(f"  {index:2d}. {diagram['name']}.mmd | {diagram['detail']}")
    print(f"[viz] view: {out_dir / 'chains.html'} (browser) or {out_dir / 'chains.md'} (VS Code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
