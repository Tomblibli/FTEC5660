#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

try:  # 它会展开 ${DASHSCOPE_API_KEY}
    load_dotenv()
except UnicodeDecodeError:  # e.g. .env saved as UTF-16 by a Windows editor
    load_dotenv(encoding="utf-16")


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.

    Design used here (parallel extraction + deterministic aggregation):
      1. ``extract``: one vision chain that reads a *single* receipt photo and
         returns a structured ``ReceiptData`` payload (item lines, discount
         lines, SUBTOTAL/小計, ROUNDING and the amount really paid). It prefers
         tool-calling structured output, then JSON mode, then a plain-text JSON
         parse, so it also works on an endpoint without function calling.
      2. ``recheck``: the same receipt photo plus the previous JSON and a
         description of the arithmetic that did not add up ("reflection").
         ``answer_queries`` only calls it for receipts that failed a check.
      3. ``answer_queries`` runs ``extract`` with LangChain's ``batch`` method,
         so every receipt is processed in parallel, and then aggregates the
         HKD totals in Python with ``Decimal``.
    """
    import os
    import sys

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek
    from pydantic import BaseModel, Field

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

    model = ChatDeepSeek(model=MODEL_NAME, temperature=0, max_retries=3, timeout=180)

    return {
        "extract": _structured_chain(extract_prompt),
        "recheck": _structured_chain(recheck_prompt),
        "model": model,
        "schema": ReceiptData,
        "prompt": extract_prompt,
    }

def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.

    Return a dict keyed by the two exact question strings above,
    each value a response containing one HKD amount, e.g. "HK$1974.30".
    """
    import os
    import sys
    from decimal import ROUND_HALF_UP

    CENT = Decimal("0.01")
    # The printed sums of these receipts agree to the cent (public_test: sum(items) - sum(discounts)
    # == subtotal and subtotal + rounding == paid, exactly), so any difference of one cent or more
    # is a misread: it marks the reading as unclean and makes _without() use its other estimate.
    TOLERANCE = Decimal("0.005")
    # How many receipt photos are sent to the vision model at the same time.
    DEFAULT_CONCURRENCY = 2

    def _concurrency() -> int:
        try:
            return max(1, int(os.environ.get("HW1_MAX_CONCURRENCY", str(DEFAULT_CONCURRENCY))))
        except (TypeError, ValueError):
            return DEFAULT_CONCURRENCY

    def _field(response: Any, name: str, default: Any = None) -> Any:
        """Read one field from a pydantic response or a plain dict."""
        if isinstance(response, dict):
            return response.get(name, default)
        return getattr(response, name, default)

    def _money(value: Any) -> Decimal:
        """Convert an extracted number to a 2-decimal HKD amount."""
        try:
            return Decimal(str(value).replace(",", "").strip()).quantize(
                CENT, rounding=ROUND_HALF_UP
            )
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0.00")

    def _line_total(lines: Any) -> tuple[Decimal, int]:
        """Sum the amounts of a line list, ignoring a minus sign if the model kept one."""
        total = Decimal("0.00")
        count = 0
        for line in lines or []:
            total += _money(_field(line, "amount", 0)).copy_abs()
            count += 1
        return total, count

    def _as_dict(response: Any) -> dict[str, Any]:
        """Best-effort plain-dict view of an extraction (used for the recheck prompt)."""
        dump = getattr(response, "model_dump", None)
        if callable(dump):
            return dump()
        if isinstance(response, dict):
            return response
        payload: dict[str, Any] = {}
        for name in (
            "store_name",
            "items",
            "discounts",
            "subtotal_after_discounts",
            "rounding",
            "amount_paid",
            "payment_method",
            "unclear_lines",
        ):
            value = getattr(response, name, None)
            if value is None:
                continue
            if isinstance(value, list):
                value = [
                    {"description": _field(item, "description", ""), "amount": _field(item, "amount", 0)}
                    for item in value
                ]
            payload[name] = value
        return payload

    def _without(summary: dict[str, Any]) -> Decimal:
        """`without the discounts` estimate of one reading: the sum of its positive item prices.

        Each receipt prints two checksums: ``sum(items) - sum(discounts) == subtotal`` and
        ``subtotal + rounding == paid``. When they hold, both definitions agree to the cent, so the
        item sum is used. When a check fails, the residual says which number to trust:

          * ``residual_lines > 0``: a discount line is missing or was read too small, or the printed
            subtotal was read too small -> the item sum is the better estimate. (Observed case:
            receipt2 came out 1.00 short because its discount total was 1.00 too small.)
          * ``residual_lines < 0`` while the subtotal is confirmed by the payment line: an item line
            was skipped -> ``subtotal + sum(discounts)`` is the better estimate.
          * ``subtotal + rounding != paid``: the printed subtotal itself is suspect -> item sum.
        """
        residual = summary["residual_lines"]
        if abs(residual) <= TOLERANCE:
            return summary["items_sum"]
        if abs(summary["residual_paid"]) > TOLERANCE:
            return summary["items_sum"]
        if residual < 0:
            return summary["subtotal"] + summary["discount_sum"]
        return summary["items_sum"]

    def _reading_score(summary: dict[str, Any]) -> Decimal:
        """How far a reading is from the two printed checksums of its receipt; smaller is better."""
        return abs(summary["residual_lines"]) + abs(summary["residual_paid"])

    def _summarise(response: Any, name: str, data_url: str) -> dict[str, Any]:
        """Turn one extraction into the numbers needed for the two totals."""
        items_sum, item_count = _line_total(_field(response, "items", []))
        discount_sum, discount_count = _line_total(_field(response, "discounts", []))
        subtotal = _money(_field(response, "subtotal_after_discounts", 0))
        rounding = _money(_field(response, "rounding", 0))
        paid = _money(_field(response, "amount_paid", 0))
        summary: dict[str, Any] = {
            "name": name,
            "data_url": data_url,
            "items_sum": items_sum,
            "item_count": item_count,
            "discount_sum": discount_sum,
            "discount_count": discount_count,
            "subtotal": subtotal,
            "rounding": rounding,
            "paid": paid,
            # Distance from the two checksums printed on this receipt.
            "residual_lines": items_sum - discount_sum - subtotal,
            "residual_paid": subtotal + rounding - paid,
            "response": response,
        }
        summary["items_ok"] = abs(summary["residual_lines"]) <= TOLERANCE
        summary["paid_ok"] = abs(summary["residual_paid"]) <= TOLERANCE
        summary["without"] = _without(summary)
        return summary

    bundle = chain if isinstance(chain, dict) else {"extract": chain}
    extract = bundle.get("extract")
    recheck = bundle.get("recheck")
    if extract is None:
        raise ValueError("chain must contain the 'extract' runnable returned by build_chain()")

    concurrency = _concurrency()
    batch_config = {"max_concurrency": concurrency}
    requests = [
        {"receipt_name": path.name, "image_data_url": image_data_url(path)} for path in images
    ]
    # Step 1 - read every receipt photo in parallel (one independent prompt per image).
    raw_results = (
        extract.batch(requests, config=batch_config, return_exceptions=True) if requests else []
    )

    summaries: list[dict[str, Any]] = []
    retry_jobs: list[dict[str, Any]] = []
    for path, request, result in zip(images, requests, raw_results):
        if isinstance(result, BaseException):
            retry_jobs.append(
                {
                    "target": None,
                    "name": path.name,
                    "data_url": request["image_data_url"],
                    "previous_json": "{}",
                    "note": (
                        f"the previous extraction call failed ({type(result).__name__}: {result}). "
                        "Read this receipt photo from scratch."
                    ),
                }
            )
            continue
        summary = _summarise(result, path.name, request["image_data_url"])
        summaries.append(summary)
        if summary["items_ok"] and summary["paid_ok"]:
            continue
        problems = []
        if not summary["items_ok"]:
            problems.append(
                f"sum(items) {summary['items_sum']} minus sum(discounts) {summary['discount_sum']} "
                f"does not equal the printed SUBTOTAL {summary['subtotal']}"
            )
        if not summary["paid_ok"]:
            problems.append(
                f"SUBTOTAL {summary['subtotal']} plus ROUNDING {summary['rounding']} does not equal "
                f"the paid amount {summary['paid']}"
            )
        retry_jobs.append(
            {
                "target": summary,
                "name": path.name,
                "data_url": request["image_data_url"],
                "previous_json": json.dumps(_as_dict(result), ensure_ascii=False, default=str),
                "note": "; ".join(problems),
            }
        )

    # Step 2 - reflection: re-read only the photos that failed a check, again in parallel.
    if retry_jobs and recheck is not None:
        retry_requests = [
            {
                "receipt_name": job["name"],
                "image_data_url": job["data_url"],
                "previous_json": job["previous_json"],
                "mismatch_note": job["note"],
            }
            for job in retry_jobs
        ]
        retried = recheck.batch(retry_requests, config=batch_config, return_exceptions=True)
        for job, result in zip(retry_jobs, retried):
            if isinstance(result, BaseException):
                continue
            fresh = _summarise(result, job["name"], job["data_url"])
            if job["target"] is None:
                summaries.append(fresh)
            elif _reading_score(fresh) < _reading_score(job["target"]):
                # Keep the re-read only when it reproduces the printed checksums better: the
                # reflection step must never turn a usable reading into a worse one.
                job["target"].update(fresh)

    # Step 3 - deterministic aggregation of the per-receipt numbers.
    total_paid = sum((summary["paid"] for summary in summaries), Decimal("0.00"))
    total_without = sum((summary["without"] for summary in summaries), Decimal("0.00"))

    print(
        f"[hw1] {len(summaries)}/{len(images)} receipts extracted "
        f"(max_concurrency={concurrency}, rechecked={len(retry_jobs)})",
        file=sys.stderr,
    )
    for summary in sorted(summaries, key=lambda item: item["name"]):
        status = "ok" if (summary["items_ok"] and summary["paid_ok"]) else "CHECK"
        detail = (
            ""
            if status == "ok"
            else (
                f" residual_lines={summary['residual_lines']}"
                f" residual_paid={summary['residual_paid']}"
            )
        )
        print(
            f"[hw1] {summary['name']}: paid={summary['paid']} items={summary['items_sum']} "
            f"({summary['item_count']} lines) discounts={summary['discount_sum']} "
            f"({summary['discount_count']} lines) subtotal={summary['subtotal']} "
            f"rounding={summary['rounding']} without={summary['without']} [{status}]{detail}",
            file=sys.stderr,
        )
    missing = sorted({path.name for path in images} - {summary["name"] for summary in summaries})
    if missing:
        print(f"[hw1] WARNING: no extraction for {', '.join(missing)}", file=sys.stderr)
    print(f"[hw1] totals: paid={total_paid}, without_discounts={total_without}", file=sys.stderr)

    # Each graded response must contain exactly one HKD amount.
    return {QUERY_1: f"HK${total_paid:.2f}", QUERY_2: f"HK${total_without:.2f}"}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:  #Decimal('1974.30')
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
