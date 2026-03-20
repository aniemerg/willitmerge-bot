"""
Rich terminal display for bot V2.

All visual output lives here so loop.py stays focused on logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from rich import box
from rich.text import Text

if TYPE_CHECKING:
    from bot.scorer import ScoredMarket

console = Console()

# ── colour palette ────────────────────────────────────────────────────────────
_GREEN  = "bold green"
_YELLOW = "yellow"
_RED    = "bold red"
_CYAN   = "cyan"
_DIM    = "dim"
_BOLD   = "bold"


# ── helpers ───────────────────────────────────────────────────────────────────

def _market_label(market: dict) -> str:
    owner = market.get("repo_owner", "?")
    name  = market.get("repo_name",  "?")
    num   = market.get("pr_number",  "?")
    return f"{owner}/{name}#{num}"


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    s = f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"
    return s


def _cents(value: float | None) -> str:
    """Format a probability as cents (e.g. 0.314 → '31¢')."""
    if value is None:
        return "—"
    return f"{value * 100:.0f}¢"


def _market_state(market: dict) -> str:
    activation_state = market.get("activation_state")
    if activation_state == "unactivated":
        return "unactivated"
    if activation_state == "activating":
        return "activating"
    if market.get("price_status") != "live":
        return str(market.get("price_status") or "unavailable")
    return "live"


# ── startup banner ────────────────────────────────────────────────────────────

def startup_banner(strategy_name: str, loop_minutes: int, dry_run: bool,
                   max_workers: int, max_trades: int, max_spend: float) -> None:
    console.print(Rule("[bold cyan]Will It Merge? Trading Bot[/bold cyan]"))
    mode = "[bold yellow]DRY RUN[/bold yellow]" if dry_run else "[bold green]LIVE[/bold green]"
    console.print(
        f"  strategy=[cyan]{strategy_name}[/cyan]  "
        f"loop=[cyan]{loop_minutes}m[/cyan]  "
        f"workers=[cyan]{max_workers}[/cyan]  "
        f"max_trades=[cyan]{max_trades}/cycle[/cyan]  "
        f"max_spend=[cyan]${max_spend:.0f}/cycle[/cyan]  "
        f"mode={mode}"
    )
    console.print()


# ── cycle start ───────────────────────────────────────────────────────────────

def cycle_start(cycle: int) -> None:
    console.print(Rule(f"[bold]Cycle {cycle}[/bold]"))


# ── scoring progress bar ──────────────────────────────────────────────────────

def make_scoring_progress() -> Progress:
    """Return a configured Progress object to use as a context manager."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Scoring markets[/cyan]"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[dim]{task.fields[errors]} errors[/dim]"),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# ── opportunity table ─────────────────────────────────────────────────────────

def _fmt_position(pos: dict | None) -> str:
    """Format a position entry as e.g. '1.50 NO' or '—'."""
    if not pos:
        return "—"
    yes = float(pos.get("yes_shares", 0))
    no  = float(pos.get("no_shares",  0))
    if yes > 0.0001:
        return f"{yes:.2f} YES"
    if no > 0.0001:
        return f"{no:.2f} NO"
    return "—"


def opportunity_table(
    results: list["ScoredMarket"],
    positions: dict[str, dict],
    edge_threshold: float,
    max_trades: int,
    max_spend: float,
    dry_run: bool,
    top_n: int = 15,
) -> None:
    """
    Print the ranked opportunity table.

    Shows the top_n markets by edge, then a summary line for the rest.
    The Decision column explains why each trade was or wasn't placed.
    """
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("Market",         style="", max_width=38)
    table.add_column("Model",          justify="right", style=_CYAN)
    table.add_column("Price",          justify="right", style=_DIM)
    table.add_column("Edge",           justify="left")
    table.add_column("Position",       justify="right", style=_DIM)
    table.add_column("Decision",       style="")

    # Separate scoreable from errors/skips
    scoreable = [r for r in results if r.edge is not None]
    above_threshold = [r for r in scoreable if r.abs_edge is not None and r.abs_edge >= edge_threshold]
    below_threshold = [r for r in scoreable if r.abs_edge is None or r.abs_edge < edge_threshold]
    errors_skips    = [r for r in results if r.edge is None]

    # Walk through the same trade-decision logic as loop.py for display purposes
    trades_shown = 0
    spend_shown  = 0.0
    rows_shown   = 0

    for result in scoreable[:top_n]:
        label = _market_label(result.market)
        model_str = _cents(result.target_prob)
        price_str = _cents(result.reference_price)

        edge_val = result.edge or 0.0
        abs_edge = abs(edge_val)
        side_hint = "→YES" if edge_val > 0 else "→NO "
        edge_str = Text(f"{abs_edge * 100:.1f}%  {side_hint}")
        if abs_edge >= edge_threshold:
            edge_str.stylize(_GREEN)
        else:
            edge_str.stylize(_DIM)

        # Determine decision
        if result.abs_edge is None or result.abs_edge < edge_threshold:
            decision = Text("below threshold", style=_DIM)
        elif result.market.get("activation_state") == "activating":
            decision = Text("market activating", style=_YELLOW)
        elif result.market.get("activation_state") == "unactivated" and not result.plans:
            decision = Text("unactivated: no opening edge", style=_DIM)
        elif _market_state(result.market) != "live" and not result.plans:
            decision = Text(f"state: {_market_state(result.market)}", style=_DIM)
        elif not result.plans:
            decision = Text("at target", style=_DIM)
        else:
            parts = []
            buy_cost = sum(p.max_cost for p in result.plans if p.action == "buy")
            if not dry_run and trades_shown >= max_trades and any(p.action == "buy" for p in result.plans):
                decision = Text("cap: max trades", style=_YELLOW)
            elif not dry_run and spend_shown + buy_cost > max_spend and buy_cost > 0:
                decision = Text("cap: spend limit", style=_YELLOW)
            else:
                prefix = "[dry-run] " if dry_run else ""
                for p in result.plans:
                    if p.action == "buy":
                        parts.append(f"BUY {p.side.upper()} ${p.max_cost:.2f}")
                    else:
                        parts.append(f"SELL {p.side.upper()} {p.shares:.2f}sh")
                decision = Text(f"{prefix}" + "  ".join(parts), style=_GREEN)
                trades_shown += 1
                spend_shown  += buy_cost

        pos_str = _fmt_position(positions.get(result.market.get("id", "")))
        table.add_row(label, model_str, price_str, edge_str, pos_str, decision)
        rows_shown += 1

    # Summary rows for the remainder
    remaining_above = len(above_threshold) - min(rows_shown, len(above_threshold))
    total_below     = len(below_threshold)
    total_errors    = len(errors_skips)

    if remaining_above > 0:
        table.add_row(
            Text(f"  … {remaining_above} more above threshold", style=_DIM),
            "", "", "", "", "",
        )
    if total_below > 0:
        table.add_row(
            Text(f"  {total_below} below threshold ({edge_threshold*100:.0f}% edge)", style=_DIM),
            "", "", "", "", "",
        )
    if total_errors > 0:
        table.add_row(
            Text(f"  {total_errors} errors / no data", style=_DIM),
            "", "", "", "", "",
        )

    console.print(table)


# ── individual trade result ───────────────────────────────────────────────────

def trade_placed(market: dict, plan, filled_shares: float | None,
                 filled_cost: float | None, tx_hash: str | None = None,
                 dry_run: bool = False) -> None:
    label  = _market_label(market)
    side   = plan.side.upper()
    prefix = "[dim][dry-run][/dim]" if dry_run else "[bold green][trade][/bold green]  "

    action_str = f"{'BUY' if plan.action == 'buy' else 'SELL'} {side}"

    if dry_run:
        amount_str = f"${plan.max_cost:.2f}" if plan.action == "buy" else f"{plan.shares:.4f}sh"
        console.print(
            f"  {prefix} [green]{label}[/green]  "
            f"{action_str}  [cyan]{amount_str}[/cyan]  "
            f"[dim]{plan.note}[/dim]"
        )
    else:
        shares_str = f"{filled_shares:.4f}sh" if filled_shares is not None else "?"
        cost_str   = f"${filled_cost:.2f}"    if filled_cost   is not None else "?"
        tx_str     = f"tx={tx_hash[:10]}…"    if tx_hash        else ""
        console.print(
            f"  {prefix} [green]{label}[/green]  "
            f"{action_str}  filled={shares_str}  cost={cost_str}  "
            f"[dim]{tx_str}  {plan.note}[/dim]"
        )


def trade_failed(market: dict, plan, error: str) -> None:
    label = _market_label(market)
    action = "BUY" if plan.action == "buy" else "SELL"
    console.print(
        f"  [bold red][error][/bold red]   {label}  {action} {plan.side.upper()}  {error}",
        style=_RED,
    )


# ── cycle summary ─────────────────────────────────────────────────────────────

def cycle_summary(
    cycle: int,
    total_markets: int,
    trades_executed: int,
    spend: float,
    bankroll: float,
    duration: float,
    loop_minutes: int,
    dry_run: bool,
) -> None:
    mode = " [dim](dry run)[/dim]" if dry_run else ""
    console.print(
        f"\n  [bold]Cycle {cycle} complete[/bold]{mode}  ·  "
        f"[cyan]{total_markets}[/cyan] markets scored  ·  "
        f"[green]{trades_executed}[/green] trade{'s' if trades_executed != 1 else ''} placed  ·  "
        f"[cyan]${spend:.2f}[/cyan] spent this cycle"
    )
    console.print(
        f"  Bankroll: [cyan]${bankroll:.2f}[/cyan]  ·  "
        f"Duration: [dim]{duration:.1f}s[/dim]  ·  "
        f"Next cycle in [dim]{loop_minutes}m[/dim]"
    )
    console.print()


# ── error / warning helpers ───────────────────────────────────────────────────

def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  {msg}", style=_YELLOW)


def error(msg: str) -> None:
    console.print(f"  [bold red]✗[/bold red]  {msg}", style=_RED)
