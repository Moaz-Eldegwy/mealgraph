"""MCP server stub.

The Model Context Protocol (MCP) is the standard for exposing tools to
any MCP-aware client (Claude Desktop, Cursor, ChatGPT, the Anthropic
SDK, …). This module is the wrapping seam: a single entry point that
turns this project's tools into an MCP server.

The official Python SDK is the ``mcp`` package, kept as an OPTIONAL
dependency so installs without MCP support do not pay the cost.

Usage (after ``pip install mcp``)::

    python -m mcp_server          # stdio transport for Claude Desktop / Cursor

To register with Claude Desktop, add to ``~/.config/claude/desktop.json``::

    {
      "mcpServers": {
        "mealgraph": {
          "command": "python",
          "args": ["-m", "mcp_server"],
          "cwd": "/path/to/mealgraph"
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict


def _build_server() -> Any:
    """Construct the MCP server lazily so the import is optional."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "MCP support requires `pip install mcp`. The rest of the system runs without it."
        ) from e

    from nutrition_formulas import full_assessment as _full_assessment
    from tools import ComputationTool, QuantitiesFinder, safe_math_eval

    server = FastMCP("mealgraph")
    qf = QuantitiesFinder()
    comp = ComputationTool()

    # ----- QuantitiesFinder ------------------------------------------------
    @server.tool()
    def quantities_finder(payload: Dict[str, Any]) -> str:
        """Linear-program meal-quantity solver.

        Args:
            payload: {"foods": [{name, calories, protein, fat, carbohydrates,
                                  estimated_g, [min_g, max_g, meal_group, estimate_weight]}, ...],
                      "targets": {calories, protein, fat, carbohydrates},
                      "meal_constraints": [...]}

        Returns:
            JSON {"quantities": {...}, "achieved": {...}}
        """
        return qf.handle_task(json.dumps(payload))

    # ----- ComputationTool (deterministic) ---------------------------------
    @server.tool()
    def nutrition_calculation(op: str, args: Dict[str, Any]) -> str:
        """Closed-form clinical formulas (BMI, BMR, TDEE, ...).

        Args:
            op: 'bmi' | 'bmr' | 'tdee' | 'calorie_target' | 'macro_split' |
                'full_assessment' | 'eval'
            args: kwargs for the chosen op (see nutrition_formulas).

        Returns:
            JSON result for the op.
        """
        return comp.handle_task(json.dumps({"op": op, **args}))

    @server.tool()
    def safe_math(expression: str) -> float:
        """Evaluate a numeric expression in a strict AST sandbox.

        Allowed: literals, + - * / // % **, unary + -, parentheses.
        Forbidden: names, calls, attributes, subscripts.
        """
        return safe_math_eval(expression)

    @server.tool()
    def assess_user(
        weight_kg: float,
        height_cm: float,
        age_years: float,
        sex: str,
        activity_level: str,
        goal: str,
    ) -> Dict[str, Any]:
        """One-shot anthropometric + nutritional assessment.

        Returns BMI, BMR, TDEE, daily_target_calories, and macro_targets
        (protein_g, fat_g, carbohydrates_g).
        """
        return _full_assessment(
            weight_kg=weight_kg,
            height_cm=height_cm,
            age_years=age_years,
            sex=sex,
            activity_level=activity_level,
            goal=goal,
        )

    return server


def main() -> None:
    server = _build_server()
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
