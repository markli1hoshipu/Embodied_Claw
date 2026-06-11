"""Slack bridge for the Embodied_Claw pipeline (spec section "## 12", subsections 13.1-13.11).

Contract with the pipeline is the filesystem mailbox ONLY (spec section 7):
  runs/<run_id>/escalations/*.question.json   -- questions from agents
  runs/<run_id>/escalations/<esc_id>.reply.*  -- replies (single int = option id, else free-form)
  runs/<run_id>/transitions.jsonl             -- one JSON line per stage transition
  runs/<run_id>/request.txt                   -- natural-language request for new runs

The bridge MUST NOT import pipeline code (spec 13.1).
"""
