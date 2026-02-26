---
name: read-notes
description: Read reference papers from /notes/ for theoretical context. Lists available PDFs and asks the user which to read before proceeding.
---

# Read Notes

Read reference papers from the `/notes/` directory to gather theoretical context for the current task.

## Step 1: List available papers

List all PDFs in `/notes/` and present them to the user as options. Ask which ones are relevant to the current task. Allow multiple selections.

## Step 2: Read selected papers

Read the selected PDFs. Extract information relevant to the current task — the caller should specify what to look for (e.g., contraction patterns, algorithmic costs, gauge fixing procedures, sampling schemes).

## Step 3: Summarize findings

Present a concise summary of what was extracted, organized by topic. This summary is consumed by the calling skill or the user.
