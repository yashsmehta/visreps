# Markdowns

This folder contains private manuscript and discussion materials. It is excluded from Git (via `.git/info/exclude`) and should never be committed or pushed.

## Structure

```
markdowns/
├── manuscript/
│   ├── structure.md   # Overall manuscript structure and outline
│   └── methods.md     # Methods section draft
├── discussion/
│   └── {date}.md      # Supervisor discussion transcripts, named by date (e.g., 17feb2026.md)
└── claude.md           # This file — context for Claude Code
```

## Guidelines

- **discussion/**: Each file is a transcript of a meeting with the supervisor, named `{DD}{mon}{YYYY}.md` (e.g., `20feb2026.md`). Use these for context on project direction and feedback.
- **manuscript/**: Working drafts of manuscript sections. `structure.md` defines the overall paper outline; `methods.md` covers the methods section.
