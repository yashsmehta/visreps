You are tackling one or more TODO items from the project's task list. The TODO number(s) are: $ARGUMENTS

The input can be a single number (e.g., `3`) or multiple space-separated numbers (e.g., `2 3`). When multiple TODOs are given, you will create one **joint** plan that addresses all of them together, not separate plans.

## Step 1: Read the TODOs

Read the file `experiments/todos.md` and find all TODO items matching the number(s) provided above. If any number doesn't match a TODO, tell the user which ones are invalid and stop.

## Step 2: Understand the Context

Before planning, gather the context you need:

- Read any files, directories, or existing experiments referenced by or related to the TODO.
- Check `experiments/coarse_grain_benefits/utils.py` and `experiments/representation_analysis/utils.py` for reusable utilities (model loading, feature extraction, label loading). **Reuse before rewriting** — never duplicate functionality that already exists.
- Check `plotters/plotter_utils.py` and `plotters/plot_helpers.py` if the TODO involves plotting.
- Look at similar completed experiments in `experiments/` for patterns and conventions.
- Read `experiments/CLAUDE.md` for experiment conventions.

## Step 3: Ultrathink and Plan

Now **ultrathink** — use your maximum extended thinking budget to reason deeply and thoroughly about the TODO(s). When multiple TODOs are given, think about how they relate and can share data loading, feature extraction, or plotting infrastructure. Consider:

1. **Goal**: What exactly do these TODO(s) ask for? What are the deliverables?
2. **Data flow**: What data do we need? Where does it come from? What transformations are required? If multiple TODOs share data needs, plan to load/compute once.
3. **Existing code**: What utilities, functions, or patterns from the codebase can we reuse?
4. **Implementation**: What is the simplest, most direct way to implement this? For multiple TODOs, should they be one script or separate scripts with shared logic?
5. **Output**: What files will be created/modified? What does the output look like?

## Step 4: Present the Plan

Present a clear, concise implementation plan structured as:

### TODO Summary
One-line description of what we're doing.

### Reusable Code
List specific functions/modules from the codebase that we'll reuse (with file paths and brief descriptions of what each provides).

### Implementation Plan
Numbered steps, each with:
- What to do (concrete action)
- Which file to create or modify
- Key code approach (pseudocode or brief description, not full code)

### File Structure
```
experiments/relevant_dir/
  new_file.py        # What it does
  figures/output.png  # Expected output
```

### Potential Issues
Anything that might go wrong or needs clarification from the user.

## Rules

- **Plan only** — do NOT start implementing. Present the plan and wait for the user to approve or adjust.
- **Simple and concise** — favor the most straightforward approach. No over-engineering.
- **Reuse existing code** — always prefer calling existing utilities over writing new ones.
- **Run from project root** — any script paths should assume execution from `/home/ymehta3/research/VisionAI/visreps/`.
- Keep the plan short and actionable. Don't pad with obvious steps.
