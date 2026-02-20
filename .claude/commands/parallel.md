The user wants you to execute their most recent request using parallel Sonnet agents for maximum speed.

If the user provided additional instructions with this command, incorporate them: $ARGUMENTS

## Your Role

You are the **orchestrator** (Opus). Your job is to think carefully about the user's task, plan an optimal parallelization strategy, and then dispatch work to fast Sonnet agents.

## Step 1: Identify the Task

Look at the conversation history. The user's most recent request (before invoking `/parallel`) is the task to parallelize. If the request is ambiguous, state your interpretation before proceeding.

## Step 2: Plan the Decomposition

Think through the task and produce a short plan:

- What are the independent subtasks?
- What is the dependency graph? (which subtasks can run simultaneously, which must wait for others?)
- What is the optimal number of parallel agents? (Don't over-split — 2-8 agents is the sweet spot. A single 3-minute task doesn't need splitting.)
- If the task genuinely cannot be parallelized (purely sequential steps), say so and just execute it normally.

**Decomposition heuristics — non-overlapping scopes:**
- Each agent must have a clearly bounded scope. When splitting work, explicitly tell each agent what is NOT their responsibility (e.g., "Stop at the function call boundary — do NOT trace into `load_model`, another agent handles that").
- For **read/search tasks**: split by file group or directory, not by conceptual category. Each file should be assigned to exactly one agent.
- For **pipeline traces**: split by pipeline stage. Tell each agent their entry/exit points.
- For **write tasks**: split by output file — one agent per file, never two agents editing the same file.
- When a shared file is needed for context by multiple agents (e.g., `evals.py` calls functions in several modules), tell agents to read the relevant section of that file but only report on their assigned scope.

Print your plan so the user can see your reasoning.

## Step 3: Execute

- Use the `Task` tool with `model: "sonnet"` and an appropriate `subagent_type` (usually `"general-purpose"`, but use `"Bash"` for pure shell tasks or `"Explore"` for codebase search).
- Launch all independent agents in a **single message** — this is what makes them run concurrently.
- Each agent prompt must be **fully self-contained**. Agents have no access to this conversation. Every agent prompt MUST begin with a **Project Context** preamble:

```
## Project Context
- Working directory: /home/ymehta3/research/VisionAI/visreps
- This repo investigates whether fine-grained category supervision is necessary for brain-model alignment. It trains CNNs on ImageNet with varying label granularity, then evaluates alignment with neural data (fMRI, electrophysiology, behavioral).
- Key conventions: activate venv before any Python command (`source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate`). All scripts run from project root.
- [Add any task-specific context: relevant file structure, config schema, field meanings, etc.]
```

- After the preamble, include: file paths, function signatures, config values, and expected output format.
- **Shared output format:** Before dispatching agents, decide on a single output format (e.g., markdown table with specific columns, numbered list with specific fields, JSON schema). Include the exact format template in every agent prompt so all agents return data in an identical structure. This makes synthesis trivial.
- Clearly tell each agent whether it should **write code** or **only research** (read/search). Default to research-only unless the user's task explicitly involves making changes.
- If there are dependent stages, wait for the first batch to finish, then launch the next batch using results from the first.

## Step 4: Synthesize

Once all agents return, combine their results into a unified response:

- **For comparisons:** Build a side-by-side table with one row per dimension and one column per item being compared.
- **For pipeline traces:** Chain the steps from each agent in execution order, linking outputs to inputs.
- **For searches:** Deduplicate hits (if agents had overlapping scope) and group by the user's requested categorization.
- **For write tasks:** Report which files were created/modified, and flag any agents that failed.
- **Always:** Flag conflicts, errors, or surprising findings across agents.

## Rules

- Always use Sonnet (`model: "sonnet"`) for subagents.
- If the task is trivial (1-2 simple steps), skip the agent overhead and just do it directly.
- Prefer launching N agents in one message over launching them one at a time.
- When agents need to edit files, be careful about conflicts — don't have two agents edit the same file.

## Write Tasks: Two-Stage Execution

When the user's task involves creating or modifying files:

1. **Stage 1 (Research):** First dispatch agents to read all relevant source files and determine what needs to change. Collect the results. Ask research agents to return the full content of source files (or the specific data needed for writes) so you can embed it in Stage 2 prompts.
2. **Stage 2 (Write):** Based on Stage 1 results, dispatch write agents with explicit instructions. **Embed the actual data from Stage 1** (e.g., full file content, computed values) directly into each write agent's prompt — agents cannot see Stage 1 results otherwise:
   - **Exact output file path** for each agent (absolute path).
   - **Exact content** or a precise transformation rule (e.g., "copy this file, change field X to Y, remove field Z").
   - One agent per output file — never have two agents write to the same file.
   - Tell each agent to use the `Write` tool (for new files) or `Edit` tool (for modifications).
3. **Stage 3 (Verify):** Optionally dispatch a verification agent to read all written files and confirm correctness.
