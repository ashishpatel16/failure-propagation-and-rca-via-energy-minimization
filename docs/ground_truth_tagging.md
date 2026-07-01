# Curating Buggy-Method Ground Truth

A reference for manually tagging the **true buggy method(s)** of each Defects4J bug.
The canonical annotations are stored in `ground_truth.csv`; the per-instance
`data/defects4j/<Project>_<id>/buggy_methods.txt` files are legacy extraction
artifacts.

This file exists because the automatic extractor (`d4j_manager._find_method_fqn`)
derives names from a fragile diff heuristic and mislabels ~42% of single-bug
instances. The rankings exported by `run_benchmark_pregen.py` do **not** need this
ground truth — it is only used by the downstream analysis that compares each
method's rank against the actual fault.

---

## The golden rule

> **The buggy method is the method whose body *encloses* the changed lines —
> never a token that appears *on* a changed line.**

Every extractor mistake violated this: it read identifiers off the changed line
(`new Range(` → `Range`, `else if (` → `if`) instead of walking *up* to the
enclosing method declaration.

---

## Procedure (per bug)

1. **Open `patch.txt`.** For each hunk note the file (`+++ b/.../Foo.java`) and the
   `-`/`+` changed lines.
2. **Open the actual buggy source file**, not just the diff — the diff context is
   often too short to show the method header. Get the buggy checkout with:
   ```
   defects4j checkout -p <Project> -v <id>b -w <dir>
   ```
3. **From the first changed line, scroll *upward*** to the nearest enclosing
   method/constructor signature at the correct brace depth. That signature's name
   is the buggy method.
4. **Repeat for every hunk.** A patch can touch several methods — list each
   (one per line), de-duplicated.
5. **Write the FQN** as `package.Class#methodName` (see Format).
6. **Validate** against `call_graph.json` (see Validation).

---

## Traps to avoid

| Mistake (extracted name) | Cause | How to avoid |
|---|---|---|
| `#Range`, `#Integer` | grabbed a `new X(...)` **call** on the changed line | A constructor *called* is never the buggy method. `new X` only matters if the *enclosing* method is `X`'s own constructor. |
| `#if` | matched `else if (...)` as a signature | Keywords (`if/for/while/else/switch/catch/try/return`) and control-flow lines are never methods. Keep scrolling up. |
| `#setMinIcon` (should be `equals`) | the buggy method was *entirely deleted*, so patch line numbers pointed at the neighbouring method | For a wholly added/removed method, the buggy method is **that** method. Read the `-` lines (buggy version) to see whose body they form. |
| (general) | multi-line signatures, annotations, generics | Anchor on the line with `returnType name(params) {`, not on `@Override`/javadoc above it. |

---

## Edge cases & naming

- **Constructor change** → method name = simple class name:
  `org.jfree.chart.renderer.GrayPaintScale#GrayPaintScale`.
  (Legitimate, and different from the `new Range()` trap — here the *enclosing*
  method really is the constructor.)
- **Static / field initializer** → no normal method node exists; treat as a
  representation gap (flag, don't invent a name).
- **Inner / nested class** → `Outer.Inner#method`, e.g. `Option.Builder#build`.
  `$` and `.` are interchangeable (the pipeline converts `$` → `.`).
- **Anonymous class** → shows as `Outer.1#method` / `Outer$1#method`; use only if
  the change is truly inside the anonymous body.
- **Change spans two methods** → list both.
- **Overloaded methods** → matching strips arguments, so `Class#foo` matches *all*
  `foo(...)` overloads. Use the "with args" variant below only if you need to
  disambiguate (and change the matcher accordingly).

---

## Canonical file format

Path: `ground_truth.csv`.

| Column | Meaning |
|---|---|
| `project`, `bug_id` | Defects4J instance identity |
| `buggy_methods` | JSON list of source-level enclosing methods without arguments |
| `buggy_nodes` | JSON list of exact, validated call-graph nodes including arguments |
| `trace_status` | `exact_match`, `trace_gap`, or `representation_gap` |

- Keep `buggy_methods` as the source-level truth even when the method was not traced.
- Put a node in `buggy_nodes` only when its complete signature occurs in that
  instance's `call_graph.json`.
- If a patch changes multiple overloads, include every changed overload separately
  in `buggy_nodes`.
- Never substitute a traced overload for a different, untraced patched overload.
- **Constructors**: `package.Class#Class` (method name repeats the simple class name).
- **Inner classes**: `package.Outer.Inner#method` (or with `$`).
- `$` and `.` are normalized when validating node names.

### Examples (corrected versions of failing cases)

```
org.jfree.chart.renderer.GrayPaintScale#getPaint                 # Chart_24 (was correct; trace gap)
org.jfree.chart.renderer.category.MinMaxCategoryRenderer#equals  # Chart_23 (was #setMinIcon)
```

> Do **not** include trailing comments in the real file — they are shown here only
> for illustration. Each line must be exactly one FQN.

## How exact nodes are matched

`buggy_nodes` is compared to result `Method` values using the full signature.
Only inner-class spelling is normalized:

```python
key = name.strip().replace('$', '.')
```

Consequently, `Class#foo(int)` does not match `Class#foo(String)`. The
argument-free `buggy_methods` column must not be used for ranking joins.

---

## Validation (do this after tagging)

For each tagged name, check whether its key appears as a node in
`call_graph.json`. Three outcomes:

1. **Match** → usable instance.
2. **No match, but the method clearly exists in source** → **trace gap**: the
   method was never executed, so it is not a graph node (e.g. Chart_24). Keep the
   correct name, mark the instance as *fault-not-in-graph*, and exclude it from
   EXAM **with a reported count** — do not force a match.
3. **No match due to a typo / wrong scope** → fix the name.

Reporting the split between (2) and (3) is required for the write-up: it
distinguishes "tagging error" from "fault genuinely not traceable," and documents
how many bugs were excluded and why.
