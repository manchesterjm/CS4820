# CS4820/5820 Academic Writing Guide

**Josh Manchester's Personal Writing Style Guide**

**Course**: CS 4820/5820 (Artificial Intelligence)
**Institution**: University of Colorado Colorado Springs
**Version**: 1.0
**Created**: November 1, 2025
**Purpose**: Maintain consistent voice and style across all academic papers

This guide captures Josh Manchester's distinctive writing style for AI coursework, research proposals, and technical reports. It is designed to help AI assistants (like Claude Code) generate text that sounds authentically like Josh's voice while maintaining academic rigor.

---

## Table of Contents

1. [Core Writing Philosophy](#core-writing-philosophy)
2. [Pedagogical Approach](#pedagogical-approach)
3. [Sentence Structure and Syntax](#sentence-structure-and-syntax)
4. [Technical Terminology](#technical-terminology)
5. [Question-Driven Writing](#question-driven-writing)
6. [Citation and Attribution Patterns](#citation-and-attribution-patterns)
7. [Numerical and Quantitative Writing](#numerical-and-quantitative-writing)
8. [Paragraph Organization](#paragraph-organization)
9. [Section Templates](#section-templates)
10. [Common Phrases and Transitions](#common-phrases-and-transitions)
11. [Formatting Conventions](#formatting-conventions)
12. [Voice and Tone](#voice-and-tone)
13. [Error Analysis and Discussion](#error-analysis-and-discussion)
14. [AI Disclosure Style](#ai-disclosure-style)

---

## Core Writing Philosophy

### Teaching While Reporting
Josh's writing serves dual purposes:
1. **Document research**: Present findings with scientific rigor
2. **Educate readers**: Ensure even non-experts can follow the logic

Every paragraph should answer: "Would someone unfamiliar with this topic understand both WHAT was done and WHY it matters?"

### Transparency First
- Acknowledge limitations openly
- Explain scope constraints upfront
- Discuss both successes and failures
- Document all assumptions and decisions

### Evidence-Based Claims
- Never make unsupported assertions
- Back every claim with data, citations, or logical reasoning
- Use phrases like "According to...", "The results show...", "This demonstrates..."

---

## Pedagogical Approach

### 1. Parenthetical Definitions (Signature Style)

**Always define technical terms inline on first use** with parenthetical explanations:

```
GOOD (Josh's style):
"I will try straightforward RNN variants (LSTM (a type of RNN (long short-term
memory)) and GRU (a type of RNN (gated recurrent unit)))"

"Handle class imbalance (far fewer positives than negatives) with weights or
focal loss (loss that down-weights easy examples)."

"Report precision (share of predicted positives that are correct), recall
(share of actual positives the model found), F1 score (precision and recall),
and ROC-AUC (How well the model ranks positives over negatives across all
thresholds)."
```

```
AVOID (too formal, assumes expert knowledge):
"The LSTM and GRU architectures were evaluated."
"Focal loss was applied to address class imbalance."
```

**Pattern**: `Term (brief plain-English explanation)`

### 2. Nested Explanations

Josh frequently uses nested parentheses to provide multiple levels of detail:

```
GOOD:
"period (time between repeats)"
"timing helper (simple feature for period (time between repeats)/spacing)"
"masking (blanking spans)"
"ablations (with/without timing helper (simple feature for period/spacing))"
```

This layered approach helps readers at different expertise levels.

### 3. Concrete Examples Before Abstraction

**Always provide specific instances before generalizing:**

```
GOOD:
"Space telescopes record how a star's brightness changes over time. When a
planet crosses in front of the star, the brightness dips slightly. Finding
these small dips is hard because real signals can be weak and noisy."
[Then later discuss RNNs for time-series in general]

"For the easy puzzle, it was 2.5 times faster than basic backtracking (0.023s
vs 0.058s). For the medium puzzle, it was 6.9 times faster (1.813s vs 12.578s)."
[Specific numbers before general conclusion]
```

### 4. Multi-Level Explanations

Explain concepts at increasing depths:

```
Level 1 (Intuition):
"The MRV heuristic selects the variable with the fewest legal values remaining."

Level 2 (Mechanism):
"The idea is to detect failures earlier by choosing the most constrained
variables first."

Level 3 (Implementation):
"This is implemented by computing legal values for each unassigned variable
based on the current partial assignment."

Level 4 (Justification):
"If a variable has only one legal value left, choosing it immediately can
prevent future backtracking."
```

---

## Sentence Structure and Syntax

### Active Voice Predominates

```
GOOD (Josh's style):
"I will build and test a Recurrent Neural Network..."
"The algorithm selects the variable with the fewest legal values..."
"Results demonstrate that PSO struggles with discrete problems."
"This improvement comes from MRV choosing the most constrained variables first."
```

```
AVOID (passive when unnecessary):
"A Recurrent Neural Network will be built and tested..."
"The variable with the fewest legal values is selected..."
"It was demonstrated by results that..."
```

### Conversational Yet Precise

Josh blends informal and formal elements:

```
GOOD:
"My goal is to train an RNN that can tell the difference between true transits
and look-alike noise."

"What makes this algorithm so efficient? The time complexity is only O(1) per
step because we just move one queen at a time."

"This is a classic example of why heuristics matter in search problems."
```

Notice:
- Contractions are OK in rhetorical questions
- "My goal", "I will", "I plan" instead of "The researcher's objective"
- Casual phrasing like "look-alike noise" alongside formal complexity analysis

### Sentence Length Variety

Mix short punchy sentences with longer explanatory ones:

```
GOOD:
"Finding these small dips is hard because real signals can be weak and noisy.
RNNs are a natural fit for time-series and can learn patterns across many time
steps. My goal is to train an RNN that can tell the difference between true
transits and look-alike noise using a modest, well-labeled subset first, then
expand if time allows."
```

Pattern: Short declarative → Medium explanatory → Longer planning statement

---

## Technical Terminology

### Consistent Abbreviation Patterns

**First mention**: Full term (abbreviation) with definition
**Subsequent mentions**: Abbreviation or short form

```
First: "long short-term memory (LSTM) network"
Later: "LSTM" or "the LSTM"

First: "Minimum Remaining Values (MRV) heuristic"
Later: "MRV" or "the MRV heuristic"
```

### Algorithm Names

Use specific, full names on first mention:

```
GOOD:
"AC-3 algorithm for arc consistency"
"Minimum Conflicts local search heuristic"
"Particle Swarm Optimization (PSO) for continuous function minimization"
```

### Hyphenation for Clarity

Josh consistently uses hyphens for compound modifiers:

```
GOOD:
"well-labeled subset"
"transit-shaped dips"
"sequence-aware diagnostics"
"timing-aware auxiliary"
"astrophysical noise examples"
"history-dependent intensity model"
```

---

## Question-Driven Writing

### Rhetorical Questions to Introduce Topics

Josh frequently begins sections or subsections with questions:

```
GOOD:
"How do we solve problems where we need to find solutions that satisfy multiple
constraints at the same time?"

"How can we represent Sudoku as a formal CSP?"

"What makes this algorithm so efficient?"

"What do these numbers tell us about the effectiveness of each approach?"

"How does the PSO algorithm work?"

"Why it works so well?"
```

**Pattern**: Use questions to:
1. Introduce new sections
2. Transition between topics
3. Frame analysis of results
4. Explain algorithm mechanics

### Self-Questioning for Explanations

```
GOOD:
"How do we set up the initial domains?
- Given cells have singleton domains (only one possible value)
- Empty cells have the full domain {1, 2, 3, 4, 5, 6, 7, 8, 9}"

"What are the key equations?
Velocity update equation:
v[i] = w*v[i] + c1*r1*(pbest[i] - x[i]) + c2*r2*(gbest - x[i])"
```

---

## Citation and Attribution Patterns

### "According to" Phrasing

Josh's signature citation style:

```
GOOD:
"According to Russell and Norvig, the time complexity is O(d^n)..."
"According to the results, AC-3 was the fastest method..."
"According to empirical studies, the algorithm typically solves..."
"According to the implementation, typically no more than 10 restarts are needed..."
"According to Vida et al. (2021), stacked LSTMs handled astrophysical noise better..."
```

**Pattern**: "According to [source], [claim]..."

### Reference Integration

Weave citations naturally into narrative:

```
GOOD:
"Vida and colleagues trained and evaluated multiple RNNs for detecting stellar
flares in Kepler and TESS photometry. Although flares are brightenings and
transits are dimmings, both problems require learning patterns in long, noisy
sequences."

"Their best-performing network stacked several long short-term memory layers
(LSTMs), used dropout for regularization, or not memorizing noise, and applied
a one-unit sigmoid output for binary classification (Vida et al., 2021)."
```

### Attribution of Ideas

When using others' work:

```
GOOD:
"From Kügler et al. (2016), I am borrowing the idea of sequence-level
diagnostics which means I will be plotting model scores across time..."

"From Du et al. (2016), I will evaluate whether the timing helper reduces
false positives..."

"Following Vida et al. (2021), I will attempt to report precision, recall,
F1 and ROC-AUC..."
```

**Pattern**: "From [source], I [will/am] [action]..."

---

## Numerical and Quantitative Writing

### Always Include Units and Context

```
GOOD (Josh's style):
"0.023s vs 0.058s" (not just "0.023 vs 0.058")
"2.5 times faster" (with comparison baseline)
"9.32/10" (with maximum score context)
"approximately 10 known planets" (with uncertainty indicator)
"~10 positive, matched negatives" (tilde for approximation)
```

### Precision Varies by Context

```
Experimental results: "0.0581 seconds", "3.974s", "9.32/10"
Estimates: "~10 known planets", "approximately n!/e solutions"
Complexity: "O(cd^3)", "O(n)"
Percentages: "100 percent success", "90.0% success rate"
```

### Comparisons with Specific Numbers

Always quantify improvements:

```
GOOD:
"For the easy puzzle, it was 2.5 times faster than basic backtracking (0.023s
vs 0.058s)."

"AC-3 was 2.2 times faster than forward checking (3.97s vs 8.83s) and 2.7 times
faster than MRV+LCV."

"AC-3's runtime only increased by about 200 times from easy to hard (0.019s to
3.97s)"
```

**Pattern**: "[X] was [N] times [comparison] than [Y] ([specific numbers])"

### Statistical Reporting

```
GOOD:
"Avg score: 7.692651e+01 +/- 1.619927e+01"
"Success rate: 90.0%"
"Pylint score: 9.32/10 (previous run: 9.32/10, +0.00)"
```

---

## Paragraph Organization

### Standard Paragraph Structure

Josh's typical paragraph flow:

1. **Topic sentence** (often a question or claim)
2. **Explanation** (mechanism or reasoning)
3. **Evidence** (data, citation, or example)
4. **Implication** (what this means for the work)

```
EXAMPLE:

[Topic] "AC-3 is the clear winner:"

[Explanation] "AC-3 was the fastest algorithm on all difficulty levels. For
the hard puzzle, it was 2.2 times faster than forward checking (3.97s vs 8.83s)
and 2.7 times faster than MRV+LCV."

[Mechanism] "What makes AC-3 so effective? It propagates constraints globally
across the entire problem instead of just locally between neighbors."

[Implication] "For many Sudoku instances, AC-3 preprocessing alone can reduce
all domains to singletons, essentially solving the puzzle without any
backtracking at all."

[Justification] "The O(cd^3) overhead per call is more than justified by the
dramatic reduction in search space size."
```

### Linking Paragraphs

Use explicit transitions between paragraphs:

```
GOOD:
"Putting the three ideas together."
"Two details from Vida et al. (2021) feed directly into my setup."
"For my work, the ESN-autoencoder paper provides two practical observations."
"Knowing this, I will try to adapt the RMTPP spirit in a lightweight way."
"Summary." [as section header]
```

---

## Section Templates

### Abstract Structure

```
[Problem statement]
[Approach overview with specific methods]
[Key results with numbers]
[Conclusion about what was achieved]

EXAMPLE:
"This report presents implementations and experimental analysis of constraint
satisfaction problem (CSP) solving techniques and metaheuristic optimization
algorithms. Part A formulates and solves Sudoku as a CSP using backtracking
with various enhancements including MRV, LCV, forward checking, and AC-3.
According to the results, AC-3 was the fastest method, solving hard puzzles
up to 200 times faster than basic backtracking. Part B applies the Minimum
Conflicts local search heuristic to the n-Queens problem for board sizes
n=8, 16, and 25. The algorithm achieved 100 percent success with empirical
O(n) scaling, solving even n=25 in milliseconds."
```

### Introduction Pattern

```
[Motivating question]
[Problem context and importance]
[This work's contribution]
[Brief methodology overview]
[Implementation note (from scratch, libraries used)]
[Citation of foundational work]

EXAMPLE:
"How do we solve problems where we need to find solutions that satisfy multiple
constraints at the same time? Constraint Satisfaction Problems (CSPs) and
optimization problems are fundamental in artificial intelligence. CSPs involve
finding assignments to variables that satisfy a set of constraints, while
optimization problems seek to minimize or maximize an objective function. This
report explores both systematic search methods for CSPs and metaheuristic
approaches for optimization.

This work implements and analyzes three problem-solving approaches:
[bulleted list]

All algorithms are implemented from scratch in Python without specialized CSP
or optimization libraries. According to Russell and Norvig (2020) and course
lecture materials, these algorithms represent the state-of-the-art..."
```

### Related Work Pattern

```
[Introductory sentence framing the section]
[For each paper: Summary → Details → Relevance to your work]
[Synthesis paragraph: "Putting the ideas together"]
[Implementation implications]

EXAMPLE:
"This section reviews how recurrent neural networks (RNNs) have been applied to
space-telescope light curves and what that means for my plan. I focus on three
works: (1) [paper 1], (2) [paper 2], and (3) [paper 3]. Together these papers
explain why [thesis]."

[Then for each paper:]
"RNNs for light-curve event detection (Vida et al., 2021). [Summary]. [Key
findings]. [Direct relevance to your work]."

[Synthesis:]
"Putting the three ideas together. Vida et al. (2021) show that [X]; I will
mirror that recipe. Kügler et al. (2016) show that [Y]; I will evaluate based
on [Y]. Du et al. (2016) provide [Z]; I will add [Z]."
```

### Methodology Pattern

```
[Brief overview paragraph]
[Subsections for each component:]
  - Preprocessing: [steps]
  - Model: [architecture details with alternatives]
  - Training: [procedures and hyperparameters]
  - Evaluation: [metrics with parenthetical definitions]
  - Optional/Advanced: [additional techniques]

EXAMPLE:
"Preprocessing: simple detrending and normalization; careful handling of gaps;
no label leakage (windows only use past/current data).

Model: start with a 2-3 layer LSTM (also test GRU (a type of RNN (gated
recurrent unit))) with dropout. Compare final-state (last hidden state) vs.
time-pooled readout (output layer)."
```

### Results and Analysis Pattern

```
[Present results in table or list]
[Analysis subsection with question header]
[For each finding:]
  - Bold header: [Key Finding Name]
  - Evidence with numbers
  - Explanation of mechanism
  - Implication or insight

EXAMPLE:
"The results in Table X show significant performance differences between the
algorithms. What do these numbers tell us about the effectiveness of each
approach?

**Basic Backtracking struggles:** On the hard puzzle (17 given cells), basic
backtracking hit the 5-minute timeout without finding a solution. This
demonstrates that naive depth-first search with no heuristics cannot handle
difficult Sudoku puzzles. [Continue...]"
```

### Experimental Plan Pattern

```
Week-by-week breakdown with specific deliverables:

EXAMPLE:
"Week 1: Assemble the small curated subset (~10 positive, matched negatives)
and write a minimal data-prep notebook.

Week 2: Train a simple baseline (e.g., logistic regression on summary stats)
to set a floor; implement the first LSTM.

Week 3: Tune window length and labeling; add GRU; pick the better recurrent
variant based on validation F1/AUC."
```

### Risks & Mitigations Pattern

```
Bulleted list with risk followed by mitigation:

EXAMPLE:
"• Too few positives: mitigate with pseudo/synthetic injections and careful
cross-validation.
• Overfitting the small set: use validation splits, early stopping, and simple
models first.
• Data cleaning surprises: keep preprocessing minimal and documented; track all
changes in the notebook."
```

---

## Common Phrases and Transitions

### Introducing Methods

```
"I will build and test..."
"My goal is to train..."
"I will try straightforward..."
"I plan to..."
"To keep scope realistic..."
"Starting with a small, well-labeled subset keeps the work focused and feasible."
```

### Explaining Mechanisms

```
"How does [X] work?"
"What makes this algorithm so efficient?"
"The idea is to..."
"This means that..."
"According to [source], [mechanism]"
"The key advantage is that..."
"This is implemented by..."
```

### Presenting Evidence

```
"According to the results..."
"The results in Table X show..."
"This demonstrates that..."
"Results will be compared against..."
"The data clearly show..."
```

### Drawing Conclusions

```
"This improvement comes from..."
"What makes [X] so effective?"
"This is a classic example of..."
"The takeaway is..."
"Together these choices..."
"Summary."
```

### Acknowledging Limitations

```
"To keep scope realistic..."
"If time allows..."
"If needed, I will..."
"Although [limitation], [workaround]"
"While [method] has [drawback], [alternative approach]"
```

### Connecting Ideas

```
"Putting the three ideas together."
"Two details from [source] feed directly into my setup."
"For my work, [paper] provides [insight]."
"Knowing this, I will..."
"Following [source], I will..."
"From [source], I am borrowing..."
```

### Future Work

```
"If time permits, scale up..."
"Expansion path (later): if time allows..."
"In the future, I plan to track whether..."
"Polish: ablations, final metrics/tables/plots..."
```

---

## Formatting Conventions

### Bold Text

Use bold for:
- Key findings in analysis: `**AC-3 is the clear winner:**`
- Important concepts on first mention: `**MRV (Minimum Remaining Values):**`
- Section-within-section headers: `**Preprocessing:**`, `**Model:**`

```
GOOD:
"**Inertia weight effects (w):** The inertia weight has a significant impact..."
"**Scalability analysis:** The results clearly demonstrate..."
```

### Italics

Use italics for:
- Technical terms from papers: `\emph{three LSTM layers with 128 units}`
- Emphasis on specific values: `\emph{Kepler}`, `\emph{TESS}`
- Mathematical notation in LaTeX: `$O(n)$`, `$\lVert w-\tilde{w}\rVert_2^2$`

### Lists and Structure

Use bulleted lists for:
- Multiple parallel items
- Options or alternatives
- Risks and mitigations

Use numbered lists for:
- Sequential steps
- Week-by-week plans
- Ordered priorities

```
EXAMPLE (bulleted):
This work implements and analyzes three problem-solving approaches:
\begin{itemize}
\item Backtracking search with MRV, LCV, forward checking, and AC-3 for Sudoku
\item Minimum Conflicts local search for n-Queens
\item Particle Swarm Optimization for benchmark functions and Sudoku
\end{itemize}
```

### Parentheses and Hyphens

**Parentheses** for:
- Definitions: `(share of predicted positives that are correct)`
- Alternative terms: `(a type of RNN (long short-term memory))`
- Units and context: `(0.023s vs 0.058s)`
- Examples: `(e.g., ~10 known planets)`
- Abbreviations: `(LSTM)`

**Hyphens** for compound modifiers:
- `well-labeled subset`
- `transit-shaped dips`
- `time-series data`
- `sequence-aware diagnostics`

### Em-dashes and Semicolons

**Em-dashes** (represented as —):
- For emphasis or elaboration
- "I will build and test a Recurrent Neural Network—small, regular dips in a star's brightness—in light-curve data."

**Semicolons** for:
- Separating dense list items: "Preprocessing: X; Model: Y; Training: Z"
- Connecting closely related independent clauses

---

## Voice and Tone

### First Person in Proposals

```
GOOD:
"I will build and test..."
"My goal is to..."
"I am borrowing the idea..."
"I will try to adapt..."
"For my work, the paper provides..."
```

### Third Person in Final Reports

Mix first person (for experimental choices) with third person (for describing results):

```
GOOD (mixed):
"This report presents implementations and experimental analysis..."
"All algorithms were implemented from scratch..."
"The results demonstrate that while PSO works reasonably well..."
```

### Hedging Language

Use appropriate hedging for uncertain claims:

```
GOOD:
"approximately 10 known planets"
"typically solves in O(n) steps"
"generally <50 lines"
"often much better with AC-3"
"roughly 90 steps"
"about 200 times from easy to hard"
```

### Definitive Language for Strong Claims

When evidence is clear, be assertive:

```
GOOD:
"AC-3 is the clear winner"
"This demonstrates that naive depth-first search cannot handle..."
"The results clearly demonstrate..."
"All algorithms were implemented from scratch"
```

---

## Error Analysis and Discussion

### Balanced Reporting

Always discuss both successes and failures:

```
GOOD:
"Performance: PSO did not successfully solve the Sudoku puzzle. After 3000
iterations (taking approximately 37 seconds), it remained stuck with an average
of 10 violations. Trial 1 performed best with only 3 violations, but trials 2
and 3 ended with 13 and 14 violations respectively."
```

### Comparative Analysis

Compare methods explicitly with numbers:

```
GOOD:
"Comparison to CSP methods: This really demonstrates the difference between
choosing the right tool versus the wrong tool for a problem. The CSP methods
from Part A solved the exact same puzzle in under 0.02 seconds with a
guaranteed perfect solution (zero violations). PSO took 37 seconds and did not
even solve the puzzle. That means PSO is approximately 1800 times slower and
it still failed to find a valid solution."
```

### Root Cause Explanation

Explain WHY things worked or didn't:

```
GOOD:
"Why PSO struggles with Sudoku: Sudoku is fundamentally a discrete combinatorial
problem with hard constraints that must be exactly satisfied. There is no
'almost solved' Sudoku—either all constraints are satisfied or they are not.
PSO was designed for continuous optimization where the objective function is
smooth and approximate solutions can be useful."
```

### Lessons Learned

Extract actionable insights:

```
GOOD:
"When metaheuristics are useful: PSO and similar metaheuristic algorithms work
best when:
\begin{itemize}
\item Approximate solutions are acceptable (versus needing exact constraint
satisfaction)
\item The search space is continuous (versus discrete like Sudoku)
\item Systematic search methods would take an impractical amount of time
\item The problem involves balancing multiple competing objectives with tradeoffs
\end{itemize}"
```

---

## AI Disclosure Style

### Transparency and Detail

Josh's AI disclosure is thorough and specific:

```
GOOD:

## AI Disclosure

This code was generated with assistance from **Claude Code (Sonnet 4.5)**,
version **claude-sonnet-4-5-20250929**.

The AI assistant helped with:
- Understanding AC-3 algorithm from textbook and lecture slides
- Implementing backtracking search with MRV and LCV heuristics
- Debugging timeout protection logic
- Writing comprehensive docstrings and comments
- Creating run_experiments.py for automated testing
- Formatting output for LaTeX tables

All code was reviewed, understood, and tested by the student.
```

**Pattern**:
1. Specific tool and version
2. Bulleted list of what AI helped with (specific tasks, not vague)
3. Statement of student responsibility

### LaTeX Disclosure

```
\section*{AI Use Disclosure}

This assignment was completed with assistance from \textbf{Claude Code (Sonnet
4.5)}, version \texttt{claude-sonnet-4-5-20250929}.

AI assistance included:
\begin{itemize}
\item Understanding algorithm concepts from lecture and textbook
\item Code implementation and debugging
\item Experiment design and analysis
\item LaTeX formatting and figure generation
\end{itemize}

All code was reviewed, understood, and tested by the student before submission.
The AI did not complete the assignment autonomously—student understanding,
testing, and decision-making were central to the process.
```

---

## Quick Reference: Voice Checklist

When writing as Josh Manchester, ensure:

- [ ] Uses rhetorical questions to introduce topics
- [ ] Defines technical terms inline with parenthetical explanations
- [ ] Provides concrete examples before abstractions
- [ ] Uses "According to [source]" for citations
- [ ] Includes specific numbers with units and comparisons
- [ ] Mixes short punchy sentences with longer explanatory ones
- [ ] Uses active voice ("I will...", "The algorithm selects...")
- [ ] Employs conversational yet precise language
- [ ] Explains WHY not just WHAT
- [ ] Discusses both successes and failures
- [ ] Uses bold for key findings, italics for emphasis
- [ ] Organizes with clear bullet points and numbered lists
- [ ] Hedges appropriately ("typically", "approximately", "often")
- [ ] Provides multi-level explanations (intuition → mechanism → implementation)
- [ ] Links ideas explicitly ("Putting these together", "From X, I borrow...")

---

## Example Transformations

### Generic → Josh's Style

**Generic**:
> "The MRV heuristic was applied to improve performance. Results showed a
> significant speedup over the baseline approach."

**Josh's Style**:
> "**MRV+LCV makes a huge difference:** Adding these heuristics dramatically
> improved performance. For the easy puzzle, it was 2.5 times faster than basic
> backtracking (0.023s vs 0.058s). For the medium puzzle, it was 6.9 times
> faster (1.813s vs 12.578s). This improvement comes from MRV choosing the most
> constrained variables first, which causes failures to occur earlier in the
> search tree. According to the results, LCV complements this by preserving
> maximum flexibility for the remaining variables."

### Technical → Pedagogical

**Too Technical**:
> "We parameterize the conditional intensity as a nonlinear function of event
> history encoded by an RNN."

**Josh's Style**:
> "Du et al. (2016) connect recurrent neural networks with temporal point
> processes by parameterizing the conditional intensity as a nonlinear function
> of event history encoded by an RNN. In other words, RMTPP turns the RNN into
> a history-dependent intensity model that handles event timing and event type
> together (Du et al., 2016). Although RMTPP has been demonstrated in other
> areas like finance and healthcare, the idea can be directly applied to
> periodic transit detection: a light curve with a planet has a regular rhythm
> (ingress to egress, repeat), while many false positives are random or
> quasiperiodic."

---

## Resources

### Example Papers (in CS4820/Term Paper/)
- `Josh_Proposal_Part_Take_2_AAAI24.tex` - Research proposal style
- `Josh_Proposal_Part_AAAI24_v2_relatedwork.tex` - Related work section
- `HW02/writeup/assignment_writeup.tex` - Technical report style

### Key Characteristics Summary
1. **Pedagogical**: Teach while reporting
2. **Transparent**: Show reasoning and limitations
3. **Precise**: Numbers with units and context
4. **Accessible**: Define all technical terms inline
5. **Structured**: Clear organization with questions and bullets
6. **Evidence-based**: "According to X" and specific data
7. **Balanced**: Discuss successes and failures equally

---

**Version History**

- **1.0** (November 1, 2025): Initial writing guide based on analysis of Josh Manchester's term papers and homework writeups

**Questions or Updates?**

This guide should evolve as writing style develops. Update with new patterns and preferences as they emerge.
