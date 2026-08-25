# Lunch and Learn: session abstract

Draft copy for the 60 minute Posit Conference session.

**Audience.** Working data science practitioners. They want to see how the pieces
fit together and whether it suits how they work. They do not want IAM policies,
billing, or image internals. Those stay in the repository for whoever needs them.

**Scheduling.** Runs the day after the joint Posit and AWS talk, so that talk can
send people here. The abstract still stands alone, because most readers were not
in the room.

---

## Title

**Positron on Amazon SageMaker: one place to do the whole job**

Alternates:

- The whole workflow, one editor
- From question to published answer, without leaving Positron
- What it looks like when the tools stop getting in the way

---

## Short abstract (program listing, ~160 words)

You know the shape of the day. The data lives in the warehouse, so you pull an
extract. The model has to run somewhere you do not control, so you hand it off.
Someone needs the numbers in a deck by Thursday, so you take a screenshot.

This session is about what changes when all of that happens in one place.

We open Positron on Amazon SageMaker and work a problem from start to finish.
Query governed data and get a data frame back. Poke at it. Build a model, keep
track of what we tried and why. Put the good one somewhere it can be called.
Publish a report that goes and gets the fresh numbers itself, so nobody asks you
to re-run it.

R or Python, whichever you already use.

It is a live demo, not slides. The example is a clinical trial, but if you have
ever waited on an extract or emailed someone a screenshot, you will recognize the
problem it solves.

---

## Long description

Sixty minutes, one editor, one problem worked all the way through.

The problem comes from a Phase III trial: some subjects are going to drop out,
and a study team would like to know which ones while there is still time to do
something. It is a good problem for this because it needs all the pieces. Real
data you are not allowed to copy. A model. And an answer that has to reach people
who are never going to open an IDE.

We start by opening Positron on Amazon SageMaker and querying the warehouse. The
thing worth watching is how little happens: no credentials to set up, no extract
to wait for, no separate tool. You write SQL, you get a data frame, and it is
sitting in your session next to everything else.

Then we spend a while just looking at the data, which is the part most demos skip.
Positron's Data Explorer sorts any column, filters the rows, and shows you the
distribution of every field, without writing code. You find things this way
that you would not find otherwise. We find two, and one of them is a query that
looks perfectly reasonable and returns less than half the right answer.

From there we build a model. Nothing exotic, but we try a few versions and keep a
record of each one, which turns out to matter more than the model does. The
record answers a question the study team actually has, which is whether the extra
data they collect early in a trial is worth collecting. It is.

The model goes somewhere it can be called, and then we call it, and the answer
comes back into the same session we have been working in the whole time.

Last, we publish. Not a screenshot, and not a file someone has to remember to
regenerate. A report that goes and gets the current numbers on its own and shows
up where the study team already looks.

Everything is live and everything is real code. The data is synthetic and the
company is invented, but the workflow is the one you would use on Monday.

If you work in finance rather than life sciences, the same repository has a
lending version. The parts underneath do not change.

---

## What you will leave with

- A clear picture of the whole workflow, and where it would fit in yours
- The parts of Positron that only make sense once you have seen them used
- A way to keep track of what you tried that your future self can read
- A report that refreshes itself, so you stop being the refresh button
- The repository, if you want to run any of it yourself

## Who this is for

Data scientists, statisticians and analysts who work with governed data and have
to show their results to somebody. R, Python, or both. You do not need to know
AWS. Nothing in the session assumes you have set any of it up.

## Format

60 minutes, live, from the editor. Ask questions as we go.
