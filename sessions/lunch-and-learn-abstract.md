# Lunch and Learn: session abstract

Draft copy for advertising the 60 minute Posit Conference session. Written as
promotional copy rather than in Simplified Technical English, which is meant for
instructions and strips persuasion by design.

**Framing.** This is a comprehensive technical deep dive into Positron on Amazon
SageMaker. The worked example comes from life sciences, but the example is the
vehicle and not the subject. Someone who works in banking, insurance, or public
sector should read this and expect to get what they came for.

**Scheduling.** This session runs the day after the joint Posit and AWS talk, so
that talk can invite people to it. The abstract still stands on its own, because
most readers will not have been in the room.

---

## Title

**Positron on Amazon SageMaker: the complete build**

Alternates:

- Positron on Amazon SageMaker, end to end
- The whole data science workflow, in one IDE, on AWS
- Positron on Amazon SageMaker: a working environment, built live

---

## Short abstract (program listing, ~150 words)

Most demonstrations of a data science platform show you the parts. This one
builds the whole thing, live, and then shows you what breaks.

Over 60 minutes we stand up a complete workflow in Positron running on Amazon
SageMaker: governed data in Amazon Athena, interactive analysis in the IDE, a
model trained and hosted on SageMaker, every experiment tracked in managed
MLflow, and a report published to Posit Connect that refreshes itself against
live data. One environment, one language runtime of your choosing, no context
switching.

We spend real time on the parts that only appear when you build this for
production. How the IAM boundary changes the way you are allowed to query. How
to keep a served model from becoming version-locked to the machine that trained
it. What it costs, and how to switch it off.

We work a clinical trial dataset, but the architecture is the point. You leave
with the public repository and can run all of it in your own account.

---

## Long description

Most data science teams assemble their workflow from parts that were never
designed to meet. The IDE is on a laptop. The governed data is somewhere else,
reached through an extract. The model runs in a system the analyst does not own.
The results are pasted into a slide. Each seam is a place where reproducibility
and momentum leak away.

This session builds the whole path as one environment, on AWS, in one sitting.

**The environment.** Positron running as a custom image on Amazon SageMaker. The
first thing worth noticing is that there is nothing to configure: the SageMaker
execution role is already present, and the AWS SDK finds it. No keys, no profile,
no setup step. We cover how the image is put together, what it contains, how it
is attached to a domain, and what it means that Positron is a full IDE for R and
Python rather than a notebook surface.

**The data.** A governed AWS Glue catalog, queried through Amazon Athena straight
into a DataFrame. Nothing is copied to the workstation. We then open the result
in Positron's Data Explorer to sort, filter, and read column distributions
without writing code, which is the fastest route from a query result to an actual
understanding of it.

**The model.** We train a classifier and log every candidate to a SageMaker
managed MLflow tracking server, then compare them on a question the business
asks rather than on a tuning parameter. The winner deploys to a SageMaker
real-time endpoint, which the analysis calls over HTTPS under the same role it
already had.

**The audience.** A Quarto report rendered against live data and published to
Posit Connect, where it re-renders on a schedule for the people who will never
open an IDE.

**Then the parts nobody demonstrates.** A read-only IAM role changes one specific
argument in every Athena call, and getting it wrong produces code that works for
you and fails for everyone else. A model served as a serialized object becomes
locked to the library versions that trained it, and there is a straightforward
way to avoid that. Reproducibility is a seed, a lockfile, a data dictionary, and
an artifact a human can read. A hosted endpoint and a tracking server both bill
by the hour, and knowing which one to stop and which one to delete is worth
knowing before the invoice.

We also spend a few minutes on a query that looks correct, passes review, runs
without an error, and reports less than half the right answer. It is a good
reminder that the environment is only as trustworthy as the analysis you run in
it.

The worked example is a Phase III clinical trial, and the data is synthetic. If
you work in financial services, the same repository carries a consumer lending
version, and the architecture underneath is identical.

---

## What you will leave with

- A complete, working architecture for data science on AWS, not a diagram of one
- How to reach governed data from an IDE with no credential handling
- A model hosted on SageMaker that your analysis calls like any other API
- Experiment tracking that answers a question worth asking
- A published artifact that refreshes itself, and the credential model behind it
- The cost controls, so nothing bills after you close your laptop
- The public repository, so you can run all of it in your own account

## Who this is for

Data scientists, statisticians, and ML engineers who want their governed data,
their compute, and their published output in one place. Useful whether you write
R, Python, or both, and whichever industry you work in. Some AWS familiarity
helps, but every step is shown.

## What this session is not

We are specific about how this architecture supports reproducibility, and about
what it actually does. We do not claim a validated or qualified environment, and
nothing here replaces your own compliance process.

## Format

60 minutes. Live, from the IDE, against real AWS services. Questions throughout,
and a repository you can clone before you leave.
