# Aurora Lending commons agent

This R Shiny app uses `commons` to answer questions about the synthetic Aurora
Lending loan portfolio in Amazon Athena. Trusted portfolio calculations are in
`measures/`; novel questions can use read-only SQL against the three exposed
Athena tables. Amazon Bedrock supplies the language model through `ellmer`.

The app uses the normal AWS credential chain. In the Positron SageMaker demo,
both Athena and Bedrock pick up the current SageMaker execution-role
credentials. No AWS keys are stored in the app.

SageMaker Studio does not expose the Linux features that `commons` uses for its
full R sandbox, so this local demo explicitly enables the package's best-effort
fallback. The agent still uses `network = "none"` and `commons` rejects
non-read-only SQL, but this fallback is not a production security boundary.

From the repository root, run:

```r
shiny::runApp("apps/aurora_lending_r")
```

The defaults are `us-east-2`, the `Athena` ODBC driver, and the
`us.anthropic.claude-sonnet-4-6` Bedrock inference profile. Override them with
`POSIT_DEMO_REGION`, `POSIT_DEMO_ATHENA_DRIVER`,
`POSIT_DEMO_ATHENA_AUTH`, or `POSIT_DEMO_BEDROCK_MODEL` as needed.

Good first questions include:

- What is the current portfolio size and realized charge-off rate?
- Which loan purposes have the highest charge-off rates?
- How does charge-off rate vary by FICO band?
- Compare charge-off rates by origination vintage and explain the caveat.

Deploying outside SageMaker requires an authorized AWS credential mechanism
for both Athena and Bedrock. Do not copy temporary credentials into this
project or deployment bundle.
