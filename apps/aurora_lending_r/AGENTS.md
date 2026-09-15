# Aurora Lending commons app

This app is scoped to the synthetic Aurora Lending loan portfolio. It exposes
only `fct_loan_performance`, `dim_borrower`, and `dim_loan_product` from the
`aurora_lending` Athena database.

Treat the existing R portfolio risk review as the trusted source for the
initial semantic layer. Preserve measure provenance and do not add new business
calculations or interpretations without a trusted source. Keep Athena access
read-only and use the standard AWS credential chain; never store credential
values in this project.

Each Shiny session must create a fresh Athena connection and a fresh commons
agent. The default Bedrock model is the US Claude Sonnet 4.6 inference profile
and can be overridden with `POSIT_DEMO_BEDROCK_MODEL`.
