#' Portfolio overview
#'
#' Summarizes the size, date range, exposure, and realized charge-off rate of
#' the full Aurora Lending portfolio.
#'
#' @return A one-row data frame with loan count, first and last origination
#'   dates, total originated principal, outstanding balance, and charge-off
#'   rate.
#' @provenance https://github.com/posit-dev/positron-sagemaker-demo/blob/238fc56f7d3ba73627a404c60572947f8beca755/reports/aurora_lending/portfolio_risk_review_r.qmd#L103-L147
#' @measure
portfolio_overview <- function(aurora_lending) {
  DBI::dbGetQuery(
    aurora_lending,
    paste(
      "SELECT COUNT(*) AS loans,",
      "MIN(origination_date) AS first_origination_date,",
      "MAX(origination_date) AS last_origination_date,",
      "CAST(SUM(principal_amount) AS double) AS originated_principal,",
      "CAST(SUM(outstanding_balance) AS double) AS outstanding_balance,",
      "AVG(CASE WHEN charged_off THEN 1.0 ELSE 0.0 END) AS chargeoff_rate",
      "FROM fct_loan_performance"
    )
  )
}

#' Loan performance by purpose
#'
#' Compares loan count, originated principal, average APR, average FICO, and
#' realized charge-off rate for every stated loan purpose.
#'
#' @return A data frame with one row per loan purpose, ordered from highest to
#'   lowest charge-off rate.
#' @provenance https://github.com/posit-dev/positron-sagemaker-demo/blob/238fc56f7d3ba73627a404c60572947f8beca755/reports/aurora_lending/portfolio_risk_review_r.qmd#L149-L180
#' @measure
performance_by_purpose <- function(aurora_lending) {
  DBI::dbGetQuery(
    aurora_lending,
    paste(
      "SELECT purpose,",
      "COUNT(*) AS loans,",
      "CAST(SUM(principal_amount) AS double) AS originated_principal,",
      "AVG(apr) AS average_apr,",
      "AVG(fico_at_origination) AS average_fico,",
      "AVG(CASE WHEN charged_off THEN 1.0 ELSE 0.0 END) AS chargeoff_rate",
      "FROM fct_loan_performance",
      "GROUP BY purpose",
      "ORDER BY chargeoff_rate DESC"
    )
  )
}

#' Charge-off rate by FICO band
#'
#' Calculates portfolio size and realized charge-off rate in the FICO bands
#' used by the Aurora Lending portfolio risk review.
#'
#' @return A data frame with one row per FICO band, loan count, and charge-off
#'   rate.
#' @provenance https://github.com/posit-dev/positron-sagemaker-demo/blob/238fc56f7d3ba73627a404c60572947f8beca755/reports/aurora_lending/portfolio_risk_review_r.qmd#L123-L130
#' @provenance https://github.com/posit-dev/positron-sagemaker-demo/blob/238fc56f7d3ba73627a404c60572947f8beca755/reports/aurora_lending/portfolio_risk_review_r.qmd#L182-L207
#' @measure
chargeoff_by_fico_band <- function(aurora_lending) {
  DBI::dbGetQuery(
    aurora_lending,
    paste(
      "SELECT CASE",
      "WHEN fico_at_origination < 620 THEN '580-619'",
      "WHEN fico_at_origination < 660 THEN '620-659'",
      "WHEN fico_at_origination < 700 THEN '660-699'",
      "WHEN fico_at_origination < 740 THEN '700-739'",
      "WHEN fico_at_origination < 780 THEN '740-779'",
      "ELSE '780+' END AS fico_band,",
      "COUNT(*) AS loans,",
      "AVG(CASE WHEN charged_off THEN 1.0 ELSE 0.0 END) AS chargeoff_rate",
      "FROM fct_loan_performance",
      "GROUP BY 1",
      "ORDER BY MIN(fico_at_origination)"
    )
  )
}

#' Charge-off rate by origination vintage and FICO band
#'
#' Calculates realized charge-off rates by origination year and FICO band for
#' examining how observed portfolio performance changes across vintages.
#'
#' @return A data frame with one row per origination year and FICO band, loan
#'   count, and charge-off rate.
#' @provenance https://github.com/posit-dev/positron-sagemaker-demo/blob/238fc56f7d3ba73627a404c60572947f8beca755/reports/aurora_lending/portfolio_risk_review_r.qmd#L182-L207
#' @measure
chargeoff_by_vintage <- function(aurora_lending) {
  DBI::dbGetQuery(
    aurora_lending,
    paste(
      "SELECT YEAR(origination_date) AS vintage,",
      "CASE",
      "WHEN fico_at_origination < 620 THEN '580-619'",
      "WHEN fico_at_origination < 660 THEN '620-659'",
      "WHEN fico_at_origination < 700 THEN '660-699'",
      "WHEN fico_at_origination < 740 THEN '700-739'",
      "WHEN fico_at_origination < 780 THEN '740-779'",
      "ELSE '780+' END AS fico_band,",
      "COUNT(*) AS loans,",
      "AVG(CASE WHEN charged_off THEN 1.0 ELSE 0.0 END) AS chargeoff_rate",
      "FROM fct_loan_performance",
      "GROUP BY 1, 2",
      "ORDER BY 1, MIN(fico_at_origination)"
    )
  )
}
