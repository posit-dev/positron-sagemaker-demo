source("agent.R", local = TRUE)

con <- connect_athena()
on.exit(DBI::dbDisconnect(con), add = TRUE)

agent <- build_agent(con)
agent$prewarm()

app_files <- c(
  "app.R",
  "agent.R",
  "DESCRIPTION",
  "README.md",
  list.files("dictionaries", recursive = TRUE, full.names = TRUE),
  list.files("measures", recursive = TRUE, full.names = TRUE),
  list.files("context", recursive = TRUE, full.names = TRUE),
  list.files("commons-cache", recursive = TRUE, full.names = TRUE)
)

rsconnect::deployApp(
  appDir = ".",
  appFiles = app_files,
  appPrimaryDoc = "app.R"
)
