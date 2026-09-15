source("agent.R", local = TRUE)

starter_questions <- shinychat::chat_greeting(paste(
  "## Explore the loan portfolio\n\n",
  "Choose a starter question or ask your own.\n\n",
  '<span class="suggestion">What is the current portfolio size and realized charge-off rate?</span>\n',
  '<span class="suggestion">Which loan purposes have the highest charge-off rates?</span>\n',
  '<span class="suggestion">Create a plot of charge-off rate by FICO band.</span>'
))

ui <- shinychat::page_chat(
  "Aurora Lending portfolio analyst",
  id = "chat",
  greeting = starter_questions,
  theme = commons::commons_theme()
)

server <- function(input, output, session) {
  con <- connect_athena()
  session$onSessionEnded(function() {
    if (DBI::dbIsValid(con)) {
      DBI::dbDisconnect(con)
    }
  })

  agent <- build_agent(con)
  commons::commons_server("chat", agent)
}

shiny::shinyApp(ui, server)
