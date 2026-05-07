#let project_report(
  title: "Raport z Etapu X",
  subtitle: "Temat Projektu: [Wstaw nazwę tematu]",
  authors: (),
  stage: 1,
  date: none,
  body
) = {
  set document(title: title, author: authors)
  set text(font: "Linux Libertine", lang: "pl", size: 11pt)
  set heading(numbering: "1.")

  set par(justify: true)

  set page(
    margin: (top: 2cm, bottom: 2cm, x: 2.5cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        align(right)[#text(size: 8pt, fill: gray)[#title - #authors.join(", ")]]
      }
    }
  )

  align(center)[
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 10pt)[AGH University of Science and Technology \ Kraków, Poland]],
      align(right)[#text(size: 10pt)[Wydział Informatyki]]
    )
    #v(1em)
    #text(size: 18pt, weight: "bold")[#title] \
    #v(0.4em)
    #text(size: 13pt, style: "italic")[#subtitle] \
    #v(0.8em)
    #text(size: 11pt)[
      *Autorzy:* #authors.join(", ")
    ]
    #v(0.4em)
    #if date != none [#date] else [#datetime.today().display()]
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt)
  ]

  body

  v(2em)
  line(length: 100%, stroke: 0.5pt)
  v(0.5em)
  bibliography("bibliography.bib", title: "Bibliografia", style: "ieee")
}

#show: project_report.with(
  title: "Raport z Etapu X:",
  subtitle: "Symulacja kolonii mrówek",
  authors: ("Jakub Grześ", "Tomasz Smyda"),
  stage: 3,
)
