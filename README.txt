🎨 Visualization Guide
1. Line Chart — Accidents per year/decade (RQ1)

Purpose: Show long-term decline in accidents (safety improvements).

File to use:

annual_summary.csv → for yearly trend

decade_summary.csv → for a smoother decade-level trend

Columns:

year, accidents, fatal_accidents (for annual)

decade, accidents, fatal_accidents (for decade)


2. Violin Plot — Fatalities by decade (RQ2)

Purpose: Show distribution of fatalities per accident across decades.

File to use:

dataset_clean_snapshot.csv (the full cleaned dataset)

Columns:

decade, fatalities

Notes: Each point = one accident, grouped by decade. Violin plot will show how the distribution shrank over time (fewer catastrophic accidents).


3. Grouped Bar Chart — Accidents by aircraft type (RQ3)

Purpose: Compare how often different aircraft models appear in accidents, and how many of those were fatal.

File to use:

aircraft_type_counts.csv

Columns:

type, accidents, fatal_accidents

Notes: Can plot bars grouped by type, with two bars: total vs fatal.


4. Horizontal Bar Chart — Top airlines (operators) by normalized accident count (RQ3)

Purpose: Show which airlines/operators appear most often, adjusted for years active.

File to use:

operator_normalized_counts.csv

Columns:

operator, accidents_per_year_active, fatal_per_year_active

Notes: Sort by accidents_per_year_active (highest to lowest). Horizontal bars make long operator names readable.


📌 Extra Notes for your teammate

All these files are inside reports/ after you run your analysis script.

They don’t need to redo the stats — just load the CSVs into Pandas (or Excel/Google Sheets if easier) and plot.

For a clean presentation:

Keep colors consistent (e.g., accidents = blue, fatal accidents = red).

Use clear titles matching your research questions.

Add plain-language captions: “Accidents have declined since 1970s”, “Widebodies have higher median fatalities”, etc.