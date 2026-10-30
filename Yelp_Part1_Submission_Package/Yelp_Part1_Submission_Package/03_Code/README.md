# Part 1 plotting code

plot_part1_data.py reads the two small reports and generates four PNG figures, three count tables, and a JSON summary.
It verifies the row totals, matching label sets, and feature-matrix dimensions reported by the pipeline.
Yelp_Part1_Data_Graphs.ipynb contains the same plotting functions and is self-contained for use in Jupyter on Pitt.

From this folder, reproduce the bundled snapshot with:

```bash
python3 plot_part1_data.py --source-dir ../04_Source_Reports --output-dir part1_outputs
```

Use --project-dir /ix1/cs1671-2026f/yelp instead of --source-dir to read the live project reports.
Dependencies are listed in requirements.txt. No classifier is trained and the large CSV/NPZ files are not loaded.
