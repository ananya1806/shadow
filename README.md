# WR Doppelgänger

- **What:** We fingerprint how 110 NFL wide receivers move, using 2021 Next Gen Stats tracking data (weeks 1–8).
- **Insight:** The fingerprint is real: from half of a receiver's games, we pick him out of 110 at 11× chance.
- **Method:** Movement-only route features (release, speed, braking, cuts) are averaged per WR and compared by cosine similarity, teammates excluded.
- **Limits:** 8 weeks of pass plays only, and just 5 doppelgänger pairs stay stable across the season, e.g. Cooper Kupp → K.J. Osborn, who costs $14.9M less per year.
- **For:** Scouts and GMs finding cheaper look-alikes, coaches picking backups, and broadcasters making player comparisons.

[Website](https://chirantan2003.github.io/shadow/) · [Notebook](demo.ipynb) · [Code](src) · [Outputs](outputs) · [Reproduce](OUT_OF_REPO.md)
