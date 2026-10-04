# data/

* `prognostic_data.csv` — BIGDATA-COVID19 blood markers (San Raffaele Hospital), from [Zenodo 4686707](https://zenodo.org/record/4686707). Kept in the repo (0.7 MB).
* `images/` and `cache/` — the 4 GB Menoufia X-ray/CT dataset ([Mendeley Data](https://data.mendeley.com/datasets/8h65ywd2jr/3)) and its 100×100 cache. **Not in git.** Run `python src/download_data.py` then `python src/prepare_images.py`.

Set `CAPSTONE_DATA` to keep data elsewhere.
