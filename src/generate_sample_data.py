"""
generate_sample_data.py
------------------------
Creates a small placeholder Fake.csv / True.csv so the whole pipeline
(train_model.py -> app.py) can be run and demoed immediately without
first downloading the real Kaggle dataset.

This is NOT a replacement for the real dataset — accuracy on this toy
data will look great (it's an easy, tiny, repetitive sample) but that's
just because the patterns are trivial. For an actual internship/college
submission, download the "Fake and Real News Dataset" from Kaggle
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
and drop Fake.csv + True.csv into the dataset/ folder, replacing these.

Run with:
    python src/generate_sample_data.py
"""

import os
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

random.seed(42)

REAL_TEMPLATES = [
    "The {org} announced on {day} that {policy} will take effect next quarter, "
    "according to officials briefed on the plan.",
    "Researchers at {org} published a peer-reviewed study in {journal} showing "
    "measurable progress on {topic} over the past five years.",
    "{country}'s finance ministry reported that GDP grew by {num} percent in the "
    "last quarter, citing gains in {sector}.",
    "City council members voted {num} to approve funding for {topic} after a "
    "public hearing that lasted several hours.",
    "The central bank kept interest rates unchanged, saying inflation data on "
    "{topic} remained within its target range.",
    "A spokesperson for {org} confirmed the merger is still under regulatory "
    "review and no final decision has been made.",
]

FAKE_TEMPLATES = [
    "SHOCKING: {org} secretly controls {topic} and mainstream media refuses to "
    "report it, insider claims.",
    "Doctors HATE this one trick that cures {topic} overnight — {org} tried to "
    "ban it!",
    "BREAKING: Anonymous leak proves {country} government is hiding the truth "
    "about {topic}, sources say.",
    "You won't believe what {org} is doing behind closed doors — this changes "
    "everything about {topic}.",
    "EXPOSED: The real reason {org} doesn't want you to know about {topic}, "
    "according to a viral post.",
    "Scientists are STUNNED after this simple method reversed {topic} in just "
    "{num} days, no research required.",
]

ORGS = ["the government", "a major tech company", "the World Health Organization",
        "a leading university", "the central bank", "a Fortune 500 firm"]
COUNTRIES = ["the United States", "India", "the United Kingdom", "Germany", "Brazil"]
TOPICS = ["the economy", "climate policy", "vaccine research", "the housing market",
          "artificial intelligence", "public health", "election security"]
SECTORS = ["manufacturing", "technology", "agriculture", "the service industry"]
JOURNALS = ["Nature", "The Lancet", "a peer-reviewed economics journal"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _fill(template: str) -> str:
    return template.format(
        org=random.choice(ORGS),
        country=random.choice(COUNTRIES),
        topic=random.choice(TOPICS),
        sector=random.choice(SECTORS),
        journal=random.choice(JOURNALS),
        day=random.choice(DAYS),
        num=random.randint(1, 9),
        policy=random.choice(TOPICS),
    )


def build_dataset(n_per_class: int = 400):
    real_rows = []
    for i in range(n_per_class):
        template = random.choice(REAL_TEMPLATES)
        text = _fill(template)
        real_rows.append({
            "title": text.split(",")[0][:70],
            "text": text,
            "subject": "politicsNews",
            "date": f"{random.randint(1, 28)}-{random.randint(1, 12)}-2023",
        })

    fake_rows = []
    for i in range(n_per_class):
        template = random.choice(FAKE_TEMPLATES)
        text = _fill(template)
        fake_rows.append({
            "title": text.split(",")[0][:70],
            "text": text,
            "subject": "News",
            "date": f"{random.randint(1, 28)}-{random.randint(1, 12)}-2023",
        })

    return pd.DataFrame(real_rows), pd.DataFrame(fake_rows)


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    true_df, fake_df = build_dataset(n_per_class=400)

    true_path = os.path.join(DATASET_DIR, "True.csv")
    fake_path = os.path.join(DATASET_DIR, "Fake.csv")

    true_df.to_csv(true_path, index=False)
    fake_df.to_csv(fake_path, index=False)

    print(f"Sample dataset created:")
    print(f"  {true_path}  ({len(true_df)} rows)")
    print(f"  {fake_path}  ({len(fake_df)} rows)")
    print("\nThis is a small synthetic dataset meant only for testing the "
          "pipeline end-to-end. Replace with the real Kaggle dataset before "
          "using this project as an actual submission.")


if __name__ == "__main__":
    main()
