# Phase 2: Python Data Engineering with Lahman Baseball Database

This repository is part of my **10-Project Data & AI Governance Roadmap**, where I demonstrate how governance, trust, and compliance come together across the AI lifecycle.
Phase 2 focuses on **building reliable ETL pipelines**, ensuring traceability, reproducibility, and privacy in raw data transformation.

---

## Project Overview

### Project 3: ETL Pipeline with Python – Baseball Stats

Using the Lahman Baseball Database (1871–2023), I designed and implemented **end-to-end ETL pipelines**:

* **Ingest:** Load raw CSV data (`Batting`, `Pitching`, `Teams`, `People`, `Salaries`) from AWS S3
* **Transform:** Clean data, correct inconsistencies, and standardize formats (dates, IDs, numeric fields)
* **Load:** Store cleaned data into **PostgreSQL / SQLite** for queryable access
* **Logging:** Track transformations, corrections, and errors for traceability

**Example transformations:**

* Standardizing playerID and teamID formats
* Handling missing or invalid batting averages
* Aggregating salary data by year and team

**Output:** Python ETL scripts, architecture diagram showing pipeline from raw CSV → cleaned DB

---

### Project 4: PII Detection & Masking – Player Data

Before exposing or using personal information, **privacy compliance is essential**:

* Detected sensitive fields in the `People` table (e.g., birthdates, names)
* Applied masking, hashing, or pseudonymization
* Logged transformations to maintain an **audit trail** for risk mitigation

**Output:** Python notebook or script + blog post explaining the approach and potential privacy risks

---

## Technologies Used

* **Python:** `pandas`, `pyarrow`, `SQLAlchemy` for ETL and transformations
* **AWS S3:** Storage for raw and processed CSV datasets
* **AWS Glue:** Optional ETL orchestration and metadata cataloging
* **AWS RDS (PostgreSQL):** Structured storage for cleaned data
* **Python Logging & AWS CloudWatch:** Track data transformations and errors
* **Visualization (optional):** Matplotlib, Seaborn for inspection of transformed data

**Alternative Tools / Platforms:**

* **ETL Orchestration:** Apache Airflow, Databricks
* **Cloud Platforms:** GCP BigQuery, Azure Data Factory + SQL Database
* **Privacy & PII:** Presidio, Azure Purview, GCP DLP

---

## Strategic Alignment with Governance Roadmap

**Data Foundations Matter**

* Built **traceable ETL pipelines** from raw CSV → Glue Catalog → PostgreSQL
* Demonstrated **repeatable workflows** for clean, queryable data

**Lineage & Observability**

* Documented **data flow** across tables: `People → Batting → Teams → Salaries`
* Logged all transformation steps for reproducibility and auditability

**Security & Privacy by Design**

* Masked sensitive fields to simulate handling of PII in real-world scenarios
* Demonstrated patterns applicable to sensitive healthcare or financial datasets

**Trust as a Product Feature**

* Provided **logs, error exports, and audit-ready reports**
* Made **data quality and privacy compliance** visible, actionable, and repeatable

**Enterprise + Public Perspective**

* **AWS-native path:** S3 + Glue + RDS + CloudWatch for full enterprise workflow
* **Local / public path:** Python scripts + SQLite/PostgreSQL for sharing or demonstration



