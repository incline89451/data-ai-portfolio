# Phase 1: SQL & Data Exploration with Lahman Baseball Database

This repository is part of my **10-Project Data & AI Governance Roadmap**, where I demonstrate how **governance, trust, and compliance** come together across the AI lifecycle.  
Phase 1 focuses on **data foundations** — making raw data queryable, exploring it with SQL, validating quality, and exposing insights through dashboards.  

---

## Project Overview

### Project 1: SQL Analytics
Using the **Lahman Baseball Database (1871–2023)**, I explored trends and insights in batting, pitching, and team performance.  
- **SQL Queries**: Joins, aggregations, window functions, and CTEs to analyze performance across eras.  
- **Examples**:  
  - Evolution of home runs and ERA by decade  
  - Career progression of Hall of Famers  
  - Wins vs. runs scored for teams  
- **Output**: SQL scripts + sample query results  

### Project 2: Data Quality Checks
Before any AI/ML work, **trust starts with data quality**. I designed validation rules for Lahman data, including:  
- Null checks (`birthYear` in People table)  
- Range checks (batting averages ≤ 1.000, ages < 120)  
- Uniqueness checks (`playerID + yearID + stint` in Batting)  
- Referential integrity (teamIDs in Batting exist in Teams)  
- Historical plausibility (no debut before 1871, none after 2023)  

**Validation implemented in:**  
- **Python + Pandas**: lightweight scripts for profiling & exporting bad rows  
- **AWS Glue**: schema inference & metadata cataloging  
- **Athena SQL**: rules run as queries against structured tables  

---

## Technologies Used

- **SQL** (Athena, PostgreSQL, SQLite): Data exploration, joins, aggregations  
- **AWS S3**: Storage for CSV datasets  
- **AWS Glue**: Crawlers + Data Catalog for schema management  
- **AWS Athena**: Serverless SQL queries against S3 data  
- **Python (Pandas, PyArrow)**: Data quality validation and error exports  
- **Tableau Public**: Visualization of analytics results and data quality dashboards  

---

## Strategic Alignment with Governance Roadmap

This phase maps to my **strategic goals for AI Governance**:

1. **Data Foundations Matter**  
   - Built queryable datasets from raw CSV → Glue Catalog → Athena (enterprise flow)  
   - Also demonstrated local workflows with PostgreSQL & Python (public-friendly)  

2. **Lineage & Observability**  
   - Documented table origins and joins (People → Batting → Teams)  
   - Established traceable queries for reproducibility  

3. **Security & Privacy by Design**  
   - Modeled IAM-controlled S3 access and Glue schema governance  
   - Demonstrated patterns that scale to sensitive healthcare/financial data  

4. **Trust as a Product Feature**  
   - Created **data quality rules** and visualized results in Tableau  
   - Made “trustworthiness” a visible deliverable, not a hidden process  

5. **Enterprise + Public Perspective**  
   - **AWS-native path** (Athena, Glue, S3, Tableau Desktop)  
   - **Public path** (Python, CSVs, Tableau Public for sharing dashboards)  



