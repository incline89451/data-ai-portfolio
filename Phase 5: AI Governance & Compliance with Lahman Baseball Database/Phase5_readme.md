## Phase 5: AI Governance & Compliance

- **Phase 5** focuses on **model drift detection, incident response, and operational governance** for AI systems.

### Project 9: Model Drift Detection & Alerts – Baseball Models

- **Objective:** Monitor player/team prediction models for drift  
- **Approach:**  
  - Simulate new player data or updated stats  
  - Detect deviations between predicted and actual outcomes  
  - Trigger automated alerts on performance drift  
- **Output:** Jupyter notebook + dashboard + blog post detailing drift detection and response  

### Project 10: AI Governance Playbook / Incident Response – Baseball Analytics

- **Objective:** Document structured response to model/data failures  
- **Approach:**  
  - Simulate misreported player stats or prediction errors  
  - Step-by-step mitigation, logging, and communication plan  
  - Establish governance processes aligned with AI best practices  
- **Output:** Markdown playbook + diagram + blog post documenting incident response  

### Technologies Used

- **Python:** `pandas`, `evidently`, `scikit-multiflow`  
- **Monitoring & Alerts:** AWS Lambda + SNS, Streamlit dashboards  
- **Visualization:** Plotly, Matplotlib  
- **Documentation:** Markdown, Mermaid, Draw.io  
- **Logging & Audit:** AWS CloudWatch, S3, Athena  

**Alternatives:** Alibi Detect, Databricks Feature Store, Confluence / Notion

### Strategic Alignment

- **Data Foundations:** Used structured datasets from Phases 1–2  
- **Lineage & Observability:** Logged data transformations, model predictions, drift metrics  
- **Security & Privacy:** Maintained anonymization and logging  
- **Trust as a Product Feature:** Alerts and dashboards make governance actionable and visible  
- **Enterprise + Public Perspective:** AWS-native path (CloudWatch + Lambda + SNS + S3 + Athena) and Python/Markdown/Streamlit path for public demo

