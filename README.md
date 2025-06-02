# Databricks2S3-Pipeline
A streaming pipeline that chunks transaction data from Databricks every second and uploads to AWS S3. It concurrently ingests these chunks for real-time pattern detection on transactions. Detected insights are batched back to S3. Checkpointing ensures fault tolerance and state management.
