# Databricks notebook source
from pyspark.sql import SparkSession
from datetime import datetime
import time
import json
import os
from pyspark.sql.functions import col, count, avg, percentile_approx
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize SparkSession
spark = SparkSession.builder \
    .master('local') \
    .appName("Assignment") \
    .getOrCreate()

# Configure AWS S3 Access
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "AKIA4CLY4WOVEXQWDD7Y")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "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")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")

# Configuration
TRANSACTIONS_FILE = "/FileStore/AssignmentFile/transactions.csv"
CUSTOMER_IMPORTANCE_FILE = "/FileStore/AssignmentFile/CustomerImportance.csv"
S3_CHUNK_FOLDER = "s3a://databricks-assignment1/chunks/"
S3_DETECT_FOLDER = "s3a://databricks-assignment1/detections/"
S3_CHECKPOINT_X = "s3a://databricks-assignment1/checkpoints/x_checkpoint.json"
S3_CHECKPOINT_Y = "s3a://databricks-assignment1/checkpoints/y_checkpoint.json"
CHUNK_SIZE = 10000

# Checkpoint Functions
def load_checkpoint(s3_path, default_key="last_chunk_index"):
    try:
        df = spark.read.text(s3_path)
        checkpoint_data = json.loads(df.collect()[0]["value"])
        return checkpoint_data
    except Exception:
        return {
            default_key: -1,
            "last_updated": "",
            "chunk_size": CHUNK_SIZE,
            "total_chunks_processed": 0
        }

def update_checkpoint(s3_path, key, value, metadata):
    checkpoint_data = {
        key: value,
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "chunk_size": CHUNK_SIZE,
        "total_chunks_processed": value + 1
    }
    checkpoint_data.update(metadata)
    spark.createDataFrame([(json.dumps(checkpoint_data),)], ["value"]) \
        .write.mode("overwrite").text(s3_path)

# Pattern Detection Function
def detect_patterns(chunk_file_path):
    df_chunk = spark.read.option("header", True).csv(chunk_file_path)
    df_customer_importance = spark.read.csv(CUSTOMER_IMPORTANCE_FILE, header=True, inferSchema=True)
    df_transaction = spark.read.option("header", True).csv(TRANSACTIONS_FILE)
    
    detections = []
    y_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # PatId1: Top 1% customers by transaction count with bottom 1% weight, merchant >50K transactions
    merchant_tx_counts = df_transaction.groupBy("merchant_id").agg(count("*").alias("tx_count"))
    eligible_merchants = merchant_tx_counts.filter(col("tx_count") > 50000).select("merchant_id")
    
    if eligible_merchants.count() > 0:
        df_eligible = df_chunk.join(eligible_merchants, "merchant_id")
        customer_counts = df_eligible.groupBy("merchant_id", "customer_id").agg(count("*").alias("customer_tx_count"))
        
        merchant_percentiles = customer_counts.groupBy("merchant_id").agg(
            percentile_approx("customer_tx_count", 0.99).alias("top_1_percentile")
        )
        
        customer_weights = df_customer_importance.groupBy("customer_id").agg(
            percentile_approx("weight", 0.01).alias("bottom_1_percentile_weight")
        )
        
        pat1_candidates = customer_counts.join(merchant_percentiles, "merchant_id") \
            .join(customer_weights, "customer_id") \
            .filter(col("customer_tx_count") >= col("top_1_percentile")) \
            .filter(col("bottom_1_percentile_weight").isNotNull())
        
        for row in pat1_candidates.collect():
            detections.append({
                "YStartTime": y_start_time,
                "detectionTime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "patternId": "PatId1",
                "ActionType": "UPGRADE",
                "customerName": row["customer_id"],
                "MerchantId": row["merchant_id"]
            })
    
    # PatId2: Avg transaction value < Rs 23 and >= 80 transactions
    customer_stats = df_chunk.groupBy("merchant_id", "customer_id").agg(
        avg("transaction_amount").alias("avg_tx_amount"),
        count("*").alias("tx_count")
    )
    pat2_candidates = customer_stats.filter((col("avg_tx_amount") < 23) & (col("tx_count") >= 80))
    
    for row in pat2_candidates.collect():
        detections.append({
            "YStartTime": y_start_time,
            "detectionTime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "patternId": "PatId2",
            "ActionType": "CHILD",
            "customerName": row["customer_id"],
            "MerchantId": row["merchant_id"]
        })
    
    # PatId3: Female customers < Male customers and female customers > 100
    gender_counts = df_chunk.groupBy("merchant_id").pivot("gender").agg(count("*").alias("count"))
    gender_counts = gender_counts.fillna(0)
    pat3_candidates = gender_counts.filter(
        (col("Female") < col("Male")) & (col("Female") > 100)
    )
    
    for row in pat3_candidates.collect():
        detections.append({
            "YStartTime": y_start_time,
            "detectionTime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "patternId": "PatId3",
            "ActionType": "DEI-NEEDED",
            "customerName": "",
            "MerchantId": row["merchant_id"]
        })
    
    return detections

# Mechanism X: Chunk Transactions and Upload to S3
def mechanism_x():
    df_transaction = spark.read.option("header", True).csv(TRANSACTIONS_FILE)
    total_count = df_transaction.count()
    total_chunks = (total_count // CHUNK_SIZE) + 1
    
    checkpoint_x = load_checkpoint(S3_CHECKPOINT_X)
    start_idx = checkpoint_x["last_chunk_index"] + 1
    
    for i in range(start_idx, total_chunks):
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        df_chunk = df_transaction.limit(end).subtract(df_transaction.limit(start))
        
        s3_path = f"{S3_CHUNK_FOLDER}chunk_{i}.csv"
        df_chunk.write.mode("overwrite").csv(s3_path, header=True)
        print(f"[{start_time}] Wrote chunk {i} to {s3_path}")
        
        update_checkpoint(S3_CHECKPOINT_X, "last_chunk_index", i, {"start_time": start_time})
        time.sleep(1)

# Mechanism Y: Process Chunks and Write Detections
def mechanism_y():
    checkpoint_y = load_checkpoint(S3_CHECKPOINT_Y)
    start_chunk = checkpoint_y["last_chunk_index"] + 1
    
    while True:
        chunk_file_path = f"{S3_CHUNK_FOLDER}chunk_{start_chunk}.csv"
        try:
            detections = detect_patterns(chunk_file_path)
            
            for i in range(0, len(detections), 50):
                chunk = detections[i:i+50]
                det_df = spark.createDataFrame(chunk, schema=StructType([
                    StructField("YStartTime", StringType(), True),
                    StructField("detectionTime", StringType(), True),
                    StructField("patternId", StringType(), True),
                    StructField("ActionType", StringType(), True),
                    StructField("customerName", StringType(), True),
                    StructField("MerchantId", StringType(), True)
                ]))
                det_df.write.mode("overwrite").option("header", True).csv(
                    f"{S3_DETECT_FOLDER}det_{start_chunk}_{i//50}.csv"
                )
            
            update_checkpoint(S3_CHECKPOINT_Y, "last_chunk_index", start_chunk, {})
            print(f"Processed chunk {start_chunk}")
            start_chunk += 1
            time.sleep(1)
        except Exception as e:
            print(f"No more chunks found. Halting at chunk {start_chunk}: {e}")
            break

# Run Mechanisms X and Y (can be run in separate notebooks or cells)
if __name__ == "__main__":
    from threading import Thread
    
    x_thread = Thread(target=mechanism_x)
    y_thread = Thread(target=mechanism_y)
    
    x_thread.start()
    y_thread.start()
    
    x_thread.join()
    y_thread.join()
