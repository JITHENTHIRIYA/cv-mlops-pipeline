import time
import logging
import boto3
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, namespace="CVMLOpsMetrics"):
        self.namespace = namespace
        self.start_time = None
        
        # Initialize CloudWatch client (will use AWS credentials from environment)
        try:
            self.cloudwatch = boto3.client('cloudwatch')
            logger.info("Successfully initialized CloudWatch client")
        except Exception as e:
            logger.warning(f"Failed to initialize CloudWatch client: {str(e)}")
            self.cloudwatch = None
    
    def start_timer(self):
        """Start a timer for measuring execution time"""
        self.start_time = time.time()
    
    def record_latency(self, endpoint_name):
        """Record API latency metric"""
        if self.start_time is None:
            logger.warning("Timer was not started")
            return
        
        latency = (time.time() - self.start_time) * 1000  # Convert to ms
        logger.info(f"Latency for {endpoint_name}: {latency:.2f} ms")
        
        if self.cloudwatch:
            try:
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=[
                        {
                            'MetricName': 'APILatency',
                            'Dimensions': [
                                {
                                    'Name': 'Endpoint',
                                    'Value': endpoint_name
                                },
                            ],
                            'Value': latency,
                            'Unit': 'Milliseconds',
                            'Timestamp': datetime.now()
                        },
                    ]
                )
            except Exception as e:
                logger.error(f"Failed to put metrics data: {str(e)}")
        
        self.start_time = None
    
    def record_prediction_count(self, prediction_type):
        """Record count of prediction requests by type"""
        if self.cloudwatch:
            try:
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=[
                        {
                            'MetricName': 'PredictionCount',
                            'Dimensions': [
                                {
                                    'Name': 'Type',
                                    'Value': prediction_type
                                },
                            ],
                            'Value': 1,
                            'Unit': 'Count',
                            'Timestamp': datetime.now()
                        },
                    ]
                )
            except Exception as e:
                logger.error(f"Failed to put metrics data: {str(e)}")