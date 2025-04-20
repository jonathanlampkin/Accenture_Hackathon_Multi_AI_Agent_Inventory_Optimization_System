"""
Messaging utilities for RabbitMQ integration.
"""
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

import pika
from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker

logger = logging.getLogger(__name__)

# Default RabbitMQ connection parameters
DEFAULT_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
DEFAULT_USERNAME = os.environ.get("RABBITMQ_USERNAME", "guest")
DEFAULT_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "guest")

# Queue names
INVENTORY_UPDATES_QUEUE = "inventory_updates"
FORECAST_REQUESTS_QUEUE = "forecast_requests"
FORECAST_RESULTS_QUEUE = "forecast_results"
ALERTS_QUEUE = "inventory_alerts"

class RabbitMQClient:
    """Client for interacting with RabbitMQ."""
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        exchange: str = "inventory_exchange",
    ):
        """Initialize RabbitMQ client.
        
        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            exchange: Exchange to use
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange = exchange
        self.connection = None
        self.channel = None
        self._connect()
        
    def _connect(self) -> None:
        """Establish connection to RabbitMQ server."""
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare exchange
            self.channel.exchange_declare(
                exchange=self.exchange,
                exchange_type="topic",
                durable=True,
            )
            
            # Declare queues
            queues = [
                INVENTORY_UPDATES_QUEUE,
                FORECAST_REQUESTS_QUEUE,
                FORECAST_RESULTS_QUEUE,
                ALERTS_QUEUE,
            ]
            
            for queue in queues:
                self.channel.queue_declare(queue=queue, durable=True)
                self.channel.queue_bind(
                    exchange=self.exchange,
                    queue=queue,
                    routing_key=queue,
                )
                
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
            
    def publish(self, routing_key: str, message: Dict[str, Any]) -> bool:
        """Publish message to RabbitMQ.
        
        Args:
            routing_key: Routing key to use
            message: Message to publish
            
        Returns:
            bool: True if message was published successfully
        """
        try:
            if not self.connection or self.connection.is_closed:
                self._connect()
                
            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type="application/json",
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
            
    def consume(self, queue: str, callback: Callable) -> None:
        """Start consuming messages from a queue.
        
        Args:
            queue: Queue to consume from
            callback: Callback function to process messages
        """
        try:
            if not self.connection or self.connection.is_closed:
                self._connect()
                
            self.channel.basic_consume(
                queue=queue,
                on_message_callback=callback,
                auto_ack=False,
            )
            logger.info(f"Started consuming from queue '{queue}'")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.channel.stop_consuming()
        except Exception as e:
            logger.error(f"Error while consuming: {e}")
            
    def close(self) -> None:
        """Close connection to RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("Closed RabbitMQ connection")
            
class BackgroundConsumer(threading.Thread):
    """Background thread for consuming messages from RabbitMQ."""
    
    def __init__(
        self,
        queue: str,
        callback: Callable,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        exchange: str = "inventory_exchange",
    ):
        """Initialize background consumer.
        
        Args:
            queue: Queue to consume from
            callback: Callback function to process messages
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            exchange: Exchange to use
        """
        super().__init__(daemon=True)
        self.queue = queue
        self.callback = callback
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange = exchange
        self.should_run = True
        
    def run(self) -> None:
        """Run consumer thread."""
        while self.should_run:
            try:
                client = RabbitMQClient(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    exchange=self.exchange,
                )
                client.consume(self.queue, self.callback)
            except AMQPConnectionError:
                logger.error("Connection to RabbitMQ lost. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in consumer thread: {e}")
                time.sleep(5)
                
    def stop(self) -> None:
        """Stop consumer thread."""
        self.should_run = False

# Sample message handlers
def handle_inventory_update(
    ch: Any, method: Any, properties: Any, body: bytes
) -> None:
    """Handle inventory update message.
    
    Args:
        ch: Channel
        method: Method
        properties: Properties
        body: Message body
    """
    try:
        message = json.loads(body)
        logger.info(f"Received inventory update: {message}")
        
        # Process the update (implement business logic here)
        product_id = message.get("product_id")
        new_quantity = message.get("new_quantity")
        
        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
        logger.info(f"Processed inventory update for product {product_id}")
    except Exception as e:
        logger.error(f"Error processing inventory update: {e}")
        # Reject message and requeue
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
def handle_forecast_request(
    ch: Any, method: Any, properties: Any, body: bytes
) -> None:
    """Handle forecast request message.
    
    Args:
        ch: Channel
        method: Method
        properties: Properties
        body: Message body
    """
    try:
        message = json.loads(body)
        logger.info(f"Received forecast request: {message}")
        
        # Process the forecast request (implement business logic here)
        product_id = message.get("product_id")
        horizon = message.get("horizon", 30)
        
        # TODO: Generate forecast
        
        # Publish result to forecast results queue
        result = {
            "product_id": product_id,
            "horizon": horizon,
            "forecast": [100, 110, 120],  # Sample forecast
            "timestamp": time.time(),
        }
        
        client = RabbitMQClient()
        client.publish(FORECAST_RESULTS_QUEUE, result)
        
        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
        logger.info(f"Processed forecast request for product {product_id}")
    except Exception as e:
        logger.error(f"Error processing forecast request: {e}")
        # Reject message and requeue
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True) 