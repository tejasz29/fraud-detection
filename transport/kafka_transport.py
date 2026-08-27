"""Kafka implementation of :class:`Transport`."""
from __future__ import annotations

import json
import logging
import time
from typing import Iterator

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaTimeoutError

from transport import Transport

logger = logging.getLogger(__name__)


class KafkaTransport(Transport):
    """Wraps ``kafka-python`` producer/consumer with connect-and-retry semantics."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "fraud-detection-group") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._producer: KafkaProducer | None = None
        self._consumer: KafkaConsumer | None = None

    def connect(self) -> None:
        self._connect_producer()
        self._connect_consumer()

    def _connect_producer(self) -> None:
        while True:
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=3,
                )
                logger.info("Connected to Kafka at %s", self.bootstrap_servers)
                return
            except (KafkaTimeoutError, OSError):
                logger.warning("Kafka not available at %s — retrying in 3s", self.bootstrap_servers)
                time.sleep(3)

    def _connect_consumer(self) -> None:
        while True:
            try:
                self._consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset="earliest",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    enable_auto_commit=True,
                    consumer_timeout_ms=1000,
                )
                logger.info("Connected to Kafka at %s (topic=%s)", self.bootstrap_servers, self.topic)
                return
            except (KafkaTimeoutError, OSError):
                logger.warning("Kafka not available — retrying in 3s")
                time.sleep(3)

    def send(self, transaction: dict) -> None:
        assert self._producer is not None
        self._producer.send(self.topic, value=transaction)

    def flush(self) -> None:
        if self._producer is not None:
            self._producer.flush()

    def receive(self, should_stop) -> Iterator[dict]:
        assert self._consumer is not None
        while not should_stop():
            batch = self._consumer.poll(timeout_ms=500)
            for messages in batch.values():
                for message in messages:
                    if should_stop():
                        return
                    yield message.value

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close()
        if self._consumer is not None:
            self._consumer.close()
