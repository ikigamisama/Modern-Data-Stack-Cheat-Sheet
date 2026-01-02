import streamlit as st

st.set_page_config(
    page_title="Kafka in Python Guide and Tips",
    page_icon="📡",
    layout="wide"
)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f2937;
    margin-bottom: 1rem;
}
.section-header {
    font-size: 1.8rem;
    font-weight: bold;
    color: #374151;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.subsection-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #4b5563;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Light mode */
[data-theme="light"] .info-box {
    background-color: #f8fafc;
    color: #0f172a;
    border-left: 4px solid #2563eb;
}

/* Dark mode */
[data-theme="dark"] .info-box {
    background-color: #0f172a;
    color: #e5e7eb;
    
.info-box-blue {
    background-color: #eff6ff;
    border-left: 4px solid #3b82f6;
}
.info-box-green {
    background-color: #f0fdf4;
    border-left: 4px solid #22c55e;
}
.info-box-yellow {
    background-color: #fefce8;
    border-left: 4px solid #eab308;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Pages:",
    [
        "🏠 Overview",
        "📡 Data Streaming",
        "📄 Log Aggregation",
        "📬 Message Queuing",
        "🌐 Web Activity Tracking",
        "🔁 Data Replication",
        "⚠️ Common Issues"
    ],
)
st.markdown('# 📊 Apache Kafka Learning Hub and Tips',
            unsafe_allow_html=True)
if page == "🏠 Overview":
    st.markdown('#### What is Apache Kafka?',
                unsafe_allow_html=True)

    st.markdown("""
    **Definition:** Apache Kafka is a distributed event streaming platform designed to handle 
    high-throughput, fault-tolerant, real-time data feeds. It acts as a message broker that enables 
    applications to publish, subscribe to, store, and process streams of records in a distributed manner.
    """)

    st.markdown('<div class="info-box info-box-blue"><b>Core Purpose:</b><ul><li><b>Decouple systems:</b> Enables independent scaling of data producers and consumers</li><li><b>Real-time processing:</b> Handles streaming data with low latency</li><li><b>Durability:</b> Persists messages to disk for fault tolerance</li><li><b>Scalability:</b> Distributes data across multiple servers (brokers)</li></ul></div>', unsafe_allow_html=True)
    st.markdown("### Key Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Topic**")
        st.write("A category or feed name to which messages are published")

        st.markdown("**🔀 Partition**")
        st.write("Ordered, immutable sequence of messages within a topic")

        st.markdown("**📤 Producer**")
        st.write("Application that publishes messages to Kafka topics")

    with col2:
        st.markdown("**📥 Consumer**")
        st.write("Application that subscribes to topics and processes messages")

        st.markdown("**🖥️ Broker**")
        st.write("Kafka server that stores and serves data")

        st.markdown("**👥 Consumer Group**")
        st.write("Set of consumers working together to consume a topic")

    # Installation
    st.markdown('### ⚙️ Installation',
                unsafe_allow_html=True)

    st.markdown(
        "Install the Python client library for interacting with Apache Kafka clusters.")

    st.code("pip install kafka-python", language="bash")

    st.info("💡 This library provides a Pythonic interface to Apache Kafka")

    # Producer Basics
    st.markdown('### 📤 Basic Producer',  unsafe_allow_html=True)
    st.markdown('<div class="info-box info-box-blue"><b>What is a Producer?</b><br>A Kafka Producer is a client application that publishes (writes) messages to Kafka topics. It determines which partition to send messages to and handles batching, compression, and retries.</div>', unsafe_allow_html=True)
    st.markdown('#### Simple Producer', unsafe_allow_html=True)

    st.code("""from kafka import KafkaProducer
    import json

    # Create producer
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Send message
    producer.send('my-topic', {'key': 'value'})

    # Flush and close
    producer.flush()
    producer.close()""", language="python")

    with st.expander("📖 Parameters Explained"):
        st.markdown("""
        - **bootstrap_servers:** List of Kafka broker addresses for initial connection
        - **value_serializer:** Function to convert message values to bytes
        - **flush():** Blocks until all pending messages are sent
        - **close():** Closes producer and releases resources
        """)

    st.markdown('#### Producer with Key',
                unsafe_allow_html=True)

    st.code("""producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        key_serializer=lambda k: k.encode('utf-8'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    producer.send('my-topic', key='my-key', value={'data': 'example'})""", language="python")

    st.markdown('<div class="info-box info-box-green"><b>Purpose of Message Keys:</b><ul><li><b>Partitioning:</b> Messages with the same key go to the same partition</li><li><b>Ordering:</b> Guarantees message order within a partition</li><li><b>Co-location:</b> Related messages are stored together</li></ul></div>', unsafe_allow_html=True)

    st.markdown('#### Producer with Callback',
                unsafe_allow_html=True)

    st.code("""def on_success(metadata):
        print(f"Message sent to {metadata.topic} partition {metadata.partition} offset {metadata.offset}")

    def on_error(e):
        print(f"Error: {e}")

    future = producer.send('my-topic', {'message': 'hello'})
    future.add_callback(on_success)
    future.add_errback(on_error)""", language="python")

    st.info("🎯 Handle asynchronous send results for monitoring and error handling")

    # Consumer Basics
    st.markdown('### 📥 Basic Consumer',
                unsafe_allow_html=True)

    st.markdown('<div class="info-box info-box-blue"><b>What is a Consumer?</b><br>A Kafka Consumer is a client application that subscribes to (reads) messages from Kafka topics. It maintains an offset (position) in each partition to track which messages have been processed.</div>', unsafe_allow_html=True)

    st.markdown('#### Simple Consumer',
                unsafe_allow_html=True)

    st.code("""from kafka import KafkaConsumer
    import json

    consumer = KafkaConsumer(
        'my-topic',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    for message in consumer:
        print(f"Topic: {message.topic}")
        print(f"Partition: {message.partition}")
        print(f"Offset: {message.offset}")
        print(f"Key: {message.key}")
        print(f"Value: {message.value}")""", language="python")

    with st.expander("📖 Parameters Explained"):
        st.markdown("""
        **auto_offset_reset:** Where to start reading when no offset exists
        - `'earliest'`: Start from beginning of partition
        - `'latest'`: Start from newest messages
        - `'none'`: Throw exception if no offset found
        
        **enable_auto_commit:** Automatically commit offsets periodically
        
        **group_id:** Consumer group name for coordinated consumption
        
        **value_deserializer:** Function to convert message bytes to objects
        """)

    st.markdown('#### Consumer with Multiple Topics', unsafe_allow_html=True)

    st.code("""consumer = KafkaConsumer(
        'topic1', 'topic2', 'topic3',
        bootstrap_servers=['localhost:9092'],
        group_id='my-group'
    )""", language="python")

    st.markdown('#### Manual Offset Commit', unsafe_allow_html=True)

    st.code("""consumer = KafkaConsumer(
        'my-topic',
        bootstrap_servers=['localhost:9092'],
        enable_auto_commit=False,
        group_id='my-group'
    )

    for message in consumer:
        # Process message
        print(message.value)
        # Manually commit
        consumer.commit()""", language="python")

    st.markdown('<div class="info-box info-box-yellow"><b>When to use manual commits:</b><ul><li>Need to ensure message processing before committing</li><li>Implementing transactional processing</li><li>Custom error handling and retry logic</li></ul></div>', unsafe_allow_html=True)

    # Configuration
    st.markdown('### 🔧 Configuration Options',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Producer Configuration", "Consumer Configuration"])

    with tab1:
        st.markdown(
            '#### Producer Configuration', unsafe_allow_html=True)

        st.code("""producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        acks='all',  # 0, 1, or 'all'
        retries=3,
        batch_size=16384,
        linger_ms=10,
        buffer_memory=33554432,
        compression_type='gzip',  # None, 'gzip', 'snappy', 'lz4', 'zstd'
        max_in_flight_requests_per_connection=5
    )""", language="python")

        st.markdown("**Key Parameters:**")

        st.markdown("""
        **acks (Acknowledgment Level):**
        - `0`: No acknowledgment (fastest, least safe)
        - `1`: Leader broker acknowledges (balanced)
        - `'all'`: All replicas acknowledge (slowest, most safe)
        
        **retries:** Number of retry attempts for failed sends
        
        **batch_size:** Maximum bytes to batch before sending (default: 16384)
        
        **linger_ms:** Time to wait before sending batch (default: 0)
        
        **compression_type:** Compress messages to reduce network bandwidth
        - `'gzip'`: Best compression, slower
        - `'snappy'`: Balanced compression/speed
        - `'lz4'`: Fast compression
        - `'zstd'`: Modern, efficient compression
        """)

    with tab2:
        st.markdown(
            '#### Consumer Configuration', unsafe_allow_html=True)

        st.code("""consumer = KafkaConsumer(
        'my-topic',
        bootstrap_servers=['localhost:9092'],
        group_id='my-group',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        auto_commit_interval_ms=5000,
        max_poll_records=500,
        max_poll_interval_ms=300000,
        session_timeout_ms=10000,
        heartbeat_interval_ms=3000
    )""", language="python")

        st.markdown("**Key Parameters:**")

        st.markdown("""
        **group_id:** Consumer group identifier for load balancing
        
        **auto_commit_interval_ms:** Frequency of automatic offset commits (default: 5000)
        
        **max_poll_records:** Maximum records returned per poll() call (default: 500)
        
        **max_poll_interval_ms:** Maximum time between poll() calls (default: 300000 = 5 min)
        
        **session_timeout_ms:** Maximum time between heartbeats (default: 10000 = 10 sec)
        
        **heartbeat_interval_ms:** Frequency of heartbeat to coordinator (default: 3000 = 3 sec)
        """)

    # Advanced Operations
    st.markdown('### 🚀 Advanced Operations',  unsafe_allow_html=True)
    st.markdown('#### Seek to Specific Offset', unsafe_allow_html=True)

    st.code("""from kafka import TopicPartition

    tp = TopicPartition('my-topic', 0)
    consumer.assign([tp])
    consumer.seek(tp, 10)  # Seek to offset 10""", language="python")

    st.markdown('<div class="info-box info-box-green"><b>Use Cases:</b><ul><li>Reprocessing historical data</li><li>Skipping corrupted messages</li><li>Time-travel debugging</li><li>Implementing custom offset management</li></ul></div>', unsafe_allow_html=True)

    st.markdown('#### Seek to Beginning/End', unsafe_allow_html=True)

    st.code("""consumer.seek_to_beginning()
    consumer.seek_to_end()""", language="python")

    st.markdown('#### Get Topic Partitions', unsafe_allow_html=True)

    st.code("""partitions = consumer.partitions_for_topic('my-topic')
    print(f"Partitions: {partitions}")""", language="python")

    st.markdown('#### Pause and Resume',  unsafe_allow_html=True)

    st.code("""# Pause consumption
    consumer.pause(*consumer.assignment())

    # Resume consumption
    consumer.resume(*consumer.assignment())""", language="python")

    st.info("🎯 Useful for backpressure handling and dynamic rate limiting")

    # Admin Operations
    st.markdown('### 👨‍💼 Admin Operations', unsafe_allow_html=True)

    st.markdown("Administrative operations for managing Kafka cluster resources like topics, partitions, and configurations programmatically.")

    st.markdown('#### Create Topics', unsafe_allow_html=True)

    st.code("""from kafka.admin import KafkaAdminClient, NewTopic

    admin = KafkaAdminClient(bootstrap_servers=['localhost:9092'])

    topic = NewTopic(
        name='new-topic',
        num_partitions=3,
        replication_factor=1
    )

    admin.create_topics([topic])
    admin.close()""", language="python")

    with st.expander("📖 Parameters Explained"):
        st.markdown("""
        - **num_partitions:** Number of partitions (parallelism level)
        - **replication_factor:** Number of copies across brokers (fault tolerance)
        
        **Best Practices:**
        - More partitions = higher throughput but more overhead
        - Replication factor ≥ 3 for production
        - Consider retention policies and compaction
        """)

    st.markdown('#### Delete Topics', unsafe_allow_html=True)
    st.code("""admin.delete_topics(['topic-to-delete'])""", language="python")
    st.warning("⚠️ Use with caution in production - this removes all data")
    st.markdown('#### List Topics',  unsafe_allow_html=True)

    st.code("""topics = admin.list_topics()
    print(f"Available topics: {topics}")""", language="python")

    # Error Handling
    st.markdown('### ⚠️ Error Handling', unsafe_allow_html=True)

    st.markdown('<div class="info-box info-box-yellow"><b>Why Error Handling Matters:</b><br>Kafka operations involve network I/O and can fail due to broker issues, network problems, or configuration errors. Proper error handling ensures system reliability.</div>', unsafe_allow_html=True)

    st.markdown('#### Producer Error Handling', unsafe_allow_html=True)

    st.code("""from kafka.errors import KafkaError

    try:
        future = producer.send('my-topic', {'data': 'value'})
        record_metadata = future.get(timeout=10)
    except KafkaError as e:
        print(f"Failed to send message: {e}")""", language="python")

    st.markdown("**Common Errors:**")
    st.markdown("""
    - `KafkaTimeoutError`: Broker unreachable or overloaded
    - `MessageSizeTooLargeError`: Message exceeds broker limits
    - `TopicAuthorizationFailedError`: Insufficient permissions
    """)

    st.markdown('#### Consumer Error Handling', unsafe_allow_html=True)

    st.code("""from kafka.errors import KafkaError

    try:
        for message in consumer:
            try:
                # Process message
                process(message.value)
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
    except KafkaError as e:
        print(f"Kafka error: {e}")""", language="python")

    st.markdown('<div class="info-box info-box-green"><b>Best Practices:</b><ul><li>Separate Kafka errors from processing errors</li><li>Implement dead letter queues for poison messages</li><li>Log errors with context (topic, partition, offset)</li><li>Monitor error rates</li></ul></div>', unsafe_allow_html=True)

    # Common Patterns
    st.markdown('### 🎯 Common Patterns', unsafe_allow_html=True)

    st.markdown('#### Batch Processing', unsafe_allow_html=True)

    st.code("""messages = []
    for message in consumer:
        messages.append(message)
        if len(messages) >= 100:
            # Process batch
            process_batch(messages)
            messages = []
            consumer.commit()""", language="python")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Benefits:**")
        st.markdown("""
        - Amortize network/database overhead
        - Enable bulk operations
        - Reduce per-message cost
        """)

    with col2:
        st.markdown("**Trade-offs:**")
        st.markdown("""
        - Increased latency
        - Higher memory usage
        - Risk of data loss
        """)

    st.markdown('#### Asynchronous Producer',
                unsafe_allow_html=True)

    st.code("""import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def send_async(producer, topic, message):
        producer.send(topic, message)

    executor = ThreadPoolExecutor(max_workers=10)
    for i in range(1000):
        executor.submit(send_async, producer, 'my-topic', {'id': i})""", language="python")

    st.info("🚀 Maximize producer throughput by parallelizing sends")

    st.markdown('#### Context Manager',
                unsafe_allow_html=True)

    st.code("""from contextlib import closing

    with closing(KafkaProducer(bootstrap_servers=['localhost:9092'])) as producer:
        producer.send('my-topic', b'message')
        producer.flush()""", language="python")

    st.success("✅ Ensures proper resource cleanup")

    # Security
    st.markdown('### 🔒 Security (SSL/SASL)',  unsafe_allow_html=True)

    st.markdown(
        "Protect data in transit and authenticate clients to prevent unauthorized access.")

    st.markdown('#### SSL Configuration',  unsafe_allow_html=True)

    st.code("""producer = KafkaProducer(
        bootstrap_servers=['localhost:9093'],
        security_protocol='SSL',
        ssl_cafile='/path/to/ca-cert',
        ssl_certfile='/path/to/client-cert',
        ssl_keyfile='/path/to/client-key'
    )""", language="python")

    with st.expander("📖 Certificates Explained"):
        st.markdown("""
        - **ssl_cafile:** Certificate Authority (validates broker identity)
        - **ssl_certfile:** Client certificate (proves client identity)
        - **ssl_keyfile:** Private key for client certificate
        """)

    st.markdown('#### SASL Authentication',  unsafe_allow_html=True)

    st.code("""producer = KafkaProducer(
        bootstrap_servers=['localhost:9093'],
        security_protocol='SASL_SSL',
        sasl_mechanism='PLAIN',
        sasl_plain_username='username',
        sasl_plain_password='password'
    )""", language="python")

    st.markdown("**SASL Mechanisms:**")
    st.markdown("""
    - `PLAIN`: Simple username/password (use with SSL)
    - `SCRAM-SHA-256/512`: Secure password-based authentication
    - `GSSAPI`: Kerberos authentication
    - `OAUTHBEARER`: OAuth 2.0 token-based authentication
    """)

    # Performance Tips
    st.markdown('### ⚡ Performance Tips', unsafe_allow_html=True)

    tips = [
        {
            "title": "Use batch sending for producers",
            "desc": "Set linger_ms=10-50 and batch_size=16384-131072",
            "benefit": "Reduce network overhead by grouping messages"
        },
        {
            "title": "Enable compression",
            "desc": "Use snappy for balanced performance, zstd for best compression",
            "benefit": "Reduce network bandwidth and broker storage"
        },
        {
            "title": "Adjust max_poll_records",
            "desc": "Lower for slow processing, higher for fast processing",
            "benefit": "Prevent consumer timeout from slow processing"
        },
        {
            "title": "Use appropriate acks setting",
            "desc": "acks='all' with min.insync.replicas=2 for critical data",
            "benefit": "Balance throughput vs. data safety"
        },
        {
            "title": "Partition your topics",
            "desc": "Partitions = number of parallel consumers needed",
            "benefit": "Enable horizontal scaling of consumers"
        },
        {
            "title": "Use consumer groups",
            "desc": "Max consumers = number of partitions",
            "benefit": "Distribute partition consumption across consumers"
        },
        {
            "title": "Monitor lag",
            "desc": "Add consumers when lag consistently increases",
            "benefit": "Ensure consumers keep up with producers"
        }
    ]

    for i, tip in enumerate(tips, 1):
        with st.expander(f"💡 Tip {i}: {tip['title']}"):
            st.markdown(f"**Recommendation:** {tip['desc']}")
            st.markdown(f"**Benefit:** {tip['benefit']}")

elif page == "📡 Data Streaming":
    st.header("📡 Data Streaming")
    st.markdown("""
**Definition:** Real-time data streaming aggregates information from multiple sources (social media, IoT devices, applications) through Kafka topics to processing engines like Spark Streaming and storage systems.

**Purpose:**

- Unify disparate data sources into a single pipeline
- Enable real-time analytics and decision-making
- Buffer high-volume data streams
- Decouple data producers from consumers

**Key Benefits:**

- **Scalability**: Handle millions of events per second
- **Fault Tolerance**: No data loss even if processing fails
- **Flexibility**: Multiple consumers can process same stream differently
- **Real-time**: Sub-second latency for time-sensitive applications

**Common Scenarios:**

- IoT sensor data aggregation
- Social media feed processing
- Financial market data distribution
- Application event streaming
""")
    st.markdown(
        "#### Example: Multi-Source IoT & Social Media Stream Aggregator")
    st.code("""import asyncio
from kafka import KafkaProducer, KafkaConsumer
from kafka.partitioner import Murmur2Partitioner
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Any

class StreamAggregator:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            partitioner=Murmur2Partitioner(),
            compression_type='snappy',
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        self._running = False

    def stream_social_media(self, platform, num_messages=100):
        # Simulate streaming from multiple social platforms
        for i in range(num_messages):
            event = {
                'event_id': f'{platform}_{i}_{int(time.time())}',
                'platform': platform,
                'user_id': f'user_{i % 1000}',
                'content': f'Streaming content from {platform}',
                'engagement': {'likes': i * 10, 'shares': i * 2, 'comments': i * 5},
                'timestamp': time.time(),
                'sentiment_score': round(random.uniform(-1, 1), 2),
                'location': {'lat': random.uniform(-90, 90), 'lon': random.uniform(-180, 180)}
            }
            # Use user_id as key for partition consistency
            self.producer.send('social-stream', key=event['user_id'], value=event)
            time.sleep(0.01)

    def stream_iot_sensors(self, device_type, num_readings=100):
        # Simulate IoT sensor data streaming
        for i in range(num_readings):
            reading = {
                'device_id': f'{device_type}_{i % 50}',
                'device_type': device_type,
                'reading': random.uniform(20, 100),
                'battery_level': random.uniform(0, 100),
                'signal_strength': random.randint(-100, -30),
                'timestamp': time.time(),
                'metadata': {'firmware': 'v2.1.3', 'location': 'warehouse_A'}
            }
            self.producer.send('iot-stream', key=reading['device_id'], value=reading)
            time.sleep(0.005)

    def stop(self):
        # Stop the producer
        self.producer.flush()
        self.producer.close()
        self._running = False
""", language='python')
    st.markdown("#### Consumer with real-time processing")
    st.code("""class StreamProcessor:
    def __init__(self, bootstrap_servers, topics):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id='stream-processor-group',
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_poll_records=500,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.batch = []
        self._running = False

    def process_stream(self):
        # Process stream with windowing and aggregation
        window_data = {'social': [], 'iot': []}
        window_start = time.time()
        self._running = True

        try:
            for message in self.consumer:
                if not self._running:
                    break

                # Window-based processing (5 second windows)
                if time.time() - window_start > 5:
                    self.aggregate_and_forward(window_data)
                    window_data = {'social': [], 'iot': []}
                    window_start = time.time()

                if message.topic == 'social-stream':
                    window_data['social'].append(message.value)
                elif message.topic == 'iot-stream':
                    window_data['iot'].append(message.value)

                self.batch.append(message)
                if len(self.batch) >= 100:
                    self.consumer.commit()
                    self.batch = []
        except Exception as e:
            print(f"Error in process_stream: {e}")
        finally:
            self.consumer.close()

    def aggregate_and_forward(self, window_data):
        # Aggregate window data and forward to Spark/analytics
        if window_data['social']:
            avg_sentiment = sum(d['sentiment_score'] for d in window_data['social']) / len(window_data['social'])
            total_events = len(window_data['social'])
            avg_likes = sum(d['engagement']['likes'] for d in window_data['social']) / total_events
            print(f"Window: {total_events} social events, Avg Sentiment: {avg_sentiment:.2f}, Avg Likes: {avg_likes:.1f}")

        if window_data['iot']:
            avg_reading = sum(d['reading'] for d in window_data['iot']) / len(window_data['iot'])
            avg_battery = sum(d['battery_level'] for d in window_data['iot']) / len(window_data['iot'])
            print(f"Window: {len(window_data['iot'])} IoT readings, Avg Reading: {avg_reading:.2f}, Avg Battery: {avg_battery:.2f}")

    def stop(self):
        # Stop the consumer
        self._running = False
        self.consumer.close()""", language='python')

    st.markdown("#### Enhanced multi-threaded processing with error handling")
    st.code("""class EnhancedStreamProcessor(StreamProcessor):
    def __init__(self, bootstrap_servers, topics, worker_threads=4):
        super().__init__(bootstrap_servers, topics)
        self.worker_threads = worker_threads
        self.processing_queue = []
        self.lock = threading.Lock()

    def process_stream_with_threads(self):
        # Process stream using thread pool for better performance
        window_data = {'social': [], 'iot': []}
        window_start = time.time()
        self._running = True

        with ThreadPoolExecutor(max_workers=self.worker_threads) as executor:
            try:
                for message in self.consumer:
                    if not self._running:
                        break

                    # Submit message processing to thread pool
                    if message.topic == 'social-stream':
                        future = executor.submit(self.process_social_message, message.value)
                    elif message.topic == 'iot-stream':
                        future = executor.submit(self.process_iot_message, message.value)

                    # Window-based processing (5 second windows)
                    if time.time() - window_start > 5:
                        self.aggregate_and_forward(window_data)
                        window_data = {'social': [], 'iot': []}
                        window_start = time.time()

                    self.batch.append(message)
                    if len(self.batch) >= 100:
                        self.consumer.commit()
                        self.batch = []
            except Exception as e:
                print(f"Error in enhanced process_stream: {e}")
            finally:
                self.consumer.close()

    def process_social_message(self, message):
        # Process individual social media message
        # Add your custom processing logic here
        # This could include NLP, sentiment analysis, etc.
        return message

    def process_iot_message(self, message):
        # Process individual IoT sensor message
        # Add your custom processing logic here
        # This could include anomaly detection, filtering, etc.
        return message

# Main execution with proper cleanup
def main():
    bootstrap_servers = ['localhost:9092']

    # Create aggregator and processor
    aggregator = StreamAggregator(bootstrap_servers)
    processor = StreamProcessor(bootstrap_servers, ['social-stream', 'iot-stream'])

    try:
        # Start streaming in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(aggregator.stream_social_media, 'twitter', 1000),
                executor.submit(aggregator.stream_social_media, 'facebook', 1000),
                executor.submit(aggregator.stream_iot_sensors, 'temperature', 1000),
                executor.submit(aggregator.stream_iot_sensors, 'pressure', 1000)
            ]

            # Start processing in a separate thread
            processing_thread = threading.Thread(target=processor.process_stream)
            processing_thread.start()

            # Wait for all streaming tasks to complete
            for future in futures:
                future.result()

            # Give processing time to finish
            time.sleep(2)
            processor.stop()
            processing_thread.join()

    except KeyboardInterrupt:
        print("Received interrupt signal, stopping...")
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        aggregator.stop()
        processor.stop()
        print("All components stopped gracefully")

if __name__ == "__main__":
    main()
""", language='python')

elif page == "📄 Log Aggregation":
    st.header("📄 Log Aggregation")
    st.markdown("""
**Definition:** Collecting logs from multiple services and applications, centralizing them through Kafka, and forwarding to processing systems like Spark and analytics platforms (ELK Stack, Splunk).

**Purpose:**

- Centralize distributed logs for easier debugging
- Enable real-time log analysis and alerting
- Reduce storage costs through efficient compression
- Provide audit trail for compliance

**Key Benefits:**

- **Unified View**: See logs from all services in one place
- **Real-time Alerting**: Detect and respond to issues immediately
- **Scalability**: Handle terabytes of logs per day
- **Decoupling**: Log producers don't depend on log processors

**Common Scenarios:**

- Microservices log aggregation
- Error pattern detection
- Security event monitoring
- Performance metrics collection

**Architecture:**

- Applications → Kafka Topics (by log level) → Log Analyzer → ELK/Splunk
""")
    st.markdown(
        "#### Example: Enterprise Log Aggregation with ELK Stack Integration")
    st.code("""
import logging
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
import hashlib
from datetime import datetime
from collections import defaultdict

class DistributedLogAggregator:
    def __init__(self, bootstrap_servers, app_name):
        self.app_name = app_name
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            acks=1,
            retries=5,
            max_in_flight_requests_per_connection=5
        )

    def emit_log(self, level, message, **kwargs):
        # Emit structured logs to Kafka
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'app_name': self.app_name,
            'level': level,
            'message': message,
            'trace_id': kwargs.get('trace_id', self._generate_trace_id()),
            'host': kwargs.get('host', 'unknown'),
            'service': kwargs.get('service', 'default'),
            'environment': kwargs.get('environment', 'production'),
            'metadata': kwargs.get('metadata', {}),
            'stack_trace': kwargs.get('stack_trace', None)
        }

        # Route by log level to different topics
        topic = f'logs-{level.lower()}'
        try:
            self.producer.send(topic, value=log_entry)
        except KafkaError as e:
            # Fallback logging
            print(f"Failed to send log: {e}")

    def _generate_trace_id(self):
        return hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()
""", language='python')
    st.code("""
class LogAnalyzer:
    def __init__(self, bootstrap_servers):
        self.consumer = KafkaConsumer(
            'logs-error', 'logs-warning', 'logs-info',
            bootstrap_servers=bootstrap_servers,
            group_id='log-analyzer',
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.error_patterns = defaultdict(int)
        self.alerts = []

    def analyze_logs(self):
        # Real-time log analysis with pattern detection
        batch = []
        alert_threshold = {'error': 10, 'warning': 50}
        window_start = time.time()

        for message in self.consumer:
            log = message.value
            batch.append(message)

            # Pattern detection
            if log['level'] in ['ERROR', 'WARNING']:
                pattern = self._extract_pattern(log['message'])
                self.error_patterns[pattern] += 1

                # Alert on threshold breach
                if self.error_patterns[pattern] >= alert_threshold.get(log['level'].lower(), 100):
                    self.trigger_alert(log, pattern, self.error_patterns[pattern])
                    self.error_patterns[pattern] = 0

            # Process in micro-batches
            if len(batch) >= 1000 or (time.time() - window_start) > 10:
                self.process_batch(batch)
                self.consumer.commit()
                batch = []
                window_start = time.time()

    def _extract_pattern(self, message):
        # Extract error pattern by removing dynamic values
        import re
        # Remove numbers, UUIDs, timestamps
        pattern = re.sub(r'\d+', 'N', message)
        pattern = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', pattern)
        return pattern[:100]

    def trigger_alert(self, log, pattern, count):
        # Trigger alert for repeated errors
        alert = {
            'alert_type': 'ERROR_SPIKE',
            'pattern': pattern,
            'count': count,
            'service': log['service'],
            'environment': log['environment'],
            'first_seen': log['timestamp'],
            'severity': 'HIGH' if log['level'] == 'ERROR' else 'MEDIUM'
        }
        print(f"🚨 ALERT: {alert}")
        # Forward to alerting system (PagerDuty, Slack, etc.)

    def process_batch(self, batch):
        # Process batch and forward to ELK/Splunk
        print(f"Processed {len(batch)} logs, {len(self.error_patterns)} unique error patterns")
        # Forward to Elasticsearch, Splunk, etc.
""", language='python')

    st.markdown("#### Usage")
    st.code("""logger = DistributedLogAggregator(['localhost:9092'], 'payment-service')
logger.emit_log('ERROR', 'Payment processing failed',
                service='payment', trace_id='abc123',
                metadata={'amount': 99.99, 'user_id': 'U12345'})

analyzer = LogAnalyzer(['localhost:9092'])
analyzer.analyze_logs()""", language='python')

elif page == "📬 Message Queuing":
    st.header("📬 Message Queuing")
    st.markdown("""**Definition:** Using Kafka as a distributed message queue to decouple producers and consumers, enabling multiple producers to send messages and multiple consumers to process them independently with guaranteed delivery.

**Purpose:**

- Decouple services for independent scaling
- Ensure reliable message delivery with retries
- Enable asynchronous processing
- Load balance work across multiple workers

**Key Benefits:**

- **Reliability**: Messages persist until successfully processed
- **Ordering**: Maintain message order within partitions
- **Priority Queues**: Route high-priority tasks to dedicated topics
- **Dead Letter Queues**: Isolate failing messages for investigation

**Common Scenarios:**

- Background job processing
- Email/notification delivery
- Payment processing
- Report generation
- Data import/export tasks

**Features:**

- Priority-based task routing
- Automatic retry with exponential backoff
- Dead letter queue for failed messages
- Multiple workers for parallel processing""")
    st.markdown(
        "#### Example: Distributed Task Queue with Priority & Dead Letter Queue")
    st.code("""from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.admin import KafkaAdminClient, NewTopic
import json
import time
from enum import Enum
from dataclasses import dataclass, asdict
import threading

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    task_id: str
    task_type: str
    priority: int
    payload: dict
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class TaskQueueProducer:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            acks='all',
            retries=3
        )
        self.ensure_topics()

    def ensure_topics(self):
        # Create priority-based topics
        admin = KafkaAdminClient(bootstrap_servers=['localhost:9092'])
        topics = [
            NewTopic('tasks-critical', num_partitions=3, replication_factor=1),
            NewTopic('tasks-high', num_partitions=3, replication_factor=1),
            NewTopic('tasks-medium', num_partitions=5, replication_factor=1),
            NewTopic('tasks-low', num_partitions=5, replication_factor=1),
            NewTopic('tasks-dlq', num_partitions=1, replication_factor=1)
        ]
        try:
            admin.create_topics(topics)
        except:
            pass

    def submit_task(self, task: Task):
        # Submit task to appropriate priority queue
        topic = f'tasks-{TaskPriority(task.priority).name.lower()}'
        future = self.producer.send(
            topic,
            key=task.task_id,
            value=asdict(task)
        )
        future.add_callback(lambda m: print(f"✓ Task {task.task_id} queued to {topic}"))
        future.add_errback(lambda e: print(f"✗ Failed to queue task: {e}"))
        return future
""", language='python')

    st.code("""class TaskQueueConsumer:
    def __init__(self, bootstrap_servers, worker_id):
        self.worker_id = worker_id
        # Subscribe to multiple priority queues with weighted polling
        self.consumers = {
            'critical': self._create_consumer(bootstrap_servers, 'tasks-critical', 'workers-critical'),
            'high': self._create_consumer(bootstrap_servers, 'tasks-high', 'workers-high'),
            'medium': self._create_consumer(bootstrap_servers, 'tasks-medium', 'workers-medium'),
            'low': self._create_consumer(bootstrap_servers, 'tasks-low', 'workers-low')
        }
        self.dlq_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def _create_consumer(self, servers, topic, group):
        return KafkaConsumer(
            topic,
            bootstrap_servers=servers,
            group_id=group,
            enable_auto_commit=False,
            max_poll_records=10,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def start_processing(self):
        # Process tasks with priority-based polling
        # Weighted polling: check critical more frequently
        poll_weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}

        while True:
            for priority, weight in poll_weights.items():
                for _ in range(weight):
                    messages = self.consumers[priority].poll(timeout_ms=100, max_records=10)

                    for tp, msgs in messages.items():
                        for message in msgs:
                            self.process_task(message, priority)
                        self.consumers[priority].commit()

    def process_task(self, message, priority):
        # Process individual task with retry logic
        task_data = message.value
        task = Task(**task_data)

        try:
            print(f"[Worker {self.worker_id}] Processing {priority} task: {task.task_id}")

            # Simulate task processing
            if task.task_type == 'email':
                self.send_email(task.payload)
            elif task.task_type == 'report':
                self.generate_report(task.payload)
            elif task.task_type == 'payment':
                self.process_payment(task.payload)

            print(f"[Worker {self.worker_id}] ✓ Completed task: {task.task_id}")

        except Exception as e:
            print(f"[Worker {self.worker_id}] ✗ Failed task: {task.task_id} - {e}")
            self.handle_failure(task, e)

    def handle_failure(self, task: Task, error):
        # Handle failed tasks with retry and DLQ
        task.retry_count += 1

        if task.retry_count < task.max_retries:
            # Retry with exponential backoff
            time.sleep(2 ** task.retry_count)
            print(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
            # Re-queue to same priority
            topic = f'tasks-{TaskPriority(task.priority).name.lower()}'
            self.dlq_producer.send(topic, value=asdict(task))
        else:
            # Send to Dead Letter Queue
            dlq_entry = {
                **asdict(task),
                'failed_at': time.time(),
                'error': str(error),
                'worker_id': self.worker_id
            }
            self.dlq_producer.send('tasks-dlq', value=dlq_entry)
            print(f"Task {task.task_id} sent to DLQ after {task.retry_count} retries")

    def send_email(self, payload):
        time.sleep(0.1)  # Simulate work

    def generate_report(self, payload):
        time.sleep(0.5)

    def process_payment(self, payload):
        time.sleep(0.2)
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Payment gateway timeout")""", language='python')

    st.code("""
# Usage: Multiple producers and consumers
producer = TaskQueueProducer(['localhost:9092'])

# Submit tasks with different priorities
for i in range(100):
    priority = random.choice([p.value for p in TaskPriority])
    task = Task(
        task_id=f'task_{i}',
        task_type=random.choice(['email', 'report', 'payment']),
        priority=priority,
        payload={'data': f'payload_{i}'}
    )
    producer.submit_task(task)

# Start multiple workers
workers = []
for worker_id in range(5):
    worker = TaskQueueConsumer(['localhost:9092'], f'W{worker_id}')
    thread = threading.Thread(target=worker.start_processing, daemon=True)
    thread.start()
    workers.append(thread)
""", language='python')

elif page == "🌐 Web Activity Tracking":
    st.header("🌐 Web Activity Tracking")
    st.markdown("""**Definition:** Tracking user activities on websites and applications, sending events through Kafka to analytics engines like Spark for real-time analysis, personalization, and reporting.

**Purpose:**

- Understand user behavior in real-time
- Enable personalized user experiences
- Detect anomalies and fraud
- Optimize conversion funnels
- Measure marketing campaign effectiveness

**Key Benefits:**

- **Real-time**: Respond to user actions immediately
- **Scalability**: Handle millions of events per second
- **Flexibility**: Multiple analytics pipelines from same data
- **Historical**: Store events for long-term analysis

**Common Scenarios:**

- Page view tracking
- Click stream analysis
- E-commerce conversion funnel
- A/B testing
- User session analysis
- Churn prediction
- Recommendation engines

**Tracked Events:**

- Page views, clicks, form submissions
- Purchases and cart additions
- Video plays, scrolls, hovers
- Search queries
- User authentication events""")
    st.markdown("#### Example: Real-time User Behavior Analytics Pipeline")
    st.code("""from kafka import KafkaProducer, KafkaConsumer
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
import uuid

class WebAnalyticsTracker:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            compression_type='lz4',
            linger_ms=10,
            batch_size=32768
        )
        self.session_cache = {}

    def track_event(self, user_id, event_type, properties):
        # Track various user events with session management
        session_id = self.session_cache.get(user_id, str(uuid.uuid4()))
        self.session_cache[user_id] = session_id

        event = {
            'event_id': str(uuid.uuid4()),
            'user_id': user_id,
            'session_id': session_id,
            'event_type': event_type,
            'timestamp': time.time(),
            'properties': properties,
            'user_agent': properties.get('user_agent', 'Unknown'),
            'ip_address': properties.get('ip', '0.0.0.0'),
            'referrer': properties.get('referrer'),
            'utm_params': properties.get('utm', {})
        }

        # Partition by user_id for session continuity
        self.producer.send('web-events', key=user_id, value=event)

    def track_page_view(self, user_id, page_url, page_title, **kwargs):
        self.track_event(user_id, 'page_view', {
            'page_url': page_url,
            'page_title': page_title,
            **kwargs
        })

    def track_click(self, user_id, element_id, element_text, **kwargs):
        self.track_event(user_id, 'click', {
            'element_id': element_id,
            'element_text': element_text,
            **kwargs
        })

    def track_purchase(self, user_id, order_id, items, total, **kwargs):
        self.track_event(user_id, 'purchase', {
            'order_id': order_id,
            'items': items,
            'total': total,
            'currency': kwargs.get('currency', 'USD'),
            **kwargs
        })""", language='python')

    st.code("""class RealTimeAnalyticsEngine:
    def __init__(self, bootstrap_servers):
        self.consumer = KafkaConsumer(
            'web-events',
            bootstrap_servers=bootstrap_servers,
            group_id='analytics-engine',
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_poll_records=1000,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        # Real-time metrics
        self.metrics = {
            'active_users': set(),
            'page_views': defaultdict(int),
            'conversions': 0,
            'revenue': 0.0,
            'session_data': defaultdict(list),
            'funnel_data': defaultdict(lambda: {'started': 0, 'completed': 0}),
            'cohorts': defaultdict(lambda: defaultdict(int))
        }

        self.window_start = time.time()
        self.anomaly_detector = AnomalyDetector()

    def process_events(self):
        # Process events with real-time aggregations
        batch = []

        for message in self.consumer:
            event = message.value
            batch.append(message)

            # Update real-time metrics
            self.update_metrics(event)

            # Sliding window (1 minute)
            if time.time() - self.window_start > 60:
                self.compute_window_metrics()
                self.window_start = time.time()
                self.metrics['active_users'].clear()

            # Process in micro-batches
            if len(batch) >= 1000:
                self.process_batch(batch)
                self.consumer.commit()
                batch = []

    def update_metrics(self, event):
        # Update real-time metrics for dashboards
        user_id = event['user_id']
        event_type = event['event_type']

        # Active users
        self.metrics['active_users'].add(user_id)

        # Page views
        if event_type == 'page_view':
            page = event['properties']['page_url']
            self.metrics['page_views'][page] += 1

            # Track user journey
            self.metrics['session_data'][event['session_id']].append({
                'page': page,
                'timestamp': event['timestamp']
            })

        # Conversions & Revenue
        elif event_type == 'purchase':
            self.metrics['conversions'] += 1
            self.metrics['revenue'] += event['properties']['total']

            # Funnel analysis
            session_id = event['session_id']
            journey = self.metrics['session_data'][session_id]
            if any('/product' in step['page'] for step in journey):
                self.metrics['funnel_data']['product_page']['completed'] += 1

        # Anomaly detection
        if self.anomaly_detector.is_anomaly(event):
            self.trigger_anomaly_alert(event)

    def compute_window_metrics(self):
        # Compute and emit window-based metrics
        active_users = len(self.metrics['active_users'])
        total_page_views = sum(self.metrics['page_views'].values())

        window_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'active_users': active_users,
            'total_page_views': total_page_views,
            'top_pages': dict(sorted(self.metrics['page_views'].items(),
                                    key=lambda x: x[1], reverse=True)[:10]),
            'conversions': self.metrics['conversions'],
            'revenue': self.metrics['revenue'],
            'conversion_rate': self.metrics['conversions'] / max(active_users, 1) * 100
        }

        print(f'''
            ╔══════════════════════════════════════╗
            ║  Real-Time Analytics Dashboard       ║
            ╠══════════════════════════════════════╣
            ║  Active Users: {active_users: 20} ║
            ║  Page Views: {total_page_views: 22} ║
            ║  Conversions: {self.metrics['conversions']: 21} ║
            ║  Revenue: ${self.metrics['revenue']: 23.2f} ║
            ║  Conv Rate: {window_metrics['conversion_rate']: 21.2f} % ║
            ╚══════════════════════════════════════╝
            ''')

        # Forward to real-time dashboard (WebSocket, Redis, etc.)
        # Forward to Spark for deeper analysis

        # Reset window metrics
        self.metrics['page_views'].clear()
        self.metrics['conversions'] = 0
        self.metrics['revenue'] = 0.0

    def process_batch(self, batch):
        # Process batch for ML models and predictions
        # Extract features for ML
        user_behaviors = defaultdict(list)
        for msg in batch:
            event = msg.value
            user_behaviors[event['user_id']].append(event)

        # Run prediction models
        for user_id, events in user_behaviors.items():
            if len(events) >= 5:
                churn_risk = self.predict_churn_risk(events)
                if churn_risk > 0.7:
                    self.trigger_retention_campaign(user_id, churn_risk)

    def predict_churn_risk(self, events):
        # Simple churn prediction based on user behavior
        # Check for decreased engagement signals
        recent_activity = [e for e in events if time.time() - e['timestamp'] < 3600]
        return max(0, 1 - (len(recent_activity) / 10))

    def trigger_retention_campaign(self, user_id, risk_score):
        print(f"🎯 High churn risk detected: User {user_id} ({risk_score:.2%})")
        # Trigger personalized retention campaign

    def trigger_anomaly_alert(self, event):
        print(f"⚠️ Anomaly detected: {event['event_type']} from {event['user_id']}")""", language='python')

    st.code("""class AnomalyDetector:
    def __init__(self):
        self.baseline = defaultdict(lambda: {'count': 0, 'avg_interval': 0})

    def is_anomaly(self, event):
        # Detect anomalous behavior patterns
        key = f"{event['user_id']}:{event['event_type']}"
        # Simple anomaly detection based on frequency
        return random.random() < 0.01  # 1% anomaly rate for demo

# Usage: Real-time tracking and analytics
tracker = WebAnalyticsTracker(['localhost:9092'])

# Simulate user activity
for i in range(1000):
    user_id = f"user_{random.randint(1, 100)}"
    tracker.track_page_view(user_id, f'/page/{random.randint(1, 20)}',
                           f'Page {i}', user_agent='Chrome/90.0')

    if random.random() < 0.3:
        tracker.track_click(user_id, f'btn_{i}', 'Click Me')

    if random.random() < 0.05:
        tracker.track_purchase(user_id, f'order_{i}',
                              [{'product': 'Item', 'qty': 1}],
                              random.uniform(10, 500))

# Start analytics engine
engine = RealTimeAnalyticsEngine(['localhost:9092'])
engine.process_events()""", language='python')

elif page == "🔁 Data Replication":
    st.header("🔁 Data Replication")
    st.markdown("""
**Definition:** Replicating data between different databases, data centers, or systems through Kafka Connect or custom Change Data Capture (CDC), ensuring data consistency across distributed environments.

**Purpose:**

- Maintain data consistency across regions
- Enable disaster recovery
- Support read scaling with replicas
- Facilitate data migration
- Implement CQRS (Command Query Responsibility Segregation)

**Key Benefits:**

- **Eventually Consistent**: Changes propagate to all replicas
- **Conflict Resolution**: Handle concurrent updates intelligently
- **Audit Trail**: Track all data changes
- **Zero Downtime**: Replicate without downtime

**Common Scenarios:**

- Multi-region database synchronization
- Database migration (MySQL → PostgreSQL)
- Data warehouse ETL
- Cache invalidation
- Microservices data sharing
- GDPR data replication

**Replication Strategies:**

- **Last-Write-Wins**: Use timestamps to resolve conflicts
- **Field-Level Merge**: Merge non-conflicting fields
- **Custom Business Logic**: Application-specific conflict resolution
- **Manual Resolution**: Queue conflicts for human review

**Architecture:**

- Source DB → CDC Producer → Kafka → Replication Consumer → Target DB
- Supports bidirectional replication with conflict detection
""")
    st.markdown(
        "#### Example: Multi-Region Database Replication with Conflict Resolution ")
    st.code("""from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import json
import time
from datetime import datetime
from enum import Enum
import hashlib

class OperationType(Enum):
    INSERT = 'INSERT'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'
    TRUNCATE = 'TRUNCATE'

class DatabaseCDCProducer:
    # Change Data Capture producer for database replication

    def __init__(self, bootstrap_servers, source_db_name, region):
        self.source_db = source_db_name
        self.region = region
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            acks='all',
            retries=10,
            max_in_flight_requests_per_connection=1,  # Maintain order
            enable_idempotence=True,
            compression_type='snappy'
        )
        self.sequence_num = 0

    def capture_change(self, operation, table, record_id, before_data, after_data):
        # Capture database change event
        self.sequence_num += 1

        change_event = {
            'event_id': self._generate_event_id(),
            'sequence_num': self.sequence_num,
            'timestamp': time.time(),
            'source_db': self.source_db,
            'region': self.region,
            'operation': operation.value,
            'table': table,
            'record_id': record_id,
            'before': before_data,
            'after': after_data,
            'metadata': {
                'transaction_id': self._get_transaction_id(),
                'schema_version': '1.0',
                'captured_at': datetime.utcnow().isoformat()
            }
        }

        # Use table:record_id as key for partitioning consistency
        partition_key = f"{table}:{record_id}"

        # Send to region-specific topic
        topic = f'db-changes-{self.region}'
        future = self.producer.send(topic, key=partition_key, value=change_event)

        # Also send to global replication topic
        self.producer.send('db-changes-global', key=partition_key, value=change_event)

        return future

    def _generate_event_id(self):
        # Generate unique event ID
        data = f"{self.source_db}:{self.region}:{time.time()}:{self.sequence_num}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_transaction_id(self):
        # "Get current database transaction ID
        return f"txn_{int(time.time() * 1000)}"

    def capture_bulk_insert(self, table, records):
        # Capture bulk insert operation
        for record in records:
            self.capture_change(
                OperationType.INSERT,
                table,
                record['id'],
                before_data=None,
                after_data=record
            )""", language='python')

    st.code("""
class DatabaseReplicationConsumer:
    # Consumer that applies changes to target database

    def __init__(self, bootstrap_servers, target_db_name, region):
        self.target_db = target_db_name
        self.region = region
        self.consumer = KafkaConsumer(
            'db-changes-global',
            bootstrap_servers=bootstrap_servers,
            group_id=f'db-replicator-{region}',
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            max_poll_records=100,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        # Track processed events to handle duplicates
        self.processed_events = set()
        self.conflict_resolver = ConflictResolver()
        self.replication_lag = {}

    def start_replication(self):
        # Start continuous replication process
        batch = []
        stats = {'replicated': 0, 'conflicts': 0, 'errors': 0}

        for message in self.consumer:
            change_event = message.value
            batch.append(message)

            # Skip if already processed (idempotency)
            if change_event['event_id'] in self.processed_events:
                continue

            # Skip changes from same region to avoid loops
            if change_event['region'] == self.region:
                continue

            try:
                # Check for conflicts
                conflict = self.detect_conflict(change_event)

                if conflict:
                    resolved_event = self.conflict_resolver.resolve(change_event, conflict)
                    self.apply_change(resolved_event)
                    stats['conflicts'] += 1
                else:
                    self.apply_change(change_event)

                self.processed_events.add(change_event['event_id'])
                stats['replicated'] += 1

                # Track replication lag
                lag = time.time() - change_event['timestamp']
                self.replication_lag[change_event['table']] = lag

            except Exception as e:
                print(f"❌ Replication error: {e}")
                self.handle_replication_error(change_event, e)
                stats['errors'] += 1

            # Commit in batches
            if len(batch) >= 100:
                self.consumer.commit()
                self.report_stats(stats)
                batch = []

    def detect_conflict(self, change_event):
        # Detect replication conflicts
        # Simulate conflict detection (e.g., concurrent updates)
        table = change_event['table']
        record_id = change_event['record_id']

        # Check if record was modified locally after source change
        local_record = self.fetch_local_record(table, record_id)

        if local_record and change_event['operation'] == 'UPDATE':
            local_timestamp = local_record.get('updated_at', 0)
            source_timestamp = change_event['timestamp']

            if local_timestamp > source_timestamp:
                return {
                    'type': 'CONCURRENT_UPDATE',
                    'local_data': local_record,
                    'remote_data': change_event['after']
                }

        return None

    def apply_change(self, change_event):
        # Apply change to target database
        operation = change_event['operation']
        table = change_event['table']
        record_id = change_event['record_id']

        if operation == 'INSERT':
            self.db_insert(table, change_event['after'])
            print(f"✓ Replicated INSERT: {table}:{record_id}")

        elif operation == 'UPDATE':
            self.db_update(table, record_id, change_event['after'])
            print(f"✓ Replicated UPDATE: {table}:{record_id}")

        elif operation == 'DELETE':
            self.db_delete(table, record_id)
            print(f"✓ Replicated DELETE: {table}:{record_id}")

    def fetch_local_record(self, table, record_id):
        # Fetch record from local database
        # Simulate database query
        return {'id': record_id, 'updated_at': time.time() - random.uniform(0, 100)}

    def db_insert(self, table, data):
        # Insert into target database
        # Simulate database insert
        pass

    def db_update(self, table, record_id, data):
        # Update in target database
        # Simulate database update
        pass

    def db_delete(self, table, record_id):
        # Delete from target database
        # Simulate database delete
        pass

    def handle_replication_error(self, change_event, error):
        # Handle replication errors with DLQ
        error_record = {
            **change_event,
            'error': str(error),
            'failed_at': time.time(),
            'target_db': self.target_db,
            'region': self.region
        }
        # Send to dead letter queue for manual review
        print(f"⚠️ Sending to DLQ: {change_event['table']}:{change_event['record_id']}")

    def report_stats(self, stats):
        # Report replication statistics
        avg_lag = sum(self.replication_lag.values()) / len(self.replication_lag) if self.replication_lag else 0

        print(f'''
            ╔════════════════════════════════════════════╗
            ║     Replication Statistics - {self.region}     ║
            ╠════════════════════════════════════════════╣
            ║  Replicated: {stats['replicated']: 27} ║
            ║  Conflicts Resolved: {stats['conflicts']: 19} ║
            ║  Errors: {stats['errors']: 31} ║
            ║  Avg Replication Lag: {avg_lag: 18.2f}s ║
            ╚════════════════════════════════════════════╝
            ''')
""", language='python')

    st.code("""class ConflictResolver:
    # Resolve replication conflicts using different strategies

    def resolve(self, remote_change, conflict):
        # Resolve conflict using configured strategy
        conflict_type = conflict['type']

        if conflict_type == 'CONCURRENT_UPDATE':
            return self.resolve_concurrent_update(remote_change, conflict)

        return remote_change

    def resolve_concurrent_update(self, remote_change, conflict):
        # Resolve concurrent update conflicts
        # Strategy 1: Last-Write-Wins (timestamp-based)
        local_data = conflict['local_data']
        remote_data = conflict['remote_data']

        # Compare timestamps
        if local_data.get('updated_at', 0) > remote_change['timestamp']:
            print(f"⚡ Conflict: Local wins (newer timestamp)")
            return None  # Keep local changes
        else:
            print(f"⚡ Conflict: Remote wins (newer timestamp)")
            return remote_change

        # Strategy 2: Field-level merge (merge non-conflicting fields)
        # Strategy 3: Custom business logic
        # Strategy 4: Manual resolution queue

""", language='python')

    st.code("""class BidirectionalReplicator:
    # Bidirectional replication between multiple regions

    def __init__(self, bootstrap_servers, regions):
        self.regions = regions
        self.producers = {}
        self.consumers = {}

        for region in regions:
            self.producers[region] = DatabaseCDCProducer(
                bootstrap_servers, f'db-{region}', region
            )
            self.consumers[region] = DatabaseReplicationConsumer(
                bootstrap_servers, f'db-{region}', region
            )

    def setup_topology(self):
        # Setup multi-region replication topology
        print('''
            Replication Topology:

            US-East ←→ Kafka ←→ EU-West
            ↓                    ↓
            US-West              APAC
            ''')

    def start_all_replicators(self):
        # Start all region replicators
        import threading

        threads = []
        for region, consumer in self.consumers.items():
            thread = threading.Thread(
                target=consumer.start_replication,
                daemon=True,
                name=f"Replicator-{region}"
            )
            thread.start()
            threads.append(thread)

        return threads""", language='python')
    st.markdown("#### Usage: Multi-region database replication")
    st.code("""regions = ['us-east', 'us-west', 'eu-west', 'apac']
replicator = BidirectionalReplicator(['localhost:9092'], regions)
replicator.setup_topology()

# Simulate database changes in US-East
us_producer = replicator.producers['us-east']

# Simulate various database operations
for i in range(50):
    # INSERT
    us_producer.capture_change(
        OperationType.INSERT,
        table='users',
        record_id=f'user_{i}',
        before_data=None,
        after_data={
            'id': f'user_{i}',
            'name': f'User {i}',
            'email': f'user{i}@example.com',
            'created_at': time.time()
        }
    )

    # UPDATE
    if i % 3 == 0:
        us_producer.capture_change(
            OperationType.UPDATE,
            table='users',
            record_id=f'user_{i}',
            before_data={'name': f'User {i}'},
            after_data={'name': f'Updated User {i}', 'updated_at': time.time()}
        )

    time.sleep(0.1)

# Start replication to all regions
threads = replicator.start_all_replicators()""", language='python')

elif page == "⚠️ Common Issues":
    st.header("⚠️ Common Issue")
    st.markdown("""
    <style>
         /* -------------------------------------------------
        Base shared styles
        ------------------------------------------------- */
        .info-box {
            padding: 1rem;
            border-radius: 0.6rem;
            margin-bottom: 1.2rem;
            line-height: 1.55;
            font-size: 0.95rem;
        }

        .issue-card {
            margin-bottom: 1.4rem;
        }

        .issue-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .label {
            font-weight: 600;
        }

        ul {
            margin-top: 0.4rem;
            padding-left: 1.2rem;
        }

        code {
            padding: 0.1rem 0.3rem;
            border-radius: 0.3rem;
            font-size: 0.85rem;
        }

        /* -------------------------------------------------
        Light theme
        ------------------------------------------------- */
        [data-theme="light"] .info-box {
            background-color: #f8fafc;
            color: #0f172a;
            border-left: 5px solid #2563eb;
        }

        [data-theme="light"] .info-box.warning {
            border-left-color: #f59e0b;
        }

        [data-theme="light"] .info-box.error {
            border-left-color: #ef4444;
        }

        [data-theme="light"] code {
            background-color: #e5e7eb;
            color: #1e293b;
        }

        /* -------------------------------------------------
        Dark theme
        ------------------------------------------------- */
        [data-theme="dark"] .info-box {
            background-color: #0f172a;
            color: #e5e7eb;
            border-left: 5px solid #38bdf8;
        }

        [data-theme="dark"] .info-box.warning {
            border-left-color: #fbbf24;
        }

        [data-theme="dark"] .info-box.error {
            border-left-color: #f87171;
        }

        [data-theme="dark"] code {
            background-color: #1e293b;
            color: #e5e7eb;
        }

        /* -------------------------------------------------
        Headings polish
        ------------------------------------------------- */
        h2, h3 {
            margin-top: 1.6rem;
        }

        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background-color: rgba(148, 163, 184, 0.3);
        }
    </style>

    <div class="issue-card info-box warning">
        <div class="issue-title">🔌 Connection Issues</div>
        <span class="label">Symptoms:</span> Cannot connect to Kafka brokers, timeout errors
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Check <code>bootstrap_servers</code> format</li>
            <li>Verify network connectivity: <code>telnet hostname 9092</code></li>
            <li>Check firewall rules and security groups</li>
            <li>Ensure broker is running: <code>ps aux | grep kafka</code></li>
        </ul>
    </div>

    <div class="issue-card info-box warning">
        <div class="issue-title">⏱️ Offset Issues</div>
        <span class="label">Symptoms:</span> Consumer skips messages or re-processes old messages
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Use <code>auto_offset_reset='earliest'</code></li>
            <li>Enable commits: <code>enable_auto_commit=True</code></li>
            <li>Verify consumer group offsets</li>
            <li>Reset offsets if corrupted</li>
        </ul>
    </div>

    <div class="issue-card info-box warning">
        <div class="issue-title">⏳ Timeout Issues</div>
        <span class="label">Symptoms:</span> Producer send timeouts, consumer poll timeouts
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Increase <code>request_timeout_ms</code></li>
            <li>Increase <code>session_timeout_ms</code></li>
            <li>Reduce <code>max_poll_records</code></li>
            <li>Check broker health and capacity</li>
        </ul>
    </div>

    <div class="issue-card info-box warning">
        <div class="issue-title">🧠 Memory Issues</div>
        <span class="label">Symptoms:</span> OutOfMemory errors, GC pauses
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Adjust producer <code>buffer_memory</code></li>
            <li>Reduce consumer <code>max_poll_records</code></li>
            <li>Enable compression</li>
            <li>Increase JVM heap size</li>
        </ul>
    </div>

    <div class="issue-card info-box warning">
        <div class="issue-title">🔄 Rebalancing Issues</div>
        <span class="label">Symptoms:</span> Frequent rebalances, processing interruptions
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Tune <code>session_timeout_ms</code></li>
            <li>Adjust <code>heartbeat_interval_ms</code></li>
            <li>Avoid long-running poll loops</li>
            <li>Monitor rebalance frequency</li>
        </ul>
    </div>

    <div class="issue-card info-box error">
        <div class="issue-title">🚫 Message Loss</div>
        <span class="label">Symptoms:</span> Messages appear to be missing
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Use <code>acks='all'</code></li>
            <li>Set <code>min.insync.replicas ≥ 2</code></li>
            <li>Enable producer idempotence</li>
            <li>Commit offsets after processing</li>
        </ul>
    </div>

    <div class="issue-card info-box warning">
        <div class="issue-title">🔁 Duplicate Messages</div>
        <span class="label">Symptoms:</span> Same message processed multiple times
        <br><br>
        <span class="label">Solutions:</span>
        <ul>
            <li>Implement idempotent processing</li>
            <li>Use manual offset commits</li>
            <li>Enable producer idempotence</li>
            <li>Set <code>max_in_flight_requests_per_connection=1</code></li>
        </ul>
    </div>

    <hr>

    <h2>📘 Glossary</h2>

    <div class="info-box">
        <ul>
            <li><b>Broker</b> – Kafka server storing data</li>
            <li><b>Topic</b> – Message category/feed</li>
            <li><b>Partition</b> – Ordered message log</li>
            <li><b>Offset</b> – Message position</li>
            <li><b>Producer</b> – Publishes messages</li>
            <li><b>Consumer</b> – Processes messages</li>
            <li><b>Consumer Group</b> – Coordinated consumers</li>
            <li><b>Replication</b> – Partition copies</li>
            <li><b>ISR</b> – In-sync replicas</li>
            <li><b>Lag</b> – Consumer delay</li>
            <li><b>Retention</b> – Message lifespan</li>
            <li><b>Compaction</b> – Latest value per key</li>
            <li><b>Rebalance</b> – Partition redistribution</li>
        </ul>
    </div>
    """,
                unsafe_allow_html=True
                )
