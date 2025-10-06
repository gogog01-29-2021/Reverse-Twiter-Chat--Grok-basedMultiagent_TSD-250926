Data Stream Manager

Low-Latency Market Data Ingestion & Distribution Architecture


1. Architecture Overview
This architecture is designed to ingest high-frequency market data from exchanges, maintain an up-to-date order book, and distribute updates to subscribers with minimal latency using AWS infrastructure and NATS messaging.

2. Market Data Ingestion
Dedicated EC2 Instance:

Use a performance-optimized EC2 instance (e.g., C6gn, M5n) with Enhanced Networking (ENA) to ingest real-time market data.

Establish direct connections or low-latency links (e.g., AWS Direct Connect) to exchanges for minimal latency.

Data Tagging: Immediately tag each data packet with metadata — e.g., exchange, symbol, and timestamp — to support intelligent routing and filtering downstream.

Order Book Maintenance: Efficiently update the in-memory order book and maintain per-symbol consistency with minimal locking or contention.

3. Data Distribution via NATS
NATS Core (Non-JetStream):

Utilize NATS core for ultra-low-latency message delivery, bypassing persistence overhead where real-time speed is paramount.

Subject-Based Hierarchical Filtering:

Design subjects as: exchange.symbol.orderbook (e.g., NYSE.AAPL.orderbook).

Support both specific subscriptions (e.g., NASDAQ.MSFT.orderbook) and wildcards (e.g., NYSE.*.orderbook), enabling tailored data streams.

Publishing Strategy:

Publish data updates only to relevant subjects, reducing unnecessary network chatter and improving subscriber performance.

4. Network Configuration
VPC & Availability Zone Optimization:

Co-locate publisher and subscriber instances within the same VPC and AZ to minimize latency due to inter-zone or inter-region hops.

Enhanced Networking:

Leverage ENA-enabled EC2 instances to maximize throughput and reduce jitter.

Use placement groups (cluster strategy) to ensure low-latency physical network locality among nodes.

5. NATS Cluster Configuration
High Availability with Minimal Latency:

Deploy a NATS cluster (3 or 5 nodes) within the same VPC for resilience and failover.

Use low-hop topology — place NATS servers as close as possible (logically and physically) to both publishers and subscribers.

Connection Optimization:

Tune client connection settings (buffer sizes, reconnection intervals, timeouts) to favor low-latency performance over fault tolerance when appropriate.

6. Scaling and Redundancy
Auto Scaling:

Implement Auto Scaling Groups (ASGs) for subscriber nodes based on:

CPU utilization

NATS message rate

Network throughput

Redundancy:

Maintain at least 3 NATS nodes in separate subnets to ensure fault tolerance.

Ensure state replication in the order book layer or downstream consumer to tolerate instance failure.

7. Application Logic and Subscriber Design
Selective Subscriptions:

Configure subscribers to connect only to relevant subjects.

Allow for wildcard subscriptions to support analytics or aggregated views.

Efficient Processing:

Minimize message handling overhead using lock-free queues, batching (when appropriate), and preallocated memory buffers.

Use concurrent processing models (e.g., async I/O, multithreading, coroutines) to fully utilize CPU resources.

8. Monitoring and Optimization
Real-Time Monitoring:

Use AWS CloudWatch and custom metrics to track:

Message processing time

Network RTT

CPU/memory pressure

Latency from ingestion to delivery

Performance Tuning:

Periodically tune:

NATS client configurations (e.g., max_pending, buffer sizes)

EC2 instance types and network settings

OS-level kernel parameters (e.g., socket buffers, TCP stack tuning)

9. Testing and Validation
Latency Benchmarks:

Implement synthetic and real-world latency tests across:

Ingestion → NATS publishing

NATS publishing → subscriber processing

End-to-end ingestion → subscriber output

Load Testing:

Simulate peak data volumes using replayed market sessions.

Use tools like Vegeta, K6, or custom producers to generate load profiles.

Dynamic Adjustment:

Integrate a feedback loop that adjusts system parameters dynamically (e.g., scaling thresholds, client buffer sizes) based on real-time performance metrics.

10. Security and Governance
Network Controls:

Use Security Groups and Network ACLs to restrict access at the subnet and port level.

Encrypt data in-transit (e.g., mTLS for NATS, if necessary).

Access Management:

Control access to EC2 instances and NATS endpoints via IAM roles, bastion hosts, or session managers.

Audit and Compliance:

Enable logging for NATS access and EC2 API calls.

Ensure market data usage adheres to exchange agreements and data retention policies.

Summary
This architecture enables low-latency, high-throughput, and scalable market data ingestion and dissemination, optimized for real-time trading systems or analytics platforms. It leverages AWS best practices, NATS's eventing power, and latency-aware design across compute, network, and application layers.