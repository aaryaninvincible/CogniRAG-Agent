# Enterprise Cloud Infrastructure Architecture Specification

## 1. System Overview and Global Topology
The enterprise platform operates an active-active multi-region deployment across AWS and GCP regions (`us-east-1`, `eu-west-1`, and `ap-southeast-1`). All mission-critical ingress traffic routes through global Anycast Cloudflare Enterprise CDNs with automated DDoS scrubbing and Web Application Firewalls (WAF).

## 2. Storage Tiers and Cost Structure
The infrastructure defines three operational storage tiers:
- **Hot Tier (NVMe SSD Multi-Attach)**: Dedicated for transactional databases and Kafka event streams. Storage pricing is fixed at **$0.08 per GB/month** with zero IOPS provision charges.
- **Warm Tier (Object Storage S3/GCS)**: Used for daily operational snapshots and vector embedding indices. Pricing is **$0.023 per GB/month**.
- **Cold Glacier Archive**: Long-term audit retention with 90-day minimum lifecycle. Pricing is **$0.004 per GB/month**.

## 3. High Availability and Failover Latency SLAs
- **Recovery Time Objective (RTO)**: Target SLA is strictly **under 45 seconds** for regional database failovers via automated consensus failover controllers.
- **Recovery Point Objective (RPO)**: Target RPO is **zero seconds** for synchronous replication groups and **< 250 milliseconds** across asynchronous transatlantic replicas.
- **Inter-Region Network Latency**: Dedicated 10 Gbps DirectConnect tunnels maintain 68ms p95 roundtrip between US-East and EU-West.

## 4. Compute Autoscaling and Egress Policies
Kubernetes worker node clusters scale automatically using KEDA event-driven autoscalers. Cross-region network data egress is capped at **$0.01 per GB** under negotiated enterprise volume agreements.
