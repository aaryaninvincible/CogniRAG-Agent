# Distributed Systems Consensus Protocol Reference: Raft Implementation

## 1. Consensus Leader Election Mechanics
In the Raft consensus cluster, each node exists in one of three primary states: `Leader`, `Follower`, or `Candidate`.
- **Heartbeat Interval**: The elected Leader broadcasts empty AppendEntries RPC heartbeats every **50 milliseconds** to maintain leadership authority.
- **Election Timeout**: Followers randomize election timeouts between **150 milliseconds and 300 milliseconds** to prevent split-vote deadlocks during concurrent partition events.
- **Quorum Requirement**: A cluster consisting of $2F + 1$ nodes can tolerate up to $F$ simultaneous failure nodes while maintaining liveness. A strict majority quorum of $\lfloor N/2 \rfloor + 1$ votes is required to commit log entries.

## 2. Log Replication and Safety Invariant
The Leader appends incoming state machine command entries to its write-ahead log and issues AppendEntries RPCs to all peers in parallel.
- An entry is deemed committed once replicated to a strict majority of nodes.
- **Leader Completeness Property**: If a log entry is committed in a given term, that entry will be present in the logs of the leaders for all higher-numbered terms.

## 3. Log Compaction and Snapshotting
To prevent logs from unbounded growth on disk, each node takes independent snapshots once the uncompacted log size reaches **50,000 entries**. The snapshot contains state machine state, last included index, and last included term.
