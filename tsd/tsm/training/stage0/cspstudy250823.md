
# Event scheduler in C++
# Memory management in C++
# Graph execution engine in C++
250823
Meaning of Graph in CSP
# PURE PYTHON VERSION (hundreds of lines needed):
class PurePythonGraph:
    def __init__(self):
        self.nodes = {}
        self.dependencies = {}
        self.execution_order = []
        self.data_buffers = {}
        
    def add_node(self, name, func, inputs):
        self.nodes[name] = func
        self.dependencies[name] = inputs
        self.data_buffers[name] = []
        
    def build_execution_order(self):
        # MANUAL DEPENDENCY RESOLUTION: Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(node):
            if node in temp_visited:
                raise Exception("Circular dependency!")
            if node in visited:
                return
                
            temp_visited.add(node)
            for dep in self.dependencies.get(node, []):
                dfs(dep)
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
            
        for node in self.nodes:
            if node not in visited:
                dfs(node)
                
        self.execution_order = order
        
    def execute_timestamp(self, timestamp, input_data):
        # MANUAL EXECUTION ORDER
        for node_name in self.execution_order:
            inputs = []
            for dep in self.dependencies[node_name]:
                # MANUAL STATE MANAGEMENT
                if dep in input_data:
                    inputs.append(input_data[dep])
                elif self.data_buffers[dep]:
                    inputs.append(self.data_buffers[dep][-1])  # Latest value
                    
            if inputs:  # Only execute if inputs available
                result = self.nodes[node_name](*inputs)
                self.data_buffers[node_name].append((timestamp, result))
                input_data[node_name] = result

# CSP VERSION (automatic):
@csp.graph
def performance_graph():
    bid = csp.curve(float, data)
    ask = csp.curve(float, data)
    
    mid = (bid + ask) / 2     # CSP automatically knows this needs bid AND ask
    spread = ask - bid        # CSP automatically knows this needs bid AND ask
    ratio = spread / mid      # CSP automatically knows this needs spread AND mid
    
    # CSP automatically:
    # 1. Builds dependency graph
    # 2. Determines execution order  
    # 3. Manages memory for each node
    # 4. Synchronizes timestamps
    # 5. Handles edge cases




Do
// RxJS - Javascript reactive programming,Apache Flink, Kafka has Graph(V,E)?
# In CSP, Graph G = (V, E) where:
# V (Vertices) = Nodes + Adapters
# E (Edges) = Data connections between nodes

@csp.graph
def graph_structure(): # csp.show_graph(my_graph, "graph.png")  # Generates actual graph visualization
    # Vertices (V):
    source = csp.curve(float, data)      # V1: Input adapter
    processor = my_calculation(source)   # V2: Processing node  
    output = csp.print("result", processor) # V3: Output adapter
    
    # Edges (E):
    # E1: source → processor
    # E2: processor → output


const { from } = require('rxjs');
const { map } = require('rxjs/operators');

from([100.0, 100.1, 100.2])
  .pipe(map(price => price * 1.01))
  .subscribe(price => console.log(`Adjusted: ${price}`));





  Event Driven Parallelism: such as LabVIEW oder Simulink.
# Event=(timestamp, value, node) tuple
# A.Data event
# B. Time Events
# C. System Events

# 1. CSP automatically handles time synchronization
@csp.node
def spread_calculator(bid: csp.ts[float], ask: csp.ts[float]) -> csp.ts[float]:
    # CSP ensures both inputs are available at same timestamp
    if csp.ticked(bid, ask):  # This is the magic!
        return ask - bid

# RxPy equivalent - much more complex
from rx import combine_latest
bid_stream.pipe(
    combine_latest(ask_stream),
    map(lambda x: x[1] - x[0])  # Manual tuple handling
)

# CSP VERSION - What happens under the hood:
@csp.node
def spread_calculator(bid: csp.ts[float], ask: csp.ts[float]) -> csp.ts[float]:
    # ALGORITHMIC LEVEL: CSP maintains internal data structures:
    # - InputBuffer<float> bid_buffer;    // C++ circular buffer
    # - InputBuffer<float> ask_buffer;    // C++ circular buffer
    # - Timestamp last_execution_time;    // C++ timestamp tracking
    
    if csp.ticked(bid, ask):  # This translates to C++:
    # if (bid_buffer.has_new_data(current_time) && 
    #     ask_buffer.has_new_data(current_time) &&
    #     bid_buffer.timestamp() == ask_buffer.timestamp()) {
    #     // Both inputs have data at SAME timestamp
        
        return ask - bid
        # C++ level: return ask_buffer.get_value() - bid_buffer.get_value();

# RxPy VERSION - Manual complexity:
from rx import combine_latest
bid_stream.pipe(
    combine_latest(ask_stream),  # Algorithm: Cartesian product of latest values
    # DATA STRUCTURE: Must maintain:
    # - latest_bid_value: Optional[float] = None
    # - latest_ask_value: Optional[float] = None
    # - bid_timestamp: Optional[datetime] = None
    # - ask_timestamp: Optional[datetime] = None
    
    map(lambda x: x[1] - x[0])  # Manual tuple unpacking: (bid, ask) -> ask - bid
    # PROBLEM: No timestamp synchronization guarantee!
    # bid from 10:00:01, ask from 10:00:02 could be combined incorrectly
)


  
# CSP processes multiple data streams concurrently
@csp.graph
def parallel_processing():
    # Multiple independent streams (like SIMD but event-driven)
    stream1 = csp.curve(float, [(start, 100.0), (start+timedelta(seconds=1), 101.0)])
    stream2 = csp.curve(float, [(start, 200.0), (start+timedelta(seconds=1), 201.0)])
    stream3 = csp.curve(float, [(start, 300.0), (start+timedelta(seconds=1), 301.0)])
    
    # Same operation on multiple streams (SIMD-like)
    result1 = csp.multiply(stream1, 1.1)
    result2 = csp.multiply(stream2, 1.1)
    result3 = csp.multiply(stream3, 1.1)


  SISD: Single Instruction, Single Data (regular CPU)
SIMD: Single Instruction, Multiple Data (vectorization)
SIMT: Single Instruction, Multiple Threads (GPU)



250825
1. Basket of Timeseries of Input? If It is not Timeseries?
What algo/Datastructues for 
Timeseries buffer for what? For Timeseries synchronization
# CSP VERSION (automatic):
@csp.node
def spread_calculator(bid: csp.ts[float], ask: csp.ts[float]) -> csp.ts[float]:
    if csp.ticked(bid, ask):  # Magic: CSP ensures both are at same timestamp
        return ask - bid

# PURE PYTHON EQUIVALENT (hundreds of lines):
import heapq
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime

class TimeseriesBuffer:
    """Manual implementation of CSP's automatic buffer management"""
    def __init__(self, max_size: int = 1000):
        self.data: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
    
    def push(self, timestamp: datetime, value: Any):
        self.timestamps.append(timestamp)
        self.data.append(value)
    
    def get_at_time(self, timestamp: datetime) -> Optional[Any]:
        try:
            idx = list(self.timestamps).index(timestamp)
            return self.data[idx]
        except ValueError:
            return None
    
    def has_data_at(self, timestamp: datetime) -> bool:
        return timestamp in self.timestamps

class EventScheduler:
    """Manual implementation of CSP's automatic event scheduling"""
    def __init__(self):
        self.event_heap: List[Tuple[datetime, str, str, Any]] = []
        self.current_time: Optional[datetime] = None
    
    def schedule_event(self, timestamp: datetime, node_id: str, input_name: str, value: Any):
        heapq.heappush(self.event_heap, (timestamp, node_id, input_name, value))
    
    def get_next_event(self) -> Optional[Tuple[datetime, str, str, Any]]:
        if self.event_heap:
            return heapq.heappop(self.event_heap)
        return None

class DependencyManager:
    """Manual implementation of CSP's automatic dependency resolution"""
    def __init__(self):
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        self.reverse_deps: Dict[str, List[str]] = defaultdict(list)
        self.execution_order: List[str] = []
    
    def add_dependency(self, node: str, depends_on: str):
        self.dependencies[node].append(depends_on)
        self.reverse_deps[depends_on].append(node)
    
    def topological_sort(self) -> List[str]:
        """Manual topological sort - CSP does this automatically"""
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for node, deps in self.dependencies.items():
            all_nodes.add(node)
            for dep in deps:
                all_nodes.add(dep)
                in_degree[node] += 1
        
        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for dependent in self.reverse_deps[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result

class TimeSynchronizer:
    """Manual implementation of CSP's automatic time synchronization"""
    def __init__(self):
        self.node_buffers: Dict[str, TimeseriesBuffer] = defaultdict(TimeseriesBuffer)
        self.node_functions: Dict[str, Callable] = {}
        self.waiting_for: Dict[str, List[str]] = defaultdict(list)  # What inputs each node waits for
        
    def register_node(self, node_id: str, inputs: List[str], func: Callable):
        self.node_functions[node_id] = func
        self.waiting_for[node_id] = inputs
    
    def can_execute(self, node_id: str, timestamp: datetime) -> bool:
        """Check if all required inputs are available at timestamp"""
        required_inputs = self.waiting_for[node_id]
        for input_name in required_inputs:
            if not self.node_buffers[input_name].has_data_at(timestamp):
                return False
        return True
    
    def execute_node(self, node_id: str, timestamp: datetime) -> Any:
        """Execute node with synchronized inputs"""
        required_inputs = self.waiting_for[node_id]
        input_values = []
        
        for input_name in required_inputs:
            value = self.node_buffers[input_name].get_at_time(timestamp)
            if value is None:
                raise RuntimeError(f"Missing input {input_name} at {timestamp}")
            input_values.append(value)
        
        result = self.node_functions[node_id](*input_values)
        self.node_buffers[node_id].push(timestamp, result)
        return result

class PurePythonCSPEngine:
    """Manual implementation requiring hundreds of lines"""
    def __init__(self):
        self.scheduler = EventScheduler()
        self.dependency_manager = DependencyManager()
        self.time_synchronizer = TimeSynchronizer()
        self.output_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def add_input_stream(self, stream_name: str, data: List[Tuple[datetime, Any]]):
        """Manual equivalent of csp.curve()"""
        for timestamp, value in data:
            self.scheduler.schedule_event(timestamp, stream_name, "data", value)
    
    def add_node(self, node_id: str, inputs: List[str], func: Callable):
        """Manual node registration"""
        self.time_synchronizer.register_node(node_id, inputs, func)
        for input_name in inputs:
            self.dependency_manager.add_dependency(node_id, input_name)
    
    def add_output(self, node_id: str, callback: Callable):
        """Manual equivalent of csp.print()"""
        self.output_callbacks[node_id].append(callback)
    
    def run(self, start_time: datetime, end_time: datetime):
        """Manual execution engine - CSP does all this automatically"""
        execution_order = self.dependency_manager.topological_sort()
        
        while True:
            event = self.scheduler.get_next_event()
            if not event or event[0] > end_time:
                break
                
            timestamp, node_id, input_name, value = event
            
            # Store input data
            if input_name == "data":
                self.time_synchronizer.node_buffers[node_id].push(timestamp, value)
            
            # Try to execute all nodes that might be ready
            for exec_node in execution_order:
                if self.time_synchronizer.can_execute(exec_node, timestamp):
                    try:
                        result = self.time_synchronizer.execute_node(exec_node, timestamp)
                        
                        # Handle outputs
                        for callback in self.output_callbacks[exec_node]:
                            callback(timestamp, result)
                            
                    except RuntimeError as e:
                        continue  # Node not ready yet

# USAGE - showing complexity vs CSP simplicity:
def pure_python_spread_calculator():
    """Equivalent to CSP's 3-line spread calculator"""
    engine = PurePythonCSPEngine()
    
    # Add input streams (equivalent to csp.curve)
    bid_data = [(datetime(2020,1,1), 100.0), (datetime(2020,1,1,0,0,1), 100.1)]
    ask_data = [(datetime(2020,1,1), 100.5), (datetime(2020,1,1,0,0,1), 100.6)]
    
    engine.add_input_stream("bid", bid_data)
    engine.add_input_stream("ask", ask_data)
    
    # Add spread calculation node (equivalent to CSP's automatic sync)
    def calc_spread(bid_val: float, ask_val: float) -> float:
        return ask_val - bid_val
    
    engine.add_node("spread", ["bid", "ask"], calc_spread)
    
    # Add output (equivalent to csp.print)
    engine.add_output("spread", lambda ts, val: print(f"{ts} SPREAD: {val}"))
    
    # Run (equivalent to csp.run)
    engine.run(datetime(2020,1,1), datetime(2020,1,2))

# CSP EQUIVALENT (3 lines):
@csp.node
def spread_calculator(bid: csp.ts[float], ask: csp.ts[float]) -> csp.ts[float]:
    if csp.ticked(bid, ask):
        return ask - bid


2.  
3. Quest Db where should we connect those?


# In CSP source: cpp/csp/core/Event.h
struct Event {
    DateTime timestamp;
    NodeDef* node;
    InputId input_id;
    std::shared_ptr<const DialectGenericType> data;
};
import csp
node = csp.curve(float, [(datetime.now(), 1.0)])
print(type(node))  # Shows internal CSP types
print(dir(node))   # Shows available methods

// From CSP C++ source (cpp/csp/core/EventType.h):
enum class EventType {
    DATA_EVENT,      // New value arrives on input
    TIMER_EVENT,     // Scheduled time-based computation  
    ALARM_EVENT,     // System lifecycle events
    SHUTDOWN_EVENT   // Engine shutdown
};
4. Event(timestamp,value,node) received from api? how they connected to timeseries?
std::priority_queue<Event> event_queue_;  // Min-heap by timestamp

5. eventqueue und api?
6. 

7. node
class Node {
protected:
    size_t node_id_;
    std::vector<size_t> input_nodes_;
    std::vector<size_t> output_nodes_;
    
public:
    virtual ~Node() = default;
    virtual void execute(std::chrono::nanoseconds timestamp) = 0;
    virtual bool ready_to_execute(std::chrono::nanoseconds timestamp) const = 0;
};






10. algorithm
std::vector<std::unique_ptr<Node>> nodes_;
    std::unordered_map<size_t, std::vector<size_t>> dependency_graph_;
    std::unordered_map<size_t, std::vector<size_t>> reverse_dependencies_;
    
    // DEPENDENCY RESOLUTION: Topological sort the what?
    std::vector<size_t> topological_sort() {
        std::vector<size_t> result;
        std::vector<int> in_degree(nodes_.size(), 0);
        
        // Calculate in-degrees
        for (const auto& [node, deps] : dependency_graph_) {
            for (size_t dep : deps) {
                in_degree[node]++;
            }
        }
11. 




# Method 2: Look at these specific files for core implementation:
# cpp/csp/core/Engine.h              - Main engine
# cpp/csp/core/Time.h               - Time handling  
# cpp/csp/core/Graph.h              - Graph structure
# cpp/csp/adapters/                 - Input/output adapters
# python/csp/impl/                  - Python bindings

# Method 3: Find installed CSP location
python -c "import csp; import os; print(os.path.dirname(csp.__file__))"

# Method 4: Use debugger to step into CSP code
import pdb; pdb.set_trace()
# Then step into csp.curve() calls

250826
1. each event (time point), CSP triggers the relevant nodes in the graph.
Nodes process the event, possibly combining it with other streams (e.g., synchronizing bid/ask by timestamp).

import csp
from datetime import datetime, timedelta

@csp.node
def spread(bid: csp.ts[float], ask: csp.ts[float]) -> csp.ts[float]:
    # This node is triggered ONLY when both bid and ask have a value at the same timestamp
    if csp.ticked(bid, ask):
        return ask - bid

@csp.graph
def my_graph():
    start = datetime(2020, 1, 1)
    bid = csp.curve(float, [
        (start, 100.0),
        (start + timedelta(seconds=1), 100.1),
    ])
    ask = csp.curve(float, [
        (start, 100.5),
        (start + timedelta(seconds=1), 100.6),
    ])
    s = spread(bid, ask)
    csp.print("spread", s)

if __name__ == "__main__":
    csp.run(my_graph, starttime=datetime(2020,1,1), endtime=timedelta(seconds=2))

# bid - ask calculation
@csp.node
def spread(bid: csp.ts[float], ask: csp.ts[float]) -> csp.ts[float]:
    if csp.ticked(bid, ask):
        return ask - bid

from datetime import datetime, timedelta

# Example bid and ask data as lists of (timestamp, value)
bid_data = [
    (datetime(2020, 1, 1, 0, 0, 0), 100.0),
    (datetime(2020, 1, 1, 0, 0, 1), 100.1),
]
ask_data = [
    (datetime(2020, 1, 1, 0, 0, 0), 100.5),
    (datetime(2020, 1, 1, 0, 0, 1), 100.6),
]

# 1. Convert to dictionaries for fast timestamp lookup
bid_dict = dict(bid_data)
ask_dict = dict(ask_data)

# 2. Find all timestamps present in both streams (synchronization)
common_timestamps = sorted(set(bid_dict.keys()) & set(ask_dict.keys()))

# 3. For each synchronized timestamp, calculate the spread
for ts in common_timestamps:
    bid = bid_dict[ts]
    ask = ask_dict[ts]
    spread = ask - bid
    print(f"{ts} SPREAD: {spread}")
2. 
3. Can we use it for multivariable / Timeseries Researh?


# 250906
1. How it implmented by python so it can implment timeseries? class node
2. Alarms in csp vs Pintos

csp.alarm	Timer interrupts or kernel alarms (e.g., sleep in Pintos)
csp.schedule_alarm	Scheduling tasks in the future (e.g., adding a thread to a sleep queue in Pintos)
csp.ticked	Event-driven execution (similar to an interrupt handler waking up a thread)
csp.graph	A directed acyclic graph (DAG) of tasks (similar to a dependency graph in task scheduling)
csp.node	A single task or process (similar to a thread or kernel task)
csp.run	The execution engine (similar to the OS scheduler running tasks)
Historical Buffers	Similar to maintaining a history of events in a kernel log or event queue
PushMode	Similar to different scheduling policies (e.g., preemptive vs. non-preemptive scheduling)