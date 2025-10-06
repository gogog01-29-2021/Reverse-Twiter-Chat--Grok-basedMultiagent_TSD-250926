# Expanded NanoGPT vs CSP Code Examples

## **1. Conceptual Differences - Code Examples**

### **NanoGPT: Static Computation Graph - Detailed Code**
```python
# Complete static graph definition - everything defined at initialization
class GPTConfig:
    block_size: int = 1024  # Context length
    vocab_size: int = 50304  # GPT-2 vocab size
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768       # Embedding dimension

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Static architecture - never changes during runtime
        self.config = config

        # Fixed computation pipeline
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),   # Position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias),     # Final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (static relationship)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # Same computation path every time - no branching based on data content
        device = idx.device
        b, t = idx.size()

        # Token and position embeddings (deterministic)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # [0, 1, 2, ..., t-1]
        tok_emb = self.transformer.wte(idx)  # [batch, seq_len, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [seq_len, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)

        # Fixed sequence of transformations
        for block in self.transformer.h:
            x = block(x)  # Each block processes entire batch identically

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training mode - compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode - only compute logits for last token
            logits = self.lm_head(x[:, [-1], :])  # [batch, 1, vocab_size]
            loss = None

        return logits, loss

# Usage pattern - batch processing only
def generate_text(model, prompt_tokens, max_new_tokens=100):
    """Static generation - processes all tokens in lockstep"""
    model.eval()
    for _ in range(max_new_tokens):
        # Get predictions for current sequence
        with torch.no_grad():
            logits, _ = model(prompt_tokens)
            # Apply temperature/sampling
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            prompt_tokens = torch.cat((prompt_tokens, next_token), dim=1)
    return prompt_tokens
```

### **CSP: Dynamic Event-Driven Graph - Detailed Code**
```python
# Dynamic graph that changes behavior based on incoming events
import csp
from datetime import datetime, timedelta
from typing import Optional

class MarketState(csp.Struct):
    """Dynamic state that changes with market conditions"""
    volatility: float
    trend: str  # "bull", "bear", "sideways"
    volume_profile: str  # "high", "low", "normal"
    session: str  # "pre_market", "market_hours", "after_hours"

@csp.node
def adaptive_signal_generator(
    price: ts[float],
    volume: ts[int],
    market_state: ts[MarketState]
) -> ts[Optional[str]]:
    """
    Dynamic node that changes its behavior based on market conditions
    - Different algorithms for different market states
    - Adapts thresholds based on volatility
    - Changes frequency based on session
    """

    with csp.state():
        s_price_buffer = []
        s_volume_buffer = []
        s_current_strategy = "conservative"
        s_signal_cooldown = 0
        s_adaptive_threshold = 0.02  # Starting threshold

    with csp.start():
        # Different buffer sizes for different strategies
        csp.set_buffering_policy(price, tick_count=100)
        csp.set_buffering_policy(volume, tick_count=50)

    # Market state changes trigger strategy adaptation
    if csp.ticked(market_state):
        if market_state.volatility > 0.05:
            s_current_strategy = "high_volatility"
            s_adaptive_threshold = 0.05  # Wider threshold for volatile markets
        elif market_state.volatility < 0.01:
            s_current_strategy = "low_volatility"
            s_adaptive_threshold = 0.005  # Tighter threshold for calm markets
        else:
            s_current_strategy = "normal"
            s_adaptive_threshold = 0.02

        print(f"{csp.now()}: Strategy changed to {s_current_strategy}, threshold={s_adaptive_threshold}")

    if csp.ticked(price) and csp.valid(volume):
        s_price_buffer.append(price)
        s_volume_buffer.append(volume)

        # Keep only recent data
        if len(s_price_buffer) > 20:
            s_price_buffer.pop(0)
            s_volume_buffer.pop(0)

        # Cooldown mechanism
        if s_signal_cooldown > 0:
            s_signal_cooldown -= 1
            return None

        # Dynamic algorithm selection based on current state
        if len(s_price_buffer) >= 5:
            if s_current_strategy == "high_volatility":
                # More conservative in volatile markets
                signal = _high_vol_strategy(s_price_buffer, s_volume_buffer, s_adaptive_threshold)
            elif s_current_strategy == "low_volatility":
                # More aggressive in calm markets
                signal = _low_vol_strategy(s_price_buffer, s_volume_buffer, s_adaptive_threshold)
            else:
                # Standard strategy
                signal = _normal_strategy(s_price_buffer, s_volume_buffer, s_adaptive_threshold)

            if signal:
                s_signal_cooldown = 5  # Dynamic cooldown
                return signal

    return None

def _high_vol_strategy(prices, volumes, threshold):
    """Conservative strategy for high volatility"""
    if len(prices) < 10:
        return None

    recent_avg = sum(prices[-10:]) / 10
    current_price = prices[-1]
    avg_volume = sum(volumes[-5:]) / 5

    # Require stronger signals and higher volume
    if current_price > recent_avg * (1 + threshold * 2) and avg_volume > 1500:
        return "STRONG_BUY"
    elif current_price < recent_avg * (1 - threshold * 2) and avg_volume > 1500:
        return "STRONG_SELL"
    return None

def _low_vol_strategy(prices, volumes, threshold):
    """Aggressive strategy for low volatility"""
    if len(prices) < 5:
        return None

    recent_avg = sum(prices[-5:]) / 5
    current_price = prices[-1]

    # More sensitive to small moves
    if current_price > recent_avg * (1 + threshold * 0.5):
        return "BUY"
    elif current_price < recent_avg * (1 - threshold * 0.5):
        return "SELL"
    return None

def _normal_strategy(prices, volumes, threshold):
    """Standard strategy"""
    if len(prices) < 7:
        return None

    recent_avg = sum(prices[-7:]) / 7
    current_price = prices[-1]
    current_volume = volumes[-1]

    if current_price > recent_avg * (1 + threshold) and current_volume > 1000:
        return "BUY"
    elif current_price < recent_avg * (1 - threshold) and current_volume > 1000:
        return "SELL"
    return None
```

---

## **2. Performance Characteristics - Detailed Code**

### **NanoGPT: GPU-Optimized Batch Processing**
```python
import torch
import time
from torch.utils.data import DataLoader

class GPTPerformanceOptimizer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def benchmark_throughput(self, batch_sizes=[1, 4, 8, 16, 32]):
        """Measure tokens/second for different batch sizes"""
        results = {}

        for batch_size in batch_sizes:
            # Create dummy input
            seq_len = 512
            dummy_input = torch.randint(0, 50000, (batch_size, seq_len), device=self.device)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            torch.cuda.synchronize()

            # Benchmark
            num_runs = 100
            start_time = time.time()

            for _ in range(num_runs):
                with torch.no_grad():
                    logits, _ = self.model(dummy_input)
            torch.cuda.synchronize()

            end_time = time.time()
            elapsed = end_time - start_time

            tokens_processed = batch_size * seq_len * num_runs
            tokens_per_second = tokens_processed / elapsed

            results[batch_size] = {
                'tokens_per_second': tokens_per_second,
                'latency_per_batch': elapsed / num_runs,
                'memory_used_gb': torch.cuda.max_memory_allocated() / 1e9
            }

        return results

    def memory_optimization_example(self):
        """Show memory optimization techniques"""

        # Gradient checkpointing to save memory
        def forward_with_checkpointing(self, x):
            # Trade compute for memory
            x = torch.utils.checkpoint.checkpoint(self.transformer.h[0], x)
            for layer in self.transformer.h[1:]:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            return x

        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        def training_step_optimized(self, batch):
            with torch.cuda.amp.autocast():
                logits, loss = self.model(batch['input'], batch['target'])

            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

### **CSP: Real-Time Event Processing**
```python
import csp
import time
import psutil
from collections import deque
from datetime import datetime, timedelta

@csp.node
def latency_critical_processor(price: ts[float]) -> ts[str]:
    """Ultra-low latency processing example"""

    with csp.state():
        s_last_price = None
        s_process_count = 0

    if csp.ticked(price):
        # Minimal processing for ultra-low latency
        s_process_count += 1

        # Simple threshold check (microsecond-level processing)
        if s_last_price is not None:
            price_change = (price - s_last_price) / s_last_price

            # Immediate decision with no complex calculations
            if abs(price_change) > 0.001:  # 0.1% change
                s_last_price = price
                return "ALERT" if price_change > 0 else "WARNING"

        s_last_price = price

    return None

@csp.node
def high_frequency_aggregator(trades: ts['Trade']) -> ts[dict]:
    """Process high-frequency trading data with minimal latency"""

    with csp.state():
        s_volume_1s = 0
        s_value_1s = 0.0
        s_trade_count = 0
        s_last_vwap = 0.0

    with csp.alarms():
        # Reset aggregation every second
        reset_alarm = csp.alarm(bool)

    with csp.start():
        # Schedule first reset
        csp.schedule_alarm(reset_alarm, timedelta(seconds=1), True)

    if csp.ticked(trades):
        # Ultra-fast aggregation
        s_volume_1s += trades.volume
        s_value_1s += trades.price * trades.volume
        s_trade_count += 1

        # Real-time VWAP calculation
        if s_volume_1s > 0:
            s_last_vwap = s_value_1s / s_volume_1s

    if csp.ticked(reset_alarm):
        # Output aggregated data and reset
        result = {
            'vwap': s_last_vwap,
            'volume': s_volume_1s,
            'trade_count': s_trade_count,
            'timestamp': csp.now()
        }

        # Reset for next second
        s_volume_1s = 0
        s_value_1s = 0.0
        s_trade_count = 0

        # Schedule next reset
        csp.schedule_alarm(reset_alarm, timedelta(seconds=1), True)

        return result
```

---

## **3. Use Case Alignment - Detailed Examples**

### **NanoGPT Best For - Comprehensive Examples**
```python
# 1. Natural Language Processing Pipeline
class NLPPipeline:
    def __init__(self, model_path):
        self.model = load_nanogpt_model(model_path)
        self.tokenizer = load_tokenizer()

    def text_generation_batch(self, prompts: list[str], max_length=100):
        """Process multiple text generation requests efficiently"""
        # Batch encode prompts
        encoded_prompts = [self.tokenizer.encode(prompt) for prompt in prompts]

        # Pad to same length for batch processing
        max_prompt_len = max(len(p) for p in encoded_prompts)
        batched_input = torch.zeros(len(prompts), max_prompt_len, dtype=torch.long)

        for i, prompt in enumerate(encoded_prompts):
            batched_input[i, :len(prompt)] = torch.tensor(prompt)

        # Generate in batch - much more efficient than one-by-one
        generated = self.model.generate(
            batched_input,
            max_new_tokens=max_length,
            temperature=0.8,
            top_k=50
        )

        # Decode results
        return [self.tokenizer.decode(seq) for seq in generated]

    def fine_tuning_example(self, training_data):
        """Fine-tune model on domain-specific data"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in training_data:
                # Forward pass
                logits, loss = self.model(batch['input'], batch['target'])

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Log metrics
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### **CSP Best For - Comprehensive Examples**
```python
# 1. Real-Time Trading Systems
@csp.node
def risk_management_system(
    positions: ts['PositionUpdate'],
    pnl: ts[float],
    market_volatility: ts[float]
) -> ts['RiskAction']:
    """Real-time risk monitoring and automatic controls"""

    with csp.state():
        s_max_drawdown = 0.0
        s_position_concentration = {}
        s_var_95 = 0.0

    with csp.alarms():
        risk_check_alarm = csp.alarm(bool)

    with csp.start():
        # Check risk every 100ms
        csp.schedule_alarm(risk_check_alarm, timedelta(milliseconds=100), True)

    if csp.ticked(positions):
        # Update position tracking
        s_position_concentration[positions.symbol] = positions.quantity

    if csp.ticked(pnl):
        # Track drawdown
        if pnl < s_max_drawdown:
            s_max_drawdown = pnl

    if csp.ticked(risk_check_alarm):
        # Comprehensive risk check
        total_exposure = sum(abs(qty) for qty in s_position_concentration.values())

        # Risk limits
        if s_max_drawdown < -10000:  # $10k max loss
            return RiskAction(action="EMERGENCY_STOP", reason="Max drawdown exceeded")

        if total_exposure > 1000000:  # $1M max exposure
            return RiskAction(action="REDUCE_POSITIONS", reason="Exposure limit")

        # Reschedule next check
        csp.schedule_alarm(risk_check_alarm, timedelta(milliseconds=100), True)

# 2. IoT Sensor Processing
@csp.node
def sensor_data_processor(
    temperature: ts[float],
    humidity: ts[float],
    pressure: ts[float],
    sensor_id: str
) -> csp.Outputs(
    anomaly_alert=ts[str],
    status_update=ts[dict],
    maintenance_needed=ts[bool]
):
    """Process IoT sensor data with anomaly detection"""

    with csp.state():
        s_temp_history = deque(maxlen=100)
        s_humidity_history = deque(maxlen=100)
        s_pressure_history = deque(maxlen=100)
        s_sensor_health = 1.0
        s_last_maintenance = datetime.now()

    # Temperature monitoring
    if csp.ticked(temperature):
        s_temp_history.append(temperature)

        # Anomaly detection
        if len(s_temp_history) >= 10:
            recent_avg = sum(list(s_temp_history)[-10:]) / 10
            if abs(temperature - recent_avg) > 5.0:  # 5 degree threshold
                csp.output(anomaly_alert=f"Temperature anomaly on {sensor_id}: {temperature}°C")

    # Sensor health assessment
    if csp.ticked(temperature, humidity, pressure):
        # Status update
        status = {
            'sensor_id': sensor_id,
            'health_score': s_sensor_health,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'timestamp': csp.now()
        }
        csp.output(status_update=status)
```

---

## **4. Hybrid Integration Example**

```python
@csp.node
def llm_trading_advisor(market_news: ts[str]) -> ts[str]:
    """Use NanoGPT for text analysis within CSP framework"""

    with csp.state():
        s_gpt_model = None  # Load model once
        s_analysis_cache = {}

    with csp.start():
        # Initialize GPT model
        import torch
        s_gpt_model = load_nanogpt_model()

    if csp.ticked(market_news):
        # Check cache first
        if market_news in s_analysis_cache:
            return s_analysis_cache[market_news]

        # Use GPT for analysis
        prompt = f"Analyze this market news for trading signals: {market_news}"

        with torch.no_grad():
            analysis = s_gpt_model.generate(prompt, max_length=100)

        # Extract trading signal
        if "bullish" in analysis.lower():
            signal = "BUY_SIGNAL"
        elif "bearish" in analysis.lower():
            signal = "SELL_SIGNAL"
        else:
            signal = "HOLD"

        # Cache result
        s_analysis_cache[market_news] = signal

        return signal

@csp.graph
def ai_powered_trading_system():
    """Combine NanoGPT intelligence with CSP real-time processing"""

    # Real-time data feeds
    market_data = market_data_feed()
    news_feed = news_feed()

    # AI-powered analysis
    llm_signals = llm_trading_advisor(news_feed)

    # Traditional technical analysis
    technical_signals = technical_analysis(market_data)

    # Combine AI and traditional signals
    combined_signals = signal_fusion(llm_signals, technical_signals)

    # Risk management
    risk_adjusted_signals = risk_filter(combined_signals, market_data)

    csp.print("ai_signals", llm_signals)
    csp.print("final_signals", risk_adjusted_signals)
```

This expanded version provides comprehensive code examples for all the conceptual sections, showing practical implementations of both NanoGPT's static computation model and CSP's dynamic event-driven architecture.