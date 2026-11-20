# CNN-Transformer Autoscaling Baseline

## Workload
- Dataset: FIFA World Cup 1998 HTTP request trace
- Selected workload: seven days, aggregated into one-minute intervals
- Evaluation: chronological held-out test period

## Forecasting
- Model: CNN-Transformer
- Input sequence length: 16 minutes
- Prediction horizon: one minute
- Scaler fitted on training data only

## Autoscaling
- Replica capacity: 100 requests/minute
- Minimum replicas: 1
- Maximum replicas: 20
- Scaling delay: zero minutes (immediate)
- Initial replica count: minimum replicas
- Reactive policy: uses the previous minute's observed demand
- Predictive policy: uses the forecast for the current minute

## Limitations
- Replica capacity is a simulation assumption.
- Scaling is assumed to take effect immediately.
- Overload represents demand exceeding simulated capacity,
  not measured application latency.
- Results apply to this workload subset and evaluation period.
