export interface ServiceHealth {
  service_id: string;
  name: string;
  tier: string;
  status: 'HEALTHY' | 'DEGRADED' | 'CRITICAL';
  request_rate: number;
  error_rate: number;
  latency_p50: number;
  latency_p90: number;
  latency_p99: number;
  cpu_utilization: number;
  memory_utilization: number;
  incident_risk_probability: number;
  risk_band: 'LOW' | 'HIGH' | 'CRITICAL';
}

export interface SecurityStatus {
  auth_enabled: boolean;
  active_key_count: number;
  client_rate_limit: {
    current_rpm: number;
    max_rpm: number;
    remaining_rpm: number;
  };
}
