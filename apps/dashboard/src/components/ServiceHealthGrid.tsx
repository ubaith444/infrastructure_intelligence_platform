import React from 'react';
import { ServiceHealth } from '../types';

export const ServiceHealthGrid: React.FC<{ services: ServiceHealth[] }> = ({ services }) => {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
      {services.map((svc) => {
        const isHealthy = svc.status === 'HEALTHY';
        const isCritical = svc.status === 'CRITICAL';
        return (
          <div
            key={svc.service_id}
            style={{
              background: '#ffffff',
              border: `1px solid ${isCritical ? '#ef4444' : isHealthy ? '#e2e8f0' : '#f59e0b'}`,
              borderRadius: '6px',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <strong>{svc.name}</strong>
              <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', background: isHealthy ? '#dcfce7' : '#fee2e2', color: isHealthy ? '#166534' : '#991b1b', fontWeight: 600 }}>
                {svc.status}
              </span>
            </div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              Req: {svc.request_rate} req/s | Err: {(svc.error_rate * 100).toFixed(2)}% | p90: {svc.latency_p90} ms
            </div>
          </div>
        );
      })}
    </div>
  );
};
