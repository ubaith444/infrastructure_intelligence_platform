import React, { useEffect, useState } from 'react';
import { ServiceHealth, SecurityStatus } from './types';
import { ServiceHealthGrid } from './components/ServiceHealthGrid';

export const App: React.FC = () => {
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [security, setSecurity] = useState<SecurityStatus | null>(null);

  const fetchData = async () => {
    try {
      const [svcRes, secRes] = await Promise.all([
        fetch('/api/services/health').then((r) => r.json()),
        fetch('/api/security/status').then((r) => r.json()),
      ]);
      if (svcRes?.services) setServices(svcRes.services);
      if (secRes) setSecurity(secRes);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', background: '#f8fafc', minHeight: '100vh', color: '#0f172a' }}>
      <header style={{ background: '#ffffff', borderBottom: '1px solid #e2e8f0', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>Infrastructure Intelligence Platform</h1>
          <span style={{ fontSize: '12px', color: '#64748b' }}>Operations Console & Anomaly Arbiter</span>
        </div>
        {security && (
          <div style={{ display: 'flex', gap: '12px', fontSize: '12px' }}>
            <span style={{ background: '#dcfce7', color: '#166534', padding: '4px 10px', borderRadius: '4px', fontWeight: 600 }}>
              AUTH ACTIVE
            </span>
          </div>
        )}
      </header>
      <main style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
        <h2 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Service Telemetry & Health</h2>
        <ServiceHealthGrid services={services} />
      </main>
    </div>
  );
};
