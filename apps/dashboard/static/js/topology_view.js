class TopologyViewer {
    constructor(canvasId, onSelectService) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.onSelectService = onSelectService;
        this.nodes = [];
        this.links = [];
        this.selectedService = null;
        this.nodePositions = {
            'edge-gateway': { x: 70, y: 190 },
            'auth-service': { x: 220, y: 70 },
            'catalog-service': { x: 220, y: 160 },
            'checkout-api': { x: 220, y: 260 },
            'order-service': { x: 370, y: 220 },
            'payment-service': { x: 500, y: 150 },
            'inventory-service': { x: 500, y: 250 },
            'notification-service': { x: 370, y: 330 },
        };

        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.render();
    }

    loadData(data) {
        this.nodes = data.nodes || [];
        this.links = data.links || [];
        this.render();
    }

    setSelected(serviceId) {
        this.selectedService = serviceId;
        this.render();
    }

    handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        for (const [id, pos] of Object.entries(this.nodePositions)) {
            const dist = Math.hypot(mouseX - pos.x, mouseY - pos.y);
            if (dist <= 22) {
                this.setSelected(id);
                if (this.onSelectService) {
                    this.onSelectService(id);
                }
                break;
            }
        }
    }

    render() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw links with arrows
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1.8;
        for (const link of this.links) {
            const src = this.nodePositions[link.source];
            const dst = this.nodePositions[link.target];
            if (src && dst) {
                ctx.beginPath();
                ctx.moveTo(src.x, src.y);
                ctx.lineTo(dst.x, dst.y);
                ctx.stroke();

                // Draw subtle directional marker
                const mx = (src.x + dst.x) / 2;
                const my = (src.y + dst.y) / 2;
                ctx.fillStyle = '#94a3b8';
                ctx.beginPath();
                ctx.arc(mx, my, 2.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        // Map status from nodes list
        const statusMap = {};
        for (const n of this.nodes) {
            statusMap[n.id] = n.status || 'HEALTHY';
        }

        // Draw nodes
        for (const [id, pos] of Object.entries(this.nodePositions)) {
            const status = statusMap[id] || 'HEALTHY';
            const isSelected = this.selectedService === id;

            let strokeColor = '#0284c7'; // Healthy blue-cyan
            let fillColor = '#ffffff';
            let statusDot = '#16a34a';

            if (status === 'CRITICAL') {
                strokeColor = '#dc2626';
                statusDot = '#dc2626';
            } else if (status === 'DEGRADED') {
                strokeColor = '#d97706';
                statusDot = '#d97706';
            }

            // Selection halo
            if (isSelected) {
                ctx.strokeStyle = '#2563eb';
                ctx.lineWidth = 4;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 22, 0, 2 * Math.PI);
                ctx.stroke();
            }

            // Node Circle
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = isSelected ? 2.5 : 2;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 16, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();

            // Status Indicator Dot inside circle
            ctx.fillStyle = statusDot;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
            ctx.fill();

            // Label
            ctx.fillStyle = isSelected ? '#1e40af' : '#0f172a';
            ctx.font = isSelected ? 'bold 11px sans-serif' : '500 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(id, pos.x, pos.y + 28);
        }
    }
}
