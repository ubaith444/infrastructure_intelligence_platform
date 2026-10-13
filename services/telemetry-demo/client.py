"""Order Service Telemetry Traffic Generator & Chaos Fault Injector.

Generates realistic application load for the demo microservice, producing
real OpenTelemetry traces, latency metrics, and error rates that stream
end-to-end through the Infrastructure Intelligence Platform.

Usage:
    # Run 50 normal order transactions
    python services/telemetry-demo/client.py --mode normal --count 50

    # Inject a 350ms latency spike fault scenario
    python services/telemetry-demo/client.py --mode latency-spike --count 20

    # Inject a 40% error burst fault scenario
    python services/telemetry-demo/client.py --mode error-burst --count 20

    # Run continuous realistic mixed traffic
    python services/telemetry-demo/client.py --mode mixed --rate 5 --count 0
"""

import argparse
import asyncio
from datetime import datetime, timezone
import random
import sys
import time
from typing import Dict, List, Optional
import httpx

CATALOG_ITEMS = [
    {"item_id": "item-101", "name": "Standard Mechanical Keyboard", "price": 89.99},
    {"item_id": "item-102", "name": "Ergonomic Optical Mouse", "price": 49.50},
    {"item_id": "item-103", "name": "USB-C Multi-Port Dock", "price": 119.00},
    {"item_id": "item-104", "name": "27-inch 4K Monitor", "price": 349.99},
    {"item_id": "item-105", "name": "Noise Cancelling Headset", "price": 199.95},
]


async def run_traffic_generator(
    target_url: str = "http://127.0.0.1:8001",
    mode: str = "normal",
    rate_rps: float = 5.0,
    count: int = 50,
):
    print("=" * 70)
    print(" [Order Service] OpenTelemetry Traffic Generator")
    print(f" Target Service: {target_url}")
    print(f" Mode:           {mode.upper()}")
    print(f" Request Rate:   {rate_rps} req/sec")
    print(f" Total Requests: {'Infinite (Ctrl+C to stop)' if count == 0 else count}")
    print("=" * 70)

    sleep_interval = 1.0 / max(0.1, rate_rps)
    sent_count = 0
    success_count = 0
    error_count = 0
    latencies: List[float] = []

    async with httpx.AsyncClient(base_url=target_url, timeout=10.0) as client:
        # Check health first
        try:
            h_res = await client.get("/health")
            if h_res.status_code == 200:
                print(f"[+] Connected to order-service: {h_res.json()}")
            else:
                print(f"[!] Warning: health check returned status {h_res.status_code}")
        except Exception as e:
            print(f"[-] Cannot connect to order-service at {target_url}: {e}")
            print("  Ensure order-service is running on port 8001.")
            return

        print("\nStreaming requests...")
        try:
            while count == 0 or sent_count < count:
                sent_count += 1
                items_to_buy = random.sample(CATALOG_ITEMS, k=random.randint(1, 3))
                order_items = [
                    {"item_id": it["item_id"], "name": it["name"], "quantity": random.randint(1, 2), "price": it["price"]}
                    for it in items_to_buy
                ]

                # Determine fault parameters based on mode
                delay_ms = 0.0
                simulate_error = False

                if mode == "normal":
                    delay_ms = random.uniform(8.0, 25.0)
                    simulate_error = False
                elif mode == "latency-spike":
                    delay_ms = random.uniform(320.0, 480.0)
                    simulate_error = random.random() < 0.05
                elif mode == "error-burst":
                    delay_ms = random.uniform(15.0, 45.0)
                    simulate_error = random.random() < 0.60
                elif mode == "traffic-surge":
                    delay_ms = random.uniform(20.0, 60.0)
                    simulate_error = random.random() < 0.10
                elif mode == "mixed":
                    # 85% normal, 10% latency, 5% error
                    roll = random.random()
                    if roll < 0.85:
                        delay_ms = random.uniform(10.0, 30.0)
                        simulate_error = False
                    elif roll < 0.95:
                        delay_ms = random.uniform(250.0, 400.0)
                        simulate_error = False
                    else:
                        delay_ms = random.uniform(15.0, 40.0)
                        simulate_error = True

                payload = {
                    "customer_id": f"cust-{random.randint(100, 999)}",
                    "items": order_items,
                    "simulate_delay_ms": delay_ms,
                    "simulate_error": simulate_error,
                }

                t0 = time.perf_counter()
                status_str = "OK"
                try:
                    res = await client.post("/orders", json=payload)
                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(duration_ms)

                    if res.status_code == 200:
                        success_count += 1
                        order_id = res.json().get("order_id", "unknown")
                        status_str = f"200 OK [{order_id}]"
                    else:
                        error_count += 1
                        status_str = f"{res.status_code} ERROR [{res.text[:30]}]"
                except Exception as ex:
                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(duration_ms)
                    error_count += 1
                    status_str = f"FAILED [{ex}]"

                avg_lat = sum(latencies[-20:]) / len(latencies[-20:]) if latencies else 0.0
                err_pct = (error_count / sent_count) * 100.0
                print(
                    f"[{sent_count:03d}] {status_str:<28} | "
                    f"Latency: {duration_ms:5.1f}ms (avg {avg_lat:5.1f}ms) | "
                    f"Err Rate: {err_pct:4.1f}%"
                )

                if mode == "traffic-surge":
                    await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("\nStopped by user.")

    print("\n" + "=" * 70)
    print(" Run Summary:")
    print(f" Total Requests:    {sent_count}")
    print(f" Successful Orders: {success_count} ({success_count/max(1,sent_count)*100:.1f}%)")
    print(f" Errors/Failures:   {error_count} ({error_count/max(1,sent_count)*100:.1f}%)")
    if latencies:
        print(f" Latency Min/Avg/Max: {min(latencies):.1f}ms / {sum(latencies)/len(latencies):.1f}ms / {max(latencies):.1f}ms")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Order Service OpenTelemetry Traffic Generator")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="Target order service URL (default: http://127.0.0.1:8001)")
    parser.add_argument(
        "--mode",
        choices=["normal", "latency-spike", "error-burst", "traffic-surge", "mixed"],
        default="normal",
        help="Traffic mode (normal, latency-spike, error-burst, traffic-surge, mixed)",
    )
    parser.add_argument("--rate", type=float, default=5.0, help="Requests per second (default: 5.0)")
    parser.add_argument("--count", type=int, default=30, help="Total requests to send, 0 for continuous (default: 30)")
    args = parser.parse_args()

    asyncio.run(
        run_traffic_generator(
            target_url=args.url,
            mode=args.mode,
            rate_rps=args.rate,
            count=args.count,
        )
    )


if __name__ == "__main__":
    main()
