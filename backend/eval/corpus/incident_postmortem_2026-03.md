# Postmortem: Fleet Manager outage on 14 March 2026

Status: Final · Author: Site Reliability · Review date: 21 March 2026

## Summary
On 14 March 2026 the Ottermere Fleet Manager could not reach any Tern-3 carts for 47 minutes, from 06:12 to 06:59 UTC. Carts finished their current legs, parked, and showed error E-630 as designed. 312 carts across 4 customer sites were affected. No cart collided with anything and no load was dropped, but customers lost roughly 230 cart-hours of work during their morning shifts.

## Impact
- Sites affected: Leeds, Tilburg, Gdańsk, and Porto.
- Carts affected: 312.
- Customer-facing duration: 47 minutes.
- Support tickets opened: 58, all closed by 16 March.
- SLA credits issued: two customers on the Scale plan received service credits under the uptime commitment.

## Root cause
The TLS certificate on the MQTT broker that carts use to talk to the Fleet Manager expired at 06:12 UTC. The certificate had been issued by hand eighteen months earlier, and the renewal reminder went to the inbox of an engineer who had since changed teams. When the certificate expired, every cart's connection was refused at the next reconnect.

## Detection and response
Synthetic monitoring alerted the on-call engineer at 06:15 UTC, three minutes after the expiry. The first 25 minutes were spent ruling out a network partition, because the dashboard showed broker health as green: the health check connected without verifying the certificate chain. A replacement certificate was issued at 06:51 UTC and connections recovered fully by 06:59 UTC.

## What went well
Carts degraded safely: every cart parked within its current aisle and no operator had to intervene physically. The E-630 behaviour shipped in firmware 4.2.1 worked as intended.

## Action items
| Item | Owner | Due |
| --- | --- | --- |
| Automate certificate issuance and renewal for every broker | Priya Nandakumar | 30 April 2026 |
| Make the broker health check verify the full certificate chain | Tomasz Wierzbicki | 4 April 2026 |
| Alert 30 days and 7 days before any certificate expires, to a team channel, not a person | Ines Carvalho | 11 April 2026 |
| Add a runbook entry for "all carts show E-630" | Site Reliability | 28 March 2026 |

## Lessons
Renewal of anything that expires must be automated or owned by a rotation, never by one person's calendar. Health checks must exercise the same path real clients use.
