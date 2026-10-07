# Tern-3 Autonomous Cart — Operator Manual

Ottermere Labs · Document OM-T3 · Revision F · Applies to firmware 4.2.x

## 1. Overview
The Tern-3 is an autonomous mobile cart built for warehouse aisles and loading docks. It follows a site map created during commissioning, avoids people and obstacles with a 360-degree lidar and two depth cameras, and reports its position to the Ottermere Fleet Manager over MQTT. One operator can supervise up to 15 carts from the Fleet Manager console.

The cart is not rated for outdoor use, for ramps steeper than 6 percent, or for floors that are wet enough to leave visible standing water.

## 2. Specifications
- Rated payload: 120 kg, evenly distributed on the top deck.
- Maximum travel speed: 1.8 m/s on open aisles; the cart limits itself to 0.6 m/s in zones marked as shared with pedestrians.
- Battery: 48 V lithium iron phosphate pack, 60 Ah.
- Runtime: up to 9.5 hours at rated payload on level floors.
- Charging: 2 hours 15 minutes from 10 percent to 80 percent on the Dock-C charger; a full charge takes about 3 hours 40 minutes.
- Deck dimensions: 1,100 mm by 700 mm.
- Unladen mass: 168 kg.
- Operating temperature: 2 °C to 40 °C.
- Ingress protection: IP54.

## 3. Safety zones
The safety controller enforces two concentric zones around the cart. When a person or obstacle enters the slow zone, which extends 1.2 m from the chassis, the cart drops to 0.3 m/s. When anything enters the stop zone, which extends 0.5 m from the chassis, the cart brakes to a full stop and waits. After the stop zone has been clear for 3 seconds the cart resumes on its own.

The red emergency stop button on each side of the cart cuts drive power immediately. Releasing the button does not restart motion; an operator must acknowledge the stop on the cart's touchscreen or in the Fleet Manager.

## 4. Error codes
| Code | Meaning | Operator action |
| --- | --- | --- |
| E-104 | Lidar view obstructed or lens dirty | Wipe the lidar window with a dry microfiber cloth and clear the alert. |
| E-221 | Drive motor overcurrent | Check the wheels for debris or wrapped film; if it repeats within an hour, take the cart out of service. |
| E-317 | Battery below the reserve threshold | The cart returns to its dock automatically; do not assign new missions until it reports 30 percent. |
| E-455 | Emergency stop engaged | Release the red button and acknowledge the stop on the touchscreen. |
| E-512 | Localization lost | Push the cart to the nearest marked re-localization tag and select Recover. |
| E-630 | Fleet Manager connection lost for more than 60 seconds | The cart finishes its current leg, parks, and waits for the connection to return. |

## 5. Maintenance schedule
- Every shift: wipe the lidar window and both depth camera lenses; check that the emergency stop buttons latch.
- Every 250 operating hours: inspect the drive wheels for flat spots and the casters for hair or film wrapping.
- Every 1,000 operating hours: replace the drive wheel treads and recalibrate the depth cameras.
- Every 2,000 operating hours or 12 months, whichever comes first: a certified Ottermere technician performs a brake test and battery health check.

Battery packs are warranted for 2,500 full charge cycles or 3 years. A pack whose capacity falls below 70 percent of its rating is replaced under warranty.

## 6. Firmware 4.2.1 release notes
Firmware 4.2.1 shortens the stop-zone resume delay from 5 seconds to 3 seconds, adds error code E-630, and fixes a bug where carts parked at Dock-C could report 0 percent charge after a power blip. Updates are pushed from the Fleet Manager during the maintenance window a site chooses; carts never update while they are carrying a load.

## 7. Contacts
Field service requests go through the Ottermere support portal. Sites on the Premium support tier can also call the 24-hour line printed on the inside of the battery door.
