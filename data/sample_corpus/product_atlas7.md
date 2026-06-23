# Atlas-7 Mobile Picking Robot

The **Atlas-7** is Nimbus Robotics' third-generation autonomous mobile robot
(AMR), designed for shelf-to-station picking in warehouses.

## Specifications

- **Payload:** up to 35 kilograms per trip.
- **Top speed:** 2.1 meters per second when carrying a load, 3.0 meters per
  second when empty.
- **Battery:** a swappable 48-volt lithium-iron-phosphate (LiFePO4) pack giving
  roughly 9 hours of continuous operation. Hot-swapping a depleted pack takes
  under 30 seconds and requires no tools.
- **Navigation:** a fused LiDAR plus stereo-camera system. The robot builds and
  continuously updates a 2D occupancy map and localizes against it.
- **Safety:** dual redundant emergency-stop circuits and a 360-degree bumper
  sensor. The robot halts within 15 centimeters when a person steps into its
  path.

## Picking arm

The optional Atlas-7P variant adds a 6-axis arm with a vacuum gripper that can
handle items up to 5 kilograms. The arm is not included in the base model.

## Firmware

Atlas-7 runs the **NimbusOS** real-time firmware. Updates are delivered
over-the-air through Conductor and are staged: a new firmware build is rolled
out to 5 percent of a fleet first, monitored for one hour, then promoted to the
rest of the fleet if no fault threshold is crossed.
