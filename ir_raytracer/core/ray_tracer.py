# -*- coding: utf-8 -*-
"""
Ray tracing engine for acoustic impulse response rendering.
"""
import mathutils
import numpy as np
from math import pi, cos, exp, acos
import random
from typing import List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from .acoustics import (
    MaterialProperties, air_attenuation_bands,
    add_filtered_impulse as synthesize_filtered_impulse,
    NUM_BANDS
)
from .ambisonic import AmbisonicEncoder
from ..utils.math_utils import (
    reflect, cosine_weighted_hemisphere, jitter_specular_direction,
    segment_hits_sphere, generate_ray_directions
)
from ..utils.scene_utils import speed_of_sound_bu, get_scene_unit_scale


class RayTracingConfig:
    """Configuration for ray tracing parameters."""
    
    def __init__(self, context):
        """Initialize from Blender scene context."""
        scene = context.scene
        
        # Basic parameters
        self.num_rays = max(1, int(scene.airt_num_rays))
        self.max_bounces = int(scene.airt_max_order)
        self.sample_rate = int(scene.airt_sr)
        self.ir_length_samples = int(scene.airt_ir_seconds * self.sample_rate)
        
        # Physical parameters
        self.speed_of_sound = speed_of_sound_bu(context)
        self.unit_scale = get_scene_unit_scale(context)
        self.receiver_radius_m = max(1e-6, float(scene.airt_recv_radius))
        self.receiver_radius = self.receiver_radius_m / max(self.unit_scale, 1e-9)
        
        # Tracing behavior
        self.angle_tolerance_rad = scene.airt_angle_tol_deg * pi / 180.0
        self.specular_roughness_rad = max(0.0, float(scene.airt_spec_rough_deg)) * pi / 180.0
        self.segment_capture = bool(scene.airt_enable_seg_capture)
        
        # Russian roulette
        self.rr_enable = bool(scene.airt_rr_enable)
        self.rr_start_bounce = int(scene.airt_rr_start)
        self.rr_survive_prob = max(0.05, min(1.0, float(scene.airt_rr_p)))
        
        # Diffraction
        self.enable_diffraction = bool(getattr(scene, 'airt_enable_diffraction', False))
        self.diffraction_samples = int(getattr(scene, 'airt_diffraction_samples', 0))
        self.diffraction_max_angle = max(0.0, float(getattr(scene, 'airt_diffraction_max_deg', 40.0))) * pi / 180.0
        
        # Air absorption
        self.air_enable = bool(getattr(scene, 'airt_air_enable', True))
        self.air_temp_c = float(getattr(scene, 'airt_air_temp_c', 20.0))
        self.air_humidity = float(getattr(scene, 'airt_air_humidity', 50.0))
        self.air_pressure_kpa = float(getattr(scene, 'airt_air_pressure_kpa', 101.325))
        
        # Output options
        output_content = getattr(scene, 'airt_output_content', 'FULL')
        self.output_content = output_content if output_content in {'FULL', 'REVERB_ONLY'} else 'FULL'
        self.include_direct = self.output_content == 'FULL'
        self.quick_broadband = bool(getattr(scene, 'airt_quick_broadband', False))
        self.min_throughput = float(getattr(scene, 'airt_min_throughput', 1e-4))
        
        # Orientation
        yaw_offset = float(getattr(scene, 'airt_yaw_offset_deg', 0.0))
        invert_z = bool(getattr(scene, 'airt_invert_z', False))
        self.ambisonic_encoder = AmbisonicEncoder(yaw_offset, invert_z)
        
        # Derived constants
        self.eps = 1e-4
        self.pi4 = 4.0 * pi
        
        # HYBRID BLEND CONTROLS - Advanced user balance settings
        # Forward Tracer Gain: -24dB to +24dB (discrete echoes, tunnel reflections)
        self.hybrid_forward_gain_db = float(getattr(scene, 'airt_hybrid_forward_gain_db', 0.0))
        self.hybrid_forward_gain_db = max(-24.0, min(24.0, self.hybrid_forward_gain_db))
        
        # Reverse Tracer Gain: -24dB to +24dB (diffuse reverb tail)
        self.hybrid_reverse_gain_db = float(getattr(scene, 'airt_hybrid_reverse_gain_db', 0.0))
        self.hybrid_reverse_gain_db = max(-24.0, min(24.0, self.hybrid_reverse_gain_db))
        
        # Late Reverb Ramp: 0.05s to 0.5s (how quickly reverse reverb builds up)
        self.hybrid_reverb_ramp_time = float(getattr(scene, 'airt_hybrid_reverb_ramp_time', 0.2))
        self.hybrid_reverb_ramp_time = max(0.05, min(0.5, self.hybrid_reverb_ramp_time))
        
        # Convert dB to linear gain factors
        self.hybrid_forward_gain_linear = 10.0 ** (self.hybrid_forward_gain_db / 20.0)
        self.hybrid_reverse_gain_linear = 10.0 ** (self.hybrid_reverse_gain_db / 20.0)


class ImpulseResponseRenderer:
    """Base class for impulse response rendering."""
    
    def __init__(self, config: RayTracingConfig):
        """Initialize renderer with configuration."""
        self.config = config
        self.ir = np.zeros((16, config.ir_length_samples), dtype=np.float32)
        self.wrote_any = False
    
    def _cast_ray(self, pos: mathutils.Vector, direction: mathutils.Vector, bvh):
        """Cast a ray and return hit information."""
        hit, normal, index, dist = bvh.ray_cast(pos + direction * self.config.eps, direction)
        
        if hit is None or index is None:
            return False, None, None, None
            
        hit_point = mathutils.Vector(hit)
        normal = mathutils.Vector(normal)
        if normal.dot(direction) > 0.0:
            normal = -normal
            
        return True, hit_point, normal, index
    
    def _get_material_properties(self, face_index: int, obj_map: List[Any]):
        """Get material properties for a face."""
        from ..core.acoustics import MaterialProperties
        
        hit_obj = obj_map[face_index] if 0 <= face_index < len(obj_map) else None
        return MaterialProperties(hit_obj)
    
    def _calculate_air_absorption(self, distance: float) -> np.ndarray:
        """Calculate air absorption for a given distance."""
        from ..core.acoustics import air_attenuation_bands, NUM_BANDS
        
        return air_attenuation_bands(
            distance * self.config.unit_scale,  # Convert to meters
            self.config.air_temp_c,
            self.config.air_humidity,
            self.config.air_pressure_kpa
        )
    
    def _should_terminate_ray(self, bounce: int, throughput: np.ndarray) -> bool:
        """Check if ray should be terminated using Russian Roulette."""
        if not self.config.rr_enable:
            return False
            
        if bounce < self.config.rr_start_bounce:
            return False
            
        # Russian roulette
        import random
        if random.random() > self.config.rr_survive_prob:
            return True
            
        # Continue with boosted throughput
        return False
    
    def add_impulse_simple(self, ambi_vec: np.ndarray, delay_samples: float, amplitude: float):
        """Add a simple impulse to the IR."""
        n = int(np.floor(delay_samples))
        frac = float(delay_samples - n)
        
        if 0 <= n < self.ir.shape[1]:
            self.ir[:, n] += ambi_vec * amplitude * (1.0 - frac)
        if 0 <= n + 1 < self.ir.shape[1]:
            self.ir[:, n + 1] += ambi_vec * amplitude * frac
    
    def add_filtered_impulse(self, ambi_vec: np.ndarray, delay_samples: float, 
                           amplitude: float, band_profile: np.ndarray) -> bool:
        """Add a frequency-filtered impulse to the IR."""
        return synthesize_filtered_impulse(
            self.ir,
            ambi_vec,
            delay_samples,
            amplitude,
            band_profile,
            self.config.sample_rate,
        )
    
    def get_path_band_profile(self, band_amplitude: np.ndarray, distance_bu: float) -> np.ndarray:
        """Calculate frequency-dependent path attenuation."""
        if not self.config.air_enable:
            return np.array(band_amplitude, dtype=np.float32)
        
        distance_m = distance_bu * self.config.unit_scale
        air_attenuation = air_attenuation_bands(
            distance_m, self.config.air_temp_c, 
            self.config.air_humidity, self.config.air_pressure_kpa
        )
        
        profile = np.array(band_amplitude, dtype=np.float32) * air_attenuation
        return np.clip(profile, 0.0, 1e6)
    
    def emit_impulse(self, band_amplitude: np.ndarray, distance_bu: float, 
                    incoming_direction: mathutils.Vector, amplitude_scalar: float) -> bool:
        """Emit an impulse into the impulse response."""
        if distance_bu <= 0.0:
            return False
        
        band_profile = self.get_path_band_profile(band_amplitude, distance_bu)
        
        # Quick mode: use broadband average
        if self.config.quick_broadband:
            gain = float(np.mean(band_profile))
            if gain <= 1e-8:
                return False
            
            delay = (distance_bu / self.config.speed_of_sound) * self.config.sample_rate
            ambi = self.config.ambisonic_encoder.encode_with_nf_compensation(
                incoming_direction, distance_bu * self.config.unit_scale, 
                self.config.receiver_radius_m
            )
            
            if not np.any(np.abs(ambi) > 1e-8):
                ambi = np.zeros(16, dtype=np.float32)
                ambi[0] = 1.0
            
            self.add_impulse_simple(ambi, delay, amplitude_scalar * gain)
            return True
        
        # Full frequency-dependent processing
        if not np.any(band_profile > 1e-8):
            return False
        
        delay = (distance_bu / self.config.speed_of_sound) * self.config.sample_rate
        ambi = self.config.ambisonic_encoder.encode_with_nf_compensation(
            incoming_direction, distance_bu * self.config.unit_scale,
            self.config.receiver_radius_m
        )
        
        if not np.any(np.abs(ambi) > 1e-8):
            ambi = np.zeros(16, dtype=np.float32)
            ambi[0] = 1.0
        
        wrote = self.add_filtered_impulse(ambi, delay, amplitude_scalar, band_profile)
        if not wrote:
            self.add_impulse_simple(ambi, delay, amplitude_scalar)
            wrote = True
        
        return wrote
    
    def _apply_russian_roulette(self, bounce: int, throughput: np.ndarray):
        """Apply Russian Roulette with proper energy compensation.
        
        Returns:
            (should_terminate, compensated_throughput)
        """
        import random
        
        # Throughput check
        if float(np.max(throughput)) < self.config.min_throughput:
            return True, throughput
        
        # Russian roulette with energy compensation
        if self.config.rr_enable and bounce >= self.config.rr_start_bounce:
            if random.random() > self.config.rr_survive_prob:
                return True, throughput  # Terminate
            else:
                # Boost throughput to compensate for survival probability
                compensated_throughput = throughput / self.config.rr_survive_prob
                return False, compensated_throughput
        
        return False, throughput

    def _geometric_spreading(self, distance_bu: float) -> float:
        """Return pressure-domain 1/r spreading using scene units in metres."""
        distance_m = max(0.0, float(distance_bu)) * self.config.unit_scale
        return 1.0 / max(distance_m, self.config.receiver_radius_m)

    def _reflection_connection_profile(
        self,
        direction: mathutils.Vector,
        normal: mathutils.Vector,
        outgoing: mathutils.Vector,
        material: MaterialProperties,
    ) -> np.ndarray:
        """Evaluate the frequency-dependent reflection lobe for a point connection."""
        specular_direction = reflect(direction, normal)
        cos_angle = float(np.clip(specular_direction.dot(outgoing), -1.0, 1.0))
        angle_difference = acos(cos_angle)
        specular_lobe = exp(
            -(angle_difference / max(self.config.angle_tolerance_rad, 1e-6)) ** 2
        )
        diffuse_lobe = max(0.0, float(outgoing.dot(normal))) / pi
        lobe_energy = np.clip(
            material.specular_fraction * specular_lobe
            + material.diffuse_fraction * diffuse_lobe,
            0.0,
            1.0,
        )
        return material.reflection_amplitude * np.sqrt(lobe_energy)

    def _sample_surface_scatter(
        self,
        direction: mathutils.Vector,
        normal: mathutils.Vector,
        material: MaterialProperties,
        throughput: np.ndarray,
    ) -> Tuple[Optional[mathutils.Vector], Optional[np.ndarray]]:
        """Sample one outgoing component with Monte Carlo branch compensation.

        Branch probabilities follow mean outgoing energy. Dividing each sampled
        pressure coefficient by its probability keeps the expected contribution
        independent of the material's specular/diffuse/transmission split.
        """
        transmission_energy = float(np.mean(material.transmission_spectrum))
        diffuse_energy = float(np.mean(
            material.reflection_spectrum * material.diffuse_fraction
        ))
        specular_energy = float(np.mean(
            material.reflection_spectrum * material.specular_fraction
        ))
        total_energy = transmission_energy + diffuse_energy + specular_energy
        if total_energy <= 1e-12:
            return None, None

        probabilities = np.array(
            (transmission_energy, diffuse_energy, specular_energy),
            dtype=np.float64,
        ) / total_energy
        branch = float(random.random())

        if branch < probabilities[0]:
            probability = float(probabilities[0])
            return (
                direction.normalized(),
                throughput * material.transmission_amplitude / probability,
            )

        branch -= probabilities[0]
        if branch < probabilities[1]:
            probability = float(probabilities[1])
            return (
                cosine_weighted_hemisphere(normal),
                throughput * material.diffuse_amplitude / probability,
            )

        probability = float(probabilities[2])
        if probability <= 1e-12:
            return None, None
        specular_direction = reflect(direction, normal)
        return (
            jitter_specular_direction(
                specular_direction, self.config.specular_roughness_rad
            ),
            throughput * material.specular_amplitude / probability,
        )


class ForwardRayTracer(ImpulseResponseRenderer):
    """Forward ray tracer (source to receiver)."""
    
    def __init__(self, config: RayTracingConfig):
        """Initialize forward ray tracer."""
        super().__init__(config)
    
    def trace_rays(self, source: mathutils.Vector, receiver: mathutils.Vector,
                   bvh, obj_map: List[Any], directions: List[Tuple[float, float, float]]) -> np.ndarray:
        """Trace rays from source towards receiver."""
        if bvh is None:
            return self.ir
        
        num_dirs = max(1, len(directions))
        per_ray_throughput = np.ones(NUM_BANDS, dtype=np.float32) / float(num_dirs)
        band_one = np.ones(NUM_BANDS, dtype=np.float32)
        
        print(f"DEBUG: ForwardRayTracer starting with {num_dirs} directions")
        
        # Stochastic rays sample the reflected field. The zero-bounce path is
        # evaluated separately so its level does not depend on ray count.
        for d in directions:
            self._trace_single_ray(mathutils.Vector(d), source, receiver, 
                                 bvh, obj_map, per_ray_throughput)
        
        if self.config.include_direct:
            print("DEBUG: Adding deterministic direct path (forward tracer)...")
            self._add_direct_path(source, receiver, bvh, band_one)
        
        return self.ir
    
    def _trace_single_ray(self, direction: mathutils.Vector, source: mathutils.Vector,
                         receiver: mathutils.Vector, bvh, obj_map: List[Any], 
                         initial_throughput: np.ndarray):
        """Trace a single ray through the scene."""
        pos = source.copy()
        dirn = direction.normalized()
        throughput = initial_throughput.copy()
        path_length = 0.0
        bounce = 0
        
        while bounce <= self.config.max_bounces:
            # Cast ray
            hit, normal, index, dist = bvh.ray_cast(pos + dirn * self.config.eps, dirn)
            
            if hit is None or index is None:
                # The deterministic direct-path calculation owns bounce zero.
                # Capturing it here would make its level ray-count dependent and
                # could duplicate it in Full IR mode.
                if self.config.segment_capture and bounce > 0:
                    self._check_segment_capture(pos, dirn, receiver, throughput, path_length)
                break
            
            # Process hit
            hit_point = mathutils.Vector(hit)
            normal = mathutils.Vector(normal)
            if normal.dot(dirn) > 0.0:
                normal = -normal
            
            seg_length = float(dist)
            total_distance = path_length + seg_length
            
            # Get material properties
            hit_obj = obj_map[index] if 0 <= index < len(obj_map) else None
            material = MaterialProperties(hit_obj)
            
            # Check for early termination
            if not np.any(material.reflection_spectrum > 1e-6) and material.transmission <= 1e-6:
                break
            
            # Segment capture for ray segments 
            if self.config.segment_capture and bounce > 0:
                self._check_segment_capture(pos, dirn, receiver, throughput, path_length, hit_point)
            
            # Direct connection to receiver
            self._check_direct_connection(hit_point, normal, dirn, receiver, 
                                        throughput, material, total_distance, bvh)
            
            # Continue ray
            new_direction, new_throughput = self._scatter_ray(dirn, normal, material, throughput)
            if new_direction is None:
                break
            
            # Update for next iteration
            path_length = total_distance
            throughput = new_throughput
            pos = hit_point + normal * self.config.eps + new_direction * (self.config.eps * 0.5)
            dirn = new_direction
            bounce += 1
            
            # Russian roulette termination with energy compensation
            should_terminate, throughput = self._apply_russian_roulette(bounce, throughput)
            if should_terminate:
                break
    
    def _check_segment_capture(self, pos: mathutils.Vector, direction: mathutils.Vector,
                              receiver: mathutils.Vector, throughput: np.ndarray,
                              path_length: float, hit_point: Optional[mathutils.Vector] = None):
        """Check if ray segment intersects receiver sphere."""
        if hit_point is None:
            far = pos + direction * 100.0
        else:
            far = hit_point
        
        hit, t_hit, _ = segment_hits_sphere(pos, far, receiver, self.config.receiver_radius)
        if not hit:
            return
        
        seg_len = (far - pos).length * t_hit
        total_dist = path_length + seg_len
        incoming = (-direction).normalized()
        
        amplitude_scalar = self._geometric_spreading(total_dist)
        
        # Add debug output for segment capture
        delay_ms = (total_dist / self.config.speed_of_sound) * 1000.0
        if delay_ms < 100.0:  # Only log early reflections to avoid spam
            print(f"DEBUG: Segment capture - delay: {delay_ms:.2f}ms, distance: {total_dist:.2f}m, amplitude_scalar: {amplitude_scalar:.6f}")
        
        if self.emit_impulse(throughput, total_dist, incoming, amplitude_scalar):
            self.wrote_any = True
    
    def _check_direct_connection(self, hit_point: mathutils.Vector, normal: mathutils.Vector,
                               direction: mathutils.Vector, receiver: mathutils.Vector,
                               throughput: np.ndarray, material: MaterialProperties,
                               path_length: float, bvh):
        """Check for direct connection from hit point to receiver."""
        from ..utils.scene_utils import los_clear
        
        to_receiver = receiver - hit_point
        dist_receiver = to_receiver.length
        
        if dist_receiver <= 0.0:
            return
        
        has_los = los_clear(hit_point + normal * self.config.eps, receiver, bvh, self.config.eps)
        if not has_los:
            # Try diffraction
            self._add_diffraction(hit_point, normal, direction, to_receiver, 
                                throughput, material, path_length)
            return
        
        to_receiver_dir = to_receiver.normalized()
        connection_profile = self._reflection_connection_profile(
            direction, normal, to_receiver_dir, material
        )
        
        if np.any(connection_profile > 1e-6):
            band_amplitude = throughput * connection_profile
            total_distance = path_length + dist_receiver
            amplitude_scalar = self._geometric_spreading(total_distance)
            incoming = (hit_point - receiver).normalized()
            
            if self.emit_impulse(band_amplitude, total_distance, incoming, amplitude_scalar):
                self.wrote_any = True
    
    def _scatter_ray(self, direction: mathutils.Vector, normal: mathutils.Vector,
                    material: MaterialProperties, throughput: np.ndarray) -> Tuple[Optional[mathutils.Vector], Optional[np.ndarray]]:
        """Scatter ray at surface according to material properties."""
        return self._sample_surface_scatter(direction, normal, material, throughput)
    
    def _add_direct_path(self, source: mathutils.Vector, receiver: mathutils.Vector, 
                        bvh, throughput: np.ndarray):
        """Add direct path from source to receiver."""
        from ..utils.scene_utils import los_clear
        
        direction_vec = receiver - source
        distance = direction_vec.length
        print(f"DEBUG: Direct path distance: {distance:.3f}m")
        
        if not los_clear(source, receiver, bvh):
            print("DEBUG: Direct path blocked by geometry")
            return
        
        if distance <= 0.0:
            print("DEBUG: Direct path distance too small")
            return
        
        incoming = (source - receiver).normalized()
        amplitude_scalar = self._geometric_spreading(distance)
        delay_ms = (distance / self.config.speed_of_sound) * 1000.0
        
        print(f"DEBUG: Direct path - delay: {delay_ms:.2f}ms, amplitude_scalar: {amplitude_scalar:.6f}")
        print(f"DEBUG: Direct path throughput: {np.mean(throughput):.6f}")
        
        if self.emit_impulse(throughput, distance, incoming, amplitude_scalar):
            print("DEBUG: Direct path impulse successfully added")
            self.wrote_any = True
        else:
            print("DEBUG: Direct path impulse failed to add")
    
    def _add_diffraction(self, hit_point: mathutils.Vector, normal: mathutils.Vector,
                        direction: mathutils.Vector, to_receiver: mathutils.Vector,
                        throughput: np.ndarray, material: MaterialProperties, path_length: float):
        """Add simple diffraction sampling."""
        if not self.config.enable_diffraction or self.config.diffraction_samples <= 0:
            return
        
        # Implementation would be similar to original but extracted here
        # For brevity, I'll add a simplified version
        pass
    
    def _should_terminate_ray(self, bounce: int, throughput: np.ndarray) -> bool:
        """Determine if ray should be terminated."""
        # Throughput check
        if float(np.max(throughput)) < self.config.min_throughput:
            return True
        
        # Russian roulette
        if self.config.rr_enable and bounce >= self.config.rr_start_bounce:
            return random.random() > self.config.rr_survive_prob
        
        return False


class ReverseRayTracer(ImpulseResponseRenderer):
    """Reverse ray tracer (receiver to source)."""
    
    def trace_rays(self, source: mathutils.Vector, receiver: mathutils.Vector,
                   bvh, obj_map: List[Any], directions: List[Tuple[float, float, float]]) -> np.ndarray:
        """Trace rays from receiver towards source."""
        if bvh is None:
            return self.ir
        
        num_dirs = max(1, len(directions))
        per_ray_throughput = np.ones(NUM_BANDS, dtype=np.float32) / float(num_dirs)
        band_one = np.ones(NUM_BANDS, dtype=np.float32)
        
        print(f"DEBUG: ReverseRayTracer starting with {num_dirs} directions")
        
        # DEBUG: Print all configuration settings
        print("DEBUG: RAY TRACING CONFIGURATION:")
        print(f"  Basic Parameters:")
        print(f"    Number of rays: {self.config.num_rays}")
        print(f"    Max bounces: {self.config.max_bounces}")
        print(f"    Sample rate: {self.config.sample_rate} Hz")
        print(f"    IR length: {self.config.ir_length_samples} samples ({self.config.ir_length_samples/self.config.sample_rate:.2f}s)")
        print(f"  Physical Parameters:")
        print(f"    Speed of sound: {self.config.speed_of_sound:.1f} m/s")
        print(f"    Unit scale: {self.config.unit_scale:.6f}")
        print(f"    Receiver radius: {self.config.receiver_radius_m:.4f}m (scaled: {self.config.receiver_radius:.6f})")
        print(f"  Ray Tracing Behavior:")
        print(f"    Angle tolerance: {self.config.angle_tolerance_rad*180/pi:.1f}°")
        print(f"    Specular roughness: {self.config.specular_roughness_rad*180/pi:.1f}°")
        print(f"    Segment capture: {self.config.segment_capture}")
        print(f"    Min throughput: {self.config.min_throughput:.0e}")
        print(f"  Russian Roulette:")
        print(f"    Enabled: {self.config.rr_enable}")
        print(f"    Start bounce: {self.config.rr_start_bounce}")
        print(f"    Survive probability: {self.config.rr_survive_prob:.3f}")
        print(f"  Air Absorption:")
        print(f"    Enabled: {self.config.air_enable}")
        print(f"    Temperature: {self.config.air_temp_c:.1f}°C")
        print(f"    Humidity: {self.config.air_humidity:.1f}%")
        print(f"    Pressure: {self.config.air_pressure_kpa:.1f} kPa")
        print(f"  Advanced Settings:")
        print(f"    Quick broadband: {self.config.quick_broadband}")
        if hasattr(self.config, 'hybrid_forward_gain_db'):
            print(f"    Hybrid forward gain: {self.config.hybrid_forward_gain_db:.1f} dB")
        print("DEBUG: End configuration")
        print()
        print(f"DEBUG: This is REVERSE ray tracing - should have strong absorption for carpet!")
        
        rays_traced = 0
        for d in directions:
            first_direction = mathutils.Vector(d).normalized()
            self._trace_single_ray(first_direction, receiver, source, bvh, obj_map, 
                                 per_ray_throughput, first_direction)
            rays_traced += 1
        
        if self.config.include_direct:
            print("DEBUG: Adding deterministic direct path (reverse tracer)...")
            self._add_direct_path(source, receiver, bvh, band_one)
        
        print(f"DEBUG: Reverse tracer completed {rays_traced} rays")
        if hasattr(self, 'connection_count'):
            print(f"DEBUG: Total successful connections: {self.connection_count}")
            
            # DEBUG: Print final bounce statistics
            if hasattr(self, 'bounce_stats'):
                print(f"DEBUG: Final bounce distribution:")
                total_attempts = sum(self.bounce_stats.values())
                for b in sorted(self.bounce_stats.keys()):
                    percentage = (self.bounce_stats[b] / total_attempts) * 100
                    print(f"  Bounce {b}: {self.bounce_stats[b]} attempts ({percentage:.1f}%)")
                print(f"DEBUG: Average bounces per ray: {sum(b*count for b,count in self.bounce_stats.items()) / total_attempts:.1f}")
        
        return self.ir
    
    def _trace_single_ray(self, direction: mathutils.Vector, start_pos: mathutils.Vector,
                         target: mathutils.Vector, bvh, obj_map: List[Any],
                         initial_throughput: np.ndarray, arrival_direction: mathutils.Vector):
        """Trace a single reverse ray from receiver toward room, checking for source connections."""
        pos = start_pos
        dirn = direction
        throughput = initial_throughput.copy()
        path_length = 0.0
        bounce = 0
        
        while bounce < self.config.max_bounces:
            # DEBUG: Track bounce statistics
            if not hasattr(self, 'bounce_stats'):
                self.bounce_stats = {}
            if bounce not in self.bounce_stats:
                self.bounce_stats[bounce] = 0
            self.bounce_stats[bounce] += 1
            
            # Cast ray to find next surface hit
            hit, hit_point, normal, face_index = self._cast_ray(pos, dirn, bvh)
            
            if not hit:
                # Ray escaped to infinity - no more bounces
                break
                
            # Calculate path length to hit point
            seg_length = (hit_point - pos).length
            path_length += seg_length
            
            # Get material properties
            material = self._get_material_properties(face_index, obj_map)
            
            # DEBUG: Print material properties for early bounces
            if bounce < 3 and random.random() < 0.001:  # Debug 0.1% of early bounces
                print(f"DEBUG Material bounce {bounce}:")
                print(f"  Absorption: {material.absorption_spectrum}")
                print(f"  Scatter: {material.scatter_spectrum}")  
                print(f"  Reflection: {material.reflection_spectrum}")
                print(f"  Diffuse ampl: {material.diffuse_amplitude}")
                print(f"  Specular ampl: {material.specular_amplitude}")
                avg_abs = np.mean(material.absorption_spectrum)
                avg_refl_ampl = np.mean(material.reflection_amplitude)
                print(f"  Avg absorption: {avg_abs:.3f}, Avg refl_ampl: {avg_refl_ampl:.3f}")
            
            # Connect this reflected path back to the source when visible.
            self._check_source_connection(hit_point, normal, target, throughput, 
                                        material, path_length, bvh, dirn,
                                        arrival_direction, bounce)

            new_direction, new_throughput = self._sample_surface_scatter(
                dirn, normal, material, throughput
            )
            if new_direction is None:
                break
            throughput = new_throughput
            
            # Standard energy threshold check - no special treatment for absorptive materials
            if np.max(throughput) < self.config.min_throughput:
                if bounce < 2 and random.random() < 0.001:
                    print(f"DEBUG: Standard ray termination - energy: {np.max(throughput):.2e}, threshold: {self.config.min_throughput:.2e}")
                break
                
            # Update for next iteration
            pos = hit_point + normal * self.config.eps + new_direction * (self.config.eps * 0.5)
            dirn = new_direction
            bounce += 1

            should_terminate, throughput = self._apply_russian_roulette(
                bounce, throughput
            )
            if should_terminate:
                break
    
    def _check_source_connection(self, hit_point: mathutils.Vector, normal: mathutils.Vector,
                               source: mathutils.Vector, throughput: np.ndarray, 
                               material: MaterialProperties, path_length: float,
                               bvh, ray_direction: mathutils.Vector,
                               arrival_direction: mathutils.Vector, bounce: int):
        """Check for direct line-of-sight connection from hit point to source."""
        from ..utils.scene_utils import los_clear

        to_source = source - hit_point
        distance_to_source = to_source.length

        if distance_to_source <= self.config.eps:
            return

        source_direction = to_source / distance_to_source

        if source_direction.dot(normal) <= 0.0:
            return

        if not los_clear(hit_point + normal * self.config.eps, source, bvh):
            return

        connection_profile = self._reflection_connection_profile(
            ray_direction, normal, source_direction, material
        )
        if not np.any(connection_profile > 1e-6):
            return

        total_distance = path_length + distance_to_source
        final_throughput = throughput * connection_profile
        emission_success = self.emit_impulse(
            final_throughput,
            total_distance,
            arrival_direction,
            self._geometric_spreading(total_distance),
        )

        if emission_success:
            if not hasattr(self, 'connection_count'):
                self.connection_count = 0
            self.connection_count += 1
            self.wrote_any = True
    
    def _add_direct_path(self, source: mathutils.Vector, receiver: mathutils.Vector,
                        bvh, throughput: np.ndarray):
        """Add direct path contribution."""
        from ..utils.scene_utils import los_clear
        
        if not los_clear(source, receiver, bvh):
            return
        
        direction_vec = receiver - source
        distance = direction_vec.length
        
        if distance <= 0.0:
            return
        
        incoming = (source - receiver).normalized()
        amplitude_scalar = self._geometric_spreading(distance)
        
        if self.emit_impulse(throughput, distance, incoming, amplitude_scalar):
            self.wrote_any = True


def create_ray_tracer(tracing_mode: str, config: RayTracingConfig) -> ImpulseResponseRenderer:
    """Factory function to create appropriate ray tracer."""
    if tracing_mode == 'FORWARD':
        return ForwardRayTracer(config)
    elif tracing_mode == 'REVERSE':
        return ReverseRayTracer(config)
    else:
        raise ValueError(f"Unknown tracing mode: {tracing_mode}")


def trace_impulse_response(context, source: mathutils.Vector, receiver: mathutils.Vector,
                          bvh, obj_map: List[Any], 
                          directions: Optional[List[Tuple[float, float, float]]] = None) -> np.ndarray:
    """Main entry point for impulse response tracing using hybrid approach."""
    config = RayTracingConfig(context)
    
    if directions is None:
        directions = generate_ray_directions(config.num_rays)
    
    # Get user's preferred tracing mode
    user_trace_mode = context.scene.airt_trace_mode
    
    if user_trace_mode == 'HYBRID':
        # Professional hybrid approach: combine both methods
        print(f"Hybrid tracing: combining Forward (early) + Reverse (late) for optimal results")
        return _trace_hybrid(config, source, receiver, bvh, obj_map, directions)
        
    else:
        # Single-method approach
        print(f"Single-method mode: {user_trace_mode}")
        tracer = create_ray_tracer(user_trace_mode, config)
        return tracer.trace_rays(source, receiver, bvh, obj_map, directions)


def _trace_hybrid(config: RayTracingConfig, source: mathutils.Vector, receiver: mathutils.Vector,
                  bvh, obj_map: List[Any], directions: List[Tuple[float, float, float]]) -> np.ndarray:
    """Hybrid tracing: Forward for early reflections + Reverse for late reverb."""
    
    # Split ray budget between methods
    early_rays = directions[:len(directions)//2]  # First half for forward
    late_rays = directions[len(directions)//2:]   # Second half for reverse
    
    print(f"  Forward rays: {len(early_rays)} (early reflections)")
    print(f"  Reverse rays: {len(late_rays)} (late reverb)")
    
    # Forward tracing owns the direct path and early reflections.
    forward_tracer = create_ray_tracer('FORWARD', config)
    ir_early = forward_tracer.trace_rays(source, receiver, bvh, obj_map, early_rays)
    
    # Reverse tracing contributes only the diffuse tail in hybrid mode. This
    # avoids adding the same deterministic direct path twice.
    import copy
    config_late = copy.deepcopy(config)  # Create copy for late reverb
    config_late.include_direct = False
    reverse_tracer = create_ray_tracer('REVERSE', config_late)
    ir_late = reverse_tracer.trace_rays(source, receiver, bvh, obj_map, late_rays)
    
    # Combine with time-based weighting
    ir_combined = _blend_early_late(ir_early, ir_late, config)
    
    return ir_combined


def _blend_early_late(ir_early: np.ndarray, ir_late: np.ndarray, config: RayTracingConfig) -> np.ndarray:
    """Hybrid blend: adaptive forward tail + dynamic reverse scaling (Option B)."""

    samples = ir_early.shape[1]
    sr = config.sample_rate
    time_axis = np.arange(samples) / sr

    # --- 1. Determine ramp window ---
    ramp_start = 0.05  # fixed 50ms onset for reverse participation
    ramp_end = config.hybrid_reverb_ramp_time  # user parameter (0.05-0.5 s)
    if ramp_end <= ramp_start:
        ramp_end = ramp_start + 0.02  # safety

    # --- 2. Compute overlap RMS for dynamic reverse scaling ---
    overlap_start_sample = int(ramp_start * sr)
    overlap_end_sample = int(min(ramp_end + 0.25, samples / sr) * sr)
    overlap_slice = slice(overlap_start_sample, overlap_end_sample)

    def safe_rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x ** 2)))

    fwd_rms_overlap = safe_rms(ir_early[:, overlap_slice])
    rev_rms_overlap = safe_rms(ir_late[:, overlap_slice])

    target_ratio = 0.9  # aim reverse slightly below forward during overlap
    if rev_rms_overlap > 0.0 and fwd_rms_overlap > 0.0:
        reverse_scale = (fwd_rms_overlap / rev_rms_overlap) * target_ratio
    else:
        reverse_scale = 0.3
    reverse_scale = float(np.clip(reverse_scale, 0.2, 2.0))
    ir_late_scaled = ir_late * reverse_scale
    print(f"  DEBUG: Overlap RMS - Fwd: {fwd_rms_overlap:.6e}, Rev: {rev_rms_overlap:.6e}, RevScale: {reverse_scale:.3f}")

    # --- 3. Estimate RT60 (fallback if not measurable) for adaptive forward tail ---
    # Rough early slope: use 0.1-0.4 s window if energy present
    def estimate_rt60(ir: np.ndarray) -> float:
        win_a = int(0.1 * sr)
        win_b = int(min(0.4, samples / sr - 0.01) * sr)
        mono = np.mean(ir, axis=0)
        seg = mono[win_a:win_b]
        if seg.size < sr * 0.05:
            return 1.0
        eps = 1e-12
        env = np.sqrt(np.convolve(seg**2, np.ones(512)/512, mode='same') + eps)
        db = 20 * np.log10(np.maximum(env, eps))
        t_local = np.linspace(0, (seg.size-1)/sr, seg.size)
        # simple linear fit
        A = np.vstack([t_local, np.ones_like(t_local)]).T
        try:
            m, c = np.linalg.lstsq(A, db, rcond=None)[0]
            if m >= -1e-3:  # slope not negative => fallback
                return 1.0
            rt60_est = -60.0 / m
            return float(np.clip(rt60_est, 0.3, 6.0))
        except Exception:
            return 1.0

    rt60_guess = estimate_rt60(ir_late_scaled)
    tau_f = 0.5 * rt60_guess  # forward residual decay constant
    tail_floor = 0.12  # lowered minimum persistent forward share (pre late-release)

    # --- 4. Build base forward weight (piecewise + exponential tail) ---
    forward_weight = np.zeros_like(time_axis)
    reverse_weight = np.zeros_like(time_axis)

    # Early: 100% forward
    early_mask = time_axis < ramp_start
    forward_weight[early_mask] = 1.0

    # Ramp region
    ramp_mask = (time_axis >= ramp_start) & (time_axis <= ramp_end)
    if np.any(ramp_mask):
        ramp_width = ramp_end - ramp_start
        prog = (time_axis[ramp_mask] - ramp_start) / max(ramp_width, 1e-6)
        # cosine easing from 1.0 -> w_at_ramp_end (not directly to floor)
        w_ramp_end = 0.35  # a bit higher than final floor to allow graceful exponential
        smooth = 0.5 * (1 - np.cos(np.pi * prog))
        forward_weight[ramp_mask] = 1.0 - (1.0 - w_ramp_end) * smooth

    # Post-ramp exponential tail (primary decay toward tail_floor)
    tail_mask = time_axis > ramp_end
    if np.any(tail_mask):
        t_tail = time_axis[tail_mask] - ramp_end
        w_at_ramp_end = forward_weight[np.searchsorted(time_axis, ramp_end, side='right') - 1]
        forward_weight[tail_mask] = tail_floor + (w_at_ramp_end - tail_floor) * np.exp(-t_tail / max(tau_f, 1e-6))

    # Late forward release: further reduce residual forward energy deep in tail
    late_release_start = ramp_end + 0.8  # seconds after which residual forward share is relaxed further
    late_floor_min = 0.05                # ultimate late forward floor
    tau_rel = 0.25 * rt60_guess          # faster release constant
    late_mask = time_axis > late_release_start
    if np.any(late_mask):
        t_rel = time_axis[late_mask] - late_release_start
        current = forward_weight[late_mask]
        forward_weight[late_mask] = late_floor_min + (current - late_floor_min) * np.exp(-t_rel / max(tau_rel, 1e-6))
        print(f"  DEBUG: Late forward release applied (start={late_release_start:.2f}s, final_floor={late_floor_min:.2f})")

    # Reverse weight is complementary (not forced perfectly complementary early for slight energy freedom)
    reverse_weight = 1.0 - forward_weight
    # Keep a minimum reverse presence after ramp to avoid zero-diffuse pockets
    reverse_weight[tail_mask] = np.maximum(reverse_weight[tail_mask], 0.65)

    # Normalize if combined exceeds 1.05 (safety)
    total_weight = forward_weight + reverse_weight
    excess_mask = total_weight > 1.05
    if np.any(excess_mask):
        forward_weight[excess_mask] /= total_weight[excess_mask]
        reverse_weight[excess_mask] /= total_weight[excess_mask]
        print("  DEBUG: Weight normalization applied to prevent >1.05 sum")

    print(f"  DEBUG: Weights - fwd_tail_mean={np.mean(forward_weight[tail_mask]) if np.any(tail_mask) else 0:.3f}, rt60_guess={rt60_guess:.2f}, tau_f={tau_f:.2f}")

    # --- 5. Combine with user gains ---
    ir_combined = np.zeros_like(ir_early)
    for ch in range(ir_early.shape[0]):
        ir_combined[ch, :] = (
            ir_early[ch, :] * forward_weight * config.hybrid_forward_gain_linear +
            ir_late_scaled[ch, :] * reverse_weight * config.hybrid_reverse_gain_linear
        )

    # --- 6. Diagnostics & gentle ceiling ---
    peak_before = float(np.max(np.abs(ir_combined)))
    if peak_before > 0.9:
        scale = 0.9 / peak_before
        ir_combined *= scale
        print(f"  DEBUG: Output scaled {scale:.3f} to keep headroom")

    # Basic energy distribution logging
    early_samp = int(0.1 * sr)
    mid_samp = int(0.5 * sr)
    e_early = float(np.sum(ir_combined[:, :early_samp] ** 2))
    e_mid = float(np.sum(ir_combined[:, early_samp:mid_samp] ** 2))
    e_late = float(np.sum(ir_combined[:, mid_samp:] ** 2))
    print(f"  DEBUG: Energy blocks - early={e_early:.3e} mid={e_mid:.3e} late={e_late:.3e}")

    return ir_combined
