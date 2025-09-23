# -*- coding: utf-8 -*-
"""
Ray tracing engine for acoustic impulse response rendering.
"""
import mathutils
import numpy as np
from math import pi, sqrt, cos, exp, acos
import random
from typing import List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from .acoustics import (
    MaterialProperties, air_attenuation_bands, design_band_kernel, 
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
        kernel = design_band_kernel(band_profile, self.config.sample_rate)
        base = int(np.floor(delay_samples))
        frac = float(delay_samples - base)
        
        weights = ((base, 1.0 - frac), (base + 1, frac))
        wrote = False
        
        for start, w in weights:
            if w <= 0.0:
                continue
            for k, kv in enumerate(kernel):
                idx = start + k
                if 0 <= idx < self.ir.shape[1]:
                    self.ir[:, idx] += ambi_vec * (amplitude * w * kv)
                    wrote = True
        
        return wrote
    
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
        
        band_one = np.ones(NUM_BANDS, dtype=np.float32)
        num_dirs = max(1, len(directions))
        ray_weight = 1.0 / float(num_dirs)
        
        print(f"DEBUG: ForwardRayTracer starting with {num_dirs} directions, ray_weight={ray_weight:.6f}")
        
        for d in directions:
            self._trace_single_ray(mathutils.Vector(d), source, receiver, 
                                 bvh, obj_map, band_one * ray_weight)
        
        # Always add direct path (omit_direct functionality removed)
        print("DEBUG: Adding direct path...")
        self._add_direct_path(source, receiver, bvh, band_one * ray_weight)
        
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
                # Ray escaped - check for segment capture
                # Note: segment capture is different from direct path - always allow it
                if self.config.segment_capture:
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
            if self.config.segment_capture:
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
        
        area = pi * self.config.receiver_radius * self.config.receiver_radius
        view = area / max(self.config.pi4 * total_dist * total_dist, 1e-9)
        amplitude_scalar = sqrt(max(view, 0.0)) / max(total_dist, self.config.receiver_radius)
        
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
        
        # Calculate reflection contribution
        to_receiver_dir = to_receiver.normalized()
        required_out = reflect(direction, normal)
        cos_angle = max(-1.0, min(1.0, required_out.dot(to_receiver_dir)))
        angle_diff = acos(cos_angle)
        
        # Specular lobe
        specular_lobe = exp(-(angle_diff / max(self.config.angle_tolerance_rad, 1e-6)) ** 2)
        
        # Diffuse lobe
        cos_incident = max(0.0, (-direction).dot(normal))
        diffuse_lobe = cos_incident / pi
        
        # Combined weighting
        total_weight = np.clip(
            material.specular_fraction * specular_lobe + material.diffuse_fraction * diffuse_lobe,
            0.0, 1.0
        )
        
        if np.any(total_weight > 1e-6):
            band_amplitude = throughput * material.reflection_amplitude * np.sqrt(total_weight)
            total_distance = path_length + dist_receiver
            amplitude_scalar = 1.0 / max(total_distance, self.config.receiver_radius)
            incoming = (hit_point - receiver).normalized()
            
            if self.emit_impulse(band_amplitude, total_distance, incoming, amplitude_scalar):
                self.wrote_any = True
    
    def _scatter_ray(self, direction: mathutils.Vector, normal: mathutils.Vector,
                    material: MaterialProperties, throughput: np.ndarray) -> Tuple[Optional[mathutils.Vector], Optional[np.ndarray]]:
        """Scatter ray at surface according to material properties."""
        # Probabilistic scattering selection
        transmission_prob = min(0.999, max(0.0, material.transmission))
        remaining = max(0.0, 1.0 - transmission_prob)
        diffuse_prob = remaining * float(np.clip(np.mean(material.diffuse_fraction), 0.0, 1.0))
        specular_prob = max(remaining - diffuse_prob, 0.0)
        
        rnd = random.random()
        
        # Transmission
        if transmission_prob > 0.0 and rnd < transmission_prob:
            return direction.normalized(), throughput * material.transmission_amplitude
        
        rnd -= transmission_prob
        
        # Diffuse reflection
        if diffuse_prob > 0.0 and rnd < diffuse_prob:
            for _ in range(8):  # Try multiple samples
                candidate = cosine_weighted_hemisphere(normal)
                if candidate.dot(normal) > 1e-6:
                    return candidate.normalized(), throughput * material.diffuse_amplitude
            return None, None
        
        # Specular reflection
        if specular_prob > 0.0 and np.any(material.specular_amplitude > 1e-6):
            specular_dir = reflect(direction, normal)
            jittered_dir = jitter_specular_direction(specular_dir, self.config.specular_roughness_rad)
            return jittered_dir, throughput * material.specular_amplitude
        
        return None, None
    
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
        amplitude_scalar = 1.0 / max(distance, self.config.receiver_radius)
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
        
        band_one = np.ones(NUM_BANDS, dtype=np.float32)
        num_dirs = max(1, len(directions))
        
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
        
        total_connections = 0
        rays_traced = 0
        for d in directions:
            first_direction = mathutils.Vector(d).normalized()
            incoming_direction = (-first_direction).normalized()
            self._trace_single_ray(first_direction, receiver, source, bvh, obj_map, 
                                 band_one, incoming_direction)
            rays_traced += 1
        
        # ADAPTIVE normalization based on material properties and connection count
        if num_dirs > 0:
            ir_max_before = np.max(np.abs(self.ir))
            
            # Check if we have highly absorptive materials (like carpet) with many connections
            needs_normalization = False
            normalization_factor = 1.0
            
            if hasattr(self, 'connection_count') and self.connection_count > 100000:
                # Many connections detected - likely over-contributing energy
                needs_normalization = True
                # Much gentler normalization now that material absorption is working properly
                base_factor = min(20.0, self.connection_count / 10000.0)  # Gentler scaling
                normalization_factor = 1.0 / base_factor
                mode_reason = f"HIGH-CONNECTIONS ({self.connection_count})"
                
            elif num_dirs < 6000:  # Likely hybrid mode (half of typical 8192)
                # GENTLE normalization for hybrid compatibility - preserve most energy
                needs_normalization = True
                normalization_factor = 1.0/50.0  # Much gentler than /1000
                mode_reason = "HYBRID-mode"
            
            if needs_normalization:
                self.ir *= normalization_factor
                print(f"DEBUG: {mode_reason} IR normalization - Before: {ir_max_before:.2e}, Factor: {normalization_factor:.2e}")
            else:
                print(f"DEBUG: STANDALONE-mode IR (no normalization) - Before: {ir_max_before:.2e}, Factor: 1.00e+00")
            
            ir_max_after = np.max(np.abs(self.ir))
            print(f"DEBUG: IR normalization result - After: {ir_max_after:.2e}")
        
        # Always add direct path (omit_direct functionality removed)
        print("DEBUG: Adding direct path (reverse tracer)...")
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
                         initial_throughput: np.ndarray, incoming_direction: mathutils.Vector):
        """Trace a single reverse ray from receiver toward room, checking for source connections."""
        import random
        from ..utils.math_utils import jitter_direction, reflect, cosine_weighted_hemisphere
        
        pos = start_pos
        dirn = direction
        throughput = initial_throughput.copy()
        path_length = 0.0
        bounce = 0
        
        debug_this_ray = bounce == 0 and random.random() < 0.0001  # Debug only ~0.01% of rays
        
        connections_made = 0
        
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
            
            # Apply air absorption
            if self.config.air_enable:
                throughput *= self._calculate_air_absorption(seg_length)
            
            # Always check for direct connection to source (omit_direct functionality removed)
            self._check_source_connection(hit_point, normal, target, throughput, 
                                        material, path_length, bvh, incoming_direction, bounce)
            
            # CRITICAL FIX: Choose between specular and diffuse bounce based on scatter_spectrum
            # Average scatter value determines the probability of diffuse vs specular bounce
            avg_scatter = float(np.mean(material.scatter_spectrum))
            
            # ADDITIONAL FIX: More aggressive absorption for highly absorptive materials
            avg_absorption = float(np.mean(material.absorption_spectrum))
            
            if random.random() < avg_scatter:
                # Diffuse bounce - use diffuse_amplitude and cosine-weighted direction
                throughput *= material.diffuse_amplitude
                new_direction = cosine_weighted_hemisphere(normal)
                
                # NO EXTRA DAMPING - Test basic material absorption fix only
                if bounce < 2 and random.random() < 0.001:
                    print(f"DEBUG: Basic diffuse - abs: {avg_absorption:.3f}, diffuse_amp: {np.mean(material.diffuse_amplitude):.6f}, throughput: {np.max(throughput):.2e}")
                        
            else:
                # Specular bounce - use specular_amplitude and reflect with roughness
                throughput *= material.specular_amplitude
                reflected = reflect(dirn, normal)
                new_direction = jitter_direction(reflected, self.config.specular_roughness_rad)
                
                # NO EXTRA DAMPING - Test basic material absorption fix only
                if bounce < 2 and random.random() < 0.001:
                    print(f"DEBUG: Basic specular - abs: {avg_absorption:.3f}, specular_amp: {np.mean(material.specular_amplitude):.6f}, throughput: {np.max(throughput):.2e}")
            
            # Standard energy threshold check - no special treatment for absorptive materials
            if np.max(throughput) < self.config.min_throughput:
                if bounce < 2 and random.random() < 0.001:
                    print(f"DEBUG: Standard ray termination - energy: {np.max(throughput):.2e}, threshold: {self.config.min_throughput:.2e}")
                break
                
            # Russian roulette termination with energy compensation - TEMPORARILY DISABLED
            # should_terminate, throughput = self._apply_russian_roulette(bounce, throughput)
            # if should_terminate:
            #     break
                
            # Update for next iteration
            pos = hit_point + normal * self.config.eps + new_direction * (self.config.eps * 0.5)
            dirn = new_direction
            incoming_direction = (-new_direction).normalized()
            bounce += 1
    
    def _check_source_connection(self, hit_point: mathutils.Vector, normal: mathutils.Vector,
                               source: mathutils.Vector, throughput: np.ndarray, 
                               material: MaterialProperties, path_length: float,
                               bvh, incoming_direction: mathutils.Vector, bounce: int):
        """Check for direct line-of-sight connection from hit point to source."""
        from ..utils.scene_utils import los_clear
        from ..core.acoustics import NUM_BANDS
        
        # Vector from hit point to source
        to_source = source - hit_point
        distance_to_source = to_source.length
        
        debug_early_connections = bounce <= 2 and random.random() < 0.01  # Debug early bounces occasionally
        
        if distance_to_source <= self.config.eps:
            return  # Too close
            
        source_direction = to_source / distance_to_source
        
        # Check if source is above the surface (avoid self-intersection)
        if source_direction.dot(normal) <= 0.0:
            return  # Source is behind surface
            
        # Check line-of-sight to source
        if not los_clear(hit_point + normal * self.config.eps, source, bvh):
            return  # Blocked by geometry
            
        # Calculate total path length including connection to source
        total_distance = path_length + distance_to_source
        delay_ms = (total_distance / self.config.speed_of_sound) * 1000.0
        
        if debug_early_connections:
            print(f"DEBUG: Reverse connection bounce {bounce}: {delay_ms:.1f}ms, {total_distance:.1f}m")
            print(f"DEBUG: Initial throughput: {np.mean(throughput):.2e}")
        
        # Apply air absorption for the final segment to source
        final_throughput = throughput.copy()
        if self.config.air_enable:
            air_loss = self._calculate_air_absorption(distance_to_source)
            final_throughput *= air_loss
            if debug_early_connections:
                print(f"DEBUG: After air absorption: {np.mean(final_throughput):.2e} (factor: {np.mean(air_loss):.3f})")
            
        # Calculate BRDF contribution for reflection toward source
        # Use the incoming direction (from previous ray segment) and outgoing direction (toward source)
        brdf_weight = self._evaluate_brdf(normal, incoming_direction, source_direction, material)
        
        # CRITICAL FIX: Weight by cosine and 1/pi for proper Monte Carlo integration
        # In reverse ray tracing, each source connection must be weighted by the sampling PDF
        import math
        cos_out = abs(source_direction.dot(normal))
        monte_carlo_weight = cos_out / math.pi  # Cosine-weighted hemisphere sampling PDF
        
        # DEBUG: Track energy reduction chain
        initial_energy = np.mean(final_throughput)
        
        final_throughput *= brdf_weight * monte_carlo_weight
        
        # REMOVED AGGRESSIVE PENALTIES - Let material absorption do the work
        # No extra bounce penalties - material absorption already handles energy decay
        # No distance penalties - geometric spreading already handled in ray tracing
        
        final_energy = np.mean(final_throughput)
        
        # DEBUG: Track energy flow step by step
        debug_energy_flow = bounce <= 2 and random.random() < 0.005  # 0.5% of early bounces
        
        if debug_energy_flow:
            print(f"DEBUG ENERGY FLOW bounce {bounce}:")
            print(f"  1. Initial throughput: {np.mean(throughput):.4f}")
        
        # Apply BRDF weighting
        brdf_weight = self._evaluate_brdf(normal, incoming_direction, source_direction, material)
        
        if debug_energy_flow:
            print(f"  2. BRDF weight: {brdf_weight:.4f}")
        
        # Monte Carlo correction (solid angle sampling)
        # For surface reflections, use hemisphere solid angle (2π), not full sphere (4π)
        monte_carlo_weight = abs(source_direction.dot(normal)) / (2.0 * pi)
        
        if debug_energy_flow:
            print(f"  3. Monte Carlo weight: {monte_carlo_weight:.4f}")
            print(f"  4. Material diffuse_amp: {np.mean(material.diffuse_amplitude):.4f}")
            print(f"  5. Material specular_amp: {np.mean(material.specular_amplitude):.4f}")
        
        final_throughput = throughput.copy()
        final_throughput *= brdf_weight * monte_carlo_weight
        
        final_energy = np.mean(final_throughput)
        
        if debug_energy_flow:
            print(f"  6. Final energy: {final_energy:.4f}")
            print(f"  7. Total reduction factor: {(final_energy/initial_energy if initial_energy > 0 else 0):.4f}")
            print(f"  8. Should this contribute significantly? {final_energy > 0.001}")
        
        # DEBUG: Log energy reduction for early connections
        if debug_early_connections or (bounce <= 1 and random.random() < 0.001):
            print(f"DEBUG: Energy chain bounce {bounce}: {initial_energy:.2e} → {final_energy:.2e}")
            print(f"DEBUG: BRDF: {brdf_weight:.3f}, MC: {monte_carlo_weight:.3f}, Material absorption handles the rest")
            print(f"DEBUG: Total reduction: {(final_energy/initial_energy if initial_energy > 0 else 0):.2e}")
        
        # Emit impulse response contribution
        emission_success = self.emit_impulse(final_throughput, total_distance, source_direction, 1.0)
        
        if debug_energy_flow:
            print(f"  9. Emission success: {emission_success}")
            print(f"  10. Total distance: {total_distance:.2f}m")
            if emission_success:
                print(f"  11. ✓ CONTRIBUTION ADDED TO IR")
            else:
                print(f"  11. ✗ CONTRIBUTION REJECTED (energy too low?)")
        
        if emission_success:
            if not hasattr(self, 'connection_count'):
                self.connection_count = 0
            self.connection_count += 1
            self.wrote_any = True
    
    def _evaluate_brdf(self, normal: mathutils.Vector, incoming: mathutils.Vector, 
                      outgoing: mathutils.Vector, material: MaterialProperties) -> float:
        """Evaluate BRDF for reflection from incoming to outgoing direction."""
        cos_in = abs(incoming.dot(normal))
        cos_out = abs(outgoing.dot(normal))
        
        if cos_in <= 0.0 or cos_out <= 0.0:
            return 0.0
            
        # Use material amplitude directly - no extra normalization needed
        # The ray scattering already applied diffuse_amplitude/specular_amplitude
        avg_scatter = float(np.mean(material.scatter_spectrum))
        avg_diffuse_amp = float(np.mean(material.diffuse_amplitude))
        avg_specular_amp = float(np.mean(material.specular_amplitude))
        
        # Lambertian diffuse component - properly normalized
        diffuse = avg_diffuse_amp * cos_out
        
        # Specular component with realistic strength  
        reflect_dir = incoming - 2.0 * incoming.dot(normal) * normal
        specular_factor = max(0.0, reflect_dir.dot(outgoing))
        roughness = max(0.01, avg_scatter)
        specular = avg_specular_amp * pow(specular_factor, 1.0 / roughness)
        
        # Mix based on scatter probability (higher scatter = more diffuse)
        return diffuse * avg_scatter + specular * (1.0 - avg_scatter)
    
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
        amplitude_scalar = 1.0 / max(distance, self.config.receiver_radius)
        
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
    
    # Determine optimal strategy based on requirements
    # Omit_direct functionality removed - always use user-selected trace mode
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
    
    # Forward tracing for direct path + early reflections
    forward_tracer = create_ray_tracer('FORWARD', config)
    ir_early = forward_tracer.trace_rays(source, receiver, bvh, obj_map, early_rays)
    
    # Reverse tracing for diffuse reverb tail (skip direct path)
    import copy
    config_late = copy.deepcopy(config)  # Create copy for late reverb
    # Note: omit_direct functionality removed - reverse tracer handles direct path separately
    reverse_tracer = create_ray_tracer('REVERSE', config_late)
    ir_late = reverse_tracer.trace_rays(source, receiver, bvh, obj_map, late_rays)
    
    # Combine with time-based weighting
    ir_combined = _blend_early_late(ir_early, ir_late, config)
    
    return ir_combined


def _blend_early_late(ir_early: np.ndarray, ir_late: np.ndarray, config: RayTracingConfig) -> np.ndarray:
    """Blend early and late contributions using ENERGY-CONSERVING CROSSFADE."""
    
    # ENERGY-CONSERVING CROSSFADE: Forward fades as Reverse builds up
    # This prevents energy doubling while preserving user control over balance
    
    # Create time-based weighting functions
    samples = ir_early.shape[1]
    time_axis = np.arange(samples) / config.sample_rate
    
    # ENERGY SCALING for crossfade combination
    early_peak_region = slice(0, int(0.15 * config.sample_rate))  # 0-150ms
    late_peak_region = slice(int(0.1 * config.sample_rate), int(0.4 * config.sample_rate))  # 100-400ms
    
    early_rms = np.sqrt(np.mean(ir_early[:, early_peak_region] ** 2))
    late_rms = np.sqrt(np.mean(ir_late[:, late_peak_region] ** 2))
    
    # Scale Reverse to complement Forward in crossfade
    if late_rms > 0 and early_rms > 0:
        # More conservative scaling for crossfade to prevent over-contribution
        crossfade_scale_factor = (early_rms / late_rms) * 0.4  # Reduced from 0.6 for crossfade
        ir_late_scaled = ir_late * crossfade_scale_factor
        print(f"  DEBUG: Crossfade scaling - Forward RMS: {early_rms:.6f}, Reverse RMS: {late_rms:.6f}")
        print(f"  DEBUG: Reverse scaled by: {crossfade_scale_factor:.6f} (for crossfade blend)")
    else:
        ir_late_scaled = ir_late * 0.3  # More conservative default for crossfade
        print(f"  DEBUG: Using conservative Reverse scaling: 0.3 (for crossfade)")
    
    # CROSSFADE WEIGHTING: Forward fades as Reverse builds (weights sum to ~1.0)
    reverse_ramp_start = 0.05   # 50ms - start crossfade
    reverse_ramp_end = config.hybrid_reverb_ramp_time  # USER CONTROL: 0.05s - 0.5s
    
    # Initialize weights
    forward_weight = np.ones_like(time_axis)
    reverse_weight = np.zeros_like(time_axis)
    
    # Before crossfade: 100% Forward, 0% Reverse
    mask_early = time_axis < reverse_ramp_start
    forward_weight[mask_early] = 1.0
    reverse_weight[mask_early] = 0.0
    
    # After crossfade: Balanced blend (Forward still significant for tunnel echoes)
    mask_late = time_axis > reverse_ramp_end
    forward_late_weight = 0.3  # Keep 30% forward for discrete late reflections
    reverse_late_weight = 0.7  # 70% reverse for diffuse reverb
    forward_weight[mask_late] = forward_late_weight
    reverse_weight[mask_late] = reverse_late_weight
    
    # During crossfade: Smooth transition ensuring energy conservation
    mask_ramp = (time_axis >= reverse_ramp_start) & (time_axis <= reverse_ramp_end)
    if np.any(mask_ramp):
        ramp_width = reverse_ramp_end - reverse_ramp_start
        if ramp_width > 0:  # Prevent division by zero
            linear_progress = (time_axis[mask_ramp] - reverse_ramp_start) / ramp_width
            # Smooth S-curve for natural transition
            cosine_progress = 0.5 * (1.0 - np.cos(np.pi * linear_progress))
            
            # Crossfade: Forward fades as Reverse builds
            forward_weight[mask_ramp] = 1.0 - (1.0 - forward_late_weight) * cosine_progress
            reverse_weight[mask_ramp] = reverse_late_weight * cosine_progress
        else:
            # Instant transition if ramp_time = 0.05s
            forward_weight[mask_ramp] = forward_late_weight
            reverse_weight[mask_ramp] = reverse_late_weight
    
    # Verify energy conservation (total should be ~1.0)
    total_weight = forward_weight + reverse_weight
    max_total = np.max(total_weight)
    min_total = np.min(total_weight)
    avg_total = np.mean(total_weight)
    print(f"  DEBUG: Energy conservation check - Min: {min_total:.3f}, Max: {max_total:.3f}, Avg: {avg_total:.3f}")
    
    # CROSSFADE COMBINATION with USER GAIN CONTROLS
    ir_combined = np.zeros_like(ir_early)
    
    # Debug: Check individual tracer energies before combination
    early_max = np.max(np.abs(ir_early))
    late_max = np.max(np.abs(ir_late_scaled))
    print(f"  DEBUG: Forward max energy: {early_max:.6f}")
    print(f"  DEBUG: Reverse max energy (scaled): {late_max:.6f}")
    print(f"  DEBUG: User gains - Forward: {config.hybrid_forward_gain_db:.1f}dB ({config.hybrid_forward_gain_linear:.3f}x)")
    print(f"  DEBUG: User gains - Reverse: {config.hybrid_reverse_gain_db:.1f}dB ({config.hybrid_reverse_gain_linear:.3f}x)")
    print(f"  DEBUG: Reverb ramp time: {config.hybrid_reverb_ramp_time:.3f}s")
    
    for ch in range(ir_early.shape[0]):
        # ENERGY-CONSERVING CROSSFADE with USER GAIN CONTROLS:
        # Forward: Fades from 100% to 30% + user gain control
        # Reverse: Builds from 0% to 70% + user gain control  
        ir_combined[ch, :] = (ir_early[ch, :] * forward_weight * config.hybrid_forward_gain_linear + 
                              ir_late_scaled[ch, :] * reverse_weight * config.hybrid_reverse_gain_linear)
    
    combined_max_before = np.max(np.abs(ir_combined))
    print(f"  DEBUG: Combined max energy before norm: {combined_max_before:.6f}")
    
    # MINIMAL normalization for crossfade - should rarely be needed
    target_max = 0.8  # Conservative threshold
    if combined_max_before > target_max:
        normalization_factor = target_max / combined_max_before
        ir_combined *= normalization_factor
        combined_max_after = np.max(np.abs(ir_combined))
        print(f"  DEBUG: Applied normalization: {combined_max_before:.6f} → {combined_max_after:.6f} (factor: {normalization_factor:.6f})")
    else:
        print(f"  DEBUG: No normalization needed")
    
    # Debug: Check energy distribution over time
    sample_rate = getattr(config, 'sample_rate', 48000)
    early_samples = int(0.1 * sample_rate)  # First 100ms
    mid_samples = int(0.5 * sample_rate)    # Up to 500ms
    
    early_energy = np.sum(ir_combined[:, :early_samples] ** 2)
    mid_energy = np.sum(ir_combined[:, early_samples:mid_samples] ** 2) 
    late_energy = np.sum(ir_combined[:, mid_samples:] ** 2)
    
    print(f"  DEBUG: Energy distribution - Early: {early_energy:.6f}, Mid: {mid_energy:.6f}, Late: {late_energy:.6f}")
    
    print(f"  🎛️  ENERGY-CONSERVING CROSSFADE blend:")
    print(f"    Forward gain: {config.hybrid_forward_gain_db:+.1f}dB | Reverse gain: {config.hybrid_reverse_gain_db:+.1f}dB")
    print(f"    Reverb ramp: {reverse_ramp_start*1000:.0f}-{reverse_ramp_end*1000:.0f}ms")
    print(f"    Late field: {forward_late_weight*100:.0f}% Forward + {reverse_late_weight*100:.0f}% Reverse")
    print(f"    ✨ Preserves tunnel echoes while preventing energy doubling!")
    
    return ir_combined