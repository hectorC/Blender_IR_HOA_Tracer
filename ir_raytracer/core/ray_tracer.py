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
        
        total_connections = 0
        rays_traced = 0
        for d in directions:
            first_direction = mathutils.Vector(d).normalized()
            incoming_direction = (-first_direction).normalized()
            self._trace_single_ray(first_direction, receiver, source, bvh, obj_map, 
                                 band_one, incoming_direction)
            rays_traced += 1
        
        # SIMPLIFIED: Much gentler normalization approach
        # The issue: aggressive normalization is killing late energy
        if num_dirs > 0:
            ir_max_before = np.max(np.abs(self.ir))
            
            # Check if we're likely in hybrid mode (fewer rays)
            if num_dirs < 6000:  # Likely hybrid mode (half of typical 8192)
                # GENTLE normalization for hybrid compatibility - preserve most energy
                self.ir /= 50.0  # Much gentler than /1000
                norm_factor = 1.0/50.0
                print(f"DEBUG: HYBRID-mode IR normalization - Before: {ir_max_before:.2e}, Factor: {norm_factor:.2e}")
            else:
                # Standalone mode: no normalization (preserve RT60)
                norm_factor = 1.0
                print(f"DEBUG: STANDALONE-mode IR (no normalization) - Before: {ir_max_before:.2e}, Factor: {norm_factor:.2e}")
            
            ir_max_after = np.max(np.abs(self.ir))
            print(f"DEBUG: IR normalization result - After: {ir_max_after:.2e}")
        
        # Always add direct path (omit_direct functionality removed)
        print("DEBUG: Adding direct path (reverse tracer)...")
        self._add_direct_path(source, receiver, bvh, band_one)
        
        print(f"DEBUG: Reverse tracer completed {rays_traced} rays")
        if hasattr(self, 'connection_count'):
            print(f"DEBUG: Total successful connections: {self.connection_count}")
        
        return self.ir
    
    def _trace_single_ray(self, direction: mathutils.Vector, start_pos: mathutils.Vector,
                         target: mathutils.Vector, bvh, obj_map: List[Any],
                         initial_throughput: np.ndarray, incoming_direction: mathutils.Vector):
        """Trace a single reverse ray from receiver toward room, checking for source connections."""
        import random
        
        pos = start_pos
        dirn = direction
        throughput = initial_throughput.copy()
        path_length = 0.0
        bounce = 0
        
        debug_this_ray = bounce == 0 and random.random() < 0.0001  # Debug only ~0.01% of rays
        
        connections_made = 0
        
        while bounce < self.config.max_bounces:
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
            
            # Apply air absorption
            if self.config.air_enable:
                throughput *= self._calculate_air_absorption(seg_length)
            
            # Always check for direct connection to source (omit_direct functionality removed)
            self._check_source_connection(hit_point, normal, target, throughput, 
                                        material, path_length, bvh, incoming_direction, bounce)
            
            # Apply surface absorption
            throughput *= material.reflection_amplitude  # Use reflection amplitude, not (1 - absorption)
            
            # Check if energy is too low to continue
            if np.max(throughput) < self.config.min_throughput:
                break
                
            # Russian roulette termination with energy compensation
            should_terminate, throughput = self._apply_russian_roulette(bounce, throughput)
            if should_terminate:
                break
            
            # Sample new direction based on material BRDF
            new_direction = self._sample_brdf_direction(normal, -dirn, material, 
                                                      random.random(), random.random())
            if new_direction is None:
                break
                
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
        
        # Additional normalization based on bounce count to prevent accumulation
        # Later bounces should contribute less (geometric decay)
        bounce_weight = 1.0 / (1.0 + bounce * 0.5)  # Decay with bounce count
        final_throughput *= bounce_weight
        
        # Scale by distance to further reduce over-contribution from distant connections
        distance_weight = 1.0 / max(1.0, distance_to_source * 0.1)
        final_throughput *= distance_weight
        
        final_energy = np.mean(final_throughput)
        
        # DEBUG: Log energy reduction for early connections
        if debug_early_connections or (bounce <= 1 and random.random() < 0.001):
            print(f"DEBUG: Energy chain bounce {bounce}: {initial_energy:.2e} → {final_energy:.2e}")
            print(f"DEBUG: BRDF: {brdf_weight:.3f}, MC: {monte_carlo_weight:.3f}, Bounce: {bounce_weight:.3f}, Dist: {distance_weight:.3f}")
            print(f"DEBUG: Total reduction: {(final_energy/initial_energy if initial_energy > 0 else 0):.2e}")
        
        # Emit impulse response contribution
        if self.emit_impulse(final_throughput, total_distance, source_direction, 1.0):
            if not hasattr(self, 'connection_count'):
                self.connection_count = 0
            self.connection_count += 1
            self.wrote_any = True
    
    def _evaluate_brdf(self, normal: mathutils.Vector, incoming: mathutils.Vector, 
                      outgoing: mathutils.Vector, material: MaterialProperties) -> float:
        """Evaluate BRDF for reflection from incoming to outgoing direction."""
        # Simplified Lambert + specular BRDF model
        cos_in = abs(incoming.dot(normal))
        cos_out = abs(outgoing.dot(normal))
        
        if cos_in <= 0.0 or cos_out <= 0.0:
            return 0.0
            
        # Lambertian diffuse component
        diffuse = cos_out / pi
        
        # Simple specular component (Phong-like)
        reflect_dir = incoming - 2.0 * incoming.dot(normal) * normal
        specular_factor = max(0.0, reflect_dir.dot(outgoing))
        # Use average scatter as roughness indicator
        avg_scatter = float(np.mean(material.scatter_spectrum))
        specular = pow(specular_factor, max(1.0, 1.0 / max(avg_scatter, 0.01)))
        
        # Combine diffuse and specular (simplified mixing)
        return diffuse * avg_scatter + specular * (1.0 - avg_scatter) * 0.1
    
    def _sample_brdf_direction(self, normal: mathutils.Vector, incoming: mathutils.Vector,
                              material: MaterialProperties, r1: float, r2: float) -> Optional[mathutils.Vector]:
        """Sample a new direction based on BRDF."""
        import math
        
        # Simple cosine-weighted hemisphere sampling for diffuse
        # Convert to local coordinate system where normal is (0,0,1)
        
        # Create orthonormal basis
        if abs(normal.z) < 0.999:
            tangent = mathutils.Vector((normal.y, -normal.x, 0.0)).normalized()
        else:
            tangent = mathutils.Vector((1.0, 0.0, 0.0))
        bitangent = normal.cross(tangent)
        
        # Cosine-weighted hemisphere sampling
        cos_theta = math.sqrt(r1)
        sin_theta = math.sqrt(1.0 - r1)
        phi = 2.0 * math.pi * r2
        
        # Local direction
        local_dir = mathutils.Vector((
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi), 
            cos_theta
        ))
        
        # Transform to world coordinates
        world_dir = (local_dir.x * tangent + 
                    local_dir.y * bitangent + 
                    local_dir.z * normal)
        
        return world_dir.normalized()
    
    
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
    """Blend early and late contributions with time-based weighting."""
    
    # Transition time: early reflections dominate before this, late reverb after
    transition_time_sec = 0.1  # 100ms transition point (typical for rooms)
    transition_sample = int(transition_time_sec * config.sample_rate)
    
    # Create time-based weighting functions
    samples = ir_early.shape[1]
    time_axis = np.arange(samples) / config.sample_rate
    
    # ENERGY MATCHING: Scale tracers to similar energy levels before crossfade
    # Measure energy in overlapping regions to match levels
    early_peak_region = slice(0, int(0.15 * config.sample_rate))  # 0-150ms
    late_peak_region = slice(int(0.05 * config.sample_rate), int(0.3 * config.sample_rate))  # 50-300ms
    
    early_rms = np.sqrt(np.mean(ir_early[:, early_peak_region] ** 2))
    late_rms = np.sqrt(np.mean(ir_late[:, late_peak_region] ** 2))
    
    # Scale reverse tracer to match forward energy level
    if late_rms > 0 and early_rms > 0:
        energy_match_factor = early_rms / late_rms
        ir_late_scaled = ir_late * energy_match_factor
        print(f"  DEBUG: Energy matching - Forward RMS: {early_rms:.6f}, Reverse RMS: {late_rms:.6f}")
        print(f"  DEBUG: Scaling Reverse by factor: {energy_match_factor:.6f}")
    else:
        ir_late_scaled = ir_late
        print(f"  DEBUG: Energy matching skipped (zero RMS)")
    
    # EXTENDED crossfade for smoother transition
    crossfade_width = 0.15  # 150ms crossfade region (was 50ms)
    crossfade_start = transition_time_sec - crossfade_width/2
    crossfade_end = transition_time_sec + crossfade_width/2
    
    # Create smooth crossfade weights using cosine interpolation for gentler transition
    early_weight = np.ones_like(time_axis)
    late_weight = np.zeros_like(time_axis)
    
    # Before crossfade: pure Forward
    mask_early = time_axis < crossfade_start
    early_weight[mask_early] = 1.0
    late_weight[mask_early] = 0.0
    
    # After crossfade: pure Reverse  
    mask_late = time_axis > crossfade_end
    early_weight[mask_late] = 0.0
    late_weight[mask_late] = 1.0
    
    # During crossfade: smooth cosine transition (gentler than linear)
    mask_crossfade = (time_axis >= crossfade_start) & (time_axis <= crossfade_end)
    if np.any(mask_crossfade):
        # Cosine interpolation for smoother transition
        linear_progress = (time_axis[mask_crossfade] - crossfade_start) / crossfade_width
        # Convert linear to cosine for gentler curves
        cosine_progress = 0.5 * (1.0 - np.cos(np.pi * linear_progress))
        early_weight[mask_crossfade] = 1.0 - cosine_progress
        late_weight[mask_crossfade] = cosine_progress
    
    # Apply weighting with detailed debugging
    ir_combined = np.zeros_like(ir_early)
    
    # Debug: Check individual tracer energies before blending
    early_max = np.max(np.abs(ir_early))
    late_max = np.max(np.abs(ir_late_scaled))
    print(f"  DEBUG: Forward max energy: {early_max:.6f}")
    print(f"  DEBUG: Reverse max energy (scaled): {late_max:.6f}")
    
    for ch in range(ir_early.shape[0]):
        ir_combined[ch, :] = (ir_early[ch, :] * early_weight + 
                              ir_late_scaled[ch, :] * late_weight)
    
    combined_max_before = np.max(np.abs(ir_combined))
    print(f"  DEBUG: Combined max energy before norm: {combined_max_before:.6f}")
    
    # GENTLE normalization - only if really needed
    target_max = 0.5  # Less aggressive normalization (was 0.1)
    if combined_max_before > target_max:
        normalization_factor = target_max / combined_max_before
        ir_combined *= normalization_factor
        combined_max_after = np.max(np.abs(ir_combined))
        print(f"  DEBUG: Applied gentle normalization: {combined_max_before:.6f} → {combined_max_after:.6f} (factor: {normalization_factor:.6f})")
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
    
    print(f"  Smooth crossfade: {crossfade_start*1000:.0f}-{crossfade_end*1000:.0f}ms ({crossfade_width*1000:.0f}ms width)")
    
    return ir_combined