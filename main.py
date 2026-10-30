"""
Project NOVA — Main Entry Point
===============================
Production 6-DoF aerospace flight simulator interactive engine.
Fully implements N-body rendering, dual control modes, navball, and mission conditions.
"""
import sys
import math
import random
from typing import Tuple, List

import numpy as np
import pygame

# --- STRICT NOVA PIPELINE IMPORTS ---
from nova.core.pipeline import SimulationPipeline, PipelineConfig, ControlInput
from nova.core.state_vector import VehicleState
from nova.core.constants import EARTH_MU, EARTH_RADIUS_EQ
from nova.physics.propulsion import EngineConfig
from nova.physics.aerodynamics import AeroConfig
from nova.physics.orbital import GravBody, elements_from_state
from nova.vehicle.mass_model import point_mass
from nova.vehicle.component_graph import ComponentGraph, ComponentNode
from nova.rendering.viewport import Viewport, ViewportConfig
from nova.rendering.celestial import (
    CelestialRenderer, CelestialConfig, CelestialBodyDef, project_eci_to_screen, _camera_basis
)

# --- PHYSICAL CONSTANTS & SYSTEM DICTIONARIES ---
AU = 1.495978707e11

BODY_RADII_M = {
    "Sun":     696000e3,
    "Mercury": 2440e3,
    "Venus":   6052e3,
    "Earth":   6371e3,
    "Moon":    1737e3,
    "Mars":    3390e3,
    "Jupiter": 69911e3,
    "Saturn":  58232e3,
    "Uranus":  25362e3,
    "Neptune": 24622e3
}

SOLAR_SYSTEM = [
    {"name": "Sun",     "mu": 1.327e20,   "pos": [0.0, 0.0, 0.0],                    "col": (255, 255, 0)},
    {"name": "Mercury", "mu": 2.203e13,   "pos": [0.0, 57.9e9, 0.0],                 "col": (169, 169, 169)},
    {"name": "Venus",   "mu": 3.248e14,   "pos": [0.0, 108.2e9, 0.0],                "col": (255, 220, 100)},
    {"name": "Earth",   "mu": 3.986e14,   "pos": [0.0, 149.6e9, 0.0],                "col": (0, 100, 255)},
    {"name": "Moon",    "mu": 4.902e12,   "pos": [384400e3, 149.6e9, 0.0],           "col": (200, 200, 200)},
    {"name": "Mars",    "mu": 4.282e13,   "pos": [0.0, 227.9e9, 0.0],                "col": (188, 74, 40)},
    {"name": "Jupiter", "mu": 1.266e17,   "pos": [0.0, 778.5e9, 0.0],                "col": (210, 150, 100)},
    {"name": "Saturn",  "mu": 3.793e16,   "pos": [0.0, 1432e9, 0.0],                 "col": (210, 190, 130)},
    {"name": "Uranus",  "mu": 5.793e15,   "pos": [0.0, 2867e9, 0.0],                 "col": (100, 220, 220)},
    {"name": "Neptune", "mu": 6.836e15,   "pos": [0.0, 4515e9, 0.0],                 "col": (63, 84, 186)}
]

def fmt_time(t_sec: float) -> str:
    d = int(t_sec / 86400)
    h = int((t_sec % 86400) / 3600)
    m = int((t_sec % 3600) / 60)
    return f"T+ {d}d {h}h {m}m"

def create_graph(mass: float) -> ComponentGraph:
    graph = ComponentGraph()
    mc = point_mass("core", mass, [0.0, 0.0, 0.0])
    graph.add_node(ComponentNode("root", "Core Payload", mc, "core"))
    graph.calculate_mass_properties()
    return graph

# ==============================================================================
# UI Displays and Configuration Pre-Flights
# ==============================================================================

def draw_compass(screen, cx: float, cy: float, radius: float, az_deg: float, el_deg: float, font):
    pygame.draw.circle(screen, (50, 50, 70), (cx, cy), radius, 2)
    pygame.draw.line(screen, (100, 100, 100), (cx-radius, cy), (cx+radius, cy), 1)
    pygame.draw.line(screen, (100, 100, 100), (cx, cy-radius), (cx, cy+radius), 1)
    for txt, ox, oy in [("N",0,-radius-10),("E",radius+10,0),("S",0,radius+10),("W",-radius-10,0)]:
        t = font.render(txt, True, (150, 150, 150))
        screen.blit(t, (cx+ox - t.get_width()//2, cy+oy - t.get_height()//2))
    px = cx + radius * math.cos(math.radians(az_deg - 90))
    py = cy + radius * math.sin(math.radians(az_deg - 90))
    pygame.draw.line(screen, (0, 255, 100), (cx, cy), (px, py), 3)

def pre_launch_loop(screen, w: int, h: int, font) -> Tuple[int, float, float]:
    preset_id = 0
    l_az, l_el = 90.0, 0.0
    confirmed = False
    cl_clk = pygame.time.Clock()

    while not confirmed:
        screen.fill((5, 8, 12))
        
        # Mouse click handling for preset selection
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        
        rect_1 = pygame.Rect(100, 100, 900, 30)
        rect_2 = pygame.Rect(100, 140, 900, 30)
        rect_3 = pygame.Rect(100, 180, 900, 30)
        
        if click:
            if rect_1.collidepoint(mouse_pos): preset_id = 0
            elif rect_2.collidepoint(mouse_pos): preset_id = 1
            elif rect_3.collidepoint(mouse_pos): preset_id = 2

        info = [
            "====== NOVA ORBITAL DEPLOYMENT CONTROL ======",
            "",
            "[1] LUNAR: Optimised for Earth-Moon transfer, moderate fuel, high Isp",
            "[2] INNER PLANETS: Higher delta-v budget for Venus or Mars",
            "[3] OUTER PLANETS: Maximum fuel fraction, low thrust, long burn time",
            "",
            f"SELECTED PRESET : [{preset_id + 1}]",
            "",
            "LAUNCH TRAJECTORY DIRECTION:",
            f"  Azimuth   (L/R arrows) : {l_az:3.0f} °",
            f"  Elevation (U/D arrows) : {l_el:3.0f} °",
            "",
            "[ENTER] CONFIRM VECTOR AND INITIATE DEPLOYMENT."
        ]
        
        by = 40
        for i, text in enumerate(info):
            color = (150, 255, 150) if "SELECTED" in text or "Azimuth" in text or "Elev" in text else (220, 220, 220)
            if i in [2, 3, 4] and (i-2) == preset_id: color = (255, 255, 50)
            txt_sfc = font.render(text, True, color)
            screen.blit(txt_sfc, (100, by))
            by += 32
            
        draw_compass(screen, w - 250, 250, 100, l_az, l_el, font)
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                sys.exit(0)
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1: preset_id = 0
                elif ev.key == pygame.K_2: preset_id = 1
                elif ev.key == pygame.K_3: preset_id = 2
                elif ev.key == pygame.K_RIGHT: l_az = (l_az + 5.0) % 360
                elif ev.key == pygame.K_LEFT: l_az = (l_az - 5.0) % 360
                elif ev.key == pygame.K_UP: l_el = min(90.0, l_el + 5.0)
                elif ev.key == pygame.K_DOWN: l_el = max(-90.0, l_el - 5.0)
                elif ev.key == pygame.K_RETURN: confirmed = True
        cl_clk.tick(30)
    return preset_id, l_az, l_el

# ==============================================================================
# Master Execution & Rendering Loop
# ==============================================================================

def main():
    pygame.init()
    sw, sh = 1280, 720
    screen = pygame.display.set_mode((sw, sh))
    pygame.display.set_caption("NOVA System - 6-DoF Physics Engine")
    
    font_sm = pygame.font.SysFont("courier", 14, bold=True)
    font_md = pygame.font.SysFont("courier", 18, bold=True)
    font_lg = pygame.font.SysFont("courier", 42, bold=True)
    clock = pygame.time.Clock()

    while True: # Outer restart wrapper
        preset_idx, az, el = pre_launch_loop(screen, sw, sh, font_md)

        bodies = []
        rend_bodies = []
        for b in SOLAR_SYSTEM:
            bodies.append(GravBody(name=b["name"], mu=b["mu"], position_eci=np.array(b["pos"], dtype=np.float64)))
            rend_bodies.append(CelestialBodyDef(name=b["name"], radius_m=BODY_RADII_M[b["name"]], position_eci=np.array(b["pos"], dtype=np.float64), color=b["col"]))
            
        if preset_idx == 0: 
            eng, m0 = EngineConfig("Lunar", 3500e3, 380.0, 310.0), 450_000.0
        elif preset_idx == 1: 
            eng, m0 = EngineConfig("Inner", 9500e3, 320.0, 290.0), 900_000.0
        else:
            eng, m0 = EngineConfig("Outer", 1800.0, 10000.0, 8000.0), 2_000_000.0
            
        pipe_cfg = PipelineConfig(dt=0.01, enable_aerodynamics=False, enable_j2=False, engine_config=eng, gravity_bodies=bodies)

        earth = next(x for x in bodies if x.name == "Earth")
        earth_surf = earth.position_eci + np.array([BODY_RADII_M[earth.name] + 150000.0, 0, 0], dtype=np.float64)
        vel_mag = math.sqrt(earth.mu / np.linalg.norm(earth_surf - earth.position_eci))
        
        la_rad, le_rad = math.radians(az), math.radians(el)
        init_vel = earth.velocity_eci if hasattr(earth, 'velocity_eci') else np.array([0,0,0], dtype=np.float64) 
        init_vel += np.array([math.cos(la_rad)*math.cos(le_rad), math.sin(la_rad)*math.cos(le_rad), math.sin(le_rad)]) * vel_mag
        
        v_st = VehicleState(position_eci=earth_surf, velocity_eci=init_vel, quaternion=np.array([1,0,0,0], dtype=np.float64), omega_body=np.zeros(3), mass=m0, time=0.0)
        pipeline = SimulationPipeline(pipe_cfg, v_st, create_graph(m0))
        viewport = Viewport(pipeline.registry, ViewportConfig(width_px=sw, height_px=sh))

        # 3D Skybox Stars
        bg_stars_3d = []
        for _ in range(2500):
            vc = np.array([random.gauss(0,1) for _ in range(3)])
            norm = np.linalg.norm(vc)
            if norm > 1e-10: bg_stars_3d.append(vc / norm)

        # Global Control State
        targets_list = ["Free", "Ship", "Earth", "Moon"] + [b["name"] for b in SOLAR_SYSTEM if b["name"] not in ("Earth", "Moon")]
        tgt_id = 1 # Start locked to Ship
        pilot_mode = True 
        
        cam_free_pos = np.array(earth_surf) + np.array([1e5, 0, 0], dtype=np.float64)
        cam_az, cam_el = 0.0, 0.4
        cam_dist = 50e3 
        cam_spd = 50000.0
        
        traj_points = []
        draw_traj = True
        is_map_view = False
        sas_enabled = False
        
        throttle_cmd = 0.0
        warp_rate = 1
        paused = False
        is_drag = False
        m_start = (0, 0)
        base_dt = 0.01

        sim_t, sys_time_acc = 0.0, 0.0
        max_alt, max_spd = 0.0, 0.0
        outcome = None
        staging_flag = False
        
        stage_anim_time = 0.0
        sas_anim_time = 0.0

        while not outcome:
            d_real = clock.tick(60) / 1000.0
            kp = pygame.key.get_pressed()
            staging_flag = False

            # --- Inputs Processing ---
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE:
                        if pilot_mode: 
                            staging_flag = True
                            stage_anim_time = 2.0
                        else: 
                            paused = not paused
                    elif ev.key == pygame.K_t: 
                        if pilot_mode: 
                            sas_enabled = not sas_enabled
                            sas_anim_time = 1.5
                        else: 
                            draw_traj = not draw_traj
                    elif ev.key == pygame.K_p: 
                        pilot_mode = not pilot_mode
                    elif ev.key == pygame.K_m: 
                        is_map_view = not is_map_view
                    elif ev.key == pygame.K_u: 
                        pilot_mode = False
                        tgt_id = 0 
                    elif ev.key == pygame.K_l:
                        tgt_id = (tgt_id + 1) % len(targets_list)
                        if targets_list[tgt_id] == "Ship": pilot_mode = True 
                        else: pilot_mode = False
                    
                    # Time Warp Keys
                    elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_PERIOD): 
                        warp_rate = min(100000, warp_rate * 2)
                    elif ev.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_COMMA): 
                        warp_rate = max(1, warp_rate // 2)
                        
                    elif ev.key == pygame.K_z: 
                        if pilot_mode: 
                            throttle_cmd = 1.0; warp_rate = 1
                    elif ev.key == pygame.K_x: 
                        if pilot_mode: 
                            throttle_cmd = 0.0; warp_rate = 1
                    
                    elif ev.key == pygame.K_RIGHTBRACKET: 
                        if tgt_id == 0: cam_spd *= 1.5 
                        else: cam_dist *= 1.5
                    elif ev.key == pygame.K_LEFTBRACKET: 
                        if tgt_id == 0: cam_spd /= 1.5 
                        else: cam_dist /= 1.5

                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1: 
                        is_drag, m_start = True, ev.pos
                elif ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button == 1: is_drag = False
                elif ev.type == pygame.MOUSEMOTION and is_drag:
                    rx, ry = ev.pos[0] - m_start[0], ev.pos[1] - m_start[1]
                    cam_az -= rx * 0.006
                    cam_el = max(-math.pi/2 + 0.001, min(math.pi/2 - 0.001, cam_el + ry * 0.006))
                    m_start = ev.pos
                elif ev.type == pygame.MOUSEWHEEL:
                    if tgt_id == 0: cam_spd *= (1.2 if ev.y > 0 else 1/1.2)
                    else: cam_dist *= (1.2 if ev.y < 0 else 1/1.2)

            # Check Click on TRJ button
            btn_rect = pygame.Rect(20, 80, 100, 25)
            if pygame.mouse.get_pressed()[0] and btn_rect.collidepoint(pygame.mouse.get_pos()):
                draw_traj = not draw_traj

            # --- Pilot Continuous Updates ---
            gb_p, gb_y, rll = 0.0, 0.0, 0.0
            cb = _camera_basis(cam_az, cam_el) 
            
            sprint_mult = 10.0 if (kp[pygame.K_LSHIFT] and not pilot_mode) else 1.0

            if tgt_id == 0 and not pilot_mode: # Free Camera Control
                dr = np.zeros(3)
                if kp[pygame.K_w]: dr += cb[:, 2]
                if kp[pygame.K_s]: dr -= cb[:, 2]
                if kp[pygame.K_a]: dr -= cb[:, 0]
                if kp[pygame.K_d]: dr += cb[:, 0]
                if kp[pygame.K_q]: dr += cb[:, 1]
                if kp[pygame.K_e]: dr -= cb[:, 1]
                if np.linalg.norm(dr) > 1e-6:
                    cam_free_pos += (dr/np.linalg.norm(dr)) * cam_spd * sprint_mult * d_real

            elif pilot_mode:
                if kp[pygame.K_w]: gb_p = 0.5   # Pitch down
                if kp[pygame.K_s]: gb_p = -0.5  # Pitch up
                if kp[pygame.K_a]: gb_y = 0.5   # Yaw left
                if kp[pygame.K_d]: gb_y = -0.5  # Yaw right
                if kp[pygame.K_q]: rll = -1.0   # Roll CCW
                if kp[pygame.K_e]: rll = 1.0    # Roll CW
                
                if kp[pygame.K_LSHIFT]: 
                    throttle_cmd = min(1.0, throttle_cmd + 0.05)
                    warp_rate = 1
                if kp[pygame.K_LCTRL]: 
                    throttle_cmd = max(0.0, throttle_cmd - 0.05)
                    warp_rate = 1
                
                if sas_enabled:
                    omg = pipeline.state.omega_body
                    gb_p = np.clip(gb_p - omg[1] * 2.0, -1.0, 1.0)
                    gb_y = np.clip(gb_y - omg[2] * 2.0, -1.0, 1.0)
                    rll  = np.clip(rll - omg[0] * 2.0, -1.0, 1.0)

            # --- Pipeline Progression Syncing ---
            if not paused:
                pipeline._config.dt = base_dt * warp_rate
                
                # Resynchronize accumulator if warp caused overshoot, preventing physics freezes
                if sim_t > sys_time_acc:
                    sys_time_acc = sim_t
                    
                sys_time_acc += d_real * warp_rate
                
                while sim_t < sys_time_acc:
                    cmd = ControlInput(throttle=throttle_cmd, gimbal_pitch=gb_p, gimbal_yaw=gb_y, aileron=rll, staging=staging_flag)
                    pipeline.tick(cmd)
                    sim_t += pipeline._config.dt
                    staging_flag = False
                    
                    s_r = pipeline.state.position_eci.copy()
                    if len(traj_points) == 0 or np.linalg.norm(s_r - traj_points[-1]) > 50000.0:
                        traj_points.append(s_r)
                        if len(traj_points) > 500: traj_points.pop(0)

            c_ship_p = pipeline.state.position_eci
            v_eci = pipeline.state.velocity_eci
            v_mag = np.linalg.norm(v_eci)
            max_spd = max(max_spd, v_mag)
            
            # --- Dynamic Camera Base Positions ---
            trg_n = targets_list[tgt_id]
            t_base_pos = cam_free_pos
            r_clamp = 1000e3  

            if is_map_view:
                t_base_pos = np.zeros(3) # Look at Sun 
                cam_dist = max(cam_dist, 500 * AU)
            else:
                if trg_n != "Free":
                    if trg_n == "Ship":
                        t_base_pos, r_clamp = c_ship_p, 20.0 
                    else:
                        for b in rend_bodies:
                            if b.name == trg_n:
                                t_base_pos, r_clamp = b.position_eci, BODY_RADII_M[b.name] * 2.0
                                break

                # Clamp Zoom
                min_zoom = r_clamp
                max_zoom = r_clamp * 500.0 if trg_n != "Free" else 1000 * AU
                cam_dist = min(max(cam_dist, min_zoom), max_zoom)
                
            final_cam_pos = t_base_pos if trg_n == "Free" else t_base_pos - cb[:, 2] * cam_dist
            f_len = cam_dist * 2.0
            
            screen.fill((0, 0, 5))
            
            # --- Projector Logic ---
            def do_proj(vt):
                return project_eci_to_screen(vt, final_cam_pos, cb, f_len, (sw/2, sh/2), sh/max(10.0, cam_dist))
                
            # Skybox Background Projection - Cast infinitely mapping directional unit array constraints properly
            for sd in bg_stars_3d:
                xc = float(np.dot(sd, cb[:, 0]))
                yc = float(np.dot(sd, cb[:, 1]))
                zc = float(np.dot(sd, cb[:, 2]))
                if zc > 1e-4:
                    sx = int((xc / zc) * f_len * (sh/max(10.0, cam_dist)) + (sw/2))
                    sy = int(-(yc / zc) * f_len * (sh/max(10.0, cam_dist)) + (sh/2))
                    if 0 <= sx < sw and 0 <= sy < sh:
                        screen.set_at((sx, sy), (255, 255, 255))

            # Render Objects and Planetary Map Outlines
            for bx in rend_bodies:
                rx = BODY_RADII_M[bx.name]
                if is_map_view and bx.name != "Sun":
                    # Draw orbital ring around sun
                    rad_orb = np.linalg.norm(bx.position_eci)
                    o_pts = []
                    for k in range(120):
                        ang = 2.0 * math.pi * k / 120
                        ov = np.array([rad_orb * math.cos(ang), rad_orb * math.sin(ang), 0.0])
                        opj = do_proj(ov)
                        if opj: o_pts.append(opj)
                    if len(o_pts) > 1: pygame.draw.lines(screen, (50, 50, 80), True, o_pts, 1)

                # Draw body wireframe
                cpx, cpy, cpz = bx.position_eci
                for idx in range(12):
                    l_t = 2.0 * math.pi * idx / 12
                    c_lx = []
                    for k in range(37):
                        phi = -math.pi / 2.0 + math.pi * k / 36.0
                        vx = np.array([cpx + rx*math.cos(phi)*math.cos(l_t), cpy + rx*math.cos(phi)*math.sin(l_t), cpz + rx*math.sin(phi)])
                        prj = do_proj(vx)
                        if prj: c_lx.append(prj)
                    if len(c_lx) > 1: pygame.draw.lines(screen, bx.color, False, c_lx, 1)

                # Labels
                tx_proj = do_proj(bx.position_eci)
                if tx_proj and 0 <= tx_proj[0] < sw and 0 <= tx_proj[1] < sh:
                    txt = font_sm.render(bx.name, True, (255,255,255))
                    screen.blit(txt, (tx_proj[0]+5, tx_proj[1]-15))

            # Ship Trails 
            if draw_traj and len(traj_points) > 1:
                trk = [do_proj(pt) for pt in traj_points]
                vt = [v for v in trk if v]
                if len(vt) > 1: pygame.draw.lines(screen, (255,255,255), False, vt, 2)
                
            # --- Ship, Thruster, Prograde, & Visual Effect Projections ---
            sp1 = do_proj(c_ship_p)
            if sp1 and v_mag > 1e-5:
                sp2 = do_proj(c_ship_p + v_eci * 0.01) 
                if sp2: 
                    ax = math.atan2(sp2[1]-sp1[1], sp2[0]-sp1[0])
                    sx, sy = sp1
                    # Small filled white triangle (8px tall)
                    pygame.draw.polygon(screen, (255,255,255), [
                        (sx+8*math.cos(ax), sy+8*math.sin(ax)),
                        (sx+8*math.cos(ax+2.6), sy+8*math.sin(ax+2.6)),
                        (sx+8*math.cos(ax-2.6), sy+8*math.sin(ax-2.6))
                    ])
                    # Visual Engine Flare 
                    if throttle_cmd > 0.0:
                        back_ax = ax + math.pi
                        ln = 8 + (throttle_cmd * 18.0)
                        pygame.draw.polygon(screen, (255, 150, 0), [
                            (sx, sy), 
                            (sx+8*math.cos(ax+2.6), sy+8*math.sin(ax+2.6)), 
                            (sx+ln*math.cos(back_ax), sy+ln*math.sin(back_ax)),
                            (sx+8*math.cos(ax-2.6), sy+8*math.sin(ax-2.6))
                        ])
                    
                    # Gimbal Indicators
                    if abs(gb_p) > 0.01 or abs(gb_y) > 0.01:
                        cx_x = sx + 25 * gb_p; cx_y = sy + 25 * gb_y
                        pygame.draw.line(screen, (255,255,0), (cx_x-3, cx_y), (cx_x+3, cx_y))
                        pygame.draw.line(screen, (255,255,0), (cx_x, cx_y-3), (cx_x, cx_y+3))

                    # Visuals: SAS Overlay Brackets
                    if sas_enabled:
                        pulse = math.sin(pygame.time.get_ticks() * 0.005) * 3
                        r_sas = max(1, int(25 + pulse))
                        # Render arc brackets tracking SAS stabilization securely
                        pygame.draw.arc(screen, (50, 255, 255), (int(sx-r_sas), int(sy-r_sas), r_sas*2, r_sas*2), 0, math.pi/2, 2)
                        pygame.draw.arc(screen, (50, 255, 255), (int(sx-r_sas), int(sy-r_sas), r_sas*2, r_sas*2), math.pi, math.pi*1.5, 2)
                        pygame.draw.line(screen, (50, 255, 255), (sx-r_sas-5, sy), (sx-r_sas+5, sy), 1)
                        pygame.draw.line(screen, (50, 255, 255), (sx+r_sas-5, sy), (sx+r_sas+5, sy), 1)
                        pygame.draw.line(screen, (50, 255, 255), (sx, sy-r_sas-5), (sx, sy-r_sas+5), 1)
                        pygame.draw.line(screen, (50, 255, 255), (sx, sy+r_sas-5), (sx, sy+r_sas+5), 1)

                    # Visuals: Staging Burst Overlay
                    if stage_anim_time > 0:
                        stage_anim_time -= d_real
                        progress = 1.0 - (stage_anim_time / 2.0)
                        r_stg = max(1, int(10 + progress * 100))
                        pygame.draw.circle(screen, (255, 150, 50), (int(sx), int(sy)), r_stg, 2)
                        pygame.draw.circle(screen, (200, 200, 255), (int(sx), int(sy)), r_stg + 5, 1)

                # Prograde/Retrograde Markers
                pr = do_proj(c_ship_p + (v_eci/v_mag) * cam_dist*0.2)
                re = do_proj(c_ship_p - (v_eci/v_mag) * cam_dist*0.2)
                if pr: 
                    pygame.draw.circle(screen, (255,255,0), (int(pr[0]), int(pr[1])), 6, 0)
                if re: 
                    r_i, r_j = int(re[0]), int(re[1])
                    pygame.draw.circle(screen, (255,255,0), (r_i, r_j), 6, 1)
                    pygame.draw.line(screen, (255,255,0), (r_i-5, r_j-5), (r_i+5, r_j+5), 2)
                    pygame.draw.line(screen, (255,255,0), (r_i+5, r_j-5), (r_i-5, r_j+5), 2)

            # --- Overlay Action Text Announcements ---
            if stage_anim_time > 1.0:
                st_txt = font_lg.render("STAGING CONFIRMED", True, (255, 200, 50))
                screen.blit(st_txt, (sw//2 - st_txt.get_width()//2, sh//4))
                
            if sas_anim_time > 0:
                sas_anim_time -= d_real
                msg = "SAS ENABLED" if sas_enabled else "SAS DISABLED"
                col = (50, 255, 255) if sas_enabled else (255, 100, 100)
                s_txt = font_lg.render(msg, True, col)
                screen.blit(s_txt, (sw//2 - s_txt.get_width()//2, sh//4 + 50))

            # --- Navball Interface Processing (Bottom Center) ---
            nx, ny, n_rad = sw // 2, sh - 80, 60
            pygame.draw.circle(screen, (30, 40, 50), (nx, ny), n_rad)
            pygame.draw.circle(screen, (150, 150, 150), (nx, ny), n_rad, 2)
            
            if v_mag > 1e-5:
                v_dir = v_eci / v_mag
                h_dir = np.cross(c_ship_p, v_eci)
                if np.linalg.norm(h_dir) > 1e-6:
                    h_dir /= np.linalg.norm(h_dir)
                r_out = np.cross(v_dir, h_dir)
                
                mrks = [
                    (v_dir, (255,255,0), "pro"), (-v_dir, (255,255,0), "ret"),
                    (h_dir, (200,50,200), "nrm"), (-h_dir, (200,50,200), "anrm"),
                    (r_out, (0,255,255), "rout"), (-r_out, (0,255,255), "rin")
                ]
                for vec, col, m_t in mrks:
                    vx_c = np.dot(vec, cb[:, 0])
                    vy_c = np.dot(vec, cb[:, 1])
                    vz_c = np.dot(vec, cb[:, 2])
                    if vz_c > 0: # Only draw front face
                        px = nx + vx_c * n_rad
                        py = ny - vy_c * n_rad 
                        if m_t in ["pro", "rin", "rout"]: pygame.draw.circle(screen, col, (int(px), int(py)), 4)
                        elif m_t == "ret": 
                            pygame.draw.line(screen, col, (px-4,py-4),(px+4,py+4), 2)
                            pygame.draw.line(screen, col, (px+4,py-4),(px-4,py+4), 2)
                        else:
                            pygame.draw.polygon(screen, col, [(px, py-5), (px-4, py+3), (px+4, py+3)])

            # --- HUD Texts & Control Blocks ---
            tl = [
                f"{fmt_time(pipeline.state.time)}  | Warp: {warp_rate}x",
                f"Locked: {targets_list[tgt_id]}",
                f"Mode: {'PILOT MODE' if pilot_mode else 'FREE CAM'}",
                f"SAS : {'ON' if sas_enabled else 'OFF'}" if pilot_mode else ""
            ]
            for ix, txt in enumerate(tl):
                if not txt: continue
                c = (150,255,150) if "PILOT" in txt else ((255,255,100) if "FREE" in txt else (200,200,200))
                screen.blit(font_md.render(txt, True, c), (20, 20 + ix*20))

            snap = pipeline.registry.latest
            if snap:
                ele = elements_from_state(snap.position_eci, snap.velocity_eci, EARTH_MU)
                rt_t = [
                    "LIVE ORBITAL ELEMENTS",
                    f"Ap:  {ele.apoapsis/1000:.1f} km",
                    f"Pe:  {ele.periapsis/1000:.1f} km",
                    f"Ecc: {ele.eccentricity:.4f}",
                    f"Inc: {math.degrees(ele.inclination):.1f}°",
                    f"Per: {ele.period/60:.1f} m",
                    f"TrA: {math.degrees(ele.true_anomaly):.1f}°"
                ]
                for ix, txt in enumerate(rt_t):
                    screen.blit(font_sm.render(txt, True, (200, 250, 255)), (sw - 230, 20 + ix*18))

            cst = "Red" if throttle_cmd > 0.66 else ("Yellow" if throttle_cmd > 0.33 else "Green")
            bc = (255,50,50) if cst=="Red" else ((255,255,50) if cst=="Yellow" else (50,255,50))
            bl = [
                f"THR: {throttle_cmd*100:3.0f}%",
                f"Gimbal P/Y: {gb_p:.1f} | {gb_y:.1f}",
                f"Vel: {v_mag:.1f} m/s",
                f"Alt: {(np.linalg.norm(c_ship_p)-EARTH_RADIUS_EQ)/1000:.1f} km",
                f"Mass:{pipeline.state.mass:.0f} kg",
                f"Spd: {cam_spd:.1f} m/s" if not pilot_mode else ""
            ]
            for ix, txt in enumerate(bl):
                screen.blit(font_md.render(txt, True, bc if "THR" in txt else (220,220,220)), (20, sh - 150 + ix*20))

            # Controls Legend Base
            leg_h = font_sm.render("PILOT: W/S=Pch A/D=Yaw Q/E=Rll Shft/Ctrl=Thr+/- Z=Fl X=Cut Spc=Stg T=SAS P=Mod", True, (130, 130, 130))
            leg_f = font_sm.render("FREE: WASDQE=Mov Shft=Spt drag=Look []=Zm scroll=Zm L=Lck U=Fre M=Map +/-=Wrp", True, (130, 130, 130))
            screen.blit(leg_h, (sw - leg_h.get_width() - 10, sh - 45))
            screen.blit(leg_f, (sw - leg_f.get_width() - 10, sh - 25))

            # HUD Trajectory Button Overlay
            trj_col = (50, 255, 50) if draw_traj else (255, 50, 50)
            trj_txt = "TRJ: ON" if draw_traj else "TRJ: OFF"
            pygame.draw.rect(screen, (50, 50, 50), (20, 100, 100, 25))
            screen.blit(font_sm.render(trj_txt, True, trj_col), (25, 105))

            if paused:
                pz = font_lg.render("PAUSED", True, (255, 80, 80))
                screen.blit(pz, (sw/2 - pz.get_width()/2, sh/2 + 150))

            pygame.display.flip()

            # --- Collision / Success Monitors ---
            s_alt = np.linalg.norm(c_ship_p)
            max_alt = max(max_alt, s_alt)
            
            for b in rend_bodies:
                dist = np.linalg.norm(c_ship_p - b.position_eci)
                rad = BODY_RADII_M[b.name]
                if dist < rad: 
                    outcome = f"FAIL_{b.name}"
                    break
                elif b.name not in ("Sun", "Earth") and dist < rad * 2.0 and ele.eccentricity < 0.5:
                    outcome = f"SUCC_{b.name}"
                    break
                    
        # --- End Screen Overlays (Collision/Success) ---
        while outcome:
            screen.fill((100,0,0) if "FAIL" in outcome else (0,100,0))
            bnm = outcome.split('_')[1].upper()
            tl = f"MISSION {'FAILED' if 'FAIL' in outcome else 'SUCCESS'} — {'COLLISION WITH' if 'FAIL' in outcome else bnm + ' REACHED'}"
            r_t1 = font_lg.render(tl, True, (255,255,255))
            screen.blit(r_t1, (sw/2 - r_t1.get_width()/2, sh/2 - 100))
            
            stxt = f"Time: {fmt_time(pipeline.state.time)} | Max V: {max_spd:.1f} m/s | Max Alt: {max_alt/1000:.1f} km"
            r_t2 = font_md.render(stxt, True, (220,220,220))
            screen.blit(r_t2, (sw/2 - r_t2.get_width()/2, sh/2))
            
            rst = font_md.render("Press R to restart | ESC to Quit", True, (255,255,255))
            screen.blit(rst, (sw/2 - rst.get_width()/2, sh/2 + 50))
            
            pygame.display.flip()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: sys.exit(0)
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE: sys.exit(0)
                    elif ev.key == pygame.K_r:
                        outcome = None 

if __name__ == "__main__":
    main()