"""
================================================================
 DEFENDER-AWARE BASKETBALL SHOT SIMULATION  (Animated, v2.0)
 Extension of Silverberg, Tran & Adcock (2003)
================================================================
 Physics
   - RK4 integration of projectile motion
   - Quadratic aerodynamic drag with horizontal wind
   - Rim / backboard collision, swish vs. rim-hit detection
   - Defender modelled as a vertical reach barrier

 Interface
   - Animated ball with fading motion trail
   - Shooter and defender drawn to scale; BOTH heights are
     slider-controlled and update live on screen
   - Separate read-out panel (no text overlapping the court)
   - Auto-Optimize: searches for the softest shot that scores
     while still clearing the defender
================================================================
 Run:  python basketball_simulation.py
 Deps: numpy, matplotlib
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Rectangle, Ellipse
import textwrap
from dataclasses import dataclass


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

@dataclass
class Params:
    # Ball (men's size 7)
    m: float = 0.624            # kg
    R: float = 0.1194           # m
    Cd: float = 0.47            # drag coefficient (sphere)

    # Environment
    g: float = 9.81             # m/s^2
    rho: float = 1.225          # kg/m^3 (sea-level air)

    # Court geometry (FIBA / NBA)
    hoop_height: float = 3.048  # 10 ft
    hoop_radius: float = 0.2286 # 18 in diameter rim
    rim_offset: float = 0.15    # rim front edge stands off the backboard
    backboard_x: float = 4.191  # free-throw line to backboard
    release_x: float = 0.15     # ball leaves slightly ahead of the shooter

    @property
    def hoop_x(self) -> float:
        """Horizontal centre of the rim."""
        return self.backboard_x - self.rim_offset - self.hoop_radius


# ============================================================
# 2. PHYSICS ENGINE
# ============================================================

def drag_force(vx, vy, wind_x, P: Params):
    """Quadratic aerodynamic drag evaluated in the air frame."""
    vxr = vx - wind_x          # velocity relative to the air
    vyr = vy
    v = np.hypot(vxr, vyr)
    if v < 1e-9:
        return 0.0, 0.0
    A = np.pi * P.R ** 2
    k = 0.5 * P.rho * P.Cd * A
    return -k * v * vxr, -k * v * vyr


def derivatives(state, wind_x, P: Params):
    """state = [x, y, vx, vy] -> d(state)/dt"""
    _, _, vx, vy = state
    Fx, Fy = drag_force(vx, vy, wind_x, P)
    return np.array([vx, vy, Fx / P.m, -P.g + Fy / P.m])


def rk4_step(state, dt, wind_x, P: Params):
    k1 = derivatives(state, wind_x, P)
    k2 = derivatives(state + 0.5 * dt * k1, wind_x, P)
    k3 = derivatives(state + 0.5 * dt * k2, wind_x, P)
    k4 = derivatives(state + dt * k3, wind_x, P)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================
# 3. TRAJECTORY SIMULATION
# ============================================================

def simulate(v0, theta_deg, wind_x, defender_x, defender_reach,
             release_height, P: Params, dt: float = 0.002,
             collide: bool = True):
    """
    Integrate one shot and classify the outcome.

    Returns a dict with the trajectory plus diagnostic quantities
    (apex, entry angle, flight time, clearance over the defender).
    """
    theta = np.radians(theta_deg)
    state = np.array([P.release_x, release_height,
                      v0 * np.cos(theta), v0 * np.sin(theta)])

    t_max = 5.0

    xs, ys, ts = [state[0]], [state[1]], [0.0]

    clears_defender = None
    defender_clearance = None
    hit_backboard = False
    hit_rim = False
    hit_ground = False
    made = False
    entry_angle = None
    x_cross = None          # where the ball crosses the rim plane on the way down
    t = 0.0

    hoop_x = P.hoop_x
    board_top = P.hoop_height + 0.90
    board_bot = P.hoop_height - 0.15

    while t < t_max:
        prev = state.copy()
        state = rk4_step(state, dt, wind_x, P)
        t += dt

        # ---- defender clearance (first crossing of the defender line)
        if clears_defender is None and state[0] >= defender_x > prev[0]:
            frac = (defender_x - prev[0]) / (state[0] - prev[0] + 1e-12)
            y_at = prev[1] + frac * (state[1] - prev[1])
            defender_clearance = y_at - P.R - defender_reach
            clears_defender = defender_clearance > 0.0

        # ---- contact with the iron (front and back rim edges)
        for rx in ([] if not collide else (hoop_x - P.hoop_radius, hoop_x + P.hoop_radius)):
            dx, dy = state[0] - rx, state[1] - P.hoop_height
            d = np.hypot(dx, dy)
            if d < P.R:
                hit_rim = True
                nx, ny = dx / (d + 1e-12), dy / (d + 1e-12)
                vn = state[2] * nx + state[3] * ny
                if vn < 0:                       # moving into the rim
                    e = 0.45                     # restitution of the iron
                    state[2] -= (1 + e) * vn * nx
                    state[3] -= (1 + e) * vn * ny
                state[0] = rx + nx * P.R * 1.001
                state[1] = P.hoop_height + ny * P.R * 1.001

        # ---- rim plane crossing (ball falling through hoop height)
        if not made and prev[1] >= P.hoop_height > state[1]:
            frac = (P.hoop_height - prev[1]) / (state[1] - prev[1] + 1e-12)
            x_at = prev[0] + frac * (state[0] - prev[0])
            if x_cross is None:
                x_cross = x_at
            if abs(x_at - hoop_x) < (P.hoop_radius - P.R):
                made = True
                entry_angle = np.degrees(np.arctan2(-state[3], abs(state[2])))
                state[2] *= 0.12          # the net kills the forward speed

        # ---- backboard collision
        if (collide and not made and board_bot <= state[1] <= board_top
                and state[0] + P.R >= P.backboard_x and state[2] > 0):
            hit_backboard = True
            state[2] = -0.55 * state[2]
            state[0] = P.backboard_x - P.R - 0.005

        xs.append(state[0]); ys.append(state[1]); ts.append(t)

        # ---- the ball has dropped clear of the net
        if made and state[1] < P.hoop_height - 0.75:
            break

        # ---- ground
        if state[1] <= P.R:
            hit_ground = True
            break

        # ---- ball has clearly left the scene
        if state[0] > P.backboard_x + 0.8 and state[3] < 0:
            break

    if clears_defender is None:          # never reached the defender
        clears_defender = False
        defender_clearance = -defender_reach

    xs = np.array(xs); ys = np.array(ys)
    blocked = not clears_defender

    if blocked:
        status = "BLOCKED"
        symbol = "\u2718"
    elif made:
        if hit_rim:
            status = "SCORE \u2014 rim in"
        elif hit_backboard:
            status = "SCORE \u2014 bank"
        else:
            status = "SCORE \u2014 swish"
        symbol = "\u2714"
    elif hit_rim:
        status = "MISS \u2014 rim out"
        symbol = "\u2718"
    elif hit_backboard:
        status = "MISS \u2014 off the board"
        symbol = "\u2718"
    else:
        status = "MISS \u2014 short / long"
        symbol = "\u2718"

    return {
        't': np.array(ts), 'x': xs, 'y': ys,
        'v0': v0, 'theta': theta_deg, 'wind': wind_x,
        'made': made, 'clears_defender': clears_defender,
        'defender_clearance': defender_clearance,
        'hit_backboard': hit_backboard, 'hit_rim': hit_rim,
        'hit_ground': hit_ground,
        'success': made and clears_defender,
        'status': status, 'symbol': symbol,
        'apex': float(ys.max()),
        'x_cross': x_cross,
        'aim_error': None if x_cross is None else x_cross - hoop_x,
        'flight_time': float(ts[-1]),
        'entry_angle': entry_angle,
    }


# ============================================================
# 4. STICK FIGURE (redrawn live when heights change)
# ============================================================

class StickFigure:
    """A stick figure whose height can be updated on the fly."""

    def __init__(self, ax, x, height, color, arms_up=False, label=''):
        self.ax = ax
        self.color = color
        self.arms_up = arms_up

        self.head = Circle((0, 0), 0.01, color=color, zorder=9)
        ax.add_patch(self.head)
        self.body,  = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.leg_l, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.leg_r, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.arm_l, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.arm_r, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.name = label
        self.label = ax.text(0, 0, label, ha='center', va='bottom',
                             fontsize=9, fontweight='bold', color=color,
                             linespacing=1.3, zorder=9)
        self.update(x, height)

    def update(self, x, height, top=None):
        """
        height : standing height (top of the head)
        top    : height the raised hands reach (defender reach / release
                 point). Defaults to a little above the head.
        """
        h = height
        top = (h + 0.40) if top is None else top
        head_r = 0.105
        head_y = h - head_r
        shoulder = h - 2 * head_r
        hip = h * 0.48

        self.head.center = (x, head_y)
        self.head.set_radius(head_r)
        self.body.set_data([x, x], [shoulder, hip])
        self.leg_l.set_data([x, x - 0.16], [hip, 0])
        self.leg_r.set_data([x, x + 0.16], [hip, 0])

        if self.arms_up:                       # contesting the shot
            self.arm_l.set_data([x, x - 0.15], [shoulder, top])
            self.arm_r.set_data([x, x + 0.15], [shoulder, top])
        else:                                  # shooting motion
            self.arm_l.set_data([x, x + 0.04], [shoulder, top - 0.10])
            self.arm_r.set_data([x, x + 0.17], [shoulder, top])

        # height read-out travels with the figure, above the hands
        self.label.set_position((x, top + 0.14))
        self.label.set_text(f"{self.name}\n{h:.2f} m")


# ============================================================
# 5. INTERACTIVE ANIMATED SIMULATOR
# ============================================================

class BasketballSimulator:

    ARM_REACH = 0.45        # release point above the shooter's head
    REACH_RATIO = 1.32      # defender reach / standing height

    def __init__(self, show=True):
        self.P = Params()

        # ---- adjustable state
        self.v0 = 6.85
        self.theta = 52.0
        self.wind_x = 0.0
        self.shooter_h = 1.85
        self.defender_x = 2.8
        self.defender_reach = 2.55

        self.result = self._run()

        # ---- figure
        self.fig = plt.figure(figsize=(15.0, 9.0))
        self.fig.patch.set_facecolor('#f7f7fa')
        try:
            self.fig.canvas.manager.set_window_title(
                "Basketball Shot Simulator  —  Silverberg et al. Extension")
        except Exception:
            pass

        self.fig.text(0.5, 0.970,
                      "Defender-Aware Basketball Shot — Animated Simulation",
                      ha='center', va='top', fontsize=16, fontweight='bold',
                      color='#111827')
        self.fig.text(0.5, 0.940,
                      "RK4 projectile integration with aerodynamic drag, "
                      "wind and a defender reach constraint",
                      ha='center', va='top', fontsize=10, color='#4b5563')

        # Court on the left, read-out panel on the right: nothing overlaps.
        self.ax = self.fig.add_axes([0.045, 0.265, 0.515, 0.655])
        self.ax_info = self.fig.add_axes([0.580, 0.265, 0.195, 0.655])
        self.ax_model = self.fig.add_axes([0.795, 0.265, 0.185, 0.655])

        self._setup_court()
        self._setup_info_panel()
        self._setup_model_panel()

        # ---- animated artists
        self.ball = Circle((self.P.release_x, self.release_height), self.P.R,
                           facecolor='#f97316', edgecolor='#7c2d12',
                           lw=1.8, zorder=12)
        self.ax.add_patch(self.ball)
        self.ball_seam, = self.ax.plot([], [], color='#7c2d12', lw=1.2,
                                       zorder=13)
        self.trail, = self.ax.plot([], [], color='#ef4444', lw=2.6,
                                   alpha=0.95, zorder=7)
        self.ghost, = self.ax.plot(self.result['x'], self.result['y'],
                                   color='#94a3b8', lw=1.2, ls=(0, (5, 4)),
                                   alpha=0.85, zorder=6)

        self._create_sliders()
        self._create_buttons()

        self.anim = None
        self.frame = 0
        self._refresh_static()
        self.start_animation()

        if show:
            plt.show()

    # --------------------------------------------------------
    @property
    def release_height(self):
        return self.shooter_h + self.ARM_REACH

    def _run(self):
        return simulate(self.v0, self.theta, self.wind_x, self.defender_x,
                        self.defender_reach, self.release_height, self.P)

    # --------------------------------------------------------
    def _setup_court(self):
        ax, P = self.ax, self.P
        ax.set_facecolor('#ffffff')
        ax.set_xlim(-0.95, P.backboard_x + 1.05)
        ax.set_ylim(0, 5.4)
        ax.set_xlabel('Horizontal distance from shooter (m)',
                      fontsize=11, labelpad=8)
        ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.25, ls=':', zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

        # ---- floor
        ax.axhspan(-0.2, 0, color='#c8a27a', zorder=1)
        ax.axhline(0, color='#7c4a1e', lw=3, zorder=2)

        hoop_x = P.hoop_x
        h = P.hoop_height

        # ---- support pole and arm
        pole_x = P.backboard_x + 0.55
        ax.plot([pole_x, pole_x], [0, h + 0.55],
                color='#64748b', lw=7, solid_capstyle='round', zorder=3)
        ax.plot([P.backboard_x, pole_x], [h + 0.55, h + 0.55],
                color='#64748b', lw=6, solid_capstyle='round', zorder=3)

        # ---- backboard (panel + shooter's square)
        ax.add_patch(Rectangle((P.backboard_x, h - 0.15), 0.06, 1.05,
                               facecolor='#dbeafe', edgecolor='#1e3a8a',
                               lw=2.5, zorder=4))
        ax.plot([P.backboard_x + 0.005, P.backboard_x + 0.005],
                [h + 0.05, h + 0.50], color='#1e3a8a', lw=3, zorder=5)

        # ---- rim: an open ellipse gives a visible "hole"
        ax.plot([P.backboard_x, hoop_x + P.hoop_radius], [h, h],
                color='#9a3412', lw=3, zorder=5)                 # rim mount
        ax.add_patch(Ellipse((hoop_x, h), 2 * P.hoop_radius, 0.11,
                             facecolor='none', edgecolor='#ea580c',
                             lw=4, zorder=8))
        ax.add_patch(Ellipse((hoop_x, h), 2 * P.hoop_radius, 0.11,
                             facecolor='#ffffff', edgecolor='none',
                             alpha=0.9, zorder=6))               # the hole

        # ---- net
        net_bottom = h - 0.38
        for f in np.linspace(-1, 1, 9):
            x_top = hoop_x + f * P.hoop_radius
            x_bot = hoop_x + f * P.hoop_radius * 0.55
            ax.plot([x_top, x_bot], [h - 0.02, net_bottom],
                    color='#9ca3af', lw=1.0, zorder=5)
        for frac in (0.33, 0.66, 1.0):
            y = h - 0.02 - frac * 0.36
            w = P.hoop_radius * (1 - 0.45 * frac)
            ax.plot([hoop_x - w, hoop_x + w], [y, y],
                    color='#9ca3af', lw=1.0, zorder=5)

        # ---- defender reach zone
        self.defender_patch = Rectangle(
            (self.defender_x - 0.22, 0), 0.44, self.defender_reach,
            facecolor='#ef4444', alpha=0.13, edgecolor='none', zorder=3)
        ax.add_patch(self.defender_patch)
        self.defender_line, = ax.plot(
            [self.defender_x - 0.34, self.defender_x + 0.34],
            [self.defender_reach] * 2,
            color='#b91c1c', lw=2, ls=(0, (4, 3)), zorder=6)
        self.reach_tag = ax.text(
            self.defender_x + 0.40, self.defender_reach, '', fontsize=9,
            color='#b91c1c', va='center', ha='left', zorder=6)

        # ---- people (labels sit above the heads, clear of the tick labels)
        self.shooter_fig = StickFigure(ax, 0.0, self.shooter_h, '#1f2937',
                                       arms_up=False, label='Shooter')
        self.shooter_fig.update(0.0, self.shooter_h, self.release_height)
        self.defender_fig = StickFigure(
            ax, self.defender_x, self.defender_reach / self.REACH_RATIO,
            '#b91c1c', arms_up=True, label='Defender')
        self.defender_fig.update(self.defender_x,
                                 self.defender_reach / self.REACH_RATIO,
                                 self.defender_reach)

    # --------------------------------------------------------
    def _setup_info_panel(self):
        ax = self.ax_info
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_facecolor('#ffffff')
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

        self.banner = ax.text(0.5, 0.965, '', ha='center', va='top',
                              fontsize=13, fontweight='bold', color='white',
                              bbox=dict(boxstyle='round,pad=0.45',
                                        facecolor='#16a34a', edgecolor='none'))
        ax.text(0.07, 0.855, 'SHOT INPUTS', fontsize=10, fontweight='bold',
                color='#6b7280')
        self.txt_inputs = ax.text(0.07, 0.815, '', fontsize=10.5, va='top',
                                  family='monospace', color='#111827',
                                  linespacing=1.6)
        ax.text(0.07, 0.500, 'FLIGHT DIAGNOSTICS', fontsize=10,
                fontweight='bold', color='#6b7280')
        self.txt_stats = ax.text(0.07, 0.460, '', fontsize=10.5, va='top',
                                 family='monospace', color='#111827',
                                 linespacing=1.6)
        self.txt_note = ax.text(0.07, 0.04, '', fontsize=9, va='bottom',
                                color='#4b5563', linespacing=1.5)

    # --------------------------------------------------------
    def _setup_model_panel(self):
        ax = self.ax_model
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)
        ax.set_facecolor('#ffffff')
        for sp in ax.spines.values():
            sp.set_color('#cbd5e1')

        ax.text(0.5, 0.965, 'MODEL', ha='center', va='top', fontsize=10,
                fontweight='bold', color='#6b7280')
        ax.text(0.5, 0.905,
                r'$m\frac{d\vec{v}}{dt}=-mg\hat{\jmath}'
                r'-\frac{1}{2}\rho C_d A |\vec{v}_r|\vec{v}_r$',
                ha='center', va='top', fontsize=11, color='#111827')
        ax.text(0.5, 0.815, r'$\vec{v}_r=\vec{v}-\vec{v}_{wind}$',
                ha='center', va='top', fontsize=10, color='#111827')

        P = self.P
        ax.text(0.07, 0.735, 'CONSTANTS', fontsize=10, fontweight='bold',
                color='#6b7280')
        ax.text(0.07, 0.695,
                f"m   = {P.m:.3f} kg\n"
                f"R   = {P.R:.4f} m\n"
                f"Cd  = {P.Cd:.2f}\n"
                f"rho = {P.rho:.3f} kg/m3\n"
                f"g   = {P.g:.2f} m/s2\n"
                f"rim = {P.hoop_height:.3f} m\n"
                f"d   = {P.backboard_x:.3f} m",
                fontsize=10, va='top', family='monospace', color='#111827',
                linespacing=1.6)

        ax.text(0.07, 0.385, 'LEGEND', fontsize=10, fontweight='bold',
                color='#6b7280')
        items = [('#ef4444', 'solid',  'flown path'),
                 ('#94a3b8', 'dashed', 'full predicted path'),
                 ('#b91c1c', 'dashed', 'defender reach line'),
                 ('#f97316', 'solid',  'ball (size 7)')]
        y = 0.320
        for colour, style, text in items:
            ls = '-' if style == 'solid' else (0, (4, 3))
            ax.plot([0.09, 0.24], [y, y], color=colour, lw=2.6, ls=ls,
                    transform=ax.transAxes, clip_on=False)
            ax.text(0.28, y, text, fontsize=9, va='center', color='#374151')
            y -= 0.055

        ax.text(0.07, 0.020,
                textwrap.fill("Success = the ball clears the reach line and "
                              "drops through the rim.", 32),
                fontsize=9, va='bottom', color='#4b5563', linespacing=1.5)

    # --------------------------------------------------------
    def _create_sliders(self):
        sl_kw = dict(track_color='#e5e7eb')
        L, W, H = 0.115, 0.250, 0.022
        R = 0.620

        ax_v = self.fig.add_axes([L, 0.160, W, H])
        self.sl_v = Slider(ax_v, 'Release speed\n(m/s)', 5.0, 12.0,
                           valinit=self.v0, valstep=0.05, color='#f97316',
                           valfmt='%.2f', **sl_kw)

        ax_a = self.fig.add_axes([L, 0.116, W, H])
        self.sl_a = Slider(ax_a, 'Launch angle\n(deg)', 25, 75,
                           valinit=self.theta, valstep=0.5, color='#f97316',
                           valfmt='%.1f', **sl_kw)

        ax_w = self.fig.add_axes([L, 0.072, W, H])
        self.sl_w = Slider(ax_w, 'Wind\n(m/s)', -6, 6, valinit=self.wind_x,
                           valstep=0.2, color='#0ea5e9', valfmt='%+.1f',
                           **sl_kw)

        ax_sh = self.fig.add_axes([L, 0.028, W, H])
        self.sl_sh = Slider(ax_sh, 'Shooter height\n(m)', 1.50, 2.20,
                            valinit=self.shooter_h, valstep=0.01,
                            color='#334155', valfmt='%.2f', **sl_kw)

        ax_dx = self.fig.add_axes([R, 0.160, W, H])
        self.sl_dx = Slider(ax_dx, 'Defender X\n(m)', 0.8, 4.0,
                            valinit=self.defender_x, valstep=0.1,
                            color='#dc2626', valfmt='%.1f', **sl_kw)

        ax_dh = self.fig.add_axes([R, 0.116, W, H])
        self.sl_dh = Slider(ax_dh, 'Defender reach\n(m)', 1.80, 3.20,
                            valinit=self.defender_reach, valstep=0.05,
                            color='#dc2626', valfmt='%.2f', **sl_kw)

        for s in (self.sl_v, self.sl_a, self.sl_w, self.sl_sh,
                  self.sl_dx, self.sl_dh):
            s.label.set_fontsize(9.5)
            s.label.set_ha('right')
            s.valtext.set_fontsize(10)
            s.valtext.set_fontweight('bold')
            s.on_changed(self._on_change)

    # --------------------------------------------------------
    def _create_buttons(self):
        R0 = 0.600
        self.btn_replay = Button(self.fig.add_axes([R0, 0.045, 0.105, 0.042]),
                                 '\u25b6  Replay', color='#22c55e',
                                 hovercolor='#16a34a')
        self.btn_replay.on_clicked(lambda e: self.start_animation())

        self.btn_opt = Button(self.fig.add_axes([R0 + 0.120, 0.045,
                                                 0.140, 0.042]),
                              '\u26a1 Auto-Optimize', color='#3b82f6',
                              hovercolor='#2563eb')
        self.btn_opt.on_clicked(lambda e: self._auto_optimize())

        self.btn_reset = Button(self.fig.add_axes([R0 + 0.275, 0.045,
                                                   0.100, 0.042]),
                                '\u21ba  Reset', color='#e5e7eb',
                                hovercolor='#cbd5e1')
        self.btn_reset.on_clicked(lambda e: self._reset())

        for b in (self.btn_replay, self.btn_opt, self.btn_reset):
            b.label.set_fontsize(11)
            b.label.set_fontweight('bold')
        self.btn_replay.label.set_color('white')
        self.btn_opt.label.set_color('white')

    # --------------------------------------------------------
    def _on_change(self, _val):
        self.v0 = self.sl_v.val
        self.theta = self.sl_a.val
        self.wind_x = self.sl_w.val
        self.shooter_h = self.sl_sh.val
        self.defender_x = self.sl_dx.val
        self.defender_reach = self.sl_dh.val
        self._resimulate()

    def _reset(self):
        self.sl_w.reset(); self.sl_sh.reset(); self.sl_dx.reset()
        self.sl_dh.reset(); self.sl_a.reset(); self.sl_v.reset()

    # --------------------------------------------------------
    def _refresh_static(self):
        """Redraw everything that depends on the slider values."""
        # defender geometry
        self.defender_patch.set_x(self.defender_x - 0.22)
        self.defender_patch.set_height(self.defender_reach)
        self.defender_line.set_data(
            [self.defender_x - 0.34, self.defender_x + 0.34],
            [self.defender_reach] * 2)
        self.reach_tag.set_position((self.defender_x + 0.40,
                                     self.defender_reach))
        self.reach_tag.set_text(f"reach {self.defender_reach:.2f} m")
        self.defender_fig.update(self.defender_x,
                                 self.defender_reach / self.REACH_RATIO,
                                 self.defender_reach)

        # shooter geometry (release point follows the shooter's height)
        self.shooter_fig.update(0.0, self.shooter_h, self.release_height)

        self.ghost.set_data(self.result['x'], self.result['y'])
        self._update_panel()

    # --------------------------------------------------------
    def _update_panel(self):
        r = self.result
        ok = r['success']
        self.banner.set_text(f"{r['symbol']}  {r['status']}")
        self.banner.get_bbox_patch().set_facecolor(
            '#16a34a' if ok else '#dc2626')

        self.txt_inputs.set_text(
            f"angle        {self.theta:6.1f} deg\n"
            f"speed        {self.v0:6.2f} m/s\n"
            f"wind         {self.wind_x:+6.1f} m/s\n"
            f"shooter h    {self.shooter_h:6.2f} m\n"
            f"release h    {self.release_height:6.2f} m\n"
            f"defender X   {self.defender_x:6.1f} m\n"
            f"defender rch {self.defender_reach:6.2f} m")

        clr = r['defender_clearance']
        entry = ('  n/a' if r['entry_angle'] is None
                 else f"{r['entry_angle']:6.1f} deg")
        self.txt_stats.set_text(
            f"apex         {r['apex']:6.2f} m\n"
            f"flight time  {r['flight_time']:6.2f} s\n"
            f"entry angle {entry}\n"
            f"clearance    {clr:+6.2f} m\n"
            f"over defender {'YES' if r['clears_defender'] else 'NO'}")

        if not r['clears_defender']:
            note = ("The ball passes the defender below the reach line — "
                    "raise the launch angle or the release speed.")
        elif not r['made']:
            note = ("Clears the defender but misses the rim. Try "
                    "Auto-Optimize for the softest scoring shot.")
        else:
            note = ("Scoring shot. Entry angles near 45-50 deg give the "
                    "largest margin for error (Silverberg et al., 2003).")
        self.txt_note.set_text(textwrap.fill(note, 30))

    # --------------------------------------------------------
    def _resimulate(self):
        self.result = self._run()
        self._refresh_static()
        self.start_animation()

    # --------------------------------------------------------
    def _aim_error(self, v, th):
        """
        Signed horizontal aim error at the rim plane (metres).
        Negative = the ball falls short of the rim centre.
        Collisions are switched off so the function stays smooth.
        """
        r = simulate(v, th, self.wind_x, self.defender_x, self.defender_reach,
                     self.release_height, self.P, dt=0.005, collide=False)
        if r['aim_error'] is not None:
            return r['aim_error']
        return -3.0 if r['apex'] < self.P.hoop_height else 3.0

    def _solve_speed(self, th, target, lo=5.0, hi=12.0, iters=22):
        """Bisect for the release speed whose aim error equals `target`."""
        f_lo = self._aim_error(lo, th) - target
        f_hi = self._aim_error(hi, th) - target
        if f_lo * f_hi > 0:
            return None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if (self._aim_error(mid, th) - target) * f_lo > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _auto_optimize(self):
        """
        Find the most forgiving shot.

        For each launch angle the release speed that sends the ball through
        the centre of the rim is found by bisection. That nominal shot is
        then disturbed by realistic execution errors (+/- 0.2 m/s in speed,
        +/- 2 deg in angle) and the fraction of disturbed shots that still
        drop through the hoop is used as a robustness score. The angle with
        the highest score wins; ties go to the softer shot. This is the
        margin-for-error criterion of Silverberg, Tran & Adcock (2003).
        """
        self.txt_note.set_text(textwrap.fill(
            "Solving the aim equation and scoring the error margin for "
            "every launch angle...", 32))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        tol = self.P.hoop_radius - self.P.R        # usable half-opening
        dv_grid = (-0.2, -0.1, 0.0, 0.1, 0.2)
        dth_grid = (-2.0, 0.0, 2.0)
        best = None                                # (score, -v, angle, speed)

        for th in np.arange(32, 70.1, 2.0):
            v_mid = self._solve_speed(th, 0.0, iters=16)
            if v_mid is None:
                continue
            r = simulate(v_mid, th, self.wind_x, self.defender_x,
                         self.defender_reach, self.release_height, self.P)
            if not r['success']:
                continue                            # blocked or unusable

            hits = 0
            for dv in dv_grid:
                for dth in dth_grid:
                    if abs(self._aim_error(v_mid + dv, th + dth)) < tol:
                        hits += 1
            score = hits / (len(dv_grid) * len(dth_grid))
            key = (score, -v_mid)
            if best is None or key > best[0]:
                best = (key, th, v_mid, score, r['entry_angle'])

        if best is None:
            self.txt_note.set_text(textwrap.fill(
                "No scoring shot exists for this defender and wind. Move the "
                "defender back or lower the reach.", 32))
            self.fig.canvas.draw_idle()
            return

        _, th, v_mid, score, entry = best
        self.sl_a.set_val(round(th, 1))
        self.sl_v.set_val(round(v_mid / 0.05) * 0.05)   # triggers _on_change
        self.txt_note.set_text(textwrap.fill(
            f"Best margin at {th:.0f} deg, {v_mid:.2f} m/s: {score * 100:.0f}% "
            f"of shots still score with +/-0.2 m/s and +/-2 deg of execution "
            f"error. Entry angle {entry:.0f} deg.", 32))
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    def start_animation(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        n = len(self.result['x'])
        self.step = max(1, n // 160)
        self.n_frames = len(range(0, n, self.step)) + 12   # hold at the end

        self.anim = animation.FuncAnimation(
            self.fig, self._update, frames=self.n_frames,
            interval=22, blit=False, repeat=False)
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    def _update(self, i):
        n = len(self.result['x'])
        idx = min(i * self.step, n - 1)
        bx = self.result['x'][idx]
        by = self.result['y'][idx]

        self.ball.center = (bx, by)
        # spinning seam: an arc across the ball whose curvature cycles
        u = np.linspace(-1, 1, 25)
        bulge = np.cos(idx * 0.12)
        self.ball_seam.set_data(bx + self.P.R * 0.92 * u,
                                by + self.P.R * 0.75 * bulge *
                                np.sqrt(np.clip(1 - u ** 2, 0, 1)))
        self.trail.set_data(self.result['x'][:idx + 1],
                            self.result['y'][:idx + 1])
        return self.ball, self.trail, self.ball_seam


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print(" Defender-Aware Basketball Shot Simulator  (v2.0)")
    print("=" * 64)
    print(" Sliders : release speed, launch angle, wind,")
    print("           shooter height, defender position, defender reach")
    print(" Replay  : re-run the current shot animation")
    print(" Optimize: softest shot that scores and clears the defender")
    print(" Reset   : restore the default configuration")
    print("=" * 64)
    BasketballSimulator()