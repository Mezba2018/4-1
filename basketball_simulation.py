"""
================================================================
 DEFENDER-AWARE BASKETBALL SHOT SIMULATION (Animated)
 Extension of Silverberg, Tran & Adcock (2003)
================================================================
 Features:
   - Real-time animated ball trajectory
   - Aerodynamic drag + wind effects
   - Defender with reach constraint
   - Hoop + backboard collision detection
   - Interactive sliders (angle, velocity, wind, defender)
   - Live score / shot result display
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, Rectangle, FancyArrow
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

@dataclass
class Params:
    # Ball
    m: float = 0.624          # kg
    R: float = 0.1194         # m
    Cd: float = 0.47          # drag coefficient
    # Environment
    g: float = 9.81           # m/s^2
    rho: float = 1.225        # kg/m^3
    # Court
    hoop_height: float = 3.048
    hoop_radius: float = 0.2286
    backboard_x: float = 4.191  # free throw distance
    # Shooter
    release_height: float = 2.0


# ============================================================
# 2. PHYSICS ENGINE
# ============================================================

def drag_force(vx, vy, wind_x, P: Params):
    """Quadratic aerodynamic drag with wind."""
    vxr = vx - wind_x
    vyr = vy
    v = np.hypot(vxr, vyr)
    if v < 1e-9:
        return 0.0, 0.0
    A = np.pi * P.R ** 2
    k = 0.5 * P.rho * P.Cd * A
    return -k * v * vxr, -k * v * vyr


def derivatives(state, wind_x, P: Params):
    """State = [x, y, vx, vy]."""
    x, y, vx, vy = state
    Fx, Fy = drag_force(vx, vy, wind_x, P)
    ax = Fx / P.m
    ay = -P.g + Fy / P.m
    return np.array([vx, vy, ax, ay])


def rk4_step(state, dt, wind_x, P: Params):
    k1 = derivatives(state, wind_x, P)
    k2 = derivatives(state + 0.5 * dt * k1, wind_x, P)
    k3 = derivatives(state + 0.5 * dt * k2, wind_x, P)
    k4 = derivatives(state + dt * k3, wind_x, P)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================
# 3. TRAJECTORY SIMULATION
# ============================================================

def simulate(v0, theta_deg, wind_x, defender_x, defender_reach, P: Params):
    """
    Returns dict with full trajectory and outcome flags.
    Simulates until ball hits ground, backboard, or reaches beyond hoop.
    """
    theta = np.radians(theta_deg)
    state = np.array([0.0, P.release_height,
                      v0 * np.cos(theta), v0 * np.sin(theta)])

    dt = 0.002
    t_max = 4.0

    xs, ys, ts = [state[0]], [state[1]], [0.0]
    clears_defender = None
    hit_backboard = False
    hit_ground = False
    made = False
    t = 0.0

    hoop_x = P.backboard_x - P.hoop_radius

    while t < t_max:
        state = rk4_step(state, dt, wind_x, P)
        t += dt
        xs.append(state[0]); ys.append(state[1]); ts.append(t)

        # Check defender clearance
        if clears_defender is None and state[0] >= defender_x:
            clears_defender = state[1] > (defender_reach + P.R)

        # Check backboard collision
        if state[0] + P.R >= P.backboard_x and not hit_backboard:
            hit_backboard = True
            # Reflect x velocity (simple elastic bounce)
            state[2] = -0.7 * state[2]
            state[0] = P.backboard_x - P.R - 0.01

        # Check made shot
        if not made and len(ys) > 1:
            if ys[-2] >= P.hoop_height > ys[-1]:
                frac = (P.hoop_height - ys[-2]) / (ys[-1] - ys[-2] + 1e-12)
                x_at = xs[-2] + frac * (xs[-1] - xs[-2])
                if abs(x_at - hoop_x) < (P.hoop_radius - P.R):
                    made = True

        # Ground
        if state[1] <= 0:
            hit_ground = True
            break

        # Stop if way past hoop
        if state[0] > P.backboard_x + 1.0:
            break

    if clears_defender is None:
        clears_defender = False

    return {
        't': np.array(ts),
        'x': np.array(xs),
        'y': np.array(ys),
        'v0': v0,
        'theta': theta_deg,
        'wind': wind_x,
        'made': made,
        'clears_defender': clears_defender,
        'hit_backboard': hit_backboard,
        'hit_ground': hit_ground,
        'success': made and clears_defender
    }


# ============================================================
# 4. INTERACTIVE ANIMATED SIMULATION
# ============================================================

class BasketballSimulator:
    def __init__(self):
        self.P = Params()

        # State
        self.v0 = 7.5
        self.theta = 52.0
        self.wind_x = 0.0
        self.defender_x = 3.0
        self.defender_reach = 2.5

        # Simulate first
        self.result = simulate(self.v0, self.theta, self.wind_x,
                               self.defender_x, self.defender_reach, self.P)

        # Build figure
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title(
            "Basketball Shot Simulator — Silverberg et al. Extension")

        # Court axes
        self.ax = self.fig.add_axes([0.05, 0.28, 0.90, 0.68])
        self._setup_court()

        # Create animated artists
        self.ball = Circle((0, self.P.release_height), self.P.R,
                           color='orange', ec='darkorange', lw=2, zorder=10)
        self.ax.add_patch(self.ball)
        self.trail, = self.ax.plot([], [], 'r-', lw=2, alpha=0.7, zorder=5)
        self.trail_full, = self.ax.plot(self.result['x'], self.result['y'],
                                        'r--', lw=1, alpha=0.25, zorder=4)

        # Result text
        self.result_text = self.ax.text(
            0.02, 0.97, '', transform=self.ax.transAxes,
            fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # Sliders
        self._create_sliders()

        # Buttons
        self._create_buttons()

        # Animation
        self.anim = None
        self.frame = 0
        self.start_animation()

        plt.show()

    # --------------------------------------------------------
    def _setup_court(self):
        ax = self.ax
        P = self.P
        ax.set_xlim(-0.3, P.backboard_x + 0.7)
        ax.set_ylim(0, 4.5)
        ax.set_xlabel('Horizontal Distance (m)', fontsize=11)
        ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_title('Defender-Aware Basketball Shot — Animated Simulation',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Ground
        ax.axhline(0, color='#5b3a1a', lw=4, zorder=1)

        # Hoop
        hoop_x = P.backboard_x - P.hoop_radius
        ax.plot([hoop_x - P.hoop_radius, hoop_x + P.hoop_radius],
                [P.hoop_height, P.hoop_height],
                color='#ff4500', lw=5, solid_capstyle='round', zorder=6)

        # Backboard
        ax.plot([P.backboard_x, P.backboard_x],
                [P.hoop_height - 0.15, P.hoop_height + 0.75],
                color='#1e3a8a', lw=6, zorder=6)
        ax.plot([P.backboard_x, P.backboard_x - 0.05],
                [P.hoop_height - 0.15, P.hoop_height - 0.15],
                color='#1e3a8a', lw=6, zorder=6)

        # Pole
        ax.plot([P.backboard_x + 0.35, P.backboard_x + 0.35],
                [0, P.hoop_height + 0.75],
                color='gray', lw=4, zorder=2)
        ax.plot([P.backboard_x, P.backboard_x + 0.35],
                [P.hoop_height + 0.75, P.hoop_height + 0.75],
                color='gray', lw=4, zorder=2)

        # Shooter (stick figure)
        self._draw_stick_figure(0, 0, 1.8, color='#222222', label='Shooter')

        # Defender
        self.defender_patch = Rectangle(
            (self.defender_x - 0.20, 0), 0.40, self.defender_reach,
            color='red', alpha=0.25, zorder=3)
        self.ax.add_patch(self.defender_patch)
        self.defender_line, = ax.plot(
            [self.defender_x - 0.20, self.defender_x + 0.20],
            [self.defender_reach, self.defender_reach],
            'r--', lw=2, zorder=4)
        self._draw_stick_figure(self.defender_x, 0, self.defender_reach,
                                color='#b91c1c', label='Defender')

    # --------------------------------------------------------
    def _draw_stick_figure(self, x, y, height, color, label=''):
        ax = self.ax
        h = height
        head_r = h * 0.10
        # Head
        ax.add_patch(Circle((x, y + h - head_r), head_r,
                            color=color, zorder=8))
        # Body
        ax.plot([x, x], [y + h - 2 * head_r, y + h * 0.40],
                color=color, lw=4, zorder=8)
        # Legs
        ax.plot([x, x - 0.10], [y + h * 0.40, y],
                color=color, lw=4, zorder=8)
        ax.plot([x, x + 0.10], [y + h * 0.40, y],
                color=color, lw=4, zorder=8)
        # Arms
        ax.plot([x, x - 0.15], [y + h * 0.80, y + h * 0.60],
                color=color, lw=3, zorder=8)
        ax.plot([x, x + 0.15], [y + h * 0.80, y + h * 0.60],
                color=color, lw=3, zorder=8)
        if label:
            ax.text(x, -0.15, label, ha='center', fontsize=9,
                    fontweight='bold', color=color)

    # --------------------------------------------------------
    def _create_sliders(self):
        # Velocity
        ax_v = self.fig.add_axes([0.10, 0.16, 0.35, 0.025])
        self.sl_v = Slider(ax_v, 'Velocity (m/s)', 5.0, 12.0,
                           valinit=self.v0, valstep=0.1, color='#f97316')

        # Angle
        ax_a = self.fig.add_axes([0.10, 0.12, 0.35, 0.025])
        self.sl_a = Slider(ax_a, 'Angle (deg)', 25, 75,
                           valinit=self.theta, valstep=0.5, color='#f97316')

        # Wind
        ax_w = self.fig.add_axes([0.10, 0.08, 0.35, 0.025])
        self.sl_w = Slider(ax_w, 'Wind (m/s)', -6, 6,
                           valinit=self.wind_x, valstep=0.2, color='#0ea5e9')

        # Defender x
        ax_dx = self.fig.add_axes([0.55, 0.16, 0.35, 0.025])
        self.sl_dx = Slider(ax_dx, 'Defender X (m)', 1.0, 5.0,
                            valinit=self.defender_x, valstep=0.1,
                            color='#dc2626')

        # Defender reach
        ax_dh = self.fig.add_axes([0.55, 0.12, 0.35, 0.025])
        self.sl_dh = Slider(ax_dh, 'Defender Reach (m)', 1.8, 3.2,
                            valinit=self.defender_reach, valstep=0.05,
                            color='#dc2626')

        self.sl_v.on_changed(self._on_change)
        self.sl_a.on_changed(self._on_change)
        self.sl_w.on_changed(self._on_change)
        self.sl_dx.on_changed(self._on_change)
        self.sl_dh.on_changed(self._on_change)

    # --------------------------------------------------------
    def _create_buttons(self):
        ax_btn = self.fig.add_axes([0.55, 0.03, 0.15, 0.045])
        self.btn_replay = Button(ax_btn, '▶ Replay', color='#22c55e',
                                 hovercolor='#16a34a')
        self.btn_replay.on_clicked(lambda e: self.start_animation())

        ax_btn2 = self.fig.add_axes([0.72, 0.03, 0.18, 0.045])
        self.btn_opt = Button(ax_btn2, '⚡ Auto-Optimize', color='#3b82f6',
                              hovercolor='#2563eb')
        self.btn_opt.on_clicked(self._auto_optimize)

    # --------------------------------------------------------
    def _on_change(self, val):
        self.v0 = self.sl_v.val
        self.theta = self.sl_a.val
        self.wind_x = self.sl_w.val
        self.defender_x = self.sl_dx.val
        self.defender_reach = self.sl_dh.val
        self._resimulate()

    # --------------------------------------------------------
    def _resimulate(self):
        self.result = simulate(self.v0, self.theta, self.wind_x,
                               self.defender_x, self.defender_reach, self.P)
        self.trail_full.set_data(self.result['x'], self.result['y'])

        # Update defender visual
        self.defender_patch.set_x(self.defender_x - 0.20)
        self.defender_patch.set_height(self.defender_reach)
        self.defender_line.set_data(
            [self.defender_x - 0.20, self.defender_x + 0.20],
            [self.defender_reach, self.defender_reach])

        self.start_animation()

    # --------------------------------------------------------
    def _auto_optimize(self):
        """Sweep angle & velocity, pick smallest v that scores & clears."""
        best = None
        for th in np.arange(35, 71, 1.0):
            for v in np.arange(5.5, 12.0, 0.1):
                r = simulate(v, th, self.wind_x,
                             self.defender_x, self.defender_reach, self.P)
                if r['success']:
                    if best is None or v < best['v0']:
                        best = r
        if best is not None:
            self.sl_a.set_val(best['theta'])
            self.sl_v.set_val(best['v0'])
            # _on_change will trigger _resimulate

    # --------------------------------------------------------
    def start_animation(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        self.frame = 0
        n = len(self.result['x'])
        # Subsample so each frame looks smooth
        self.step = max(1, n // 200)
        self.n_frames = len(range(0, n, self.step))

        self.anim = animation.FuncAnimation(
            self.fig, self._update, frames=self.n_frames,
            interval=20, blit=False, repeat=False)

    # --------------------------------------------------------
    def _update(self, i):
        idx = min(i * self.step, len(self.result['x']) - 1)
        bx = self.result['x'][idx]
        by = self.result['y'][idx]

        self.ball.center = (bx, by)
        self.trail.set_data(self.result['x'][:idx + 1],
                            self.result['y'][:idx + 1])

        # Result text
        r = self.result
        status = "✓ SCORE!" if r['success'] else \
                 ("✗ Missed" if r['made'] is False else "✗ Blocked")
        if r['hit_backboard']:
            status += "  (backboard)"
        if not r['clears_defender']:
            status += "  — DEFENDER BLOCKED"
        if r['hit_ground'] and not r['made']:
            status += "  — ground"

        self.result_text.set_text(
            f"θ = {self.theta:.1f}°   v = {self.v0:.2f} m/s   "
            f"wind = {self.wind_x:+.1f} m/s\n"
            f"Defender X = {self.defender_x:.1f} m, "
            f"Reach = {self.defender_reach:.2f} m\n"
            f"Clears defender: {'YES' if r['clears_defender'] else 'NO'}\n"
            f"Result: {status}"
        )
        return self.ball, self.trail, self.result_text


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Basketball Shot Simulator — Interactive Animation")
    print("=" * 60)
    print("Controls:")
    print("  • Sliders: adjust velocity, angle, wind, defender")
    print("  • Replay : re-run current shot")
    print("  • Auto-Optimize: find smallest v that scores")
    print("=" * 60)
    BasketballSimulator()