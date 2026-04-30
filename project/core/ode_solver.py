"""
ode_solver.py — Solvers numéricos para integrar la EDO del sistema.

Resuelve el sistema:
    dx/dt = f_θ(x, y)
    dy/dt = g_θ(x, y)

dado un NeuralField que define f_θ y g_θ, una condición inicial (x0, y0),
y un intervalo de tiempo [t_start, t_end].

Métodos disponibles:
    - Euler explícito:  Simple, didáctico, muestra errores de discretización.
    - RK4 (Runge-Kutta): Preciso y estable, usa scipy.integrate.solve_ivp.
    - Paso individual:  Para animación en tiempo real (un Δt a la vez).

Uso:
    >>> from core.neural_field import NeuralField
    >>> field = NeuralField()  # Círculo por defecto
    >>> solver = ODESolver(field)
    >>> t, xs, ys = solver.solve_rk4(1.0, 0.0, (0, 6.28), dt=0.01)
"""

import numpy as np
from scipy.integrate import solve_ivp


class ODESolver:
    """
    Integrador numérico para el sistema dinámico definido por un NeuralField.

    Encapsula múltiples métodos de integración (Euler, RK4) y proporciona
    tanto resolución completa como paso a paso para animación en tiempo real.

    Attributes:
        field (NeuralField): Campo vectorial que define la EDO.
    """

    def __init__(self, neural_field):
        """
        Args:
            neural_field (NeuralField): Red neuronal que define dx/dt, dy/dt.
        """
        self.field = neural_field

    def solve_euler(self, x0, y0, t_span, dt):
        """
        Resuelve la EDO con el método de Euler explícito.

        Esquema:  x_{n+1} = x_n + Δt · f(x_n, y_n)
                  y_{n+1} = y_n + Δt · g(x_n, y_n)

        Es el método más simple. Muestra claramente el error de
        discretización cuando Δt es grande — útil para enseñar.

        Args:
            x0 (float): Posición inicial en x.
            y0 (float): Posición inicial en y.
            t_span (tuple): Intervalo (t_inicio, t_fin).
            dt (float): Paso de tiempo Δt.

        Returns:
            tuple: (t, xs, ys) — arrays numpy con el tiempo y las coordenadas.
        """
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))
        t = np.linspace(t_start, t_end, n_steps + 1)

        xs = np.zeros(n_steps + 1)
        ys = np.zeros(n_steps + 1)
        xs[0], ys[0] = x0, y0

        for i in range(n_steps):
            dx, dy = self.field.forward(xs[i], ys[i])
            xs[i + 1] = xs[i] + dt * dx
            ys[i + 1] = ys[i] + dt * dy

        return t, xs, ys

    def solve_rk4(self, x0, y0, t_span, dt):
        """
        Resuelve la EDO con Runge-Kutta 4/5 (RK45) vía scipy.

        Método adaptativo de orden 4 con control de error de orden 5.
        Mucho más preciso que Euler para el mismo tamaño de paso.

        Args:
            x0 (float): Posición inicial en x.
            y0 (float): Posición inicial en y.
            t_span (tuple): Intervalo (t_inicio, t_fin).
            dt (float): Paso máximo de tiempo.

        Returns:
            tuple: (t, xs, ys) — arrays numpy con el tiempo y las coordenadas.
        """
        def ode_system(t, state):
            x, y = state
            dx, dy = self.field.forward(x, y)
            return [dx, dy]

        n_points = int(np.ceil((t_span[1] - t_span[0]) / dt)) + 1
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            ode_system,
            t_span,
            [x0, y0],
            method='RK45',
            t_eval=t_eval,
            max_step=dt,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"El solver RK45 falló: {sol.message}")

        return sol.t, sol.y[0], sol.y[1]

    def solve_step(self, x, y, dt):
        """
        Ejecuta un solo paso de Euler (para animación en tiempo real).

        Args:
            x (float): Posición actual en x.
            y (float): Posición actual en y.
            dt (float): Paso de tiempo Δt.

        Returns:
            tuple: (x_new, y_new) — nueva posición tras un paso.
        """
        dx, dy = self.field.forward(x, y)
        return x + dt * dx, y + dt * dy

    def solve(self, x0, y0, t_span, dt, method='rk4'):
        """
        Método genérico que selecciona el solver por nombre.

        Args:
            x0, y0: Condición inicial.
            t_span: Intervalo de tiempo.
            dt: Paso de tiempo.
            method: 'euler' o 'rk4'.

        Returns:
            tuple: (t, xs, ys).
        """
        solvers = {
            'euler': self.solve_euler,
            'rk4': self.solve_rk4,
        }
        solver_fn = solvers.get(method.lower())
        if solver_fn is None:
            raise ValueError(
                f"Método '{method}' no reconocido. "
                f"Opciones: {list(solvers.keys())}"
            )
        return solver_fn(x0, y0, t_span, dt)
