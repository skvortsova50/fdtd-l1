import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FDTD2D_TMz:
    """
    2D FDTD для TMz хвиль (Ez, Hx, Hy)
    Без провідності, зі струмовим джерелом Jz
    """

    def __init__(self, Nx=120, Ny=120, dx=1e-3, dy=1e-3, Nt=600):
        # --- Фізичні константи ---
        # self.c0 = 3e8
        # self.mu0 = 4e-7 * np.pi
        # self.eps0 = 1 / (self.mu0 * self.c0**2)

        # --- Параметри сітки ---
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        S = 1
        self.dt = S / (np.sqrt((1/dx**2) + (1/dy**2)))
        self.Nt = Nt

        # --- Поля (Yee-схема) ---
        self.Ez = np.zeros((Nx, Ny))
        self.Hx = np.zeros((Nx, Ny-1))
        self.Hy = np.zeros((Nx-1, Ny))

        # --- Джерело ---
        self.src_x, self.src_y = Nx//2, Ny//2
        self.t0, self.spread = 40, 12

        # --- Датчик ---
        self.probe_x, self.probe_y = Nx//2 + 20, Ny//2
        self.signal = []

    # -------------------------------------------------
    def Jz(self, n):
        """Гаусовий імпульс струму"""
        return np.exp(-((n - self.t0) / self.spread)**2)

    # -------------------------------------------------
    def update_H(self):
        """Оновлення Hx, Hy"""
        self.Hx -= (self.dt / self.dy) * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy += (self.dt / self.dx) * (self.Ez[1:, :] - self.Ez[:-1, :])

    # -------------------------------------------------
    def update_E(self):
        """Оновлення Ez"""
        curl_H = (
            (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx
            - (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
        )
        self.Ez[1:-1, 1:-1] += self.dt  * curl_H

    # -------------------------------------------------
    def source(self, n):
        """Струмове джерело Jz"""
        self.Ez[self.src_x, self.src_y] -= self.dt * self.Jz(n)

    # -------------------------------------------------
    def step(self, n):
        """Один часовий крок"""
        self.update_H()
        self.update_E()
        self.source(n)
        self.signal.append(self.Ez[self.probe_x, self.probe_y])

    # -------------------------------------------------
    def animate(self):
        """Анімація поширення поля"""

        fig, ax = plt.subplots()
        im = ax.imshow(self.Ez.T, cmap="turbo", vmin=-1, vmax=1, origin="lower")
        plt.colorbar(im, ax=ax)
        ax.set_title("2D TMz FDTD: Ez(x,y)")

        def update(n):
            self.step(n)
            im.set_array(15000 * self.Ez.T)
            # , Ez_max = {np.max(self.Ez)}, Ez_min = {np.min(self.Ez)}
            ax.set_title(f"2D TMz FDTD: Ez(x,y), n = {n}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=self.Nt, interval=20)
        plt.show()

    # -------------------------------------------------
    def spectrum(self):
        """Часова форма та спектр"""

        signal = np.array(self.signal)
        t = np.arange(len(signal)) * self.dt

        plt.figure()
        plt.plot(t, signal)
        plt.xlabel("Час (с)")
        plt.ylabel("Ez (датчик)")
        plt.title("Часова форма сигналу")
        plt.grid()
        plt.show()

        S = np.fft.fft(signal - np.mean(signal))
        freq = np.fft.fftfreq(len(S), self.dt)

        plt.figure()
        plt.plot(freq[:len(freq)//2], np.abs(S[:len(S)//2]))
        plt.xlabel("Частота (Гц)")
        plt.ylabel("|S(f)|")
        plt.title("Спектр сигналу")
        plt.grid()
        plt.show()


# =====================================================
# ЗАПУСК ЛАБОРАТОРНОЇ
# =====================================================

if __name__ == "__main__":
    sim = FDTD2D_TMz(Nx=120, Ny=120, Nt=600)
    sim.animate()
    sim.spectrum()
