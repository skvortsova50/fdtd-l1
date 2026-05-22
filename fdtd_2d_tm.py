import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FDTD2D_TMz:
    def __init__(self, Nx=120, Ny=120, dx=1e-3, dy=1e-3, Nt=1200):

        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy

        S = 1
        self.dt = S / np.sqrt((1/dx**2) + (1/dy**2))
        self.Nt = Nt

        self.Ez = np.zeros((Nx, Ny))
        self.Hx = np.zeros((Nx, Ny-1))
        self.Hy = np.zeros((Nx-1, Ny))

        self.src_x, self.src_y = Nx//2, Ny//2
        self.t0, self.spread = 50, 25

        self.probe_x, self.probe_y = Nx//2 + 10, Ny//2
        self.signal = []

    def Jz(self, n):
        return np.sin(0.08 * n) * np.exp(-((n - self.t0) / self.spread)**2)

    def update_H(self):
        self.Hx -= (self.dt / self.dy) * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy += (self.dt / self.dx) * (self.Ez[1:, :] - self.Ez[:-1, :])

    def update_E(self):

        curl_H = (
            (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx
            - (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
        )

        self.Ez[1:-1, 1:-1] += self.dt * curl_H

        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

    def source(self, n):
        self.Ez[self.src_x, self.src_y] += self.dt * self.Jz(n)

    def step(self, n):
        self.update_H()
        self.update_E()
        self.source(n)
        self.signal.append(self.Ez[self.probe_x, self.probe_y])

    def animate(self):

        fig, ax = plt.subplots()

        im = ax.imshow(
            self.Ez.T,
            cmap="turbo",
            origin="lower",
            vmin=-1,
            vmax=1,
            animated=True
        )

        plt.colorbar(im, ax=ax)
        ax.set_title("TMz PEC Resonator (evolution)")

        def update(n):
            self.step(n)

            field = self.Ez.T
            m = np.max(np.abs(field)) + 1e-12

            im.set_array(field / m)

            ax.set_title(f"TMz Resonator | step = {n}")
            return [im]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=self.Nt,
            interval=20,
            blit=True
        )

        plt.show()

    def spectrum(self):

        signal = np.array(self.signal)
        signal = signal - np.mean(signal)

        t = np.arange(len(signal)) * self.dt

        plt.figure()
        plt.plot(t, signal)
        plt.title("Probe signal")
        plt.grid()
        plt.show()

        S = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(S), self.dt)

        plt.figure()
        plt.plot(freq[:len(freq)//2], np.abs(S[:len(S)//2]))
        plt.title("Resonance spectrum")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    sim = FDTD2D_TMz(Nx=120, Ny=120, Nt=1200)
    sim.animate()
    sim.spectrum()
