import numpy as np
import matplotlib.pyplot as plt

class ScopeViewer:
    def __init__(self):
        plt.ion()  # interactive mode for live updates
        self.fig, self.ax = plt.subplots()
        self.traces = {}  # channel -> Line2D
        self.time_base = None
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid()
        self.fig.tight_layout()
        self.fig.suptitle("Scope Viewer")
        self.ax.legend()

    def set_time_base(self, time: np.ndarray):
        """Sets or updates the shared time axis."""
        self.time_base = time
        self.ax.set_xlim(time[0], time[-1])

    def update_trace(self, channel: int, voltage: np.ndarray):
        """
        Update or add a trace for a given channel.
        Assumes `self.time_base` has been set.
        """
        if self.time_base is None:
            raise ValueError("Time base must be set before adding traces.")

        if channel not in self.traces:
            (line,) = self.ax.plot(self.time_base, voltage, label=f"CH{channel}")
            self.traces[channel] = line
            self.ax.legend(loc="upper right")
        else:
            line = self.traces[channel]
            line.set_ydata(voltage)

        self._rescale_y()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _rescale_y(self):
        """Auto-scales Y-axis to fit all visible traces."""
        if not self.traces:
            return
        all_y = np.concatenate([line.get_ydata() for line in self.traces.values()])
        ymin, ymax = np.min(all_y), np.max(all_y)
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        margin = 0.05 * (ymax - ymin)
        self.ax.set_ylim(ymin - margin, ymax + margin)
