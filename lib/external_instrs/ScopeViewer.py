import numpy as np
import matplotlib.pyplot as plt

class ScopeViewer:
    def __init__(self):
        plt.ion()  # interactive mode for live updates
        self.fig, self.ax = plt.subplots()
        self.traces = {}  # channel -> Line2D
        self.time_base = None

        # Fixed channel colors (CH1–CH4)
        self.channel_colors = {
            1: "tab:blue",
            2: "tab:orange",
            3: "tab:green",
            4: "tab:red",
        }

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid()
        self.fig.tight_layout()
        self.fig.suptitle("Scope Viewer")
        self.ax.legend()

    def set_time_base(self, time: np.ndarray):
        """Sets or updates the shared time axis, clearing old traces."""
        self.time_base = np.array(time)
        self.ax.set_xlim(time[0], time[-1])

        # Remove old traces tied to the previous time base
        for line in self.traces.values():
            line.remove()
        self.traces.clear()

    def update_trace(self, channel: int, voltage: np.ndarray):
        """
        Update or add a trace for a given channel.
        Assumes `self.time_base` has been set.
        """
        voltage = np.array(voltage)
        if self.time_base is None:
            raise ValueError("Time base must be set before adding traces.")

        # Ensure channel is within the supported range
        if channel not in self.channel_colors:
            raise ValueError(f"Unsupported channel {channel}. Max 4 channels allowed.")

        if channel not in self.traces:
            (line,) = self.ax.plot(
                self.time_base,
                voltage,
                label=f"CH{channel}",
                color=self.channel_colors[channel]
            )
            self.traces[channel] = line
            self.ax.legend(loc="upper right")
        else:
            self.traces[channel].set_ydata(voltage)

        self._rescale_y()

        if self.time_base.size == voltage.size:
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
