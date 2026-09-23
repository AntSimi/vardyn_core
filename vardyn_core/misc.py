import re

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class VideoAnimation(FuncAnimation):
    def _repr_html_(self, *args, **kwargs):
        """To get video in html and have a player."""
        content = self.to_html5_video()
        return re.sub(r'width="[0-9]*"\sheight="[0-9]*"', 'width="100%" height="100%"', content)

    def save(self, *args, **kwargs):
        if args[0].endswith("gif"):
            # In this case gif is used to create thumbnail which is not used but consume same time than video
            # So we create an empty file, to save time
            with open(args[0], "w") as _:
                pass
            return
        return super().save(*args, **kwargs)


def anim_ax(domain, dpi=80, subplots=(1, 1), **kwargs):
    if "figsize" not in kwargs:
        kwargs["figsize"] = (16, 9)
    fig = plt.figure(dpi=dpi, **kwargs)
    nb_x, nb_y = subplots
    axes = list()
    for i in range(nb_x):
        for j in range(nb_y):
            ax = fig.add_axes([i / nb_x, j / nb_y, 1 / nb_x, 1 / nb_y], aspect="equal")
            ax.set_xlim(domain.lon_min, domain.lon_max)
            ax.set_ylim(domain.lat_min, domain.lat_max)

            ax.tick_params(axis="x", which="major", pad=-15, zorder=10)
            ax.tick_params(axis="y", which="major", pad=-25, zorder=10)
            axes.append(ax)
    return fig, axes, axes[0].text(domain.lon_min + 1, domain.lat_max - 1, "")


def shift(x):
    diff = (x[1:] - x[:-1]) / 2
    return np.concatenate(([x[0] - diff[1]], x[1:] - diff, [x[-1] + diff[-1]]))


def grid_animation(ax, ds, field, cax=[0.05, 0.1, 0.9, 0.01], label="", hist=False, factor=1000, **kwargs):
    data = ds[field]
    lons, lats, times = ds["lon"], ds["lat"], ds["time"]
    if "cmap" not in kwargs:
        kwargs["cmap"] = "Spectral_r"
    pcolormesh = ax.pcolorfast(shift(lons), shift(lats), data[0], zorder=-1, **kwargs)
    ax.grid(True)
    if cax is not None:
        cb = plt.colorbar(pcolormesh, cax=ax.figure.add_axes(cax), orientation="horizontal")
        cb.set_label(label)
    if hist:
        cax_hist = [cax[0], cax[1] + cax[3], cax[2], 0.1]
        ax_hist = ax.figure.add_axes(cax_hist, xticks=[], yscale="log", frameon=False)
        ax_hist.set_xlim(cb.vmin, cb.vmax)
        vmax = data[0].size / 20
        ax_hist.set_ylim(10, vmax - vmax % 1e3)
        ax_hist.grid(ls="--")
        bins = np.linspace(cb.vmin, cb.vmax, 201)
        hist_line = ax_hist.plot([], [], "k")[0]

    def update_(i_frame):
        artists = list()
        if isinstance(i_frame, np.datetime64):
            i_frame = abs(times - i_frame).argmin()
        data_ = data[i_frame] * factor
        pcolormesh.set_array(data_)
        artists.append(pcolormesh)
        if hist:
            y, x = np.histogram(data_[~data_.mask] if hasattr(data_, "mask") else data_, bins=bins)
            x = (x[:-1] + x[1:]) / 2
            y[-1] += (data_ >= bins[-1]).sum()
            y[0] += (data_ <= bins[0]).sum()
            hist_line.set_data(x, y)
            artists.append(hist_line)
        return artists

    return update_


def animation(
    ds, field, state, label="Heigt (mm)", date_label="%Y/%m/%d", factor=1000, dpi=80, others=None, return_args=False, hist=False, **kwargs
):
    data = ds[field]
    lons, lats, times = ds["lon"], ds["lat"], ds["time"].values
    fig, axes, txt = anim_ax(state, dpi=dpi, figsize=kwargs.pop("figsize", (16, 9)))
    ax = axes[0]

    if "cmap" not in kwargs:
        kwargs["cmap"] = "Spectral_r"
    pcolormesh = ax.pcolorfast(shift(lons), shift(lats), data[0], zorder=-1, **kwargs)
    ax.grid(True)
    cb = plt.colorbar(pcolormesh, cax=fig.add_axes([0.05, 0.1, 0.9, 0.01]), orientation="horizontal")
    cb.set_label(label)

    if hist:
        ax_hist = fig.add_axes([0.05, 0.11, 0.9, 0.1], xticks=[], yscale="log", frameon=False)
        ax_hist.set_xlim(cb.vmin, cb.vmax)
        vmax = data[0].size / 20
        ax_hist.set_ylim(10, vmax - vmax % 1e3)
        ax_hist.grid(ls="--")
        bins = np.linspace(cb.vmin, cb.vmax, 201)
        hist_line = ax_hist.plot([], [], "k")[0]

    if others is not None:
        updates = list()
        for other in others:
            updates.append(other(ax, times))
        ax.legend(loc="upper right")

    def update_(i_frame):
        artists = list()
        if others is not None:
            for update in updates:
                artists.extend(update(i_frame))
        data_ = data[i_frame] * factor
        pcolormesh.set_array(data_)
        date = times[i_frame]
        txt.set_text(date)  # .strftime(date_label))
        artists.append(pcolormesh), artists.append(txt)
        if hist:
            y, x = np.histogram(data_[~data_.mask] if hasattr(data_, "mask") else data_, bins=bins)
            x = (x[:-1] + x[1:]) / 2
            y[-1] += (data_ >= bins[-1]).sum()
            y[0] += (data_ <= bins[0]).sum()
            hist_line.set_data(x, y)
            artists.append(hist_line)
        return artists

    if return_args:
        return ax, dict(fig=fig, func=update_, frames=np.arange(times.size), interval=125)
    else:
        return VideoAnimation(fig=fig, func=update_, frames=np.arange(times.size), interval=125)
