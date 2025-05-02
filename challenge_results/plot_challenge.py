import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    SCALARS,
    EventAccumulator,
)

from scipy.ndimage import gaussian_filter


def parse_interfile_header(header_path):
    metadata = {}
    with open(header_path, "r") as f:
        for line in f:
            line = line.strip().lstrip("!")
            if ":=" in line:
                key, value = line.split(":=", 1)
                key = key.strip().lower()
                value = value.strip()
                metadata[key] = value
    return metadata


def load_interfile_image(header_path: Path):
    meta = parse_interfile_header(header_path)

    # Get the binary data file name
    data_path = header_path.parent / meta["name of data file"]

    # Image dimensions (assumes 3D)
    shape = tuple(int(meta[f"matrix size [{i}]"]) for i in (3, 2, 1))  # (z, y, x)
    voxel_size = tuple(
        float(meta[f"scaling factor (mm/pixel) [{i}]"]) for i in (3, 2, 1)
    )  # (z, y, x)

    # Data type
    num_bytes = int(meta["number of bytes per pixel"])
    num_format = meta["number format"]
    byte_order = meta.get("imagedata byte order", "LITTLEENDIAN")

    # Determine numpy dtype
    if num_format == "float" and num_bytes == 4:
        dtype = np.float32
    elif num_format == "float" and num_bytes == 8:
        dtype = np.float64
    elif num_format == "signed integer":
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[num_bytes]
    elif num_format == "unsigned integer":
        dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[num_bytes]
    else:
        raise ValueError(
            f"Unsupported number format: {num_format} with {num_bytes} bytes"
        )

    # Handle endianness
    if byte_order == "LITTLEENDIAN":
        dtype = np.dtype("<" + dtype().dtype.char)
    elif byte_order == "BIGENDIAN":
        dtype = np.dtype(">" + dtype().dtype.char)
    else:
        raise ValueError(f"Unknown byte order: {byte_order}")

    # Read binary data
    image = np.fromfile(data_path, dtype=dtype).reshape(shape)

    return image, meta, voxel_size


log = logging.getLogger(Path(__file__).stem)
TAGS = {"RMSE_whole_object", "RMSE_background", "AEM_VOI"}
TAG_BLACKLIST = {"AEM_VOI_VOI_whole_object"}

LNAME = [
    ("Siemens_Vision600_ZrNEMAIQ", "Vision600_ZrNEMA"),
    ("NeuroLF_Esser_Dataset", "NeuroLF_Esser"),
    ("NeuroLF_Hoffman_Dataset", "NeuroLF_Hoffman"),
    ("Mediso_NEMA_IQ_lowcounts", "Mediso_NEMA_lowcounts"),
    ("Siemens_Vision600_Hoffman", "Vision600_Hoffman"),
    ("Siemens_Vision600_thorax", "Vision600_thorax"),
    ("Siemens_Vision600_ZrNEMAIQ", "Vision600_ZrNEMA"),
    ("GE_D690_NEMA_IQ", "D690_NEMA"),
    ("GE_DMI4_NEMA_IQ", "DMI4_NEMA"),
    ("GE_DMI3_Torso", "DMI3_Torso"),
    ("Mediso_NEMA_IQ", "Mediso_NEMA"),
    ("Siemens_mMR_ACR", "mMR_ACR"),
    ("Siemens_mMR_NEMA_IQ", "mMR_NEMA"),
    ("Siemens_mMR_NEMA_IQ_lowcounts", "mMR_NEMA_lowcounts"),
]
LNAME = {v: k for k, v in LNAME}

TNAME = [
    ("Siemens_Vision600_ZrNEMAIQ", "Vision600_ZrNEMA"),
    ("NeuroLF_Esser_Dataset", "NeuroLF_Esser"),
    ("Mediso_NEMA_IQ_lowcounts", "Mediso_NEMA_lowcounts"),
    ("Siemens_Vision600_Hoffman", "Vision600_Hoffman"),
    ("GE_D690_NEMA_IQ", "D690_NEMA"),
    ("GE_DMI4_NEMA_IQ", "DMI4_NEMA"),
]
TNAME = {v: k for k, v in TNAME}


def scalars(ea: EventAccumulator, tag: str) -> list[tuple[float, float]]:
    """[(value, time), ...]"""
    steps = [s.step for s in ea.Scalars(tag)]
    assert steps == sorted(steps)
    return [(scalar.value, scalar.wall_time) for scalar in ea.Scalars(tag)]


# ------------------------------------------------------------------------------

data_sets = sorted([x.stem for x in list(Path("ALG1l").glob("*"))])

for i_d, data_set in enumerate(data_sets):
    print(data_set)

    header_file = Path("../data") / LNAME[data_set] / "PETRIC" / "reference_image.hv"
    if header_file.exists():
        print(f"Loading {header_file}")
        img, hdr, voxsize = load_interfile_image(header_file)

        img_sm = gaussian_filter(img, 3)
        sl0 = np.argmax(img_sm.sum(axis=(1, 2)))
        sl1 = np.argmax(img_sm.sum(axis=(0, 2)))
        sl2 = np.argmax(img_sm.sum(axis=(0, 1)))

        vmax = 1.5 * img_sm.max()

    # for i_alg, alg in enumerate(["ALG1", "ALG2", "ALG3"]):
    for i_alg, alg in enumerate(["ALG1l"]):

        tensorboard_logfiles = sorted(
            list((Path(f"{alg}") / f"{data_set}").glob("events*"))
        )

        for i_f, tensorboard_logfile in enumerate(tensorboard_logfiles):
            ea = EventAccumulator(str(tensorboard_logfile), size_guidance={SCALARS: 0})
            ea.Reload()

            try:
                start_scalar = ea.Scalars("reset")[0]
            except KeyError:
                print(
                    f"KeyError: reset: not using accurate relative time for {tensorboard_logfile}"
                )
                start = 0.0
            else:
                assert start_scalar.value == 0
                assert start_scalar.step == -1
                start = start_scalar.wall_time

            tag_names: set[str] = {
                tag
                for tag in ea.Tags()["scalars"]
                if any(tag.startswith(i) for i in TAGS)
            }

            # get the length of the tags that start with AEM_VOI and are not in the blacklist
            num_cols = len(
                {
                    tag
                    for tag in tag_names
                    if tag.startswith("AEM_VOI") and tag not in TAG_BLACKLIST
                }
            )

            num_cols = max(num_cols, 4)

            if i_f == 0 and i_alg == 0:
                fig, ax = plt.subplots(
                    2,
                    num_cols,
                    figsize=(num_cols * 3, 6),
                    layout="constrained",
                )

            if skip := TAG_BLACKLIST & tag_names:
                log.warning("skipping tags: %s", skip)
                tag_names -= skip
            tags = {tag: scalars(ea, tag) for tag in tag_names}

            col0 = 0
            col1 = 0

            # loop over the dict tags
            for i, tag in enumerate(sorted(tags.keys())):
                # convert to numpy array
                values = np.array(tags[tag])
                # get the time (min) and value
                # time = (values[:, 1] - (start or values[0, 1])) / 60
                time = (values[:, 1] - values[0, 1]) / 60
                value = values[:, 0]
                # plot the values
                color = plt.cm.tab10(i_alg)
                alpha = 1.0 - (i_f * 0.2)  # Decrease alpha as i_f increases
                alpha = max(alpha, 0.2)  # Ensure alpha doesn't go below 0.2
                if tag.startswith("RMSE"):
                    row = 0

                    label = f"{alg} r{i_f + 1}"

                    ax[row, col0].semilogy(
                        time, value, "-", label=label, color=color, alpha=alpha
                    )
                    ax[row, col0].set_title(tag, fontsize="medium")

                    if i_alg == 0:
                        ax[row, col0].axhline(0.01, color="k")

                    col0 += 1
                else:
                    row = 1
                    ax[row, col1].semilogy(time, value, "-", color=color, alpha=alpha)
                    ax[row, col1].set_title(tag, fontsize="medium")

                    if i_alg == 0:
                        ax[row, col1].axhline(0.005, color="k")

                    col1 += 1

        if header_file.exists():
            kws = dict(cmap="Greys", vmin=0, vmax=vmax)
            im0 = ax[0, 2].imshow(img[sl0, :, :], **kws)
            im1 = ax[0, 3].imshow(img[:, sl1, :], aspect=voxsize[0] / voxsize[2], **kws)

    ax[0, 0].legend(loc="upper right", fontsize="x-small", ncol=3)

    for axx in ax[0, :2]:
        axx.grid(ls=":", which="both")  # Show grid for both major and minor ticks
        axx.set_ylim(5e-4, 1)
        axx.set_xlabel("wall time [min]")

    for axx in ax[1, :]:
        axx.grid(ls=":", which="both")  # Show grid for both major and minor ticks
        axx.set_ylim(5e-4, 1)
        axx.set_xlabel("wall time [min]")

    for i in range(4, num_cols):
        ax[0, i].set_axis_off()
    for i in range(col1, num_cols):
        ax[1, i].set_axis_off()

    for i in [2, 3]:
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

    fig.suptitle(f"{data_set}", fontsize="large")

    if data_set in TNAME.keys():
        prefix = "test"
    else:
        prefix = "train"

    fig.savefig(f"{prefix}_{data_set}.pdf")
    fig.show()
