import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    SCALARS,
    EventAccumulator,
)

log = logging.getLogger(Path(__file__).stem)
TAGS = {"RMSE_whole_object", "RMSE_background", "AEM_VOI"}
TAG_BLACKLIST = {"AEM_VOI_VOI_whole_object"}

LNAME = [
    ("Siemens_Vision600_ZrNEMAIQ", "Vision600_ZrNEMA"),
    ("NeuroLF_Esser_Dataset", "NeuroLF_Esser"),
    ("Mediso_NEMA_IQ_lowcounts", "Mediso_NEMA_lowcounts"),
    ("Siemens_Vision600_Hoffman", "Vision600_Hoffman"),
    ("GE_D690_NEMA_IQ", "D690_NEMA"),
    ("GE_DMI4_NEMA_IQ", "DMI4_NEMA"),
]
LNAME = {v: k for k, v in LNAME}


def scalars(ea: EventAccumulator, tag: str) -> list[tuple[float, float]]:
    """[(value, time), ...]"""
    steps = [s.step for s in ea.Scalars(tag)]
    assert steps == sorted(steps)
    return [(scalar.value, scalar.wall_time) for scalar in ea.Scalars(tag)]


# ------------------------------------------------------------------------------

data_sets = sorted([x.stem for x in list(Path("ALG1").glob("*"))])

for data_set in data_sets:
    print(data_set)
    fig, ax = plt.subplots(2, 6, figsize=(18, 6), layout="constrained")

    for i_alg, alg in enumerate(["ALG1", "ALG2", "ALG3"]):

        tensorboard_logfile = list((Path(f"{alg}") / f"{data_set}").glob("events*"))[0]

        ea = EventAccumulator(str(tensorboard_logfile), size_guidance={SCALARS: 0})
        ea.Reload()

        try:
            start_scalar = ea.Scalars("reset")[0]
        except KeyError:
            start = 0.0
        else:
            assert start_scalar.value == 0
            assert start_scalar.step == -1
            start = start_scalar.wall_time

        tag_names: set[str] = {
            tag for tag in ea.Tags()["scalars"] if any(tag.startswith(i) for i in TAGS)
        }
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
            time = values[:, 1] / 60
            time -= time[0]  # make the first time 0
            value = values[:, 0]
            # plot the values
            if tag.startswith("RMSE"):
                row = 0
                ax[row, col0].loglog(time, value, ".-", label=alg)
                ax[row, col0].set_title(tag, fontsize="medium")

                if i_alg == 0:
                    ax[row, col0].axhline(0.01, color="k")

                col0 += 1
            else:
                row = 1
                ax[row, col1].loglog(time, value, ".-")
                ax[row, col1].set_title(tag, fontsize="medium")

                if i_alg == 0:
                    ax[row, col1].axhline(0.005, color="k")

                col1 += 1

    for axx in ax.ravel():
        axx.grid(ls=":", which="both")  # Show grid for both major and minor ticks
        axx.set_xlim(1 / 6, 60)
        axx.set_ylim(1e-4, 1)
        ax[0, 0].legend(loc="upper right")
        axx.set_xlabel("wall time [min]")

    for i in range(col0, 6):
        ax[0, i].set_axis_off()
    for i in range(col1, 6):
        ax[1, i].set_axis_off()

    fig.suptitle(f"{data_set}", fontsize="large")

    if data_set in LNAME.keys():
        prefix = "test"
    else:
        prefix = "train"

    fig.savefig(f"{prefix}_{data_set}.pdf")
    fig.show()

    # ---------------------------------------------------------------------------

    ## Retrieve the list of image keys
    # image_keys = ea.Tags()["images"]
    # print("Image keys:", image_keys)

    ## Display images for each key
    # for image_key in image_keys:
    #    images = ea.Images(image_key)
    #    for i, img in enumerate(images):
    #        # Decode the image (img.encoded_image_string is a byte string)
    #        image_array = plt.imread(
    #            io.BytesIO(img.encoded_image_string), format="jpeg"
    #        )

    #        print(f"Image key: {image_key}, Step: {img.step}")

    #    ## Display the image
    #    # plt.figure(figsize=(6, 6))
    #    # plt.imshow(image_array)
    #    # plt.axis("off")
    #    # plt.title(f"{image_key} - Step {img.step}")
    #    # plt.show()
