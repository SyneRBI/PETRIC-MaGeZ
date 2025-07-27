from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter, binary_erosion


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

# ------------------------------------------------------------------------------

data_sets = sorted([x.stem for x in list(Path("../output_grad").glob("*"))])

for i_d, data_set in enumerate(data_sets):
    header_file = Path("../data") / LNAME[data_set] / "PETRIC" / "reference_image.hv"
    img, hdr, voxsize = load_interfile_image(header_file)

    img_sm = gaussian_filter(img, 3)
    sl0 = np.argmax(img_sm.sum(axis=(1, 2)))
    sl1 = np.argmax(img_sm.sum(axis=(0, 2)))
    sl2 = np.argmax(img_sm.sum(axis=(0, 1)))
    vmax = 1.5 * img_sm.max()

    osem_img, _, _ = load_interfile_image(
        Path("../data") / LNAME[data_set] / "OSEM_image.hv"
    )

    # read the data fidelity and prior gradient images for OSEM and ref recon

    df_grad_osem, _, _ = load_interfile_image(
        Path("../output_grad") / data_set / "df_grad_osem.hv"
    )

    prior_grad_osem, _, _ = load_interfile_image(
        Path("../output_grad") / data_set / "prior_grad_osem.hv"
    )

    grad_osem = df_grad_osem + prior_grad_osem

    df_grad_ref, _, _ = load_interfile_image(
        Path("../output_grad") / data_set / "df_grad_ref.hv"
    )

    prior_grad_ref, _, _ = load_interfile_image(
        Path("../output_grad") / data_set / "prior_grad_ref.hv"
    )

    grad_ref = df_grad_ref + prior_grad_ref

    mask = binary_erosion(img > 0.05 * vmax, iterations=3)

    print(data_set)
    print(
        "||df_grad(mask*osem)|| / ||prior_grad(mask*osem)||",
        np.linalg.norm(df_grad_osem * mask) / np.linalg.norm(prior_grad_osem * mask),
    )

    kws = dict(vmin=0, vmax=vmax, cmap="Greys")

    gmax = np.abs(grad_osem * mask).max()
    kwsg = dict(vmin=-gmax, vmax=gmax, cmap="seismic")

    fig, ax = plt.subplots(3, 6, figsize=(6 * 2.5, 3 * 2.5), layout="constrained")

    im00 = ax[0, 0].imshow(osem_img[sl0, :, :], **kws)
    im01 = ax[0, 1].imshow(osem_img[:, sl1, :], **kws)
    im02 = ax[0, 2].imshow(img[sl0, :, :], **kws)
    im03 = ax[0, 3].imshow(img[:, sl1, :], **kws)

    im10 = ax[1, 0].imshow(df_grad_osem[sl0, :, :], **kwsg)
    im11 = ax[1, 1].imshow(df_grad_osem[:, sl1, :], **kwsg)
    im12 = ax[1, 2].imshow(prior_grad_osem[sl0, :, :], **kwsg)
    im13 = ax[1, 3].imshow(prior_grad_osem[:, sl1, :], **kwsg)
    im14 = ax[1, 4].imshow(grad_osem[sl0, :, :], **kwsg)
    im15 = ax[1, 5].imshow(grad_osem[:, sl1, :], **kwsg)

    im20 = ax[2, 0].imshow(df_grad_ref[sl0, :, :], **kwsg)
    im21 = ax[2, 1].imshow(df_grad_ref[:, sl1, :], **kwsg)
    im22 = ax[2, 2].imshow(prior_grad_ref[sl0, :, :], **kwsg)
    im23 = ax[2, 3].imshow(prior_grad_ref[:, sl1, :], **kwsg)
    im24 = ax[2, 4].imshow(grad_ref[sl0, :, :], **kwsg)
    im25 = ax[2, 5].imshow(grad_ref[:, sl1, :], **kwsg)

    ax[0, 0].set_title("OSEM tra", fontsize="medium")
    ax[0, 1].set_title("OSEM cor", fontsize="medium")
    ax[0, 2].set_title("MAP ref. tra", fontsize="medium")
    ax[0, 3].set_title("MAP ref. cor", fontsize="medium")

    ax[1, 0].set_title("OSEM data fid. grad. tra.", fontsize="medium")
    ax[1, 1].set_title("OSEM data fid. grad. cor.", fontsize="medium")
    ax[1, 2].set_title("OSEM prior grad. tra.", fontsize="medium")
    ax[1, 3].set_title("OSEM prior grad. cor.", fontsize="medium")
    ax[1, 4].set_title("OSEM total grad. tra.", fontsize="medium")
    ax[1, 5].set_title("OSEM total grad. cor.", fontsize="medium")

    ax[2, 0].set_title("MAP ref. data fid. grad. tra.", fontsize="medium")
    ax[2, 1].set_title("MAP ref. data fid. grad. cor.", fontsize="medium")
    ax[2, 2].set_title("MAP ref. prior grad. tra.", fontsize="medium")
    ax[2, 3].set_title("MAP ref. prior grad. cor.", fontsize="medium")
    ax[2, 4].set_title("MAP ref. total grad. tra.", fontsize="medium")
    ax[2, 5].set_title("MAP ref. total grad. cor.", fontsize="medium")

    # add colorbars with location bottom
    fig.colorbar(im00, ax=ax[0, 0], orientation="horizontal", fraction=0.04)
    fig.colorbar(im01, ax=ax[0, 1], orientation="horizontal", fraction=0.04)
    fig.colorbar(im02, ax=ax[0, 2], orientation="horizontal", fraction=0.04)
    fig.colorbar(im03, ax=ax[0, 3], orientation="horizontal", fraction=0.04)

    fig.colorbar(im10, ax=ax[1, 0], orientation="horizontal", fraction=0.04)
    fig.colorbar(im11, ax=ax[1, 1], orientation="horizontal", fraction=0.04)
    fig.colorbar(im12, ax=ax[1, 2], orientation="horizontal", fraction=0.04)
    fig.colorbar(im13, ax=ax[1, 3], orientation="horizontal", fraction=0.04)
    fig.colorbar(im14, ax=ax[1, 4], orientation="horizontal", fraction=0.04)
    fig.colorbar(im15, ax=ax[1, 5], orientation="horizontal", fraction=0.04)

    fig.colorbar(im20, ax=ax[2, 0], orientation="horizontal", fraction=0.04)
    fig.colorbar(im21, ax=ax[2, 1], orientation="horizontal", fraction=0.04)
    fig.colorbar(im22, ax=ax[2, 2], orientation="horizontal", fraction=0.04)
    fig.colorbar(im23, ax=ax[2, 3], orientation="horizontal", fraction=0.04)
    fig.colorbar(im24, ax=ax[2, 4], orientation="horizontal", fraction=0.04)
    fig.colorbar(im25, ax=ax[2, 5], orientation="horizontal", fraction=0.04)

    for axx in ax.ravel():
        axx.set_xticks([])
        axx.set_yticks([])

    ax[0, -2].set_axis_off()
    ax[0, -1].set_axis_off()

    fig.savefig(f"gradients_{data_set}.png", dpi=300)
    fig.show()
