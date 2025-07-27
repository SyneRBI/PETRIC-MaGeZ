import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path


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


if __name__ == "__main__":

    method = "training"

    if method == "training":
        data = [
            "GE_DMI3_Torso",
            "Mediso_NEMA_IQ",
            "NeuroLF_Hoffman_Dataset",
            "Siemens_mMR_ACR",
            "Siemens_mMR_NEMA_IQ",
            "Siemens_mMR_NEMA_IQ_lowcounts",
            "Siemens_Vision600_thorax",
        ]
    elif method == "test":
        data = [
            "GE_D690_NEMA_IQ",
            "GE_DMI4_NEMA_IQ",
            "Mediso_NEMA_IQ_lowcounts",
            "Siemens_Vision600_Hoffman",
            "NeuroLF_Esser_Dataset",
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(
        len(data),
        3,
        figsize=(5.5, 11.0 * len(data) / 7),
        layout="constrained",
    )

    for i, data_set in enumerate(data):
        header_file = Path("data") / data_set / "PETRIC" / "reference_image.hv"
        print(header_file)
        img, hdr, voxsize = load_interfile_image(header_file)

        print("Image shape:", img.shape)
        print("Data type:", img.dtype)
        print("Min/Max values:", img.min(), img.max())

        img_sm = gaussian_filter(img, 3)
        vmax = 1.5 * img_sm.max()

        # sum img over axis 1 and 2 and pick the slice that has the highest sum
        sl0 = np.argmax(img_sm.sum(axis=(1, 2)))
        sl1 = np.argmax(img_sm.sum(axis=(0, 2)))
        sl2 = np.argmax(img_sm.sum(axis=(0, 1)))

        kws = dict(cmap="Greys", vmin=0, vmax=vmax)

        im0 = ax[i, 0].imshow(img[sl0, :, :], **kws)
        asp1 = voxsize[0] / voxsize[2]
        im1 = ax[i, 1].imshow(img[:, sl1, :], aspect=asp1, **kws)
        asp2 = voxsize[0] / voxsize[1]
        im2 = ax[i, 2].imshow(img[:, :, sl2], aspect=asp2, **kws)

        ax[i, 0].set_ylabel(data_set, fontsize="x-small")

        cbar = fig.colorbar(im2, ax=ax[i, 2], location="right", fraction=0.03)
        cbar.ax.tick_params(labelsize="x-small")

        ax[i, 0].set_title(f"z={sl0}", fontsize="small")
        ax[i, 1].set_title(f"y={sl1}", fontsize="small")
        ax[i, 2].set_title(f"x={sl2}", fontsize="small")

    for axx in ax.ravel():
        axx.set_xticks([])
        axx.set_yticks([])

    fig.show()
    fig.savefig(f"petric_data_{method}.png", dpi=300)
