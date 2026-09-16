from io import BytesIO

import PIL.Image


def fig_to_image(fig) -> PIL.Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    return PIL.Image.open(buffer)
