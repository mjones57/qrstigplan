import numpy as np

from qrstigplan.codec.payload import MarkerEllipse, MarkerPayload, PayloadTooLarge
from qrstigplan.codec.qr import fits_version1_l


class LegacyCodec:
    """Uses the original text format from thesis"""

    def encode(self, payload: MarkerPayload) -> str:
        e = payload.ellipse
        values: list[int] = []
        for i, (x, y) in enumerate(payload.nodes.tolist()):
            if e is not None and i == e.f1:
                values.extend([x, y, e.s])
            elif e is not None and i == e.f1 + 1:
                values.extend([x, y])
            else:
                values.extend([x, y, 0])

        text = " ".join(map(str, values))
        if not fits_version1_l(text):
            raise PayloadTooLarge("Payload too large for QR code spec")

        return text

    def decode(self, text: str) -> MarkerPayload:
        values = [int(v) for v in text.split(" ")]
        nodes: list[tuple[int, int]] = []
        ellipse: MarkerEllipse | None = None

        i = 0
        while i < len(values):
            # read values and add the node
            x, y, v = values[i : i + 3]
            nodes.append((x, y))

            # if no ellipse
            if v == 0:
                i += 3
                continue

            # if there is an ellipse, instantiate it
            if ellipse is not None:
                raise ValueError("Ellipse already exists")
            ellipse = MarkerEllipse(f1=len(nodes) - 1, s=v)
            i += 3

            # add f2 node as well
            if i < len(values):
                x, y = values[i : i + 2]
                nodes.append((x, y))
                i += 2

        payload = MarkerPayload(np.array(nodes, dtype=np.int64).reshape(-1, 2), ellipse)

        if self.encode(payload) != text:
            raise ValueError(f"Incorrectly formatted text: {text!r}")

        return payload

    def fits(self, payload: MarkerPayload) -> bool:
        try:
            self.encode(payload)
            return True
        except PayloadTooLarge:
            return False
