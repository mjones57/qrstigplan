"""Use qrcode to validate if given string fits QR code payload constraints."""

import qrcode
from qrcode.constants import ERROR_CORRECT_L
from qrcode.exceptions import DataOverflowError


def fits_version1_l(text: str) -> bool:
    """
    Test if text fits a Version 1 QR code at error correction level L.
    Return True if it fits.
    """
    qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_L)
    qr.add_data(text)
    try:
        qr.make(fit=False)
        return True
    except DataOverflowError:
        return False
