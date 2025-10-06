
def normalize_intrinsics(K, H=256, W=256):
    K[..., 0, 0] /= W
    K[..., 1, 1] /= H
    K[..., 0, 2] /= W
    K[..., 1, 2] /= H
    return K