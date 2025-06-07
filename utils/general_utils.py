import torch
import torch.nn.functional as F


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] +
        r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


# -- The underlying functions are taken from pytorch3d
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types
            # `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12,
                        m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types
            # `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2,
                        m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types
            # `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]
                        ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand
            # types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12,
                        q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs
    # is small, the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same
    # (up to a sign), forall i; we pick the best-conditioned one
    # (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # Normalize the batch of direction vectors
    if direction_vectors.dim() != 2 or direction_vectors.size(1) != 3:
        raise ValueError(
            "Direction vectors should be a tensor of shape (N, 3)")
    direction_vectors = direction_vectors / torch.norm(
        direction_vectors, dim=-1, keepdim=True
    )
    # Create a batch of arbitrary vectors that are not collinear with the
    # direction vectors
    v1 = (
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        .to(direction_vectors.device)
        .expand(direction_vectors.shape[0], -1)
        .clone()
    )
    # is_collinear = torch.all(torch.abs(
    # direction_vectors - v1) < 1e-5, dim=-1)
    is_collinear = torch.all(
        torch.abs(torch.abs(direction_vectors)) - torch.abs(v1) < 1e-3, dim=-1
    )
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(
        direction_vectors.device
    )
    # Calculate the first orthogonal vectors
    v1 = torch.linalg.cross(direction_vectors, v1)
    # v1 = torch.cross(direction_vectors, v1)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True))
    # Calculate the second orthogonal vectors by taking the cross product
    # v2 = torch.cross(direction_vectors, v1)
    v2 = torch.linalg.cross(direction_vectors, v1)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True))
    # Create the batch of rotation matrices with the direction vectors as
    # the last columns
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices
